//! Three front-end stages that need no model at all: a speech-gated leveler,
//! dynamic-range compression and spectral noise reduction.
//!
//! All three exist because of the same room. A meeting is recorded with one
//! microphone on a table, and the person nearest it is 20 dB louder than the
//! person at the far end — so `preprocess::normalize`, which applies a single
//! gain to the whole file, either leaves the far speaker under the noise floor
//! or pushes the near one into the limiter. The leveler and the compressor
//! are the stages that actually fix that, by narrowing the distance between
//! the two before the single gain is applied: the leveler slowly, over whole
//! phrases, and the compressor quickly, over the peaks the leveler is too slow
//! to see.
//!
//! The noise reduction is here for a different reason: the GTCRN denoiser in
//! [`crate::pipeline::denoise`] is *optional*, an extra download, and a
//! shelled-out binary. A recording made on a machine where it was never
//! installed previously got no noise handling beyond an 80 Hz high-pass. This
//! is spectral subtraction against a noise profile estimated from the
//! recording itself — weaker than GTCRN, but it is always available, it needs
//! no model, and it costs one STFT.
//!
//! None of them ever touches `original.wav`; like everything in the front
//! end, they run on a copy on the way to `processed.wav`.

use crate::pipeline::audio_profile::{amplitude_to_dbfs, dbfs_to_amplitude, rms};
use crate::pipeline::spectrum;
use serde::{Deserialize, Serialize};

/// Analysis/synthesis frame for the noise reduction. 512 samples is 32 ms at
/// 16 kHz — long enough to resolve the low end of a voice, short enough that
/// a plosive is not smeared across a syllable.
pub const NR_FRAME_LEN: usize = 512;

/// Ceiling on the compressor's automatic makeup gain. Without it, a recording
/// that is nearly all silence produces an enormous "average reduction" and the
/// makeup lifts the room tone to speech level.
const MAX_MAKEUP_DB: f32 = 12.0;

/// The makeup stage will not push the loudest sample past this. It is a gain
/// stage, not a limiter, and it must never be the thing that introduces
/// clipping.
const MAKEUP_CEILING: f32 = 0.891; // -1 dBFS

// ------------------------------------------------------------ compression

/// A feed-forward compressor with a soft knee, in the usual terms. Every one
/// of these is exposed in Settings because the right amount of compression
/// depends on the room, and too much of it makes a quiet room breathe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorOptions {
    /// Level above which gain reduction starts.
    pub threshold_dbfs: f32,
    /// How much of every dB above the threshold survives: 1.0 is no
    /// compression at all, 4.0 means 4 dB in becomes 1 dB out.
    pub ratio: f32,
    /// Width of the soft transition around the threshold. A hard knee (0)
    /// audibly grabs at the signal; a few dB of knee does not.
    pub knee_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    /// Put back the loudness the reduction took out. Left on, the stage is
    /// level-neutral, which is what keeps it from interacting with whatever
    /// the gain stage after it decides to do.
    pub auto_makeup: bool,
}

impl Default for CompressorOptions {
    fn default() -> Self {
        Self {
            // Deliberately gentle. This is an ASR front end, not a
            // broadcast chain: the goal is to lift the far end of the table
            // above the noise floor, not to flatten the meeting.
            threshold_dbfs: -24.0,
            ratio: 2.5,
            knee_db: 6.0,
            // Slow enough not to chew the attack off a consonant, fast enough
            // to catch a raised voice.
            attack_ms: 5.0,
            release_ms: 120.0,
            auto_makeup: true,
        }
    }
}

/// What the compressor actually did — reported into `metrics.json`, because
/// "the compressor was enabled" and "the compressor did something" are
/// different claims and only the second one is evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CompressorReport {
    pub max_reduction_db: f32,
    /// Averaged over the samples that were actually reduced, not over the
    /// whole file — otherwise the silence between turns drags it to zero.
    pub average_reduction_db: f32,
    pub makeup_db: f32,
    /// Peak-to-RMS distance before and after. Narrowing this is the entire
    /// point of the stage, so it is the number to look at.
    pub crest_before_db: f32,
    pub crest_after_db: f32,
}

fn crest_db(samples: &[f32]) -> f32 {
    let peak = samples.iter().map(|s| s.abs()).fold(0f32, f32::max);
    let level = rms(samples);
    if peak <= 0.0 || level <= 0.0 {
        return 0.0;
    }
    amplitude_to_dbfs(peak) - amplitude_to_dbfs(level)
}

/// Static gain curve: the gain, in dB, that a steady input at `input_db`
/// would end up with. Negative is reduction.
fn static_gain_db(input_db: f32, options: &CompressorOptions) -> f32 {
    let ratio = options.ratio.max(1.0);
    let knee = options.knee_db.max(0.0);
    let over = input_db - options.threshold_dbfs;

    if knee > 0.0 && 2.0 * over.abs() <= knee {
        // Quadratic interpolation across the knee — the standard soft-knee
        // curve, continuous in value and slope at both edges.
        let x = over + knee / 2.0;
        (1.0 / ratio - 1.0) * x * x / (2.0 * knee)
    } else if over > 0.0 {
        over * (1.0 / ratio - 1.0)
    } else {
        0.0
    }
}

/// Compresses in place and reports what it did.
///
/// The envelope is smoothed on the *gain*, not on the input level, which is
/// what gives attack and release their usual meaning: the gain falls quickly
/// when the signal jumps and recovers slowly afterwards.
pub fn compress(
    samples: &mut [f32],
    sample_rate: u32,
    options: &CompressorOptions,
) -> CompressorReport {
    let mut report = CompressorReport {
        crest_before_db: crest_db(samples),
        ..CompressorReport::default()
    };
    if samples.is_empty() || sample_rate == 0 || options.ratio <= 1.0 {
        report.crest_after_db = report.crest_before_db;
        return report;
    }

    let coefficient = |ms: f32| -> f32 {
        let seconds = (ms.max(0.0) / 1000.0) * sample_rate as f32;
        if seconds <= 0.0 {
            0.0
        } else {
            (-1.0 / seconds).exp()
        }
    };
    let attack = coefficient(options.attack_ms);
    let release = coefficient(options.release_ms);

    // Primed from the loudest sample in the opening window rather than
    // started at unity gain. Left at 0 dB the envelope has to attack from
    // scratch, so a recording that opens loud passes its first milliseconds
    // through uncompressed — an unreduced spike at the very start of every
    // meeting, which then sets the peak the gain stage afterwards works
    // around. This is an offline compressor over a file that already exists,
    // so unlike a live one it is allowed to look ahead by a few milliseconds.
    let priming_window = ((sample_rate as f32 * 0.020) as usize).clamp(1, samples.len());
    let opening_peak = samples[..priming_window]
        .iter()
        .map(|s| s.abs())
        .fold(0f32, f32::max);
    let mut envelope_db = static_gain_db(amplitude_to_dbfs(opening_peak), options);
    let mut reduction_sum = 0.0f64;
    let mut reduced_samples = 0u64;
    let mut max_reduction = 0.0f32;

    for sample in samples.iter_mut() {
        let input_db = amplitude_to_dbfs(*sample);
        let target = static_gain_db(input_db, options);
        // More reduction is applied at the attack rate, less at the release
        // rate. Both are one-pole smoothers on the dB value.
        let coefficient = if target < envelope_db { attack } else { release };
        envelope_db = target + coefficient * (envelope_db - target);

        if envelope_db < -0.01 {
            reduction_sum += -envelope_db as f64;
            reduced_samples += 1;
            if -envelope_db > max_reduction {
                max_reduction = -envelope_db;
            }
        }
        *sample *= dbfs_to_amplitude(envelope_db);
    }

    report.max_reduction_db = max_reduction;
    report.average_reduction_db = if reduced_samples == 0 {
        0.0
    } else {
        (reduction_sum / reduced_samples as f64) as f32
    };

    if options.auto_makeup && report.average_reduction_db > 0.0 {
        // Only the reduction that was actually applied is given back, and
        // never more than the headroom left below the ceiling. Clamping the
        // samples instead would make the makeup stage a clipper, which is
        // exactly the distortion the whole front end exists to avoid.
        let peak = samples.iter().map(|s| s.abs()).fold(0f32, f32::max);
        let headroom_db = if peak > 0.0 {
            amplitude_to_dbfs(MAKEUP_CEILING) - amplitude_to_dbfs(peak)
        } else {
            0.0
        };
        let makeup = report
            .average_reduction_db
            .min(MAX_MAKEUP_DB)
            .min(headroom_db.max(0.0));
        if makeup > 0.0 {
            let gain = dbfs_to_amplitude(makeup);
            for sample in samples.iter_mut() {
                *sample = (*sample * gain).clamp(-1.0, 1.0);
            }
            report.makeup_db = makeup;
        }
    }

    report.crest_after_db = crest_db(samples);
    report
}

// --------------------------------------------------------------- leveling

/// The leveler's decision rate. 50 ms is fine enough that a gain change lands
/// within a syllable of where it should, and coarse enough that an hour-long
/// meeting is 72 000 decisions rather than 57 million.
const LEVELER_HOP_MS: f32 = 50.0;

/// Share of windows, from the quietest end, that the noise floor is read
/// from. Lower than the noise reducer's third because this is a *level*, not
/// a spectrum: it only needs the pauses, and a meeting that talks over itself
/// may not have a third of its time in them.
const LEVELER_FLOOR_PERCENTILE: f32 = 0.15;

/// The floor is never taken as lower than this. A recording padded with
/// digital silence would otherwise read its floor at the bottom of the scale
/// and call every breath and chair creak speech.
const LEVELER_MIN_FLOOR_DBFS: f32 = -80.0;

/// A slow, speech-gated gain rider: the stage that brings the far end of the
/// table up to the near end, phrase by phrase.
///
/// The compressor cannot do this on its own. It works on peaks over a few
/// milliseconds, so a far talker who is *consistently* 20 dB down sits under
/// its threshold the whole time and is never touched. The leveler works on
/// the level of whole phrases instead, and only while someone is talking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelerOptions {
    /// Most the leveler will ever lift a window. Every dB of boost is a dB of
    /// that talker's room noise too, so this is the cap on how much hiss a
    /// quiet talker can bring with them.
    pub max_boost_db: f32,
    /// Most it will ever turn a window down.
    pub max_cut_db: f32,
    /// Length of the level measurement. Long enough to span the gaps between
    /// words, so a talker's level is a phrase's and not a syllable's.
    pub window_ms: f32,
    /// How far above the noise floor a window must be to count as speech.
    pub speech_margin_db: f32,
    /// Time constant for the gain going up.
    pub rise_ms: f32,
    /// Time constant for the gain going down.
    pub fall_ms: f32,
    /// How far ahead of a speech onset its gain is already applied. This is
    /// an offline stage over a finished file, so it can see the next talker
    /// coming — which is what keeps a quiet talker's first syllable from
    /// arriving before the boost does.
    pub lookahead_ms: f32,
    /// How long a pause keeps the last talker's gain before it starts to
    /// relax. Shorter than a breath between sentences would make the gain
    /// dip and come back inside one person's turn.
    pub hold_ms: f32,
    /// Time constant for a held gain relaxing back to 0 dB once the pause
    /// outlasts `hold_ms`.
    pub pause_release_ms: f32,
}

impl Default for LevelerOptions {
    fn default() -> Self {
        Self {
            // 12 up and 12 down is 24 dB of range, enough for a 20–26 dB
            // table. The reference sits between the talkers (see `level`),
            // so the spread is split between cutting the near end and lifting
            // the far one rather than all spent on boost — and boost is the
            // half that raises noise.
            max_boost_db: 12.0,
            max_cut_db: 12.0,
            window_ms: 400.0,
            // 10 dB over the floor is a voice; less than that is a cough, a
            // keyboard, or the room itself breathing.
            speech_margin_db: 10.0,
            // Down faster than up: a loud talker who starts after a quiet one
            // must not be let through at the quiet one's boost, while a slow
            // rise is harmless because the lookahead starts it early.
            rise_ms: 250.0,
            fall_ms: 100.0,
            lookahead_ms: 300.0,
            hold_ms: 250.0,
            // Quick enough that a long pause after the far talker is back
            // near its own level within about a second — the pause is where
            // whisper hallucinates, and it is the last place to add gain. It
            // costs nothing between one talker's sentences: the hold and the
            // lookahead meet across any pause shorter than about half a
            // second (nearer a second once the analysis window's own reach
            // is counted), so there the gain never starts to relax at all.
            pause_release_ms: 300.0,
        }
    }
}

/// What the leveler did, for `metrics.json`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LevelerReport {
    pub noise_floor_dbfs: f32,
    /// The level every speech window was steered towards.
    pub reference_dbfs: f32,
    /// Share of windows classified as speech. Near zero means the stage had
    /// nothing to hold on to and left the recording alone.
    pub speech_share: f32,
    /// Averaged over speech windows only, like the compressor's reduction.
    pub mean_speech_gain_db: f32,
    pub max_boost_db: f32,
    pub max_cut_db: f32,
    /// Distance between the loud (90th percentile) and quiet (10th) ends of
    /// the speech windows, before and after. Narrowing this is the entire
    /// point of the stage, so it is the number to look at.
    pub speech_spread_before_db: f32,
    pub speech_spread_after_db: f32,
    pub windows: usize,
}

/// Level in dBFS of a window centred on each hop, over a sliding sum so the
/// whole pass is O(n) however long the window.
fn window_levels_db(samples: &[f32], hop: usize, half_window: usize) -> Vec<f32> {
    let n = samples.len();
    let hops = n.div_ceil(hop);
    let mut levels = Vec::with_capacity(hops);
    let (mut lo, mut hi, mut sum) = (0usize, 0usize, 0.0f64);
    for k in 0..hops {
        let centre = (k * hop + hop / 2).min(n - 1);
        let want_lo = centre.saturating_sub(half_window);
        let want_hi = (centre + half_window).min(n);
        while hi < want_hi {
            sum += (samples[hi] as f64) * (samples[hi] as f64);
            hi += 1;
        }
        while lo < want_lo {
            sum -= (samples[lo] as f64) * (samples[lo] as f64);
            lo += 1;
        }
        let len = (hi - lo).max(1) as f64;
        levels.push(amplitude_to_dbfs((sum.max(0.0) / len).sqrt() as f32));
    }
    levels
}

fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = ((sorted.len() - 1) as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted[at]
}

fn spread_db(levels: &[f32]) -> f32 {
    percentile(levels, 0.9) - percentile(levels, 0.1)
}

/// Levels in place and reports what it did.
///
/// 1. Measure each hop's level over `window_ms`, and read the noise floor
///    from the quietest windows. A window more than `speech_margin_db` over
///    the floor is speech.
/// 2. The reference is the mean, in dB, of the speech windows. It is taken
///    from the recording rather than being an absolute target because the
///    leveler's job is the *distance* between talkers, not the level of the
///    file: an absolute target would spend the whole boost budget lifting a
///    quiet microphone, leave the near/far gap exactly where it was, and do
///    work that normalise does anyway. The mean sits between the talkers, so
///    the correction is shared between cut and boost.
/// 3. Each speech window's target gain is `reference - level`, clamped to the
///    boost and cut caps. A pause never computes a gain of its own — its
///    level is the room's, and steering the room to the reference is exactly
///    how a naive AGC pumps up hiss. It holds the last talker's gain for
///    `hold_ms`, then relaxes towards 0 dB; within `lookahead_ms` of the next
///    speech it takes that speech's gain instead, so the ramp is finished
///    before the first syllable rather than during it.
/// 4. The gain curve is smoothed in dB — down at `fall_ms`, up at `rise_ms` —
///    capped so no sample it touches is lifted past the -1 dBFS ceiling the
///    makeup stage uses, and interpolated per sample so there is no zipper.
///
/// Sorting the per-hop levels for the percentiles is O(h log h) in hops, one
/// per 50 ms — noise next to the O(n) passes over the samples.
pub fn level(samples: &mut [f32], sample_rate: u32, options: &LevelerOptions) -> LevelerReport {
    let mut report = LevelerReport::default();
    if samples.is_empty() || sample_rate == 0 {
        return report;
    }

    let n = samples.len();
    let hop = ((sample_rate as f32 * LEVELER_HOP_MS / 1000.0) as usize).max(1);
    let half_window =
        ((sample_rate as f32 * options.window_ms.max(0.0) / 2000.0) as usize).max(hop / 2).max(1);
    let levels = window_levels_db(samples, hop, half_window);
    let hops = levels.len();
    report.windows = hops;

    let floor = percentile(&levels, LEVELER_FLOOR_PERCENTILE).max(LEVELER_MIN_FLOOR_DBFS);
    report.noise_floor_dbfs = floor;
    let gate = floor + options.speech_margin_db.max(0.0);
    let speech: Vec<bool> = levels.iter().map(|&l| l > gate).collect();
    let speech_levels: Vec<f32> = levels
        .iter()
        .zip(&speech)
        .filter(|(_, &is)| is)
        .map(|(&l, _)| l)
        .collect();
    if speech_levels.is_empty() {
        // Silence, or a recording with no pauses to tell the room from the
        // talker. Either way there is nothing to steer by.
        return report;
    }
    let reference = speech_levels.iter().sum::<f32>() / speech_levels.len() as f32;
    report.reference_dbfs = reference;
    report.speech_share = speech_levels.len() as f32 / hops as f32;
    report.speech_spread_before_db = spread_db(&speech_levels);

    let max_boost = options.max_boost_db.max(0.0);
    let max_cut = options.max_cut_db.max(0.0);
    let hops_for = |ms: f32| (ms.max(0.0) / LEVELER_HOP_MS).round() as usize;
    let coefficient = |ms: f32| if ms <= 0.0 { 0.0 } else { (-LEVELER_HOP_MS / ms).exp() };

    // Forward: speech sets the gain, pauses hold it and then let it go.
    let hold = hops_for(options.hold_ms);
    let relax = coefficient(options.pause_release_ms);
    let mut target = vec![0.0f32; hops];
    let mut last = 0.0f32;
    let mut since_speech = usize::MAX;
    for k in 0..hops {
        if speech[k] {
            last = (reference - levels[k]).clamp(-max_cut, max_boost);
            since_speech = 0;
        } else {
            since_speech = since_speech.saturating_add(1);
            if since_speech > hold {
                last *= relax;
            }
        }
        target[k] = last;
    }
    // Backward: the pause just before speech already carries its gain.
    let lookahead = hops_for(options.lookahead_ms);
    let mut next: Option<(usize, f32)> = None;
    for k in (0..hops).rev() {
        if speech[k] {
            next = Some((k, target[k]));
        } else if let Some((at, gain)) = next {
            if at - k <= lookahead {
                target[k] = gain;
            }
        }
    }

    // Smoothed in dB, primed from the first target rather than from 0 dB,
    // for the same reason the compressor is.
    let rise = coefficient(options.rise_ms);
    let fall = coefficient(options.fall_ms);
    let mut gains = Vec::with_capacity(hops);
    let mut gain = target[0];
    for &want in &target {
        let c = if want > gain { rise } else { fall };
        gain = want + c * (gain - want);
        gains.push(gain);
    }

    // Each sample is interpolated between the two hop centres around it, so
    // its gain is at most the larger of those two. Capping every hop by the
    // loudest sample it can reach therefore caps every sample.
    let centre = |k: usize| (k * hop + hop / 2).min(n - 1);
    let bracket = |i: usize| -> (usize, usize) {
        if i < hop / 2 {
            return (0, 0);
        }
        let left = ((i - hop / 2) / hop).min(hops - 1);
        (left, (left + 1).min(hops - 1))
    };
    let mut reach_peak = vec![0.0f32; hops];
    for (i, sample) in samples.iter().enumerate() {
        let (left, right) = bracket(i);
        let a = sample.abs();
        reach_peak[left] = reach_peak[left].max(a);
        reach_peak[right] = reach_peak[right].max(a);
    }
    let ceiling_db = amplitude_to_dbfs(MAKEUP_CEILING);
    for (gain, &peak) in gains.iter_mut().zip(&reach_peak) {
        // Only a boost is capped. A sample already over the ceiling is not
        // the leveler's to fix, and cutting is always safe.
        if *gain > 0.0 && peak > 0.0 {
            *gain = gain.min((ceiling_db - amplitude_to_dbfs(peak)).max(0.0));
        }
    }

    for (i, sample) in samples.iter_mut().enumerate() {
        let (left, right) = bracket(i);
        let g = if left == right {
            gains[left]
        } else {
            let (c0, c1) = (centre(left), centre(right));
            let t = if c1 > c0 {
                ((i.saturating_sub(c0)) as f32 / (c1 - c0) as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            gains[left] + (gains[right] - gains[left]) * t
        };
        *sample *= dbfs_to_amplitude(g);
    }

    let mut gain_sum = 0.0f32;
    for (k, &g) in gains.iter().enumerate() {
        if speech[k] {
            gain_sum += g;
        }
        report.max_boost_db = report.max_boost_db.max(g);
        report.max_cut_db = report.max_cut_db.max(-g);
    }
    report.mean_speech_gain_db = gain_sum / speech_levels.len() as f32;

    let after: Vec<f32> = window_levels_db(samples, hop, half_window)
        .into_iter()
        .zip(&speech)
        .filter(|(_, &is)| is)
        .map(|(l, _)| l)
        .collect();
    report.speech_spread_after_db = spread_db(&after);
    report
}

// -------------------------------------------------------- noise reduction

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseReductionOptions {
    /// 0.0 does nothing, 1.0 subtracts as hard as this is willing to.
    ///
    /// It maps onto two things at once: how much of the estimated noise is
    /// subtracted, and how far a bin is allowed to fall. Turning one knob
    /// rather than three is deliberate — the failure mode of spectral
    /// subtraction is "musical noise", warbling artefacts that appear when a
    /// user pushes an over-subtraction factor they have no way to judge.
    pub strength: f32,
}

impl Default for NoiseReductionOptions {
    fn default() -> Self {
        Self { strength: 0.5 }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct NoiseReductionReport {
    /// How far the estimated noise floor dropped, in dB. This is the claim
    /// the stage is making, and it is measured rather than assumed.
    pub noise_floor_delta_db: f32,
    /// Speech-band level before and after, so a run that quietened the *voice*
    /// rather than the room is visible instead of hidden.
    pub speech_level_delta_db: f32,
    pub frames: usize,
}

/// Share of frames, taken from the quietest end, that are treated as "the
/// room" when building the noise profile. A meeting has pauses everywhere —
/// between turns, between sentences, between words — so the quietest third of
/// its frames is a good estimate of what the room sounds like with nobody
/// talking.
const QUIET_FRAME_SHARE: f32 = 0.30;

/// Per-bin noise magnitude, averaged over the quietest frames of the
/// recording.
///
/// Selecting frames by their broadband energy — rather than taking a low
/// percentile within each bin independently — matters more than it looks.
/// A per-bin percentile has no notion of *when* a bin was quiet, so a
/// component that is present the whole time at a steady level is indeed
/// classified as noise. That is right for a fan and wrong for a voice, and
/// the two are only distinguishable by the fact that speech stops.
///
/// The limitation this leaves is worth stating rather than discovering: a
/// perfectly stationary signal cannot be separated from the room by any
/// method that defines the room as "what is there when nothing is happening".
/// A recording that is one continuous unbroken tone will have that tone
/// subtracted. Meetings are not that.
///
/// No VAD model is involved, which is the point — this stage has to work on a
/// machine where nothing optional was ever installed.
fn noise_profile(samples: &[f32], frame_len: usize, max_frames: usize) -> Vec<f32> {
    let n = frame_len.next_power_of_two().max(4);
    let hop = n / 2;
    if samples.len() < n {
        return Vec::new();
    }
    let total_frames = (samples.len() - n) / hop + 1;
    let stride = (total_frames / max_frames.max(1)).max(1);
    let indices: Vec<usize> = (0..total_frames).step_by(stride).collect();
    if indices.is_empty() {
        return Vec::new();
    }

    // Pass one is cheap: frame energies only, no transform.
    let mut energies: Vec<f32> = indices
        .iter()
        .map(|&index| rms(&samples[index * hop..index * hop + n]))
        .collect();
    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff_at = ((sorted.len() as f32 * QUIET_FRAME_SHARE) as usize).clamp(1, sorted.len()) - 1;
    let cutoff = sorted[cutoff_at];

    // Pass two transforms only the frames that survived.
    let window: Vec<f32> = spectrum::hann(n).into_iter().map(|w| w.sqrt()).collect();
    let bins = n / 2 + 1;
    let mut accumulator = vec![0.0f32; bins];
    let mut kept = 0usize;
    for (slot, &index) in indices.iter().enumerate() {
        if energies[slot] > cutoff {
            continue;
        }
        let frame = &samples[index * hop..index * hop + n];
        for (bin, magnitude) in spectrum::magnitudes(frame, &window).iter().enumerate() {
            accumulator[bin] += magnitude;
        }
        kept += 1;
    }
    energies.clear();

    if kept == 0 {
        return Vec::new();
    }
    for slot in accumulator.iter_mut() {
        *slot /= kept as f32;
    }
    accumulator
}

/// Level of the noise floor and of the speech band, in dB, from a magnitude
/// spectrum estimate. Used only to report what the stage achieved.
fn floor_and_speech_db(samples: &[f32], sample_rate: u32) -> (f32, f32) {
    let spectrum = spectrum::average_spectrum(samples, sample_rate, 512, 400, |_, _| true);
    if spectrum.frames == 0 {
        return (amplitude_to_dbfs(0.0), amplitude_to_dbfs(0.0));
    }
    // Above the voice band is where a room's broadband noise is unmixed with
    // speech; 300 Hz to 3.4 kHz is the voice band itself.
    let floor = spectrum.band_energy(5_000.0, spectrum.nyquist_hz());
    let speech = spectrum.band_energy(300.0, 3_400.0);
    (amplitude_to_dbfs(floor), amplitude_to_dbfs(speech))
}

/// Spectral subtraction against a self-estimated noise profile.
///
/// Returns the enhanced signal; the input is left untouched so a caller can
/// compare the two, and so a disappointing result costs nothing.
pub fn reduce_noise(
    samples: &[f32],
    sample_rate: u32,
    options: &NoiseReductionOptions,
) -> (Vec<f32>, NoiseReductionReport) {
    let strength = options.strength.clamp(0.0, 1.0);
    if strength <= 0.0 || samples.len() < NR_FRAME_LEN * 4 {
        return (samples.to_vec(), NoiseReductionReport::default());
    }

    let profile = noise_profile(samples, NR_FRAME_LEN, 2_000);
    if profile.is_empty() {
        return (samples.to_vec(), NoiseReductionReport::default());
    }

    // Over-subtraction between 1.0 and 2.5, and a spectral floor between 0.25
    // and 0.05. Keeping a floor rather than zeroing a bin is what stops the
    // residual from warbling: the noise is pushed down, not punched out.
    let over_subtraction = 1.0 + 1.5 * strength;
    let spectral_floor = 0.25 - 0.20 * strength;

    let (floor_before, speech_before) = floor_and_speech_db(samples, sample_rate);
    let mut frames = 0usize;
    let mut gains: Vec<f32> = Vec::new();

    let out = spectrum::stft_process(samples, NR_FRAME_LEN, |magnitudes, _| {
        frames += 1;
        gains.clear();
        gains.reserve(magnitudes.len());
        for (bin, magnitude) in magnitudes.iter().enumerate() {
            let noise = profile.get(bin).copied().unwrap_or(0.0) * over_subtraction;
            let gain = if *magnitude <= 1e-9 {
                1.0
            } else {
                ((*magnitude - noise) / *magnitude).max(spectral_floor)
            };
            gains.push(gain.clamp(spectral_floor, 1.0));
        }
        // Smooth the gain across neighbouring bins. An unsmoothed mask is
        // where musical noise comes from: isolated bins flipping between
        // fully open and fully closed frame to frame.
        for (bin, magnitude) in magnitudes.iter_mut().enumerate() {
            let lo = bin.saturating_sub(1);
            let hi = (bin + 1).min(gains.len() - 1);
            let smoothed = (gains[lo] + gains[bin] + gains[hi]) / 3.0;
            *magnitude *= smoothed;
        }
    });

    let (floor_after, speech_after) = floor_and_speech_db(&out, sample_rate);
    let report = NoiseReductionReport {
        noise_floor_delta_db: floor_before - floor_after,
        speech_level_delta_db: speech_before - speech_after,
        frames,
    };
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sample_rate: u32, seconds: f32, freq: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    fn noise(n: usize, amplitude: f32) -> Vec<f32> {
        let mut state = 987_654_321u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((state >> 16) as f32 / 32_768.0) - 1.0) * amplitude
            })
            .collect()
    }

    /// A near speaker then a far one, the case the compressor exists for:
    /// 26 dB apart, which is an ordinary table microphone in an ordinary
    /// meeting room.
    fn two_speakers(sample_rate: u32) -> Vec<f32> {
        let mut samples = tone(sample_rate, 1.0, 220.0, 0.6);
        samples.extend(tone(sample_rate, 1.0, 220.0, 0.03));
        samples
    }

    /// Distance in dB between the loud half and the quiet half.
    fn speaker_gap_db(samples: &[f32]) -> f32 {
        let half = samples.len() / 2;
        amplitude_to_dbfs(rms(&samples[..half])) - amplitude_to_dbfs(rms(&samples[half..]))
    }

    /// Speech that stops, which is what makes a meeting separable from its
    /// room: four bursts with silence between them, under a constant hiss.
    fn intermittent_speech(sample_rate: u32, hiss_amplitude: f32) -> Vec<f32> {
        let mut samples = Vec::new();
        for _ in 0..4 {
            samples.extend(tone(sample_rate, 0.25, 400.0, 0.35));
            samples.extend(vec![0.0f32; sample_rate as usize / 4]);
        }
        let hiss = noise(samples.len(), hiss_amplitude);
        samples
            .iter()
            .zip(hiss.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    #[test]
    fn the_static_curve_is_flat_below_the_knee_and_sloped_above_it() {
        let options = CompressorOptions {
            threshold_dbfs: -20.0,
            ratio: 4.0,
            knee_db: 0.0,
            ..CompressorOptions::default()
        };
        assert_eq!(static_gain_db(-40.0, &options), 0.0);
        assert_eq!(static_gain_db(-20.0, &options), 0.0);
        // 12 dB over a 4:1 threshold comes out 3 dB over: 9 dB of reduction.
        assert!((static_gain_db(-8.0, &options) + 9.0).abs() < 1e-3);
    }

    #[test]
    fn a_soft_knee_starts_reducing_before_the_threshold() {
        let options = CompressorOptions {
            threshold_dbfs: -20.0,
            ratio: 4.0,
            knee_db: 8.0,
            ..CompressorOptions::default()
        };
        // Inside the lower half of the knee: some reduction, but less than
        // the hard-knee curve would give at the same point.
        let inside = static_gain_db(-22.0, &options);
        assert!(inside < 0.0 && inside > -1.0, "knee gain was {inside}");
        // Below the knee entirely: untouched.
        assert_eq!(static_gain_db(-30.0, &options), 0.0);
        // The curve is continuous where the knee ends.
        let at_edge = static_gain_db(-16.0, &options);
        let just_past = static_gain_db(-15.9, &options);
        assert!((at_edge - just_past).abs() < 0.1);
    }

    #[test]
    fn compression_narrows_the_gap_between_a_near_and_a_far_speaker() {
        let sr = 16_000;
        let mut samples = two_speakers(sr);
        let before = speaker_gap_db(&samples);
        let report = compress(&mut samples, sr, &CompressorOptions::default());
        let after = speaker_gap_db(&samples);

        assert!(report.max_reduction_db > 3.0, "{report:?}");
        assert!(before > 25.0, "the fixture should start 26 dB apart, got {before}");
        // The whole point of the stage: the far speaker is closer to the near
        // one afterwards, which is what puts them both in whisper's range.
        assert!(
            after < before - 8.0,
            "gap only closed from {before} dB to {after} dB"
        );
    }

    #[test]
    fn the_opening_of_a_recording_is_compressed_like_the_rest_of_it() {
        let sr = 16_000;
        // Starts loud, with no run-up. An envelope starting at unity gain
        // would let the first milliseconds through untouched.
        let mut samples = tone(sr, 1.0, 220.0, 0.8);
        compress(&mut samples, sr, &CompressorOptions::default());
        let opening = samples[..160].iter().map(|s| s.abs()).fold(0f32, f32::max);
        let settled = samples[8_000..8_160]
            .iter()
            .map(|s| s.abs())
            .fold(0f32, f32::max);
        assert!(
            opening <= settled * 1.2,
            "opening peaked at {opening} against a settled {settled}"
        );
    }

    #[test]
    fn a_ratio_of_one_is_a_no_op() {
        let sr = 16_000;
        let original = two_speakers(sr);
        let mut samples = original.clone();
        let report = compress(
            &mut samples,
            sr,
            &CompressorOptions {
                ratio: 1.0,
                ..CompressorOptions::default()
            },
        );
        assert_eq!(samples, original);
        assert_eq!(report.max_reduction_db, 0.0);
        assert_eq!(report.makeup_db, 0.0);
    }

    #[test]
    fn quiet_audio_below_the_threshold_is_left_alone() {
        let sr = 16_000;
        let original = tone(sr, 0.5, 300.0, 0.01); // -40 dBFS peak
        let mut samples = original.clone();
        let report = compress(&mut samples, sr, &CompressorOptions::default());
        assert_eq!(report.max_reduction_db, 0.0);
        assert_eq!(report.average_reduction_db, 0.0);
        for (a, b) in original.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn auto_makeup_puts_the_level_back_and_can_be_turned_off() {
        let sr = 16_000;
        let mut with_makeup = two_speakers(sr);
        let loud = compress(&mut with_makeup, sr, &CompressorOptions::default());
        assert!(loud.makeup_db > 0.0);

        let mut without = two_speakers(sr);
        let quiet = compress(
            &mut without,
            sr,
            &CompressorOptions {
                auto_makeup: false,
                ..CompressorOptions::default()
            },
        );
        assert_eq!(quiet.makeup_db, 0.0);
        assert!(rms(&with_makeup) > rms(&without));
    }

    #[test]
    fn makeup_gain_never_exceeds_its_cap_or_clips() {
        let sr = 16_000;
        // Brutal settings on a hot signal: the reduction is huge, so the
        // makeup would run away without the cap.
        let mut samples = tone(sr, 1.0, 220.0, 0.95);
        let report = compress(
            &mut samples,
            sr,
            &CompressorOptions {
                threshold_dbfs: -60.0,
                ratio: 20.0,
                knee_db: 0.0,
                ..CompressorOptions::default()
            },
        );
        assert!(report.makeup_db <= MAX_MAKEUP_DB);
        assert!(samples.iter().all(|s| s.abs() <= 1.0));
    }

    #[test]
    fn compressing_silence_reports_nothing_and_changes_nothing() {
        let mut samples = vec![0.0f32; 1_000];
        let report = compress(&mut samples, 16_000, &CompressorOptions::default());
        assert_eq!(report, CompressorReport::default());
        assert!(samples.iter().all(|&s| s == 0.0));
        // And an empty buffer is not a panic.
        let mut empty: Vec<f32> = Vec::new();
        compress(&mut empty, 16_000, &CompressorOptions::default());
    }

    /// A meeting table as the leveler sees it: near, pause, far, pause, near,
    /// pause, far — the far talker 20 dB down, and a low hiss under all of it
    /// so the pauses have a floor to measure. Returns the samples and the
    /// (start, end) of every talker and every pause.
    fn table(sample_rate: u32) -> (Vec<f32>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let talk = 2 * sample_rate as usize;
        let pause = 3 * sample_rate as usize;
        let mut samples = Vec::new();
        let (mut talkers, mut pauses) = (Vec::new(), Vec::new());
        for turn in 0..4 {
            let amplitude = if turn % 2 == 0 { 0.3 } else { 0.03 };
            let start = samples.len();
            samples.extend(tone(sample_rate, 2.0, 220.0, amplitude));
            talkers.push((start, start + talk));
            if turn < 3 {
                pauses.push((samples.len(), samples.len() + pause));
                samples.extend(vec![0.0f32; pause]);
            }
        }
        let hiss = noise(samples.len(), 0.002);
        for (s, h) in samples.iter_mut().zip(hiss) {
            *s += h;
        }
        (samples, talkers, pauses)
    }

    fn level_db(samples: &[f32], (start, end): (usize, usize)) -> f32 {
        amplitude_to_dbfs(rms(&samples[start..end]))
    }

    #[test]
    fn leveling_brings_the_far_end_of_the_table_up_to_the_near_end() {
        let sr = 16_000;
        let (mut samples, talkers, _) = table(sr);
        let before = level_db(&samples, talkers[0]) - level_db(&samples, talkers[1]);
        let report = level(&mut samples, sr, &LevelerOptions::default());
        let after = level_db(&samples, talkers[0]) - level_db(&samples, talkers[1]);

        assert!(before > 19.0, "the fixture should start 20 dB apart, got {before}");
        assert!(after < 8.0, "gap only closed from {before} dB to {after} dB");
        // The second pair too: the gain follows the talkers, it does not
        // just land on one setting for the whole file.
        let later = level_db(&samples, talkers[2]) - level_db(&samples, talkers[3]);
        assert!(later < 8.0, "second exchange still {later} dB apart");
        assert!(report.speech_spread_after_db < report.speech_spread_before_db - 8.0, "{report:?}");
        assert!(report.speech_share > 0.2 && report.speech_share < 0.8, "{report:?}");
    }

    #[test]
    fn leveling_does_not_pump_up_the_room_in_the_pauses() {
        let sr = 16_000;
        let (original, _, pauses) = table(sr);
        let mut samples = original.clone();
        level(&mut samples, sr, &LevelerOptions::default());

        // The middle of each pause, away from the hold after one talker and
        // the lookahead before the next: that is the room on its own, and
        // the leveler must not be steering it towards speech level.
        let margin = sr as usize / 2;
        for &(start, end) in &pauses {
            let middle = (start + margin, end - margin);
            let raised = level_db(&samples, middle) - level_db(&original, middle);
            assert!(
                raised < 4.0,
                "pause at {start} was raised by {raised} dB against a boost cap of 12"
            );
        }
    }

    #[test]
    fn leveling_never_boosts_past_its_cap() {
        let sr = 16_000;
        // 34 dB down: more than the cap can close, so the cap has to hold.
        let mut samples = tone(sr, 2.0, 220.0, 0.5);
        samples.extend(vec![0.0f32; 3 * sr as usize]);
        samples.extend(tone(sr, 2.0, 220.0, 0.01));
        samples.extend(vec![0.0f32; 3 * sr as usize]);
        let hiss = noise(samples.len(), 0.0005);
        for (s, h) in samples.iter_mut().zip(hiss) {
            *s += h;
        }
        let original = samples.clone();
        let options = LevelerOptions::default();
        let report = level(&mut samples, sr, &options);

        assert!(report.max_boost_db <= options.max_boost_db + 1e-4, "{report:?}");
        assert!(report.max_cut_db <= options.max_cut_db + 1e-4, "{report:?}");
        let cap = dbfs_to_amplitude(options.max_boost_db) * 1.0001;
        for (a, b) in original.iter().zip(&samples) {
            assert!(b.abs() <= a.abs() * cap + 1e-7);
        }
    }

    #[test]
    fn leveling_never_lifts_a_sample_past_the_ceiling() {
        let sr = 16_000;
        // A near talker right up at the ceiling, then a far one whose boost
        // would carry its occasional transient straight into clipping.
        let mut samples = tone(sr, 2.0, 220.0, 0.85);
        samples.extend(vec![0.0f32; 2 * sr as usize]);
        let quiet_start = samples.len();
        samples.extend(tone(sr, 2.0, 220.0, 0.02));
        for i in (quiet_start..samples.len()).step_by(1_000) {
            samples[i] = 0.6;
        }
        samples.extend(vec![0.0f32; 2 * sr as usize]);
        let hiss = noise(samples.len(), 0.0005);
        for (s, h) in samples.iter_mut().zip(hiss) {
            *s += h;
        }
        let report = level(&mut samples, sr, &LevelerOptions::default());
        assert!(report.speech_share > 0.0);
        let peak = samples.iter().map(|s| s.abs()).fold(0f32, f32::max);
        assert!(peak <= MAKEUP_CEILING + 1e-4, "peak reached {peak}");
    }

    #[test]
    fn leveling_silence_or_nothing_changes_nothing() {
        let mut silence = vec![0.0f32; 16_000];
        let report = level(&mut silence, 16_000, &LevelerOptions::default());
        assert!(silence.iter().all(|&s| s == 0.0));
        assert_eq!(report.speech_share, 0.0);
        assert_eq!(report.max_boost_db, 0.0);

        let mut empty: Vec<f32> = Vec::new();
        let report = level(&mut empty, 16_000, &LevelerOptions::default());
        assert_eq!(report, LevelerReport::default());
    }

    #[test]
    fn a_recording_with_no_pauses_is_left_alone() {
        // No pause means no floor, which means no way to tell a talker from
        // the room — so the leveler does nothing rather than guess.
        let original = tone(16_000, 2.0, 300.0, 0.2);
        let mut samples = original.clone();
        let report = level(&mut samples, 16_000, &LevelerOptions::default());
        assert_eq!(samples, original);
        assert_eq!(report.speech_share, 0.0);
    }

    #[test]
    fn noise_reduction_lowers_the_floor_of_a_noisy_recording() {
        let sr = 16_000;
        let noisy = intermittent_speech(sr, 0.05);

        let (cleaned, report) = reduce_noise(&noisy, sr, &NoiseReductionOptions::default());
        assert_eq!(cleaned.len(), noisy.len());
        assert!(report.frames > 0);
        assert!(
            report.noise_floor_delta_db > 3.0,
            "floor only moved {} dB",
            report.noise_floor_delta_db
        );
        // And the voice survived. The floor number alone would look just as
        // good if the stage had simply turned the whole recording down, so
        // this is the assertion that makes the other one mean anything.
        assert!(
            report.speech_level_delta_db < report.noise_floor_delta_db,
            "speech fell {} dB against a floor drop of {} dB",
            report.speech_level_delta_db,
            report.noise_floor_delta_db
        );
    }

    #[test]
    fn a_stronger_setting_removes_more_than_a_weaker_one() {
        let sr = 16_000;
        let noisy = intermittent_speech(sr, 0.05);
        let (_, gentle) = reduce_noise(&noisy, sr, &NoiseReductionOptions { strength: 0.2 });
        let (_, hard) = reduce_noise(&noisy, sr, &NoiseReductionOptions { strength: 1.0 });
        assert!(hard.noise_floor_delta_db > gentle.noise_floor_delta_db);
    }

    #[test]
    fn the_noise_profile_is_built_from_the_pauses_not_the_speech() {
        let sr = 16_000;
        let noisy = intermittent_speech(sr, 0.05);
        let profile = noise_profile(&noisy, NR_FRAME_LEN, 2_000);
        assert!(!profile.is_empty());

        // The 400 Hz speech bin must not be sitting up in the profile: if it
        // is, the estimator has decided the voice is part of the room and
        // will subtract it. 16 kHz over 512 is 31.25 Hz per bin, so 400 Hz is
        // bin 13 — compared here against a neighbouring bin with only hiss.
        let speech_bin = profile[13];
        let quiet_bin = profile[40];
        assert!(
            speech_bin < quiet_bin * 3.0,
            "the profile absorbed the voice: {speech_bin} vs {quiet_bin}"
        );
    }

    #[test]
    fn zero_strength_returns_the_input_untouched() {
        let sr = 16_000;
        let samples = tone(sr, 1.0, 400.0, 0.3);
        let (out, report) = reduce_noise(&samples, sr, &NoiseReductionOptions { strength: 0.0 });
        assert_eq!(out, samples);
        assert_eq!(report, NoiseReductionReport::default());
    }

    #[test]
    fn audio_too_short_to_profile_is_returned_as_it_came() {
        let samples = vec![0.2f32; 100];
        let (out, report) = reduce_noise(&samples, 16_000, &NoiseReductionOptions::default());
        assert_eq!(out, samples);
        assert_eq!(report.frames, 0);
    }
}
