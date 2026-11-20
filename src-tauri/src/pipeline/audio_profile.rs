//! Measuring how good a recording actually is, in terms a user can act on.
//!
//! The raw-artifact rule keeps `original.wav` untouched, which means the only lever a user has
//! over audio quality is the room and the microphone — and they only get that
//! lever if someone tells them, in plain words, what is wrong. Everything here
//! is pure analysis: it reads samples and produces numbers plus advice, and
//! never writes audio.
//!
//! The numbers are true dBFS (`20*log10(amplitude)`), not the 0..1 linear
//! scale the live meter used to expose — a linear meter reads "half" at
//! -6 dBFS and makes a badly quiet recording look merely quiet.

use crate::pipeline::noise_class::{self, NoiseReport};
use crate::pipeline::spectrum;
use hound::{SampleFormat, WavReader};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Floor for the dBFS conversion. Digital silence is `-inf` dB, which
/// serialises to `null` in JSON and poisons every average it touches, so it is
/// clamped to a value far below anything a real microphone produces.
pub const SILENCE_DBFS: f32 = -100.0;

/// Analysis frame length. 20 ms is short enough to sit inside a pause between
/// words (which is what makes the noise-floor percentile meaningful) and long
/// enough that one frame's RMS is a stable number at 16 kHz and up.
const FRAME_MS: usize = 20;

/// A sample this close to full scale was almost certainly limited by the ADC.
const CLIP_THRESHOLD: f32 = 0.999;

/// Frames within this much of the noise floor are counted as silence.
const SILENCE_MARGIN_DB: f32 = 3.0;

const QUIET_PEAK_DBFS: f32 = -30.0;
const CLIP_PERCENT_LIMIT: f32 = 0.1;
const NOISY_SNR_DB: f32 = 15.0;
const DC_OFFSET_LIMIT: f32 = 0.01;
const MOSTLY_SILENT_PERCENT: f32 = 80.0;
const MIN_USEFUL_SAMPLE_RATE: u32 = 16_000;

/// The telephone band, and near enough the band a speech recogniser actually
/// works in. Below it is room, above it is sibilance and hiss.
const SPEECH_BAND_LO_HZ: f32 = 300.0;
const SPEECH_BAND_HI_HZ: f32 = 3_400.0;

/// Analysis frame for the speech-band SNR. 1024 samples is 64 ms at 16 kHz,
/// giving 15.6 Hz bins — plenty to resolve a 300 Hz band edge, and short
/// enough that a frame is either speech or pause rather than half of each.
const SNR_FRAME: usize = 1_024;

/// Cap on frames per spectrum. Two spectra are averaged (the room and the
/// speech), so this is at most 600 FFTs for a recording of any length —
/// a 60-minute meeting is ~57 million samples and must not cost 56,000 of
/// them.
const MAX_SNR_FRAMES: usize = 300;

/// Frames within this much of the 90th-percentile level are treated as
/// speech when building the speech spectrum.
const SPEECH_MARGIN_DB: f32 = 3.0;

/// Speech-band SNR at or above which nothing is deducted from the score. Below
/// roughly this, word error rate starts climbing on every ASR system there is.
const GOOD_SPEECH_SNR_DB: f32 = 20.0;

/// The level strip the UI draws. Fixed so a 10-minute meeting and a 3-hour one
/// render identically — the strip is a shape, not a ruler, and 120 buckets is
/// about as fine as a few hundred pixels can show.
pub const MAX_TIMELINE_BUCKETS: usize = 120;

/// Below this, a bucket stops describing a passage of the meeting and starts
/// describing a syllable, so a short recording gets fewer, wider buckets
/// rather than 120 meaningless ones.
const MIN_TIMELINE_BUCKET_MS: i64 = 250;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProfile {
    pub duration_ms: i64,
    pub sample_rate: u32,
    pub channels: u16,
    pub peak_dbfs: f32,
    pub rms_dbfs: f32,
    /// 10th percentile of frame RMS — a far better estimate than the global
    /// minimum, which any single dropped sample drives to digital silence.
    pub noise_floor_dbfs: f32,
    /// 90th percentile of frame RMS — the level of the recording *while
    /// someone was talking*, which is the number the overall RMS is not: that
    /// one is dragged down by every pause in the meeting.
    pub speech_dbfs: f32,
    pub estimated_snr_db: f32,
    /// `peak_dbfs - rms_dbfs`. Wide means the loud moments tower over the
    /// quiet ones — one person close to the microphone and the rest of the
    /// table far from it — which is what the compressor exists to narrow.
    pub crest_factor_db: f32,
    /// Speech-to-noise measured **only across 300–3400 Hz**, which predicts
    /// transcription accuracy far better than the broadband
    /// [`AudioProfile::estimated_snr_db`] beside it.
    ///
    /// A recording can have an excellent broadband SNR and still be unusable:
    /// if the energy separating "speech" from "room" is all air-conditioning
    /// rumble at 40 Hz or preamp hiss at 12 kHz, the broadband number is large
    /// and the recogniser — which only ever looks at the voice band — sees
    /// nothing but noise. Restricting the measurement to the band the model
    /// actually reads is what makes it honest.
    pub speech_band_snr_db: f32,
    pub clipped_samples: u64,
    pub clipped_percent: f32,
    pub silence_percent: f32,
    pub dc_offset: f32,
    /// What the background noise actually *is*, and what to do about it.
    pub noise: NoiseReport,
    /// 0..=100. See [`quality_score`] for exactly what is deducted and why.
    pub quality_score: u8,
    /// "excellent" | "good" | "fair" | "poor" — [`quality_score`] bucketed,
    /// so the UI never has to invent its own thresholds.
    pub quality_grade: String,
    /// At most [`MAX_TIMELINE_BUCKETS`] evenly spread buckets, so the UI can
    /// draw a level strip for a recording of any length the same way.
    pub timeline: Vec<ProfileFrame>,
    /// Plain-language advice for the UI. Empty means nothing to report.
    pub warnings: Vec<String>,
}

/// One bucket of the level strip. Deliberately three numbers and a timestamp:
/// enough to draw a waveform-ish overview and to mark where the clipping was,
/// small enough that 120 of them cost nothing to serialise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ProfileFrame {
    pub start_ms: i64,
    pub rms_dbfs: f32,
    pub peak_dbfs: f32,
    pub clipped_percent: f32,
}

impl AudioProfile {
    /// True when nothing in the recording would hurt transcription.
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

/// `20*log10(amplitude)`, with digital silence and any non-finite input
/// folded onto [`SILENCE_DBFS`] instead of `-inf` / `NaN`.
pub fn amplitude_to_dbfs(amplitude: f32) -> f32 {
    let a = amplitude.abs();
    if !a.is_finite() || a <= 0.0 {
        return SILENCE_DBFS;
    }
    (20.0 * a.log10()).max(SILENCE_DBFS)
}

pub fn dbfs_to_amplitude(dbfs: f32) -> f32 {
    if dbfs <= SILENCE_DBFS {
        return 0.0;
    }
    10f32.powf(dbfs / 20.0)
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Linear-interpolated percentile of an already-sorted slice. `p` is 0..1.
fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return SILENCE_DBFS;
    }
    let pos = (p.clamp(0.0, 1.0) as f64) * (sorted.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let frac = (pos - lower as f64) as f32;
    sorted[lower] + (sorted[upper] - sorted[lower]) * frac
}

/// Per-frame RMS in dBFS, one entry per [`FRAME_MS`] window.
fn frame_levels(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let frame = ((sample_rate as usize * FRAME_MS) / 1000).max(1);
    samples
        .chunks(frame)
        .map(|chunk| amplitude_to_dbfs(rms(chunk)))
        .collect()
}

/// Estimates the noise floor and the speech level from the distribution of
/// frame energies: quiet frames are pauses, loud frames are speech. This works
/// without a VAD model, which matters because the profile has to be available
/// before any model is installed.
fn floor_and_speech(frame_dbfs: &mut Vec<f32>) -> (f32, f32) {
    if frame_dbfs.is_empty() {
        return (SILENCE_DBFS, SILENCE_DBFS);
    }
    frame_dbfs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (percentile(frame_dbfs, 0.10), percentile(frame_dbfs, 0.90))
}

/// Speech-to-noise across 300–3400 Hz only.
///
/// Two averaged spectra are compared: one over the frames that sit at the
/// noise floor (the room between words) and one over the frames near the
/// speech level. Both are capped at [`MAX_SNR_FRAMES`] and strided across the
/// whole recording by [`spectrum::average_spectrum`], so the cost is a fixed
/// number of FFTs whether this is a two-minute stand-up or a three-hour
/// workshop.
///
/// Every failure is a `0.0`, never an error: no separation between floor and
/// speech, no qualifying frames, no energy in the band. A profiling stage that
/// cannot run reports less; it does not cost the user their recording.
fn speech_band_snr(
    samples: &[f32],
    sample_rate: u32,
    noise_floor_dbfs: f32,
    speech_dbfs: f32,
) -> f32 {
    if sample_rate == 0 || samples.len() < SNR_FRAME * 2 {
        return 0.0;
    }
    let quiet_ceiling = dbfs_to_amplitude(noise_floor_dbfs + SILENCE_MARGIN_DB);
    let loud_floor = dbfs_to_amplitude(speech_dbfs - SPEECH_MARGIN_DB);
    if loud_floor <= quiet_ceiling {
        // Silence, or a recording with no dynamic range at all: there are no
        // two populations of frames to compare.
        return 0.0;
    }

    let room = spectrum::average_spectrum(samples, sample_rate, SNR_FRAME, MAX_SNR_FRAMES, |_, e| {
        e <= quiet_ceiling
    });
    let speech =
        spectrum::average_spectrum(samples, sample_rate, SNR_FRAME, MAX_SNR_FRAMES, |_, e| {
            e >= loud_floor
        });
    if room.frames == 0 || speech.frames == 0 {
        return 0.0;
    }

    let room_band = room.band_energy(SPEECH_BAND_LO_HZ, SPEECH_BAND_HI_HZ);
    let speech_band = speech.band_energy(SPEECH_BAND_LO_HZ, SPEECH_BAND_HI_HZ);
    if room_band <= 0.0 || speech_band <= 0.0 {
        return 0.0;
    }
    // Strictly this is (speech+noise)/noise rather than speech/noise — the
    // speech frames contain the room too. At the levels that matter the
    // difference is under a dB, and erring high here is the safe direction: it
    // never invents a problem that is not there.
    (20.0 * (speech_band / room_band).log10()).max(0.0)
}

/// The level strip: at most [`MAX_TIMELINE_BUCKETS`] evenly spread buckets.
///
/// Bucket boundaries are computed from the sample index rather than
/// accumulated, so rounding cannot drift across a long recording and the last
/// bucket always ends exactly at the end of the audio.
fn build_timeline(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    duration_ms: i64,
) -> Vec<ProfileFrame> {
    let channels = channels.max(1) as usize;
    let total_frames = samples.len() / channels;
    if total_frames == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let by_duration = (duration_ms / MIN_TIMELINE_BUCKET_MS).max(1) as usize;
    let buckets = by_duration.min(MAX_TIMELINE_BUCKETS).min(total_frames);

    let mut out = Vec::with_capacity(buckets);
    for index in 0..buckets {
        let start = index * total_frames / buckets;
        let end = ((index + 1) * total_frames / buckets)
            .max(start + 1)
            .min(total_frames);
        let slice = &samples[start * channels..end * channels];

        let mut peak = 0f32;
        let mut clipped = 0usize;
        for &s in slice {
            let a = s.abs();
            if a > peak {
                peak = a;
            }
            if a >= CLIP_THRESHOLD {
                clipped += 1;
            }
        }
        out.push(ProfileFrame {
            start_ms: (start as i64 * 1000) / sample_rate as i64,
            rms_dbfs: amplitude_to_dbfs(rms(slice)),
            peak_dbfs: amplitude_to_dbfs(peak),
            clipped_percent: if slice.is_empty() {
                0.0
            } else {
                (clipped as f32 / slice.len() as f32) * 100.0
            },
        });
    }
    out
}

/// A single 0..=100 number for "will this recording transcribe well".
///
/// It starts at 100 and *deducts*, from measurements that actually predict
/// transcription accuracy rather than from anything that merely sounds
/// technical. Every deduction is listed here so the number is auditable
/// instead of magic:
///
/// | deduction | when | at most |
/// |---|---|---|
/// | mostly silence | more than 80% of frames sit at the noise floor | 40 |
/// | clipping | above 0.1% of samples at full scale, 18 points per further % | 35 |
/// | speech-band SNR | below 20 dB, 1.5 points per dB short | 30 |
/// | quiet | peak below -30 dBFS, 1.25 points per dB short | 25 |
/// | named noise | 10 points per finding, scaled by its confidence | 25 |
/// | sample rate | below 16 kHz | 15 |
///
/// The weights follow the failure modes in order of how much they cost a
/// transcript: a recording that is nearly all silence or badly clipped is
/// unrecoverable, poor speech-band SNR loses words, a quiet-but-clean
/// recording mostly survives because the front end normalises it, and a named
/// noise finding is the mildest because it is advice about the *next* meeting.
/// An empty recording scores 0 — there is nothing to transcribe.
///
/// The total of every deduction exceeds 100 on purpose: a recording with four
/// serious problems is not meaningfully better than one with five, and the
/// clamp says so.
pub fn quality_score(profile: &AudioProfile) -> u8 {
    if profile.duration_ms == 0 {
        return 0;
    }
    let mut score = 100.0f32;

    if profile.sample_rate < MIN_USEFUL_SAMPLE_RATE {
        score -= 15.0;
    }
    if profile.silence_percent > MOSTLY_SILENT_PERCENT {
        score -= 40.0;
    }
    if profile.clipped_percent > CLIP_PERCENT_LIMIT {
        score -= ((profile.clipped_percent - CLIP_PERCENT_LIMIT) * 18.0).clamp(0.0, 35.0);
    }
    if profile.peak_dbfs < QUIET_PEAK_DBFS {
        score -= ((QUIET_PEAK_DBFS - profile.peak_dbfs) * 1.25).clamp(0.0, 25.0);
    }
    if profile.speech_band_snr_db < GOOD_SPEECH_SNR_DB {
        score -= ((GOOD_SPEECH_SNR_DB - profile.speech_band_snr_db) * 1.5).clamp(0.0, 30.0);
    }

    let noise: f32 = profile
        .noise
        .findings
        .iter()
        .map(|f| 10.0 * f.confidence.clamp(0.0, 1.0))
        .sum();
    score -= noise.min(25.0);

    // NaN would survive `clamp` on the wrong side of the comparison, so the
    // guard is explicit: an unmeasurable recording scores zero, not garbage.
    if !score.is_finite() {
        return 0;
    }
    score.clamp(0.0, 100.0).round() as u8
}

/// [`quality_score`] as a word. The bands are wide because the score is an
/// estimate, and a user reading "72" should not think it means 72 of anything.
pub fn quality_grade(score: u8) -> &'static str {
    match score {
        85..=100 => "excellent",
        70..=84 => "good",
        50..=69 => "fair",
        _ => "poor",
    }
}

/// Measures `samples` (mono or interleaved, normalised to -1.0..1.0).
pub fn analyze(samples: &[f32], sample_rate: u32, channels: u16) -> AudioProfile {
    let frames = if channels > 0 {
        samples.len() / channels as usize
    } else {
        samples.len()
    };
    let duration_ms = if sample_rate > 0 {
        (frames as i64 * 1000) / sample_rate as i64
    } else {
        0
    };

    let mut peak = 0f32;
    let mut sum = 0f64;
    let mut sum_sq = 0f64;
    let mut clipped = 0u64;
    for &s in samples {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
        sum += s as f64;
        sum_sq += (s as f64) * (s as f64);
        if a >= CLIP_THRESHOLD {
            clipped += 1;
        }
    }
    let count = samples.len().max(1) as f64;
    let dc_offset = (sum / count) as f32;
    let overall_rms = (sum_sq / count).sqrt() as f32;

    let mut levels = frame_levels(samples, sample_rate.max(1));
    let total_frames = levels.len();
    let (noise_floor_dbfs, speech_dbfs) = floor_and_speech(&mut levels);
    // `levels` is sorted, so the silent frames are a prefix — a binary search
    // beats rescanning the whole vector on a long meeting.
    let silence_cutoff = noise_floor_dbfs + SILENCE_MARGIN_DB;
    let silent_frames = levels.partition_point(|&l| l <= silence_cutoff);
    let silence_percent = if total_frames == 0 {
        0.0
    } else {
        (silent_frames as f32 / total_frames as f32) * 100.0
    };

    let clipped_percent = if samples.is_empty() {
        0.0
    } else {
        (clipped as f64 / samples.len() as f64 * 100.0) as f32
    };

    // The spectral stages assume one channel at `sample_rate`; handing them
    // an interleaved buffer would describe a signal at twice the rate with
    // both microphones spliced together. Downmixing costs an allocation, so it
    // only happens when there is actually more than one channel — which is
    // never, for anything `audio.rs` wrote.
    let downmixed: Option<Vec<f32>> = if channels > 1 {
        let n = channels as usize;
        Some(
            samples
                .chunks(n)
                .map(|frame| frame.iter().sum::<f32>() / frame.len().max(1) as f32)
                .collect(),
        )
    } else {
        None
    };
    let mono: &[f32] = downmixed.as_deref().unwrap_or(samples);

    let peak_dbfs = amplitude_to_dbfs(peak);
    let rms_dbfs = amplitude_to_dbfs(overall_rms);

    let mut profile = AudioProfile {
        duration_ms,
        sample_rate,
        channels,
        peak_dbfs,
        rms_dbfs,
        noise_floor_dbfs,
        speech_dbfs,
        estimated_snr_db: (speech_dbfs - noise_floor_dbfs).max(0.0),
        crest_factor_db: peak_dbfs - rms_dbfs,
        speech_band_snr_db: speech_band_snr(mono, sample_rate, noise_floor_dbfs, speech_dbfs),
        clipped_samples: clipped,
        clipped_percent,
        silence_percent,
        dc_offset,
        // `clipped_percent` is handed over rather than recomputed, so the
        // profile and the classifier can never disagree about the same file.
        noise: noise_class::classify(mono, sample_rate, clipped_percent),
        quality_score: 0,
        quality_grade: String::new(),
        timeline: build_timeline(samples, sample_rate, channels, duration_ms),
        warnings: Vec::new(),
    };

    profile.quality_score = quality_score(&profile);
    profile.quality_grade = quality_grade(profile.quality_score).to_string();

    // The classifier's advice comes last, after the level and clipping
    // warnings: those name a problem the user can fix in the Windows sound
    // panel in ten seconds, and should not be buried under a paragraph about
    // room acoustics.
    let mut warnings = warnings_for(&profile);
    warnings.extend(profile.noise.advice());
    profile.warnings = warnings;
    profile
}

/// The advice the UI shows. Deliberately written for someone who has never
/// heard of dBFS: every entry names the problem and what to do about it.
pub fn warnings_for(profile: &AudioProfile) -> Vec<String> {
    let mut warnings = Vec::new();

    if profile.duration_ms == 0 {
        warnings.push(
            "This recording is empty — no audio was captured. Check that the right microphone was selected and that Windows is not blocking microphone access.".to_string(),
        );
        return warnings;
    }

    if profile.sample_rate < MIN_USEFUL_SAMPLE_RATE {
        warnings.push(format!(
            "This microphone recorded at {} Hz. Speech recognition needs at least {} Hz, so words will be missed. Pick a different microphone in Settings.",
            profile.sample_rate, MIN_USEFUL_SAMPLE_RATE
        ));
    }

    // Checked before the level warnings: on an almost-empty recording the peak
    // and SNR numbers are technically true but say nothing useful.
    if profile.silence_percent > MOSTLY_SILENT_PERCENT {
        warnings.push(
            "Almost all of this recording is silence. Check that the microphone was not muted and that it was the device actually picking up the room.".to_string(),
        );
    }

    if profile.clipped_percent > CLIP_PERCENT_LIMIT {
        warnings.push(format!(
            "The audio is distorting on the loudest parts ({:.1}% of it is clipped). Lower the microphone level in Windows sound settings, or move the microphone slightly further from whoever is speaking.",
            profile.clipped_percent
        ));
    } else if profile.peak_dbfs < QUIET_PEAK_DBFS {
        warnings.push(
            "The microphone level is very low — move closer to the microphone, or raise the input level in Windows sound settings. Quiet audio is the most common cause of missed words."
                .to_string(),
        );
    }

    if profile.silence_percent <= MOSTLY_SILENT_PERCENT
        && profile.estimated_snr_db < NOISY_SNR_DB
        && profile.peak_dbfs > SILENCE_DBFS
    {
        warnings.push(
            "There is a lot of background noise compared to the speech. Close doors and windows, move away from fans or air conditioning, and put the microphone closer to the people talking."
                .to_string(),
        );
    }

    if profile.dc_offset.abs() > DC_OFFSET_LIMIT {
        warnings.push(
            "This microphone adds a constant electrical offset to the signal. Cadence removes it automatically before transcribing, but another microphone or an updated audio driver would give cleaner recordings."
                .to_string(),
        );
    }

    warnings
}

/// Reads a WAV file and profiles it. Accepts every format `hound` can decode,
/// not just the 16-bit PCM `audio.rs` writes, so an imported recording can be
/// profiled too.
pub fn profile_wav(path: &Path) -> Result<AudioProfile, String> {
    let mut reader = WavReader::open(path)
        .map_err(|e| format!("could not open {}: {e}", path.display()))?;
    let spec = reader.spec();
    let samples = read_normalised(&mut reader, spec)?;
    Ok(analyze(&samples, spec.sample_rate, spec.channels))
}

fn read_normalised(
    reader: &mut WavReader<std::io::BufReader<std::fs::File>>,
    spec: hound::WavSpec,
) -> Result<Vec<f32>, String> {
    match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string()),
        SampleFormat::Int => {
            // hound yields integer samples sign-extended into an i32, so one
            // divisor derived from the bit depth normalises every depth.
            let scale = (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale).map_err(|e| e.to_string()))
                .collect()
        }
    }
}

/// Profiles a meeting's untouched recording. `data_dir` is the app data
/// directory; the recording is `meetings/<id>/original.wav`.
pub fn profile_meeting(data_dir: &Path, meeting_id: &str) -> Result<AudioProfile, String> {
    let wav = crate::storage::meeting_dir(data_dir, meeting_id)?.join("original.wav");
    if !wav.exists() {
        return Err(format!(
            "No recording found at {}. Record the meeting before checking its audio quality.",
            wav.display()
        ));
    }
    profile_wav(&wav)
}

/// Audio-quality report for one meeting's `original.wav`.
#[tauri::command]
pub fn profile_meeting_audio(
    app: tauri::AppHandle,
    meeting_id: String,
) -> Result<AudioProfile, String> {
    use tauri::Manager;
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    profile_meeting(&data_dir, &meeting_id)
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

    #[test]
    fn dbfs_matches_known_values() {
        assert!((amplitude_to_dbfs(1.0) - 0.0).abs() < 1e-4);
        assert!((amplitude_to_dbfs(0.5) + 6.0206).abs() < 1e-3);
        assert!((amplitude_to_dbfs(0.1) + 20.0).abs() < 1e-3);
        // Sign is irrelevant — dBFS is a magnitude.
        assert_eq!(amplitude_to_dbfs(-0.5), amplitude_to_dbfs(0.5));
    }

    #[test]
    fn digital_silence_never_becomes_infinite_or_nan() {
        assert_eq!(amplitude_to_dbfs(0.0), SILENCE_DBFS);
        assert_eq!(amplitude_to_dbfs(f32::NAN), SILENCE_DBFS);
        assert_eq!(amplitude_to_dbfs(f32::NEG_INFINITY), SILENCE_DBFS);
        assert!(amplitude_to_dbfs(1e-30).is_finite());
    }

    #[test]
    fn dbfs_round_trips_through_amplitude() {
        for db in [-60.0f32, -20.0, -6.0, 0.0] {
            let back = amplitude_to_dbfs(dbfs_to_amplitude(db));
            assert!((back - db).abs() < 1e-3, "{db} -> {back}");
        }
        assert_eq!(dbfs_to_amplitude(SILENCE_DBFS), 0.0);
    }

    #[test]
    fn rms_of_a_sine_is_peak_over_root_two() {
        let samples = tone(16_000, 1.0, 440.0, 1.0);
        assert!((rms(&samples) - 0.70710678).abs() < 1e-3);
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn percentile_interpolates_between_neighbours() {
        let sorted = [0.0f32, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile(&sorted, 0.0), 0.0);
        assert_eq!(percentile(&sorted, 1.0), 40.0);
        assert_eq!(percentile(&sorted, 0.5), 20.0);
        assert!((percentile(&sorted, 0.25) - 10.0).abs() < 1e-4);
        assert_eq!(percentile(&[], 0.5), SILENCE_DBFS);
    }

    #[test]
    fn noise_floor_ignores_a_single_dropped_sample() {
        // A quiet room with one instantaneous dropout: a global minimum would
        // report digital silence as the floor, the percentile should not.
        let sr = 16_000;
        let mut samples = tone(sr, 2.0, 300.0, 0.02);
        samples[1234] = 0.0;
        let mut levels = frame_levels(&samples, sr);
        let (floor, _) = floor_and_speech(&mut levels);
        assert!(floor > -50.0, "floor was {floor}");
    }

    #[test]
    fn snr_separates_loud_speech_from_a_quiet_floor() {
        let sr = 16_000;
        let mut samples = tone(sr, 1.0, 200.0, 0.001); // room tone
        samples.extend(tone(sr, 1.0, 200.0, 0.3)); // talking
        let profile = analyze(&samples, sr, 1);
        assert!(
            profile.estimated_snr_db > 30.0,
            "snr was {}",
            profile.estimated_snr_db
        );
        assert!(profile.noise_floor_dbfs < -40.0);
    }

    #[test]
    fn a_silent_recording_is_reported_as_silence_not_nan() {
        let profile = analyze(&vec![0.0f32; 16_000], 16_000, 1);
        assert_eq!(profile.peak_dbfs, SILENCE_DBFS);
        assert_eq!(profile.rms_dbfs, SILENCE_DBFS);
        assert_eq!(profile.estimated_snr_db, 0.0);
        assert!(profile.silence_percent > 99.0);
        assert!(profile.warnings.iter().any(|w| w.contains("silence")));
        assert!(profile.duration_ms == 1000);
    }

    #[test]
    fn clipping_is_counted_and_warned_about() {
        let sr = 16_000;
        let mut samples = tone(sr, 1.0, 300.0, 0.5);
        for s in samples.iter_mut().take(2_000) {
            *s = 1.0;
        }
        let profile = analyze(&samples, sr, 1);
        assert_eq!(profile.clipped_samples, 2_000);
        assert!((profile.clipped_percent - 12.5).abs() < 0.1);
        assert!(profile.warnings.iter().any(|w| w.contains("distorting")));
    }

    #[test]
    fn a_quiet_recording_gets_level_advice_and_a_loud_one_does_not() {
        let sr = 16_000;
        let quiet = analyze(&tone(sr, 1.0, 300.0, 0.01), sr, 1);
        assert!(quiet.peak_dbfs < QUIET_PEAK_DBFS);
        assert!(quiet.warnings.iter().any(|w| w.contains("level is very low")));

        let healthy = analyze(&tone(sr, 1.0, 300.0, 0.5), sr, 1);
        assert!(!healthy
            .warnings
            .iter()
            .any(|w| w.contains("level is very low")));
    }

    #[test]
    fn dc_offset_is_measured_and_warned_about() {
        let sr = 16_000;
        let samples: Vec<f32> = tone(sr, 1.0, 300.0, 0.3).iter().map(|s| s + 0.05).collect();
        let profile = analyze(&samples, sr, 1);
        assert!((profile.dc_offset - 0.05).abs() < 1e-3);
        assert!(profile
            .warnings
            .iter()
            .any(|w| w.contains("electrical offset")));
    }

    #[test]
    fn a_low_sample_rate_is_called_out() {
        let profile = analyze(&tone(8_000, 1.0, 300.0, 0.4), 8_000, 1);
        assert!(profile.warnings.iter().any(|w| w.contains("8000 Hz")));
    }

    #[test]
    fn duration_accounts_for_interleaved_channels() {
        // 2 channels x 16000 frames of stereo = one second, not two.
        let profile = analyze(&vec![0.2f32; 32_000], 16_000, 2);
        assert_eq!(profile.duration_ms, 1000);
        assert_eq!(profile.channels, 2);
    }

    #[test]
    fn an_empty_recording_reports_only_that() {
        let profile = analyze(&[], 16_000, 1);
        assert_eq!(profile.duration_ms, 0);
        assert_eq!(profile.warnings.len(), 1);
        assert!(profile.warnings[0].contains("empty"));
    }

    #[test]
    fn a_healthy_recording_has_nothing_to_say() {
        // Loud, quiet-floored, centred speech-like signal.
        let sr = 16_000;
        let mut samples = tone(sr, 0.4, 180.0, 0.00005);
        samples.extend(tone(sr, 2.0, 180.0, 0.4));
        let profile = analyze(&samples, sr, 1);
        assert!(profile.is_clean(), "unexpected warnings: {:?}", profile.warnings);
    }

    #[test]
    fn profiles_a_wav_file_end_to_end() {
        let dir = std::env::temp_dir().join("cadence-profile-wav");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for s in tone(16_000, 1.0, 300.0, 0.5) {
            writer.write_sample((s * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();

        let profile = profile_wav(&path).unwrap();
        assert_eq!(profile.sample_rate, 16_000);
        assert_eq!(profile.channels, 1);
        assert!((profile.peak_dbfs - amplitude_to_dbfs(0.5)).abs() < 0.2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Deterministic broadband noise. A fixed seed keeps every assertion below
    /// reproducible, which a real PRNG would not.
    fn hiss(n: usize, amplitude: f32) -> Vec<f32> {
        let mut state = 987_654_321u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((state >> 16) as f32 / 32_768.0) - 1.0) * amplitude
            })
            .collect()
    }

    /// Alternating half-second bursts of `freq` over a constant hiss floor —
    /// the shape of a meeting: talking, pause, talking.
    fn bursts(sr: u32, seconds: f32, freq: f32, amplitude: f32, floor: f32) -> Vec<f32> {
        let total = (sr as f32 * seconds) as usize;
        let mut out = hiss(total, floor);
        let burst = (sr / 2) as usize;
        for (i, s) in out.iter_mut().enumerate() {
            if (i / burst) % 2 == 0 {
                let t = i as f32 / sr as f32;
                *s += amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }
        out
    }

    #[test]
    fn probe() {
        let sr = 16_000;
        let voice = analyze(&bursts(sr, 4.0, 1_000.0, 0.3, 0.003), sr, 1);
        let rumble = analyze(&bursts(sr, 4.0, 60.0, 0.3, 0.003), sr, 1);
        println!(
            "voice: broadband {:.1} band {:.1} score {} grade {} speech {:.1} crest {:.1} findings {:?}",
            voice.estimated_snr_db, voice.speech_band_snr_db, voice.quality_score,
            voice.quality_grade, voice.speech_dbfs, voice.crest_factor_db,
            voice.noise.findings.iter().map(|f| (f.kind, f.confidence)).collect::<Vec<_>>()
        );
        println!(
            "rumble: broadband {:.1} band {:.1} score {} grade {} findings {:?}",
            rumble.estimated_snr_db, rumble.speech_band_snr_db, rumble.quality_score,
            rumble.quality_grade,
            rumble.noise.findings.iter().map(|f| (f.kind, f.confidence)).collect::<Vec<_>>()
        );
        let mut clean = tone(sr, 0.4, 800.0, 0.00005);
        clean.extend(tone(sr, 2.0, 800.0, 0.4));
        let c = analyze(&clean, sr, 1);
        println!("clean: band {:.1} score {} grade {} warnings {:?}", c.speech_band_snr_db, c.quality_score, c.quality_grade, c.warnings);
        let mut bad = hiss(sr as usize * 3, 0.02);
        for (i, s) in bad.iter_mut().enumerate() {
            if (i / 8000) % 2 == 0 {
                let t = i as f32 / sr as f32;
                *s += 0.03 * (2.0 * std::f32::consts::PI * 300.0 * t).sin();
            }
        }
        let b = analyze(&bad, sr, 1);
        println!("bad: peak {:.1} band {:.1} score {} grade {} findings {:?}", b.peak_dbfs, b.speech_band_snr_db, b.quality_score, b.quality_grade,
            b.noise.findings.iter().map(|f| (f.kind, f.confidence)).collect::<Vec<_>>());
        let clipped = {
            let sr = 16_000;
            let mut s = tone(sr, 2.0, 500.0, 0.9);
            for v in s.iter_mut() { *v = v.clamp(-0.7, 0.7) / 0.7; }
            analyze(&s, sr, 1)
        };
        println!("clipped: clip% {:.2} score {} grade {}", clipped.clipped_percent, clipped.quality_score, clipped.quality_grade);
    }

    #[test]
    fn a_missing_meeting_recording_is_a_clear_error() {
        let dir = std::env::temp_dir().join("cadence-profile-missing");
        let _ = std::fs::remove_dir_all(&dir);
        let err = profile_meeting(&dir, "nope").unwrap_err();
        assert!(err.contains("No recording found"));
    }

    #[test]
    fn meeting_profile_rejects_invalid_ids_before_reading_audio() {
        let dir = std::env::temp_dir().join("cadence-profile-invalid-ids");
        let outside = dir.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        // If validation regresses, this would reach WAV decoding instead.
        let wav = outside.join("original.wav");
        std::fs::write(&wav, b"preserve this recording").unwrap();
        for id in ["../outside", "..\\outside", "C:\\outside", "a:stream", "CON", "a ", ""] {
            let err = profile_meeting(&dir, id).unwrap_err();
            assert!(err.contains("invalid meeting path component"), "{id:?}: {err}");
        }
        assert_eq!(std::fs::read(wav).unwrap(), b"preserve this recording");
        assert!(!dir.join("meetings").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn meeting_profile_reads_valid_recording_without_modifying_it() {
        let dir = std::env::temp_dir().join("cadence-profile-valid-meeting");
        let meeting = crate::storage::meeting_dir(&dir, "valid-meeting-123").unwrap();
        std::fs::create_dir_all(&meeting).unwrap();
        let wav = meeting.join("original.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
        for sample in tone(16_000, 1.0, 300.0, 0.5) {
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
        let original = std::fs::read(&wav).unwrap();
        let profile = profile_meeting(&dir, "valid-meeting-123").unwrap();
        assert_eq!(profile.duration_ms, 1000);
        assert_eq!(profile.sample_rate, 16_000);
        assert_eq!(profile.channels, 1);
        assert!((profile.peak_dbfs - amplitude_to_dbfs(0.5)).abs() < 0.2);
        assert_eq!(std::fs::read(wav).unwrap(), original);
        assert_eq!(std::fs::read_dir(meeting).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
