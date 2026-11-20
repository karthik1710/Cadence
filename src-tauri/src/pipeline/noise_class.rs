//! Working out what the background noise in a recording actually *is*.
//!
//! The audio profile could already say "there is a lot of background noise
//! compared to the speech", which is true and almost useless: it is the same
//! sentence whether the problem is a projector fan, a ground loop, someone
//! typing, or a hard-walled room with a two-second tail. Those have four
//! different fixes and only one of them is "move the microphone closer".
//!
//! So this classifies the noise into named causes and says what to do about
//! each. It is deliberately a **signal-processing classifier, not a model**:
//! every decision below comes from measurable properties of the spectrum and
//! the envelope. That keeps it deterministic and auditable, adds no download,
//! no ONNX session and no memory-tier decision, and means it works on a fresh
//! install before anything optional has been fetched. A learned acoustic
//! scene classifier would be more accurate on ambiguous rooms; it is not worth
//! a model dependency for advice a person reads once and acts on.
//!
//! Everything here is pure analysis. Nothing writes audio.

use crate::pipeline::audio_profile::{amplitude_to_dbfs, dbfs_to_amplitude, rms, SILENCE_DBFS};
use crate::pipeline::spectrum::{self, AverageSpectrum};
use serde::{Deserialize, Serialize};

/// Frame for the envelope measurements (transients, decay). 20 ms matches the
/// audio profile, so the two describe the same frames.
const ENVELOPE_FRAME_MS: usize = 20;

/// Frame for the spectral measurements. 2048 samples is 128 ms at 16 kHz,
/// giving 7.8 Hz bins — enough to tell 50 Hz mains from 60 Hz mains, which a
/// shorter frame cannot do.
const SPECTRAL_FRAME: usize = 2048;

/// Cap on spectral frames analysed. Noise is stationary by definition, so a
/// few hundred frames spread across the meeting describe it as well as all of
/// them would.
const MAX_SPECTRAL_FRAMES: usize = 600;

/// How far above the noise floor a frame may sit and still count as "the
/// room" rather than "someone talking".
const QUIET_MARGIN_DB: f32 = 6.0;

/// A room this far below the speech is not worth describing.
///
/// Without this gate the spectral tests fire on every recording ever made,
/// because *every* room tone has a shape — white-ish noise 60 dB down is
/// still, technically, hiss. What the user needs to know is not the colour of
/// the noise but whether it is loud enough to matter, so the colour is only
/// reported once the level says it does. This is the difference between a
/// classifier and a horoscope.
const QUIET_ROOM_SNR_DB: f32 = 30.0;

/// A named cause of background noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseKind {
    /// A ground loop or unshielded cable: 50 or 60 Hz and its harmonics.
    MainsHum,
    /// Air conditioning, a projector, a laptop fan — steady, low, broadband.
    Fan,
    /// Below the bottom of a voice: traffic, building services, desk thumps.
    Rumble,
    /// Preamp or cheap-capsule hiss, and the top-heavy noise of a mic gain
    /// turned up to compensate for distance.
    Hiss,
    /// Typing, mouse clicks, handling the microphone, papers.
    Impulsive,
    /// A hard room with a long tail. The worst of these for transcription,
    /// because it smears one word into the next.
    Reverberant,
    /// Other people talking: speech-band energy that is there in the pauses.
    Babble,
    /// The input was driven past what the converter could represent.
    Clipping,
}

impl NoiseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NoiseKind::MainsHum => "mains_hum",
            NoiseKind::Fan => "fan",
            NoiseKind::Rumble => "rumble",
            NoiseKind::Hiss => "hiss",
            NoiseKind::Impulsive => "impulsive",
            NoiseKind::Reverberant => "reverberant",
            NoiseKind::Babble => "babble",
            NoiseKind::Clipping => "clipping",
        }
    }

    /// Short name for the UI.
    pub fn label(self) -> &'static str {
        match self {
            NoiseKind::MainsHum => "Electrical hum",
            NoiseKind::Fan => "Fan or air conditioning",
            NoiseKind::Rumble => "Low-frequency rumble",
            NoiseKind::Hiss => "Microphone hiss",
            NoiseKind::Impulsive => "Typing or handling noise",
            NoiseKind::Reverberant => "Room echo",
            NoiseKind::Babble => "Other people talking",
            NoiseKind::Clipping => "Distortion",
        }
    }

    /// Whether the pipeline can do anything about it, which is the difference
    /// between advice worth giving and a counsel of despair.
    fn advice(self) -> &'static str {
        match self {
            NoiseKind::MainsHum => "This is usually a power or cabling problem, not a room one. Try a different USB port, unplug the laptop charger for the meeting, or move the microphone cable away from power cables.",
            NoiseKind::Fan => "Move the microphone away from the air conditioning vent, the projector and the laptop's own fan vents, or put something soft under it. Cadence filters some of this out, but a metre of distance does more.",
            NoiseKind::Rumble => "Something is shaking the desk or the building. Put the microphone on a soft mat rather than directly on the table, and keep laptops and feet off the same surface.",
            NoiseKind::Hiss => "The microphone gain is probably turned up too far to make up for distance. Move the microphone closer to the people talking and lower the input level in Windows sound settings.",
            NoiseKind::Impulsive => "Someone is typing near the microphone, or it is being handled. Move it away from keyboards and papers, and avoid touching it while people are speaking.",
            NoiseKind::Reverberant => "This room echoes, which is the hardest problem here for transcription because it blurs one word into the next. Move the microphone closer to whoever is speaking, and if the room has bare walls and a glass table, soft furnishings or a different room will help more than any setting.",
            NoiseKind::Babble => "There is other speech in the background — an open door, a corridor, or a nearby conversation. Close the door and move the microphone away from it: the transcript cannot tell those voices from the meeting.",
            NoiseKind::Clipping => "The input level is too high and the loudest speech is being cut off. Lower the microphone level in Windows sound settings, or move the microphone slightly further away.",
        }
    }
}

/// One identified cause, with the measurement behind it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseFinding {
    pub kind: NoiseKind,
    pub label: String,
    /// 0.0..1.0. Not a probability — a scaled distance past the threshold at
    /// which this stops being ambiguous. Ordering findings by it is the point.
    pub confidence: f32,
    /// The measurement, so a sceptical user can see why it said this.
    pub detail: String,
    pub advice: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseReport {
    /// Ordered by confidence, most confident first.
    pub findings: Vec<NoiseFinding>,
    /// Frames judged to be room rather than speech, and what share that was.
    pub quiet_frames: usize,
    pub quiet_share: f32,
    /// Mains frequency, when hum was found: 50 or 60.
    pub mains_hz: Option<f32>,
    /// Rough time for the room to fall 20 dB after speech stops. `None` when
    /// the recording had too few clean decays to measure. An estimate from
    /// the envelope, not a calibrated RT60.
    pub decay_ms: Option<f32>,
    /// Impulsive events per minute.
    pub transients_per_minute: f32,
    /// Spectral flatness of the room tone: near 1 is broadband hiss, near 0
    /// is a tonal noise like hum.
    pub noise_flatness: f32,
    /// Energy above 4 kHz over energy below it, in dB, for the room tone.
    pub noise_tilt_db: f32,
    pub noise_floor_dbfs: f32,
    /// How far the speech sits above the room. Above [`QUIET_ROOM_SNR_DB`]
    /// the room is quiet enough that none of the spectral findings are
    /// reported at all.
    pub speech_to_noise_db: f32,
}

impl NoiseReport {
    pub fn dominant(&self) -> Option<NoiseKind> {
        self.findings.first().map(|f| f.kind)
    }

    /// The advice lines, for the audio profile to fold into its warnings.
    pub fn advice(&self) -> Vec<String> {
        self.findings
            .iter()
            .map(|f| format!("{}: {} {}", f.label, f.detail, f.advice))
            .collect()
    }
}

/// Maps a measurement onto 0..1 confidence: nothing at `threshold`, full at
/// `certain`. Keeps every threshold in this file written as two readable
/// numbers rather than a hand-tuned curve.
fn confidence(value: f32, threshold: f32, certain: f32) -> f32 {
    if certain <= threshold {
        return if value >= threshold { 1.0 } else { 0.0 };
    }
    ((value - threshold) / (certain - threshold)).clamp(0.0, 1.0)
}

/// Per-frame RMS in dBFS over non-overlapping [`ENVELOPE_FRAME_MS`] frames.
fn envelope_db(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let frame = ((sample_rate as usize * ENVELOPE_FRAME_MS) / 1000).max(1);
    samples
        .chunks(frame)
        .map(|chunk| amplitude_to_dbfs(rms(chunk)))
        .collect()
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return SILENCE_DBFS;
    }
    let index = ((sorted.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

/// Counts short, loud events against a quiet background — typing, clicks, a
/// hand on the microphone.
///
/// The test is deliberately narrow: a frame that jumps well above the one
/// before it and has fallen back by the frame after. Speech does not do that
/// — a syllable is several frames long — so ordinary talking does not
/// register, which is what keeps this from firing on every meeting.
fn count_transients(levels: &[f32], floor_db: f32) -> usize {
    if levels.len() < 3 {
        return 0;
    }
    let mut count = 0usize;
    for i in 1..levels.len() - 1 {
        let rise = levels[i] - levels[i - 1];
        let fall = levels[i] - levels[i + 1];
        if levels[i] > floor_db + 12.0 && rise > 12.0 && fall > 8.0 {
            count += 1;
        }
    }
    count
}

/// Median time for the level to fall 20 dB after speech stops.
///
/// This is an envelope decay estimate, not a calibrated reverberation time:
/// a real RT60 needs an impulse or an interrupted noise source and a
/// calibrated room. It is still the right shape of measurement for the
/// question being asked, which is "does this room smear words together".
fn decay_ms(levels: &[f32], floor_db: f32) -> Option<f32> {
    let loud = floor_db + 25.0;
    let max_frames = (1_500 / ENVELOPE_FRAME_MS).max(1);
    let mut decays: Vec<f32> = Vec::new();

    let mut i = 1usize;
    while i + 1 < levels.len() {
        // A decay starts at a loud frame that is followed by a quieter one.
        if levels[i] < loud || levels[i + 1] >= levels[i] {
            i += 1;
            continue;
        }
        let start_db = levels[i];
        let mut frames = 0usize;
        let mut settled = false;
        while frames < max_frames && i + frames + 1 < levels.len() {
            frames += 1;
            let level = levels[i + frames];
            if level <= start_db - 20.0 {
                settled = true;
                break;
            }
            // Someone started talking again before it finished decaying, so
            // this is not a clean measurement of the room.
            if level > start_db {
                break;
            }
        }
        if settled {
            decays.push((frames * ENVELOPE_FRAME_MS) as f32);
            i += frames;
        }
        i += 1;
    }

    // Fewer than a handful of clean decays is not evidence about a room.
    if decays.len() < 5 {
        return None;
    }
    decays.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(decays[decays.len() / 2])
}

/// Which mains frequency the hum is on, and how strongly, by comparing each
/// candidate against its own harmonics.
fn mains_hum(spectrum: &AverageSpectrum) -> (Option<f32>, f32) {
    let score = |fundamental: f32| -> f32 {
        // The harmonics matter: a ground loop puts energy at 50/100/150, and
        // requiring more than the fundamental alone is what stops a single
        // low-frequency bin of fan noise from being read as hum.
        let first = spectrum.tonal_ratio(fundamental, 8.0);
        let second = spectrum.tonal_ratio(fundamental * 2.0, 8.0);
        let third = spectrum.tonal_ratio(fundamental * 3.0, 10.0);
        first.min(6.0) * 0.5 + second.min(6.0) * 0.3 + third.min(6.0) * 0.2
    };
    let fifty = score(50.0);
    let sixty = score(60.0);
    if fifty.max(sixty) < 2.0 {
        return (None, 0.0);
    }
    if fifty >= sixty {
        (Some(50.0), fifty)
    } else {
        (Some(60.0), sixty)
    }
}

/// Classifies the background noise in `samples`.
///
/// `clipped_percent` comes from the audio profile rather than being
/// recomputed, so the two never disagree about the same recording.
pub fn classify(samples: &[f32], sample_rate: u32, clipped_percent: f32) -> NoiseReport {
    let mut report = NoiseReport::default();
    if samples.is_empty() || sample_rate == 0 {
        return report;
    }

    let levels = envelope_db(samples, sample_rate);
    let mut sorted = levels.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor_db = percentile(&sorted, 0.10);
    let speech_db = percentile(&sorted, 0.90);
    let quiet_cutoff_db = floor_db + QUIET_MARGIN_DB;
    let quiet_cutoff = dbfs_to_amplitude(quiet_cutoff_db);

    report.noise_floor_dbfs = floor_db;
    report.speech_to_noise_db = (speech_db - floor_db).max(0.0);

    let duration_minutes = (samples.len() as f32 / sample_rate as f32 / 60.0).max(1.0 / 60.0);
    report.transients_per_minute = count_transients(&levels, floor_db) as f32 / duration_minutes;
    report.decay_ms = decay_ms(&levels, floor_db);

    // The room tone: the spectrum of only those frames nobody was talking in.
    let mut quiet_frames = 0usize;
    let mut considered = 0usize;
    let room = spectrum::average_spectrum(
        samples,
        sample_rate,
        SPECTRAL_FRAME,
        MAX_SPECTRAL_FRAMES,
        |_, energy| {
            considered += 1;
            let quiet = energy <= quiet_cutoff;
            if quiet {
                quiet_frames += 1;
            }
            quiet
        },
    );
    report.quiet_frames = quiet_frames;
    report.quiet_share = if considered == 0 {
        0.0
    } else {
        quiet_frames as f32 / considered as f32
    };

    let mut findings: Vec<NoiseFinding> = Vec::new();
    let push = |findings: &mut Vec<NoiseFinding>, kind: NoiseKind, confidence: f32, detail: String| {
        findings.push(NoiseFinding {
            kind,
            label: kind.label().to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            detail,
            advice: kind.advice().to_string(),
        });
    };

    if clipped_percent > 0.1 {
        push(
            &mut findings,
            NoiseKind::Clipping,
            confidence(clipped_percent, 0.1, 2.0),
            format!("{clipped_percent:.2}% of the samples are at full scale."),
        );
    }

    if let Some(decay) = report.decay_ms {
        if decay > 300.0 {
            push(
                &mut findings,
                NoiseKind::Reverberant,
                confidence(decay, 300.0, 900.0),
                format!("Sound takes about {decay:.0} ms to fade after someone stops speaking."),
            );
        }
    }

    if report.transients_per_minute > 8.0 {
        push(
            &mut findings,
            NoiseKind::Impulsive,
            confidence(report.transients_per_minute, 8.0, 60.0),
            format!(
                "About {:.0} short bursts of noise per minute.",
                report.transients_per_minute
            ),
        );
    }

    // Everything below needs a room tone to look at, and needs that room to
    // be loud enough to be worth describing. With almost no pauses in the
    // recording there is nothing to measure, and guessing from speech frames
    // would produce confident nonsense.
    if room.frames >= 4 && report.speech_to_noise_db < QUIET_ROOM_SNR_DB {
        report.noise_flatness = room.flatness();
        report.noise_tilt_db = room.tilt_db();

        let (mains, hum_score) = mains_hum(&room);
        if let Some(hz) = mains {
            report.mains_hz = Some(hz);
            push(
                &mut findings,
                NoiseKind::MainsHum,
                confidence(hum_score, 2.0, 5.0),
                format!("A steady {hz:.0} Hz tone and its harmonics are in the background."),
            );
        }

        let rumble = room.band_fraction(0.0, 80.0);
        if rumble > 0.30 {
            push(
                &mut findings,
                NoiseKind::Rumble,
                confidence(rumble, 0.30, 0.70),
                format!(
                    "{:.0}% of the background noise is below 80 Hz, under the bottom of a voice.",
                    rumble * 100.0
                ),
            );
        }

        // A fan sits above the rumble band and below the voice, and is
        // broadband rather than tonal — which is what separates it from hum
        // living in the same part of the spectrum.
        let fan = room.band_fraction(80.0, 400.0);
        if fan > 0.30 && report.noise_flatness > 0.15 {
            push(
                &mut findings,
                NoiseKind::Fan,
                confidence(fan, 0.30, 0.65),
                format!(
                    "{:.0}% of the background noise is steady and low-pitched, between 80 and 400 Hz.",
                    fan * 100.0
                ),
            );
        }

        let hiss = room.band_fraction(4_000.0, room.nyquist_hz());
        if hiss > 0.20 {
            push(
                &mut findings,
                NoiseKind::Hiss,
                confidence(hiss, 0.20, 0.55),
                format!(
                    "{:.0}% of the background noise is high-pitched hiss above 4 kHz.",
                    hiss * 100.0
                ),
            );
        }

        // Speech-shaped energy in the frames where nobody in the meeting was
        // talking. The flatness test is what keeps broadband hiss, which also
        // has energy in this band, from being read as a voice.
        let babble = room.band_fraction(300.0, 3_400.0);
        if babble > 0.45 && report.noise_flatness < 0.40 && report.quiet_share > 0.05 {
            push(
                &mut findings,
                NoiseKind::Babble,
                confidence(babble, 0.45, 0.80),
                format!(
                    "{:.0}% of the background noise falls in the frequency range of speech.",
                    babble * 100.0
                ),
            );
        }
    }

    findings.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    report.findings = findings;
    report
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
        let mut state = 24_680u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((state >> 16) as f32 / 32_768.0) - 1.0) * amplitude
            })
            .collect()
    }

    /// A running mean, which is a crude but genuine low-pass: the first null
    /// lands at `sample_rate / width`.
    fn smooth(samples: &[f32], width: usize) -> Vec<f32> {
        let width = width.max(1);
        let mut out = Vec::with_capacity(samples.len());
        let mut sum = 0.0f32;
        for i in 0..samples.len() {
            sum += samples[i];
            if i >= width {
                sum -= samples[i - width];
            }
            out.push(sum / width.min(i + 1) as f32);
        }
        out
    }

    /// Top-heavy broadband noise, the shape a preamp or a mic gain turned too
    /// far up actually produces: white noise minus a low-passed copy of
    /// itself, which rises steadily towards Nyquist.
    fn hiss_noise(sample_rate: u32, seconds: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        let white = noise(n, 1.0);
        let low = smooth(&white, 3);
        let high: Vec<f32> = white.iter().zip(low.iter()).map(|(a, b)| a - b).collect();
        let peak = high.iter().map(|s| s.abs()).fold(1e-9f32, f32::max);
        high.iter().map(|s| s / peak * amplitude).collect()
    }

    /// Band-limited *noise* — white noise low-passed at `hi` and high-passed
    /// at `lo` — rather than a stack of tones.
    ///
    /// This distinction is the whole test: a sum of 24 sine waves has a
    /// spectral flatness near zero, so the classifier correctly refuses to
    /// call it a fan. Real fan noise is broadband, and a fixture that is not
    /// broadband tests nothing.
    fn band_noise(sample_rate: u32, seconds: f32, lo: f32, hi: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        let white = noise(n, 1.0);
        let low_passed = smooth(&white, (sample_rate as f32 / hi).round() as usize);
        let below = smooth(&low_passed, (sample_rate as f32 / lo).round() as usize);
        let band: Vec<f32> = low_passed
            .iter()
            .zip(below.iter())
            .map(|(a, b)| a - b)
            .collect();
        let peak = band.iter().map(|s| s.abs()).fold(1e-9f32, f32::max);
        band.iter().map(|s| s / peak * amplitude).collect()
    }

    /// Speech bursts with pauses between them, over a background. The pauses
    /// are what the classifier measures the room in.
    fn meeting(sample_rate: u32, background: &[f32]) -> Vec<f32> {
        let mut speech = Vec::new();
        for _ in 0..8 {
            speech.extend(tone(sample_rate, 0.4, 300.0, 0.35));
            speech.extend(vec![0.0f32; (sample_rate as f32 * 0.4) as usize]);
        }
        speech
            .iter()
            .enumerate()
            .map(|(i, s)| s + background.get(i % background.len().max(1)).copied().unwrap_or(0.0))
            .collect()
    }

    fn kinds(report: &NoiseReport) -> Vec<NoiseKind> {
        report.findings.iter().map(|f| f.kind).collect()
    }

    #[test]
    fn a_clean_quiet_recording_reports_nothing() {
        let sr = 16_000;
        let samples = meeting(sr, &noise(sr as usize, 0.0002));
        let report = classify(&samples, sr, 0.0);
        assert!(
            report.findings.is_empty(),
            "unexpected findings: {:?}",
            kinds(&report)
        );
        assert!(report.dominant().is_none());
        assert!(report.quiet_share > 0.2, "share was {}", report.quiet_share);
    }

    #[test]
    fn a_ground_loop_is_identified_as_mains_hum_at_the_right_frequency() {
        let sr = 16_000;
        let mut hum = tone(sr, 4.0, 50.0, 0.02);
        for (i, s) in tone(sr, 4.0, 100.0, 0.012).iter().enumerate() {
            hum[i] += s;
        }
        for (i, s) in tone(sr, 4.0, 150.0, 0.008).iter().enumerate() {
            hum[i] += s;
        }
        let report = classify(&meeting(sr, &hum), sr, 0.0);
        assert!(kinds(&report).contains(&NoiseKind::MainsHum), "{:?}", kinds(&report));
        assert_eq!(report.mains_hz, Some(50.0));
    }

    #[test]
    fn sixty_hertz_mains_is_told_apart_from_fifty() {
        let sr = 16_000;
        let mut hum = tone(sr, 4.0, 60.0, 0.02);
        for (i, s) in tone(sr, 4.0, 120.0, 0.012).iter().enumerate() {
            hum[i] += s;
        }
        let report = classify(&meeting(sr, &hum), sr, 0.0);
        assert_eq!(report.mains_hz, Some(60.0));
    }

    #[test]
    fn air_conditioning_is_identified_as_a_fan_and_not_as_hum() {
        let sr = 16_000;
        let fan = band_noise(sr, 4.0, 90.0, 380.0, 0.05);
        let report = classify(&meeting(sr, &fan), sr, 0.0);
        assert!(kinds(&report).contains(&NoiseKind::Fan), "{:?}", kinds(&report));
        assert!(!kinds(&report).contains(&NoiseKind::MainsHum));
    }

    #[test]
    fn preamp_hiss_is_identified_by_its_spectral_tilt() {
        let sr = 16_000;
        let hiss = hiss_noise(sr, 4.0, 0.05);
        let report = classify(&meeting(sr, &hiss), sr, 0.0);
        assert!(kinds(&report).contains(&NoiseKind::Hiss), "{:?}", kinds(&report));
        assert!(report.noise_tilt_db > 0.0);
    }

    #[test]
    fn typing_is_counted_as_impulsive_noise() {
        let sr = 16_000;
        let mut samples = meeting(sr, &noise(sr as usize, 0.0005));
        // One 20 ms click every half second: unmistakably typing, and
        // nothing like the shape of a syllable.
        let click_len = (sr as f32 * 0.02) as usize;
        let mut at = 0usize;
        while at + click_len < samples.len() {
            for slot in samples[at..at + click_len].iter_mut() {
                *slot += 0.25;
            }
            at += sr as usize / 2;
        }
        let report = classify(&samples, sr, 0.0);
        assert!(
            kinds(&report).contains(&NoiseKind::Impulsive),
            "{:?} at {} per minute",
            kinds(&report),
            report.transients_per_minute
        );
        assert!(report.transients_per_minute > 8.0);
    }

    #[test]
    fn ordinary_speech_is_not_mistaken_for_typing() {
        let sr = 16_000;
        let report = classify(&meeting(sr, &noise(sr as usize, 0.0002)), sr, 0.0);
        assert!(
            report.transients_per_minute < 8.0,
            "speech registered {} transients per minute",
            report.transients_per_minute
        );
    }

    #[test]
    fn a_room_with_a_long_tail_is_reported_as_reverberant() {
        let sr = 16_000;
        // Speech bursts with an exponential tail spliced on: a 600 ms decay.
        let mut samples: Vec<f32> = Vec::new();
        for _ in 0..10 {
            samples.extend(tone(sr, 0.3, 300.0, 0.4));
            let tail = tone(sr, 0.9, 300.0, 0.4);
            let decay_samples = (sr as f32 * 0.6) as usize;
            for (i, s) in tail.iter().enumerate() {
                let factor = 10f32.powf(-(i as f32) / decay_samples as f32);
                samples.push(s * factor);
            }
        }
        let report = classify(&samples, sr, 0.0);
        assert!(
            kinds(&report).contains(&NoiseKind::Reverberant),
            "{:?}, decay {:?}",
            kinds(&report),
            report.decay_ms
        );
        assert!(report.decay_ms.unwrap() > 300.0);
    }

    #[test]
    fn a_dry_room_is_not_reported_as_reverberant() {
        let sr = 16_000;
        let report = classify(&meeting(sr, &noise(sr as usize, 0.0002)), sr, 0.0);
        assert!(!kinds(&report).contains(&NoiseKind::Reverberant));
    }

    #[test]
    fn clipping_is_carried_through_from_the_audio_profile() {
        let sr = 16_000;
        let report = classify(&meeting(sr, &noise(sr as usize, 0.0002)), sr, 1.5);
        assert_eq!(report.dominant(), Some(NoiseKind::Clipping));
        assert!(report.findings[0].detail.contains("1.50%"));
    }

    #[test]
    fn findings_come_back_most_confident_first() {
        let sr = 16_000;
        let fan = band_noise(sr, 4.0, 90.0, 380.0, 0.08);
        let report = classify(&meeting(sr, &fan), sr, 5.0);
        assert!(report.findings.len() >= 2);
        for pair in report.findings.windows(2) {
            assert!(pair[0].confidence >= pair[1].confidence);
        }
    }

    #[test]
    fn every_finding_carries_a_measurement_and_something_to_do_about_it() {
        let sr = 16_000;
        let report = classify(&meeting(sr, &band_noise(sr, 4.0, 90.0, 380.0, 0.08)), sr, 0.0);
        assert!(!report.findings.is_empty());
        for finding in &report.findings {
            assert!(!finding.label.is_empty());
            assert!(!finding.detail.is_empty(), "{:?} has no measurement", finding.kind);
            assert!(!finding.advice.is_empty(), "{:?} has no advice", finding.kind);
            assert!((0.0..=1.0).contains(&finding.confidence));
        }
        assert_eq!(report.advice().len(), report.findings.len());
    }

    #[test]
    fn silence_and_empty_input_are_handled_without_panicking() {
        assert!(classify(&[], 16_000, 0.0).findings.is_empty());
        assert!(classify(&vec![0.0f32; 16_000], 16_000, 0.0).findings.is_empty());
        assert!(classify(&vec![0.1f32; 100], 0, 0.0).findings.is_empty());
    }

    #[test]
    fn a_recording_with_no_pauses_skips_the_spectral_findings() {
        let sr = 16_000;
        // Wall-to-wall speech: there is no room tone to measure, so the
        // classifier must decline rather than describe the voice as noise.
        let report = classify(&tone(sr, 4.0, 300.0, 0.4), sr, 0.0);
        assert!(!kinds(&report).contains(&NoiseKind::Fan));
        assert!(!kinds(&report).contains(&NoiseKind::Babble));
        assert!(!kinds(&report).contains(&NoiseKind::MainsHum));
    }

    #[test]
    fn confidence_scales_between_its_two_thresholds() {
        assert_eq!(confidence(0.0, 1.0, 2.0), 0.0);
        assert_eq!(confidence(1.5, 1.0, 2.0), 0.5);
        assert_eq!(confidence(9.0, 1.0, 2.0), 1.0);
        // A degenerate range is a step function, not a division by zero.
        assert_eq!(confidence(5.0, 1.0, 1.0), 1.0);
        assert_eq!(confidence(0.5, 1.0, 1.0), 0.0);
    }
}
