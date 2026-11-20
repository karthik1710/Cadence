//! The ASR front-end ("Audio preprocessing").
//!
//! `original.wav` is sacred — nothing here ever writes to it. Everything
//! below produces a *new* file, so a bad preprocessing choice costs a rerun,
//! never a meeting.
//!
//! Beyond format conversion, three cheap fixes buy real accuracy on the audio
//! a laptop or USB microphone in a meeting room actually produces:
//!
//! * **DC offset removal** — some USB capsules sit off-centre, which wastes
//!   headroom and makes every downstream level measurement wrong.
//! * **80 Hz high-pass** — air conditioning, projector fans, desk thumps and
//!   handling noise all live below the ~85 Hz bottom of a human voice. Removing
//!   them costs one biquad per sample and stops whisper.cpp hallucinating over
//!   rumble in the pauses.
//! * **Loudness normalisation to a target RMS with peak limiting** — the
//!   single biggest win. Whisper was trained on normalised audio; a quiet
//!   laptop microphone at -45 dBFS is the most common cause of missed words,
//!   and the pipeline previously fed whisper whatever level the mic happened
//!   to produce.

use crate::pipeline::audio_profile::{amplitude_to_dbfs, dbfs_to_amplitude, rms};
use crate::pipeline::enhance;
use crate::settings::Settings;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Never boost by more than this. Without a cap, normalising a near-silent
/// recording would lift its noise floor to speech level and hand whisper pure
/// hiss to transcribe.
const MAX_BOOST_DB: f32 = 30.0;

/// Which steps of the chain run. Every one is user-toggleable because
/// each is a judgement call about someone else's room and microphone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessOptions {
    pub remove_dc: bool,
    /// High-pass corner in Hz; `0.0` disables the filter.
    pub high_pass_hz: f32,
    pub normalize: bool,
    pub target_rms_dbfs: f32,
    /// Ceiling the limiter holds after normalisation, so gain never clips.
    pub peak_ceiling_dbfs: f32,
    /// Even out the distance across a meeting table. `None` disables the
    /// stage — see [`crate::settings::Settings::effective_compressor`].
    pub compressor: Option<enhance::CompressorOptions>,
    /// Slow speech-gated gain riding across the table. `None` disables the
    /// stage.
    pub leveler: Option<enhance::LevelerOptions>,
    /// Model-free spectral subtraction. `None` disables the stage. On by
    /// default as the fallback for a machine without GTCRN, which supersedes
    /// it wherever it is installed.
    pub noise_reduction: Option<enhance::NoiseReductionOptions>,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            remove_dc: true,
            high_pass_hz: 80.0,
            normalize: true,
            target_rms_dbfs: -20.0,
            peak_ceiling_dbfs: -1.0,
            // On: cheap, and it is what lifts the far end of a meeting table
            // above the noise floor.
            compressor: Some(enhance::CompressorOptions::default()),
            // On: O(n), and the compressor alone cannot lift a talker who is
            // consistently 20 dB down — it only ever sees peaks.
            leveler: Some(enhance::LevelerOptions::default()),
            // On: it is the only noise handling a machine without the
            // optional GTCRN model gets. Where GTCRN is installed the
            // pipeline drops it via `without_noise_reduction`, so the two
            // never both run.
            noise_reduction: Some(enhance::NoiseReductionOptions::default()),
        }
    }
}

impl PreprocessOptions {
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            remove_dc: settings.preprocess_remove_dc,
            high_pass_hz: settings.preprocess_high_pass_hz.max(0.0),
            normalize: settings.preprocess_normalize,
            target_rms_dbfs: settings.preprocess_target_rms_dbfs,
            peak_ceiling_dbfs: -1.0,
            compressor: settings.effective_compressor(),
            // No Settings field yet, so it is always on with its defaults. A
            // toggle can be wired later the way the compressor's is.
            leveler: Some(enhance::LevelerOptions::default()),
            noise_reduction: settings.effective_noise_reduction(),
        }
    }

    /// The same chain with every *gain* stage removed. Used when the GTCRN
    /// denoiser runs next: normalising first would hand it a boosted noise
    /// floor to fight, and leveling or compressing first would hand it a
    /// signal whose noise floor moves, which is exactly what a noise
    /// estimator cannot cope with. All three are applied to its output
    /// instead, by [`apply_gain`] — so the leveler runs exactly once, after
    /// the denoiser.
    ///
    /// Model-free noise reduction deliberately stays: it is a *cleaning*
    /// stage, not a level one, and running it before the denoiser is the same
    /// order it would have had in a single pass.
    pub fn without_gain(&self) -> Self {
        Self {
            normalize: false,
            compressor: None,
            leveler: None,
            ..*self
        }
    }

    /// The same chain with the model-free denoiser removed. The caller that
    /// wants this is one that has a *better* denoiser to run — spending a full
    /// STFT on spectral subtraction before GTCRN sees the audio would be
    /// paying twice to remove the same room tone.
    pub fn without_noise_reduction(&self) -> Self {
        Self {
            noise_reduction: None,
            ..*self
        }
    }
}

/// What the chain actually did, for `metrics.json` and for explaining a
/// surprising result to a user.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PreprocessReport {
    pub input_sample_rate: u32,
    pub input_channels: u16,
    pub removed_dc_offset: f32,
    pub applied_gain_db: f32,
    pub input_rms_dbfs: f32,
    pub output_rms_dbfs: f32,
    pub output_peak_dbfs: f32,
    /// `None` when the stage did not run. "Enabled" and "did something" are
    /// different claims, and only the second one is evidence.
    pub compressor: Option<enhance::CompressorReport>,
    pub noise_reduction: Option<enhance::NoiseReductionReport>,
    /// Defaulted so a report written before the leveler existed still loads.
    #[serde(default)]
    pub leveler: Option<enhance::LevelerReport>,
}

/// Downmixes to mono and resamples to 16kHz via linear interpolation.
///
/// Audio preprocessing for ASR input doesn't need broadcast-grade sinc
/// resampling — whisper.cpp is robust to this — and a hand-written linear
/// resampler avoids pulling in a heavier DSP dependency with its own
/// API-stability risk.
///
/// Format conversion only. [`preprocess`] is the full chain; this stays as it
/// was so the benchmark harness can measure conversion on its own.
pub fn resample_to_16k_mono(input_path: &Path, output_path: &Path) -> Result<(), String> {
    let mut reader = WavReader::open(input_path).map_err(|e| e.to_string())?;
    let spec = reader.spec();

    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "unsupported WAV format for preprocessing: {:?}/{}-bit (expected 16-bit PCM, which is all audio.rs ever writes)",
            spec.sample_format, spec.bits_per_sample
        ));
    }

    let channels = spec.channels as usize;
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    let mono: Vec<f32> = samples
        .chunks(channels.max(1))
        .map(|frame| {
            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum as f32) / (frame.len().max(1) as f32)
        })
        .collect();

    let resampled = if spec.sample_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        linear_resample(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)
    };

    let out_spec = WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(output_path, out_spec).map_err(|e| e.to_string())?;
    for sample in resampled {
        let clamped = sample.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(clamped).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(())
}

/// The full front-end: read any WAV, downmix, resample to 16 kHz, then apply
/// whichever of DC removal / high-pass / normalisation is enabled.
pub fn preprocess(
    input_path: &Path,
    output_path: &Path,
    options: &PreprocessOptions,
) -> Result<PreprocessReport, String> {
    let (mut samples, input_rate, input_channels) = read_mono_16k(input_path)?;
    let mut report = apply_chain(&mut samples, TARGET_SAMPLE_RATE, options);
    report.input_sample_rate = input_rate;
    report.input_channels = input_channels;
    write_mono_16k(output_path, &samples)?;
    Ok(report)
}

/// Runs only the level stages over an already 16 kHz mono file. This is the
/// second half of [`preprocess`], applied after denoising.
///
/// "Level stages" means the leveler, the compressor and normalise, and
/// nothing else: DC
/// removal and the high-pass already ran before the denoiser, and re-running
/// the model-free noise reduction over audio a *model* has just cleaned would
/// spend a whole STFT chasing a room tone that is no longer there.
pub fn apply_gain(
    input_path: &Path,
    output_path: &Path,
    options: &PreprocessOptions,
) -> Result<PreprocessReport, String> {
    let gain_only = PreprocessOptions {
        remove_dc: false,
        high_pass_hz: 0.0,
        noise_reduction: None,
        ..*options
    };
    preprocess(input_path, output_path, &gain_only)
}

/// The DSP itself, split out so it is testable without touching the disk.
///
/// The order is **DC removal → high-pass → noise reduction → leveler →
/// compressor → normalise**, and it is not arbitrary:
///
/// * DC removal and the high-pass come first because everything after them
///   measures levels, and a constant offset or a shelf of rumble makes every
///   one of those measurements describe the room's plumbing rather than its
///   speech.
/// * Noise reduction comes before the leveler, so the leveler's speech/pause
///   decision is made against a lower, cleaner floor: the further speech
///   sits above the room, the fewer quiet words are mistaken for the room.
/// * The leveler comes before the compressor, so the compressor only has the
///   fast peaks the slow leveler cannot follow left to catch, rather than
///   fighting the leveler over the same phrase-length level. Both come after
///   noise reduction for the same reason: each raises whatever is quiet, and
///   left to itself would happily raise hiss.
/// * The compressor comes before normalise, so normalise is still the stage
///   that sets the final level and holds the peak ceiling. Compressing after
///   it would undo the ceiling the limiter had just enforced.
pub fn apply_chain(
    samples: &mut [f32],
    sample_rate: u32,
    options: &PreprocessOptions,
) -> PreprocessReport {
    let input_rms = rms(samples);
    let mut report = PreprocessReport {
        input_sample_rate: sample_rate,
        input_channels: 1,
        input_rms_dbfs: amplitude_to_dbfs(input_rms),
        ..PreprocessReport::default()
    };

    if options.remove_dc {
        report.removed_dc_offset = remove_dc_offset(samples);
    }
    if options.high_pass_hz > 0.0 && options.high_pass_hz < sample_rate as f32 / 2.0 {
        Biquad::high_pass(options.high_pass_hz, sample_rate as f32).process(samples);
    }
    if let Some(nr) = options.noise_reduction.as_ref() {
        // `reduce_noise` builds a new signal rather than working in place, so
        // that a disappointing result costs nothing to discard. Here the
        // caller owns the buffer, so the cleaned signal is copied back over
        // it — same length by construction, but zipped rather than indexed so
        // a future change to that guarantee cannot become an out-of-bounds.
        let (cleaned, nr_report) = enhance::reduce_noise(samples, sample_rate, nr);
        for (slot, value) in samples.iter_mut().zip(cleaned) {
            *slot = value;
        }
        report.noise_reduction = Some(nr_report);
    }
    if let Some(leveler) = options.leveler.as_ref() {
        report.leveler = Some(enhance::level(samples, sample_rate, leveler));
    }
    if let Some(compressor) = options.compressor.as_ref() {
        report.compressor = Some(enhance::compress(samples, sample_rate, compressor));
    }
    if options.normalize {
        report.applied_gain_db =
            normalize(samples, options.target_rms_dbfs, options.peak_ceiling_dbfs);
    }

    report.output_rms_dbfs = amplitude_to_dbfs(rms(samples));
    report.output_peak_dbfs =
        amplitude_to_dbfs(samples.iter().map(|s| s.abs()).fold(0f32, f32::max));
    report
}

/// Subtracts the mean, returning the offset that was removed.
pub fn remove_dc_offset(samples: &mut [f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| s as f64).sum();
    let mean = (sum / samples.len() as f64) as f32;
    if mean == 0.0 {
        return 0.0;
    }
    for s in samples.iter_mut() {
        *s -= mean;
    }
    mean
}

/// Direct-form-I RBJ cookbook biquad. One filter, written by hand, is a lot
/// less risk than a DSP crate for the one shape this pipeline needs.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// Butterworth-ish 2nd-order high-pass (Q = 1/sqrt(2), i.e. maximally flat
    /// passband) at `cutoff_hz`.
    pub fn high_pass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let x0 = *s;
            let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x0;
            self.y2 = self.y1;
            self.y1 = y0;
            *s = y0;
        }
    }
}

/// Scales `samples` so their RMS lands on `target_rms_dbfs`, then holds the
/// peak at or below `ceiling_dbfs`. Returns the gain actually applied, in dB.
///
/// The peak constraint wins over the RMS target: a recording that is already
/// hot enough to clip is left alone rather than "normalised" into distortion.
pub fn normalize(samples: &mut [f32], target_rms_dbfs: f32, ceiling_dbfs: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let current_rms = rms(samples);
    let peak = samples.iter().map(|s| s.abs()).fold(0f32, f32::max);
    if current_rms <= 0.0 || peak <= 0.0 {
        return 0.0;
    }

    let target_rms = dbfs_to_amplitude(target_rms_dbfs);
    let ceiling = dbfs_to_amplitude(ceiling_dbfs);
    let mut gain = target_rms / current_rms;
    // Never push the loudest sample past the ceiling, and never lift a nearly
    // silent recording's noise floor to speech level.
    gain = gain
        .min(ceiling / peak)
        .min(dbfs_to_amplitude(MAX_BOOST_DB));

    if !gain.is_finite() || (gain - 1.0).abs() < 1e-4 {
        return 0.0;
    }
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
    amplitude_to_dbfs(gain).max(-MAX_BOOST_DB)
}

fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 {
        return Vec::new();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

/// Reads any WAV `hound` can decode as mono 16 kHz f32 in -1.0..1.0.
pub fn read_mono_16k(path: &Path) -> Result<(Vec<f32>, u32, u16), String> {
    let mut reader =
        WavReader::open(path).map_err(|e| format!("could not open {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
        SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale).map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, String>>()?
        }
    };

    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len().max(1) as f32)
        .collect();

    let resampled = if spec.sample_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        linear_resample(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)
    };
    Ok((resampled, spec.sample_rate, spec.channels))
}

/// Writes 16-bit mono PCM at 16 kHz — the format every downstream native tool
/// in this pipeline expects.
pub fn write_mono_16k(path: &Path, samples: &[f32]) -> Result<(), String> {
    let bytes = encode_mono_16k(samples)?;
    std::fs::write(path, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))
}

/// The same WAV, in memory. Code-switched STT cuts dozens of blocks out
/// of one recording and posts each to whisper-server; none of them is an
/// artifact worth keeping, so none of them touches the disk.
pub fn encode_mono_16k(samples: &[f32]) -> Result<Vec<u8>, String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buffer = std::io::Cursor::new(Vec::new());
    {
        let mut writer =
            WavWriter::new(&mut buffer, spec).map_err(|e| format!("could not encode WAV: {e}"))?;
        for &s in samples {
            // 32767, not 32768: scaling by 32768 makes an exact -1.0/+1.0
            // sample wrap on the positive side.
            let value = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            writer.write_sample(value).map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
    }
    Ok(buffer.into_inner())
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
    fn upsamples_to_expected_length() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let out = linear_resample(&input, 8000, 16000);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn identity_when_same_rate() {
        let input = vec![0.0, 1.0, 2.0];
        let out = linear_resample(&input, 16000, 16000);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn downsamples_to_expected_length() {
        let input: Vec<f32> = (0..16000).map(|i| i as f32).collect();
        let out = linear_resample(&input, 16000, 8000);
        assert_eq!(out.len(), 8000);
    }

    // ------------------------------------------------------------ DC offset

    #[test]
    fn dc_removal_centres_the_signal() {
        let mut samples: Vec<f32> = tone(16_000, 0.5, 200.0, 0.3)
            .iter()
            .map(|s| s + 0.2)
            .collect();
        let removed = remove_dc_offset(&mut samples);
        assert!((removed - 0.2).abs() < 1e-3, "removed {removed}");
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 1e-4, "mean after removal {mean}");
    }

    #[test]
    fn dc_removal_is_a_no_op_on_centred_or_empty_input() {
        let mut empty: Vec<f32> = Vec::new();
        assert_eq!(remove_dc_offset(&mut empty), 0.0);
        let mut centred = vec![0.5f32, -0.5, 0.5, -0.5];
        assert_eq!(remove_dc_offset(&mut centred), 0.0);
        assert_eq!(centred, vec![0.5, -0.5, 0.5, -0.5]);
    }

    // ---------------------------------------------------------- high-pass

    #[test]
    fn high_pass_kills_rumble_and_keeps_speech() {
        let sr = 16_000u32;
        let reference = amplitude_to_dbfs(0.5 / std::f32::consts::SQRT_2);
        // One biquad is 12 dB/octave, so 30 Hz (HVAC, desk thumps) lands about
        // 17 dB down and 10 Hz (footfall, door slams) about 36 dB down, while
        // speech is untouched. That is the trade: cheap, and enough.
        for (freq, min_attenuation_db) in [(10.0f32, 30.0f32), (30.0, 15.0)] {
            let mut rumble = tone(sr, 1.0, freq, 0.5);
            Biquad::high_pass(80.0, sr as f32).process(&mut rumble);
            // Ignore the filter's start-up transient when measuring.
            let attenuation = reference - amplitude_to_dbfs(rms(&rumble[sr as usize / 2..]));
            assert!(
                attenuation > min_attenuation_db,
                "{freq} Hz was only attenuated by {attenuation} dB"
            );
        }

        let mut voice = tone(sr, 1.0, 1_000.0, 0.5);
        Biquad::high_pass(80.0, sr as f32).process(&mut voice);
        let voice_rms = rms(&voice[sr as usize / 2..]);
        assert!(voice_rms > 0.33, "1 kHz was attenuated to {voice_rms}");
    }

    #[test]
    fn high_pass_is_minus_three_db_at_its_corner() {
        let sr = 16_000u32;
        let mut at_corner = tone(sr, 1.0, 80.0, 0.5);
        Biquad::high_pass(80.0, sr as f32).process(&mut at_corner);
        let gain_db = amplitude_to_dbfs(rms(&at_corner[sr as usize / 2..]))
            - amplitude_to_dbfs(0.5 / std::f32::consts::SQRT_2);
        assert!((gain_db + 3.0).abs() < 0.6, "corner gain was {gain_db} dB");
    }

    #[test]
    fn high_pass_removes_a_constant_offset_too() {
        let mut dc = vec![0.4f32; 8_000];
        Biquad::high_pass(80.0, 16_000.0).process(&mut dc);
        assert!(rms(&dc[4_000..]).abs() < 1e-3);
    }

    // --------------------------------------------------------- normalise

    #[test]
    fn normalisation_hits_the_target_rms() {
        let mut samples = tone(16_000, 1.0, 300.0, 0.02);
        let gain_db = normalize(&mut samples, -20.0, -1.0);
        assert!(gain_db > 0.0);
        assert!((amplitude_to_dbfs(rms(&samples)) + 20.0).abs() < 0.5);
    }

    #[test]
    fn normalisation_attenuates_something_too_loud() {
        let mut samples = tone(16_000, 1.0, 300.0, 0.9);
        let gain_db = normalize(&mut samples, -20.0, -1.0);
        assert!(gain_db < 0.0, "expected attenuation, got {gain_db} dB");
        assert!((amplitude_to_dbfs(rms(&samples)) + 20.0).abs() < 0.5);
    }

    #[test]
    fn normalisation_never_clips() {
        // A signal with a big crest factor: hitting the RMS target naively
        // would push the peak well past full scale.
        let mut samples = tone(16_000, 1.0, 300.0, 0.05);
        samples[500] = 0.95;
        samples[501] = -0.95;
        normalize(&mut samples, -20.0, -1.0);
        let peak = samples.iter().map(|s| s.abs()).fold(0f32, f32::max);
        assert!(peak <= dbfs_to_amplitude(-1.0) + 1e-4, "peak {peak}");
    }

    #[test]
    fn normalisation_will_not_amplify_silence_into_hiss() {
        let mut samples = tone(16_000, 1.0, 300.0, 1e-6);
        let gain_db = normalize(&mut samples, -20.0, -1.0);
        assert!(gain_db <= MAX_BOOST_DB + 1e-3, "boosted by {gain_db} dB");
    }

    #[test]
    fn normalising_true_silence_or_nothing_is_a_no_op() {
        let mut silence = vec![0.0f32; 100];
        assert_eq!(normalize(&mut silence, -20.0, -1.0), 0.0);
        assert!(silence.iter().all(|&s| s == 0.0));
        let mut empty: Vec<f32> = Vec::new();
        assert_eq!(normalize(&mut empty, -20.0, -1.0), 0.0);
    }

    // ------------------------------------------------------------- chain

    #[test]
    fn the_chain_reports_what_it_did() {
        let sr = 16_000u32;
        let mut samples: Vec<f32> = tone(sr, 1.0, 500.0, 0.02)
            .iter()
            .zip(tone(sr, 1.0, 25.0, 0.1).iter())
            .map(|(v, r)| v + r + 0.05)
            .collect();
        let report = apply_chain(&mut samples, sr, &PreprocessOptions::default());

        assert!((report.removed_dc_offset - 0.05).abs() < 5e-3);
        assert!(report.applied_gain_db > 0.0);
        assert!(report.output_peak_dbfs <= -1.0 + 0.1);
        assert!((report.output_rms_dbfs + 20.0).abs() < 1.0);
    }

    #[test]
    fn every_stage_can_be_turned_off() {
        let sr = 16_000u32;
        let original = tone(sr, 0.2, 30.0, 0.3);
        let mut samples = original.clone();
        let options = PreprocessOptions {
            remove_dc: false,
            high_pass_hz: 0.0,
            normalize: false,
            compressor: None,
            leveler: None,
            noise_reduction: None,
            ..PreprocessOptions::default()
        };
        let report = apply_chain(&mut samples, sr, &options);
        assert_eq!(samples, original);
        assert_eq!(report.applied_gain_db, 0.0);
        assert_eq!(report.removed_dc_offset, 0.0);
        assert!(report.compressor.is_none());
        assert!(report.leveler.is_none());
        assert!(report.noise_reduction.is_none());
    }

    #[test]
    fn without_gain_drops_every_level_stage_and_nothing_else() {
        let options = PreprocessOptions {
            noise_reduction: Some(enhance::NoiseReductionOptions::default()),
            ..PreprocessOptions::default()
        }
        .without_gain();
        assert!(!options.normalize);
        // The compressor and the leveler are gain stages too: leaving either
        // in would hand the GTCRN denoiser a signal whose noise floor moves.
        assert!(options.compressor.is_none());
        assert!(options.leveler.is_none());
        // Cleaning stages stay.
        assert!(options.remove_dc);
        assert_eq!(options.high_pass_hz, 80.0);
        assert!(options.noise_reduction.is_some());
    }

    #[test]
    fn without_noise_reduction_drops_only_the_denoiser() {
        let options = PreprocessOptions {
            noise_reduction: Some(enhance::NoiseReductionOptions { strength: 0.9 }),
            ..PreprocessOptions::default()
        }
        .without_noise_reduction();
        assert!(options.noise_reduction.is_none());
        assert!(options.remove_dc);
        assert_eq!(options.high_pass_hz, 80.0);
        assert!(options.normalize);
        assert!(options.compressor.is_some());
        assert!(options.leveler.is_some());
    }

    #[test]
    fn the_chain_runs_dc_then_high_pass_then_noise_reduction_then_leveler_then_compressor_then_normalise(
    ) {
        // Each assertion below pins one *adjacency* in the order, by checking
        // a number that could only have been measured at that point in the
        // chain — rather than by asserting on the samples, which would only
        // say the output changed.
        let sr = 16_000u32;
        let mut source: Vec<f32> = Vec::new();
        // A near and a far talker taking turns, with pauses between them so
        // the leveler has a floor to tell them from.
        for burst in 0..6 {
            let amplitude = if burst % 2 == 0 { 0.05 } else { 0.008 };
            source.extend(tone(sr, 0.5, 1_000.0, amplitude));
            source.extend(vec![0.0f32; sr as usize / 4]);
        }
        // Broadband room tone for the denoiser to find, 25 Hz of rumble for
        // the high-pass to remove, and a constant offset for DC removal.
        let mut state = 7u32;
        let rumble = tone(sr, source.len() as f32 / sr as f32, 25.0, 0.02);
        for (i, s) in source.iter_mut().enumerate() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let hiss = (((state >> 16) as f32 / 32_768.0) - 1.0) * 0.004;
            *s += hiss + rumble.get(i).copied().unwrap_or(0.0) + 0.05;
        }

        let nr = Some(enhance::NoiseReductionOptions { strength: 1.0 });
        let run_keeping_audio = |options: &PreprocessOptions| {
            let mut samples = source.clone();
            let report = apply_chain(&mut samples, sr, options);
            (report, samples)
        };
        let run = |options: &PreprocessOptions| run_keeping_audio(options).0;
        let crest = |r: &PreprocessReport| r.output_peak_dbfs - r.output_rms_dbfs;
        // What the leveler would measure as the floor, if handed this audio.
        let floor_seen_in = |mut samples: Vec<f32>| {
            enhance::level(&mut samples, sr, &enhance::LevelerOptions::default()).noise_floor_dbfs
        };

        let (cleaned_only, cleaned_audio) = run_keeping_audio(&PreprocessOptions {
            normalize: false,
            compressor: None,
            leveler: None,
            noise_reduction: None,
            ..PreprocessOptions::default()
        });
        let (cleaned_and_denoised, denoised_audio) = run_keeping_audio(&PreprocessOptions {
            normalize: false,
            compressor: None,
            leveler: None,
            noise_reduction: nr,
            ..PreprocessOptions::default()
        });
        let up_to_the_leveler = run(&PreprocessOptions {
            normalize: false,
            compressor: None,
            noise_reduction: nr,
            ..PreprocessOptions::default()
        });
        let up_to_the_compressor = run(&PreprocessOptions {
            normalize: false,
            noise_reduction: nr,
            ..PreprocessOptions::default()
        });
        let full = run(&PreprocessOptions {
            noise_reduction: nr,
            ..PreprocessOptions::default()
        });

        // DC removal ran *before* the high-pass: it still had the whole offset
        // to find. Had the filter gone first there would have been nothing
        // left to remove and this would read ~0.
        assert!(
            (full.removed_dc_offset - 0.05).abs() < 5e-3,
            "removed {}",
            full.removed_dc_offset
        );

        // Noise reduction ran *before* the leveler: the floor the leveler
        // steered by is the denoised signal's, not the merely-filtered one's.
        let leveler = full.leveler.expect("the leveler ran");
        let denoised_floor = floor_seen_in(denoised_audio);
        let filtered_floor = floor_seen_in(cleaned_audio);
        assert!(
            (leveler.noise_floor_dbfs - denoised_floor).abs() < 0.05,
            "leveler saw a floor of {} dBFS, denoised audio has {}",
            leveler.noise_floor_dbfs,
            denoised_floor
        );
        assert!(
            (denoised_floor - filtered_floor).abs() > 1.0,
            "the denoiser did not move the floor, so this test proves nothing"
        );
        assert!(
            (crest(&cleaned_and_denoised) - crest(&cleaned_only)).abs() > 0.05,
            "the denoiser did nothing, so this test proves nothing"
        );

        // The leveler ran *before* the compressor: the crest the compressor
        // reports at its input is the leveled signal's, not the denoised one's.
        let compressor = full.compressor.expect("the compressor ran");
        assert!(
            (compressor.crest_before_db - crest(&up_to_the_leveler)).abs() < 0.05,
            "compressor saw {} dB of crest, leveled audio has {}",
            compressor.crest_before_db,
            crest(&up_to_the_leveler)
        );
        assert!(
            (crest(&up_to_the_leveler) - crest(&cleaned_and_denoised)).abs() > 0.05,
            "the leveler did nothing, so this test proves nothing"
        );

        // The compressor ran *before* normalise: the gain normalise chose is
        // exactly the gain that takes the *compressed* level to the target.
        let expected_gain = PreprocessOptions::default().target_rms_dbfs
            - up_to_the_compressor.output_rms_dbfs;
        assert!(
            (full.applied_gain_db - expected_gain).abs() < 0.05,
            "normalise applied {} dB, expected {} dB from the compressed level",
            full.applied_gain_db,
            expected_gain
        );

        // And normalise, being last, is still what holds the peak ceiling.
        assert!(
            full.output_peak_dbfs <= PreprocessOptions::default().peak_ceiling_dbfs + 0.1,
            "peak escaped the ceiling at {} dBFS",
            full.output_peak_dbfs
        );
        assert!(full.noise_reduction.is_some());
    }

    #[test]
    fn every_stage_is_on_by_default_and_gtcrn_still_drops_the_model_free_denoiser() {
        let options = PreprocessOptions::default();
        assert_eq!(options.compressor, Some(enhance::CompressorOptions::default()));
        assert_eq!(options.leveler, Some(enhance::LevelerOptions::default()));
        assert_eq!(
            options.noise_reduction,
            Some(enhance::NoiseReductionOptions::default())
        );
        // The GTCRN path strips it, so the two denoisers never both run.
        assert!(options.without_noise_reduction().noise_reduction.is_none());
    }

    #[test]
    fn apply_gain_runs_the_level_stages_and_skips_the_denoiser() {
        let dir = std::env::temp_dir().join("cadence-preprocess-gain-stages");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.wav");
        let output = dir.join("out.wav");
        // Loud enough for the compressor to have something to reduce.
        let mut samples = tone(16_000, 1.0, 400.0, 0.5);
        samples.extend(tone(16_000, 1.0, 400.0, 0.02));
        write_mono_16k(&input, &samples).unwrap();

        let options = PreprocessOptions {
            noise_reduction: Some(enhance::NoiseReductionOptions::default()),
            ..PreprocessOptions::default()
        };
        let report = apply_gain(&input, &output, &options).unwrap();
        // The compressor and the leveler are level stages, so they run...
        assert!(report.compressor.is_some());
        assert!(report.leveler.is_some());
        // ...and the denoiser is not, so it does not: GTCRN has already
        // cleaned this audio and a second pass would only cost an STFT.
        assert!(report.noise_reduction.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_drive_the_options() {
        let settings = Settings {
            preprocess_high_pass_hz: 120.0,
            preprocess_normalize: false,
            preprocess_target_rms_dbfs: -18.0,
            ..Settings::default()
        };
        let options = PreprocessOptions::from_settings(&settings);
        assert_eq!(options.high_pass_hz, 120.0);
        assert!(!options.normalize);
        assert_eq!(options.target_rms_dbfs, -18.0);
        // Defaults still hold for anything the user did not touch.
        assert_eq!(
            PreprocessOptions::from_settings(&Settings::default()),
            PreprocessOptions::default()
        );
    }

    #[test]
    fn preprocess_writes_a_16k_mono_file_and_leaves_the_original_alone() {
        let dir = std::env::temp_dir().join("cadence-preprocess-chain");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("original.wav");
        let output = dir.join("processed.wav");

        // Stereo, 44.1 kHz, quiet — the awkward case.
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&input, spec).unwrap();
        for s in tone(44_100, 1.0, 440.0, 0.02) {
            let v = (s * 32767.0) as i16;
            writer.write_sample(v).unwrap();
            writer.write_sample(v).unwrap();
        }
        writer.finalize().unwrap();
        let before = std::fs::read(&input).unwrap();

        let report = preprocess(&input, &output, &PreprocessOptions::default()).unwrap();
        assert_eq!(report.input_sample_rate, 44_100);
        assert_eq!(report.input_channels, 2);
        assert!(report.applied_gain_db > 10.0);

        let out_reader = WavReader::open(&output).unwrap();
        assert_eq!(out_reader.spec().channels, 1);
        assert_eq!(out_reader.spec().sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(out_reader.spec().bits_per_sample, 16);

        // the untouched recording stays byte-identical.
        assert_eq!(std::fs::read(&input).unwrap(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_gain_only_touches_the_level() {
        let dir = std::env::temp_dir().join("cadence-preprocess-gain");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.wav");
        let output = dir.join("out.wav");
        // A 30 Hz tone would be destroyed by the high-pass; apply_gain must
        // leave it alone and only change the level.
        write_mono_16k(&input, &tone(16_000, 1.0, 30.0, 0.02)).unwrap();

        let report = apply_gain(&input, &output, &PreprocessOptions::default()).unwrap();
        assert!(report.applied_gain_db > 10.0);
        let (samples, _, _) = read_mono_16k(&output).unwrap();
        assert!((amplitude_to_dbfs(rms(&samples)) + 20.0).abs() < 0.6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wav_round_trips_through_16_bit_without_wrapping() {
        let dir = std::env::temp_dir().join("cadence-preprocess-roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full-scale.wav");
        write_mono_16k(&path, &[1.0, -1.0, 0.5, 0.0, 2.0, -2.0]).unwrap();
        let (samples, _, _) = read_mono_16k(&path).unwrap();
        assert!((samples[0] - 1.0).abs() < 1e-3);
        assert!((samples[1] + 1.0).abs() < 1e-3);
        assert!((samples[2] - 0.5).abs() < 1e-3);
        assert_eq!(samples[3], 0.0);
        // Out-of-range input is clamped, never wrapped to the opposite sign.
        assert!(samples[4] > 0.9);
        assert!(samples[5] < -0.9);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
