//! Measuring how good a recording actually is, in terms a user can act on.
//!
//! §10 keeps `original.wav` untouched, which means the only lever a user has
//! over audio quality is the room and the microphone — and they only get that
//! lever if someone tells them, in plain words, what is wrong. Everything here
//! is pure analysis: it reads samples and produces numbers plus advice, and
//! never writes audio.
//!
//! The numbers are true dBFS (`20*log10(amplitude)`), not the 0..1 linear
//! scale the live meter used to expose — a linear meter reads "half" at
//! -6 dBFS and makes a badly quiet recording look merely quiet.

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
    pub estimated_snr_db: f32,
    pub clipped_samples: u64,
    pub clipped_percent: f32,
    pub silence_percent: f32,
    pub dc_offset: f32,
    /// Plain-language advice for the UI. Empty means nothing to report.
    pub warnings: Vec<String>,
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

    let mut profile = AudioProfile {
        duration_ms,
        sample_rate,
        channels,
        peak_dbfs: amplitude_to_dbfs(peak),
        rms_dbfs: amplitude_to_dbfs(overall_rms),
        noise_floor_dbfs,
        estimated_snr_db: (speech_dbfs - noise_floor_dbfs).max(0.0),
        clipped_samples: clipped,
        clipped_percent,
        silence_percent,
        dc_offset,
        warnings: Vec::new(),
    };
    profile.warnings = warnings_for(&profile);
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
            "This microphone adds a constant electrical offset to the signal. EchoMind removes it automatically before transcribing, but another microphone or an updated audio driver would give cleaner recordings."
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

/// Profiles a meeting's untouched recording (§10). `data_dir` is the app data
/// directory; the recording is `meetings/<id>/original.wav`.
pub fn profile_meeting(data_dir: &Path, meeting_id: &str) -> Result<AudioProfile, String> {
    let wav = data_dir
        .join("meetings")
        .join(meeting_id)
        .join("original.wav");
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
        let dir = std::env::temp_dir().join("echomind-profile-wav");
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

    #[test]
    fn a_missing_meeting_recording_is_a_clear_error() {
        let dir = std::env::temp_dir().join("echomind-profile-missing");
        let _ = std::fs::remove_dir_all(&dir);
        let err = profile_meeting(&dir, "nope").unwrap_err();
        assert!(err.contains("No recording found"));
    }
}
