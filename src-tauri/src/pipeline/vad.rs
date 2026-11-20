//! Voice activity detection — the pipeline stage that was specified but
//! never built.
//!
//! It earns its place twice over:
//!
//! 1. **Speech ratio.** How much of a recording is actually someone talking is
//!    a quality signal in its own right: a two-hour file that is 4% speech
//!    usually means the wrong device was captured, and it explains a thin
//!    transcript without anyone having to listen back.
//! 2. **Skipping silence before STT.** whisper.cpp costs the same per second
//!    whether or not anyone is speaking, so dropping the gaps goes straight to
//!    the processing-time budget.
//!
//! ## The timestamp trap
//!
//! Feeding whisper speech-only audio means every timestamp it returns is on
//! the *trimmed* timeline, while the transcript, the diarization alignment,
//! the MoM evidence links and click-to-play in the UI all assume the original
//! recording's timeline. [`Timeline`] is the mapping back, and it is why this
//! module — not sherpa-onnx — does the actual cutting: sherpa-onnx-vad writes
//! a speech-only WAV but hands back no way to relate it to the input, so
//! trusting its file would mean guessing at the offsets. Instead this module
//! parses the segment list sherpa-onnx prints and cuts the audio itself, in
//! samples, so the map is exact by construction.
//!
//! Residual inexactness (and the reason trimming is off by default — see
//! `Settings::vad_trim_silence`): a whisper segment that *straddles* a splice
//! gets correct endpoints but a stretched interior, because the silence
//! between them no longer exists to place words in.

use crate::pipeline::models::SttSegment;
use crate::pipeline::paths;
use crate::pipeline::preprocess;
use crate::pipeline::proc::{self, StageMetrics};
use crate::models_manager;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Catalog id of the Silero VAD weights. Installed with the required set,
/// but a run without it still succeeds — see [`is_available`].
pub const VAD_MODEL_ID: &str = "vad-silero";

const VAD_BINARY: &str = "sherpa-onnx-vad.exe";

/// Below this, the recording is mostly not speech and worth flagging.
const LOW_SPEECH_RATIO: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechSegment {
    pub start_ms: i64,
    pub end_ms: i64,
}

impl SpeechSegment {
    pub fn duration_ms(&self) -> i64 {
        (self.end_ms - self.start_ms).max(0)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VadReport {
    pub segments: Vec<SpeechSegment>,
    pub total_ms: i64,
    pub speech_ms: i64,
    /// 0.0..1.0 — the share of the recording that is speech.
    pub speech_ratio: f32,
}

/// `sherpa-onnx-vad.exe` ships with the sherpa-onnx runtime, so it runs from
/// that runtime's folder.
pub fn binary_path(data_dir: &Path) -> PathBuf {
    paths::runtime_binary(data_dir, models_manager::Runtime::Sherpa, VAD_BINARY)
}

pub fn model_path(data_dir: &Path) -> PathBuf {
    match models_manager::entry(VAD_MODEL_ID) {
        Some(entry) => entry.path_in(&paths::models_dir(data_dir)),
        None => paths::models_dir(data_dir).join("vad").join("silero_vad.onnx"),
    }
}

/// True when both the sherpa-onnx VAD binary and the Silero weights are
/// installed. The model is part of the installed set, but the pipeline still
/// runs without it: a missing file is a skip, never a lost meeting.
pub fn is_available(data_dir: &Path) -> bool {
    binary_path(data_dir).exists() && model_path(data_dir).exists()
}

/// Runs sherpa-onnx's VAD over a 16 kHz mono WAV and returns the speech
/// segments it found.
///
/// The tool insists on an output WAV argument (it writes a speech-only copy),
/// so one is written to `scratch_wav` and immediately discarded — the segment
/// list, which it prints to **stderr** as `"<start> -- <end>"` lines, is the
/// part this pipeline needs.
pub fn detect(
    data_dir: &Path,
    threads: usize,
    wav_path: &Path,
    scratch_wav: &Path,
) -> Result<(VadReport, StageMetrics), String> {
    let binary = binary_path(data_dir);
    let model = model_path(data_dir);
    if !binary.exists() || !model.exists() {
        return Err(format!(
            "voice activity detection is not installed ({} or {} is missing)",
            binary.display(),
            model.display()
        ));
    }

    let mut command = Command::new(binary);
    command
        .arg(format!("--silero-vad-model={}", model.display()))
        .arg(format!("--vad-num-threads={}", threads.max(1)))
        .arg(wav_path)
        .arg(scratch_wav);

    let measured = proc::run_measured(command)?;
    let _ = std::fs::remove_file(scratch_wav);
    if !measured.success {
        return Err(format!(
            "sherpa-onnx VAD exited with {}: {}",
            measured.status,
            measured.stderr.trim()
        ));
    }

    let total_ms = wav_duration_ms(wav_path)?;
    let segments = parse_segments(&measured.stderr);
    Ok((report(segments, total_ms), measured.metrics))
}

fn wav_duration_ms(path: &Path) -> Result<i64, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| format!("could not open {}: {e}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Ok(0);
    }
    Ok((reader.duration() as i64 * 1000) / spec.sample_rate as i64)
}

/// sherpa-onnx prints one segment per line as `"0.070 -- 1.644"`, the same
/// shape its diarization tool uses, mixed in with its config dump — hence the
/// tolerant per-line parse rather than a strict format.
fn parse_segments(output: &str) -> Vec<SpeechSegment> {
    output.lines().filter_map(parse_segment_line).collect()
}

fn parse_segment_line(line: &str) -> Option<SpeechSegment> {
    let (start, end) = line.trim().split_once(" -- ")?;
    let start_sec: f64 = start.trim().parse().ok()?;
    // Anything after the end time (a speaker token, say) is not ours to read.
    let end_sec: f64 = end.trim().split_whitespace().next()?.parse().ok()?;
    if end_sec < start_sec {
        return None;
    }
    Some(SpeechSegment {
        start_ms: (start_sec * 1000.0).round() as i64,
        end_ms: (end_sec * 1000.0).round() as i64,
    })
}

fn report(segments: Vec<SpeechSegment>, total_ms: i64) -> VadReport {
    let speech_ms: i64 = segments.iter().map(|s| s.duration_ms()).sum();
    let speech_ratio = if total_ms > 0 {
        (speech_ms as f32 / total_ms as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    VadReport {
        segments,
        total_ms,
        speech_ms,
        speech_ratio,
    }
}

/// Quality advice derived from the speech ratio, in the same plain language as
/// [`crate::pipeline::audio_profile`]'s warnings.
pub fn speech_ratio_warning(ratio: f32) -> Option<String> {
    if ratio >= LOW_SPEECH_RATIO {
        return None;
    }
    Some(format!(
        "Only about {}% of this recording contains speech. If that is unexpected, the microphone may have been picking up the wrong device or was too far from the people talking.",
        (ratio * 100.0).round() as i64
    ))
}

// ------------------------------------------------------------- the mapping

/// The kept regions of a recording, in samples on the **original** timeline,
/// plus the arithmetic to move a timestamp between the original and the
/// trimmed (speech-only) timelines.
///
/// Samples, not milliseconds, are the unit on purpose: the trimmed file is
/// built by concatenating exactly these sample ranges, so cumulative offsets
/// are exact rather than accumulating a rounding error per segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timeline {
    sample_rate: u32,
    kept: Vec<(usize, usize)>,
}

impl Timeline {
    /// Pads each segment (so a trim never clips a word onset), clamps to the
    /// recording, and merges anything that now overlaps or touches.
    pub fn build(
        segments: &[SpeechSegment],
        pad_ms: i64,
        total_samples: usize,
        sample_rate: u32,
    ) -> Self {
        let pad = ms_to_samples(pad_ms.max(0), sample_rate);
        let mut ranges: Vec<(usize, usize)> = segments
            .iter()
            .filter_map(|seg| {
                let start = ms_to_samples(seg.start_ms, sample_rate).saturating_sub(pad);
                let end = (ms_to_samples(seg.end_ms, sample_rate) + pad).min(total_samples);
                let start = start.min(total_samples);
                (end > start).then_some((start, end))
            })
            .collect();
        ranges.sort_unstable();

        let mut kept: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            match kept.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => kept.push((start, end)),
            }
        }
        Self { sample_rate, kept }
    }

    /// A timeline that keeps everything — the identity mapping. Used whenever
    /// VAD is unavailable or trimming is off, so callers need no special case.
    pub fn identity(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            kept: Vec::new(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.kept.is_empty()
    }

    pub fn segment_count(&self) -> usize {
        self.kept.len()
    }

    pub fn kept_samples(&self) -> usize {
        self.kept.iter().map(|(s, e)| e - s).sum()
    }

    pub fn trimmed_duration_ms(&self) -> i64 {
        samples_to_ms(self.kept_samples(), self.sample_rate)
    }

    /// Maps a timestamp on the trimmed timeline back to the original one.
    ///
    /// Clamps rather than failing: a timestamp before the first kept sample
    /// lands on the start of the first kept region, and one past the end lands
    /// on the end of the last — whisper occasionally reports an end time a few
    /// milliseconds past the audio it was given, and that must not throw away
    /// a transcript segment.
    pub fn to_original_ms(&self, trimmed_ms: i64) -> i64 {
        if self.kept.is_empty() {
            return trimmed_ms;
        }
        let target = ms_to_samples(trimmed_ms.max(0), self.sample_rate);
        let mut consumed = 0usize;
        for &(start, end) in &self.kept {
            let len = end - start;
            if target < consumed + len {
                return samples_to_ms(start + (target - consumed), self.sample_rate);
            }
            consumed += len;
        }
        samples_to_ms(self.kept.last().map(|(_, e)| *e).unwrap_or(0), self.sample_rate)
    }

    /// The inverse. `None` when the timestamp falls in a gap that was cut out
    /// (including before the first kept region or after the last), because
    /// that instant genuinely does not exist on the trimmed timeline.
    pub fn to_trimmed_ms(&self, original_ms: i64) -> Option<i64> {
        if self.kept.is_empty() {
            return Some(original_ms);
        }
        let target = ms_to_samples(original_ms.max(0), self.sample_rate);
        let mut consumed = 0usize;
        for &(start, end) in &self.kept {
            if target < start {
                return None;
            }
            if target < end {
                return Some(samples_to_ms(consumed + (target - start), self.sample_rate));
            }
            consumed += end - start;
        }
        None
    }

    /// Moves a whole STT result back onto the original timeline.
    ///
    /// Applied to `start_ms`/`end_ms` and to every word timestamp, so
    /// alignment, MoM evidence links and click-to-play all keep pointing at
    /// the moment the words were actually said.
    pub fn remap_stt(&self, segments: &mut [SttSegment]) {
        if self.kept.is_empty() {
            return;
        }
        for segment in segments.iter_mut() {
            segment.start_ms = self.to_original_ms(segment.start_ms);
            segment.end_ms = self.to_original_ms(segment.end_ms).max(segment.start_ms);
            for word in segment.words.iter_mut() {
                word.start_ms = self.to_original_ms(word.start_ms);
                word.end_ms = self.to_original_ms(word.end_ms).max(word.start_ms);
            }
        }
    }
}

fn ms_to_samples(ms: i64, sample_rate: u32) -> usize {
    if ms <= 0 {
        return 0;
    }
    ((ms as i128 * sample_rate as i128) / 1000) as usize
}

fn samples_to_ms(samples: usize, sample_rate: u32) -> i64 {
    if sample_rate == 0 {
        return 0;
    }
    ((samples as i128 * 1000) / sample_rate as i128) as i64
}

/// Writes the speech-only WAV that STT will actually read, and returns the
/// timeline that maps its timestamps back. The input file is left untouched.
pub fn write_trimmed(
    input_wav: &Path,
    output_wav: &Path,
    segments: &[SpeechSegment],
    pad_ms: i64,
) -> Result<Timeline, String> {
    let (samples, _, _) = preprocess::read_mono_16k(input_wav)?;
    let timeline = Timeline::build(
        segments,
        pad_ms,
        samples.len(),
        preprocess::TARGET_SAMPLE_RATE,
    );
    if timeline.is_identity() {
        return Err("voice activity detection found no speech to keep".to_string());
    }
    let mut trimmed = Vec::with_capacity(timeline.kept_samples());
    for &(start, end) in &timeline.kept {
        trimmed.extend_from_slice(&samples[start..end]);
    }
    preprocess::write_mono_16k(output_wav, &trimmed)?;
    Ok(timeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    fn seg(start_ms: i64, end_ms: i64) -> SpeechSegment {
        SpeechSegment { start_ms, end_ms }
    }

    #[test]
    fn parses_sherpa_segment_lines() {
        let parsed = parse_segment_line("0.070 -- 1.644").unwrap();
        assert_eq!(parsed.start_ms, 70);
        assert_eq!(parsed.end_ms, 1644);
    }

    #[test]
    fn ignores_the_config_dump_and_anything_malformed() {
        let stderr = "\
VadModelConfig(silero_vad=SileroVadModelConfig(model=\"x.onnx\", threshold=0.5))
0.070 -- 1.644
2.502 -- 5.868
Saved to out.wav
not a segment
5.0 -- oops";
        let segments = parse_segments(stderr);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[1].start_ms, 2502);
        assert!(parse_segment_line("").is_none());
        // An end before the start is nonsense, not a zero-length segment.
        assert!(parse_segment_line("5.0 -- 1.0").is_none());
    }

    #[test]
    fn speech_ratio_is_speech_over_total() {
        let r = report(vec![seg(0, 1_000), seg(2_000, 3_000)], 10_000);
        assert_eq!(r.speech_ms, 2_000);
        assert!((r.speech_ratio - 0.2).abs() < 1e-6);
        // A zero-length recording is 0%, never a division by zero.
        assert_eq!(report(vec![], 0).speech_ratio, 0.0);
    }

    #[test]
    fn a_mostly_silent_recording_gets_a_warning() {
        assert!(speech_ratio_warning(0.9).is_none());
        assert!(speech_ratio_warning(LOW_SPEECH_RATIO).is_none());
        let warning = speech_ratio_warning(0.04).unwrap();
        assert!(warning.contains("4%"), "{warning}");
    }

    // ------------------------------------------------------- the mapping

    /// 10s of audio, speech at 1-2s and 5-7s, no padding.
    fn timeline() -> Timeline {
        Timeline::build(
            &[seg(1_000, 2_000), seg(5_000, 7_000)],
            0,
            10 * SR as usize,
            SR,
        )
    }

    #[test]
    fn trimmed_duration_is_the_sum_of_kept_speech() {
        let t = timeline();
        assert_eq!(t.segment_count(), 2);
        assert_eq!(t.trimmed_duration_ms(), 3_000);
    }

    #[test]
    fn trimmed_timestamps_map_back_to_the_original_timeline() {
        let t = timeline();
        // Start of the trimmed file is the start of the first kept segment.
        assert_eq!(t.to_original_ms(0), 1_000);
        // 500 ms in is still inside the first segment.
        assert_eq!(t.to_original_ms(500), 1_500);
        // 1000 ms in is the first sample of the *second* segment — the splice.
        assert_eq!(t.to_original_ms(1_000), 5_000);
        assert_eq!(t.to_original_ms(1_001), 5_001);
        assert_eq!(t.to_original_ms(2_999), 6_999);
    }

    #[test]
    fn timestamps_outside_the_trimmed_file_clamp_to_its_edges() {
        let t = timeline();
        // Before the start (whisper can report a small negative offset).
        assert_eq!(t.to_original_ms(-500), 1_000);
        // Exactly at, and past, the end of the trimmed audio.
        assert_eq!(t.to_original_ms(3_000), 7_000);
        assert_eq!(t.to_original_ms(9_999), 7_000);
    }

    #[test]
    fn original_timestamps_round_trip_through_the_trimmed_timeline() {
        let t = timeline();
        for original in [1_000i64, 1_001, 1_500, 1_999, 5_000, 6_000, 6_999] {
            let trimmed = t
                .to_trimmed_ms(original)
                .unwrap_or_else(|| panic!("{original} ms should exist on the trimmed timeline"));
            assert_eq!(
                t.to_original_ms(trimmed),
                original,
                "round trip failed for {original} ms"
            );
        }
    }

    #[test]
    fn timestamps_in_a_removed_gap_have_no_trimmed_position() {
        let t = timeline();
        assert_eq!(t.to_trimmed_ms(0), None); // before the first segment
        assert_eq!(t.to_trimmed_ms(999), None); // just before it
        assert_eq!(t.to_trimmed_ms(3_000), None); // inside the cut gap
        assert_eq!(t.to_trimmed_ms(4_999), None); // just before the second
        assert_eq!(t.to_trimmed_ms(7_000), None); // exclusive end boundary
        assert_eq!(t.to_trimmed_ms(9_000), None); // after the last segment
    }

    #[test]
    fn segment_boundaries_are_exact_in_both_directions() {
        let t = timeline();
        assert_eq!(t.to_trimmed_ms(1_000), Some(0));
        assert_eq!(t.to_trimmed_ms(1_999), Some(999));
        assert_eq!(t.to_trimmed_ms(5_000), Some(1_000));
        assert_eq!(t.to_trimmed_ms(6_999), Some(2_999));
    }

    #[test]
    fn an_identity_timeline_changes_nothing() {
        let t = Timeline::identity(SR);
        assert!(t.is_identity());
        assert_eq!(t.to_original_ms(1_234), 1_234);
        assert_eq!(t.to_trimmed_ms(1_234), Some(1_234));
        assert_eq!(t.trimmed_duration_ms(), 0);
    }

    #[test]
    fn padding_widens_segments_and_merges_what_it_joins() {
        // 200 ms of padding closes the 400 ms gap between these two.
        let t = Timeline::build(
            &[seg(1_000, 2_000), seg(2_400, 3_000)],
            200,
            10 * SR as usize,
            SR,
        );
        assert_eq!(t.segment_count(), 1);
        assert_eq!(t.to_original_ms(0), 800);
        assert_eq!(t.trimmed_duration_ms(), 2_400);
    }

    #[test]
    fn padding_never_runs_off_either_end_of_the_recording() {
        let total = 3 * SR as usize;
        let t = Timeline::build(&[seg(50, 2_950)], 500, total, SR);
        assert_eq!(t.to_original_ms(0), 0);
        assert_eq!(t.trimmed_duration_ms(), 3_000);
        assert_eq!(t.kept_samples(), total);
    }

    #[test]
    fn overlapping_and_unsorted_segments_are_merged() {
        let t = Timeline::build(
            &[seg(5_000, 6_000), seg(1_000, 2_000), seg(1_500, 3_000)],
            0,
            10 * SR as usize,
            SR,
        );
        assert_eq!(t.segment_count(), 2);
        assert_eq!(t.trimmed_duration_ms(), 3_000);
        assert_eq!(t.to_original_ms(0), 1_000);
        assert_eq!(t.to_original_ms(2_000), 5_000);
    }

    #[test]
    fn zero_length_and_out_of_range_segments_are_dropped() {
        let t = Timeline::build(
            &[seg(1_000, 1_000), seg(20_000, 21_000), seg(2_000, 3_000)],
            0,
            10 * SR as usize,
            SR,
        );
        assert_eq!(t.segment_count(), 1);
        assert_eq!(t.to_original_ms(0), 2_000);
    }

    #[test]
    fn no_speech_at_all_yields_the_identity_timeline() {
        let t = Timeline::build(&[], 200, 10 * SR as usize, SR);
        assert!(t.is_identity());
    }

    #[test]
    fn many_segments_do_not_accumulate_drift() {
        // 500 one-second segments, each separated by a second of silence, at a
        // sample rate that does not divide evenly into the frame length — the
        // shape that would expose millisecond-domain rounding.
        let segments: Vec<SpeechSegment> = (0..500)
            .map(|i| seg(i * 2_000, i * 2_000 + 1_000))
            .collect();
        let t = Timeline::build(&segments, 0, 1_000 * 44_100, 44_100);
        assert_eq!(t.segment_count(), 500);
        assert_eq!(t.trimmed_duration_ms(), 500_000);
        // The last segment starts at 998_000 ms in the original and at
        // 499_000 ms in the trimmed file; no drift is allowed in between.
        assert_eq!(t.to_original_ms(499_000), 998_000);
        assert_eq!(t.to_trimmed_ms(998_500), Some(499_500));
    }

    #[test]
    fn remapping_moves_stt_segments_and_their_words() {
        use crate::pipeline::models::SttWord;
        let t = timeline();
        let mut segments = vec![
            SttSegment {
                start_ms: 0,
                end_ms: 900,
                text: "first".to_string(),
                words: vec![SttWord {
                    start_ms: 100,
                    end_ms: 400,
                    text: "first".to_string(),
                }],
                language: None,
            },
            SttSegment {
                start_ms: 1_000,
                end_ms: 2_500,
                text: "second".to_string(),
                words: Vec::new(),
                language: None,
            },
        ];
        t.remap_stt(&mut segments);
        assert_eq!(segments[0].start_ms, 1_000);
        assert_eq!(segments[0].end_ms, 1_900);
        assert_eq!(segments[0].words[0].start_ms, 1_100);
        assert_eq!(segments[0].words[0].end_ms, 1_400);
        assert_eq!(segments[1].start_ms, 5_000);
        assert_eq!(segments[1].end_ms, 6_500);
        // Ordering survives the remap — align.rs depends on it.
        assert!(segments[0].end_ms <= segments[1].start_ms);
    }

    #[test]
    fn remapping_with_an_identity_timeline_is_a_no_op() {
        let mut segments = vec![SttSegment {
            start_ms: 10,
            end_ms: 20,
            text: "x".to_string(),
            words: Vec::new(),
            language: None,
        }];
        Timeline::identity(SR).remap_stt(&mut segments);
        assert_eq!(segments[0].start_ms, 10);
        assert_eq!(segments[0].end_ms, 20);
    }

    #[test]
    fn trimming_a_real_file_matches_the_timeline_it_returns() {
        let dir = std::env::temp_dir().join("cadence-vad-trim");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("processed.wav");
        let output = dir.join("speech.wav");

        // 10 s: a marker tone only during 1-2 s and 5-7 s, silence elsewhere.
        let total = 10 * SR as usize;
        let mut samples = vec![0.0f32; total];
        for i in 0..total {
            let t = i as f32 / SR as f32;
            if (1.0..2.0).contains(&t) || (5.0..7.0).contains(&t) {
                samples[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            }
        }
        preprocess::write_mono_16k(&input, &samples).unwrap();
        let before = std::fs::read(&input).unwrap();

        let timeline =
            write_trimmed(&input, &output, &[seg(1_000, 2_000), seg(5_000, 7_000)], 0).unwrap();
        assert_eq!(timeline.trimmed_duration_ms(), 3_000);

        let (trimmed, _, _) = preprocess::read_mono_16k(&output).unwrap();
        assert_eq!(trimmed.len(), timeline.kept_samples());
        // Every kept sample is speech, so nothing in the trimmed file is
        // digital silence for long — the cut landed where the timeline says.
        assert!(crate::pipeline::audio_profile::rms(&trimmed) > 0.3);
        // the input file is never modified.
        assert_eq!(std::fs::read(&input).unwrap(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trimming_refuses_when_there_is_no_speech_to_keep() {
        let dir = std::env::temp_dir().join("cadence-vad-nospeech");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.wav");
        preprocess::write_mono_16k(&input, &vec![0.0f32; SR as usize]).unwrap();
        let err = write_trimmed(&input, &dir.join("out.wav"), &[], 0).unwrap_err();
        assert!(err.contains("no speech"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn availability_is_false_in_an_empty_data_dir() {
        let dir = std::env::temp_dir().join("cadence-vad-missing");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!is_available(&dir));
        assert!(model_path(&dir).to_string_lossy().contains("silero"));
    }
}
