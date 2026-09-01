use crate::pipeline::models::SttSegment;
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::proc::{self, StageMetrics};
use crate::pipeline::PipelineConfig;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Deserialize)]
struct WhisperJsonOffsets {
    from: i64,
    to: i64,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonSegment {
    offsets: WhisperJsonOffsets,
    text: String,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonOutput {
    transcription: Vec<WhisperJsonSegment>,
}

/// Transcribes `wav_path` (16kHz mono PCM — see preprocess.rs) via
/// whisper.cpp.
///
/// The `--max-len 80 --split-on-word --max-context 0` flags are
/// load-bearing, not tuning knobs — see docs/models.md: without them
/// whisper.cpp has been observed to silently collapse a real, multi-second
/// span of speech into a short, wrong sentence. `--language auto` is
/// required per master prompt §8 — meetings code-switch between English and
/// Tamil mid-sentence, so the language can never be hard-coded.
pub fn transcribe(
    paths: &ModelPaths,
    config: &PipelineConfig,
    wav_path: &Path,
) -> Result<(Vec<SttSegment>, StageMetrics), String> {
    // whisper.cpp appends `.json` itself, so pass the base path without it.
    let output_base = wav_path.with_extension("");

    let mut command = Command::new(paths.whisper_cli());
    command
        .arg("-m")
        .arg(&paths.stt_model)
        .arg("-f")
        .arg(wav_path)
        .arg("--language")
        .arg("auto")
        .arg("--max-len")
        .arg("80")
        .arg("--split-on-word")
        .arg("--max-context")
        .arg("0")
        .arg("-t")
        .arg(config.stt_threads.to_string())
        .arg("--output-json")
        .arg("--output-file")
        .arg(&output_base);

    let measured = proc::run_measured(command)?;
    if !measured.success {
        return Err(format!(
            "whisper-cli.exe exited with {}: {}",
            measured.status,
            last_lines(&measured.stderr, 4)
        ));
    }

    let json_path = wav_path.with_extension("json");
    let raw = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("could not read whisper output {json_path:?}: {e}"))?;
    let segments = parse_whisper_json(&raw)?;
    Ok((segments, measured.metrics))
}

fn parse_whisper_json(raw: &str) -> Result<Vec<SttSegment>, String> {
    let parsed: WhisperJsonOutput =
        serde_json::from_str(raw).map_err(|e| format!("could not parse whisper JSON: {e}"))?;

    Ok(parsed
        .transcription
        .into_iter()
        .filter_map(|seg| {
            let text = seg.text.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(SttSegment {
                start_ms: seg.offsets.from,
                end_ms: seg.offsets.to,
                text,
                words: Vec::new(),
            })
        })
        .collect())
}

fn last_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(count)..].join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whisper_output() {
        let raw = r#"{
            "transcription": [
                {"offsets": {"from": 0, "to": 2400}, "text": " We should move this to the next sprint."},
                {"offsets": {"from": 2400, "to": 2600}, "text": "   "},
                {"offsets": {"from": 2600, "to": 5000}, "text": "இந்த feature ready ஆச்சா?"}
            ]
        }"#;
        let segments = parse_whisper_json(raw).unwrap();
        // The blank segment is dropped; the others keep their offsets and text.
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].text, "We should move this to the next sprint.");
        assert_eq!(segments[1].text, "இந்த feature ready ஆச்சா?");
    }

    #[test]
    fn reports_unparseable_output() {
        assert!(parse_whisper_json("not json").is_err());
    }

    #[test]
    fn summarises_the_tail_of_an_error() {
        let text = "line one\n\nline two\nline three\nline four\nline five";
        assert_eq!(last_lines(&text, 2), "line four | line five");
        assert_eq!(last_lines("", 3), "");
    }
}
