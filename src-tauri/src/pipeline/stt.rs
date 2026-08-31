use crate::pipeline::models::SttSegment;
use crate::pipeline::paths::ModelPaths;
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
) -> Result<Vec<SttSegment>, String> {
    // whisper.cpp appends `.json` itself, so pass the base path without it.
    let output_base = wav_path.with_extension("");

    let status = Command::new(paths.whisper_cli())
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
        .arg(&output_base)
        .status()
        .map_err(|e| format!("failed to launch whisper-cli.exe: {e}"))?;

    if !status.success() {
        return Err(format!("whisper-cli.exe exited with status {status}"));
    }

    let json_path = wav_path.with_extension("json");
    let raw = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("could not read whisper output {json_path:?}: {e}"))?;
    let parsed: WhisperJsonOutput =
        serde_json::from_str(&raw).map_err(|e| format!("could not parse whisper JSON: {e}"))?;

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
