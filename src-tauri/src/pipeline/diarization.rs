use crate::pipeline::models::DiarizationSegment;
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::PipelineConfig;
use std::path::Path;
use std::process::Command;

/// Runs sherpa-onnx's offline speaker diarization CLI and parses its
/// stdout, which prints one line per segment in the form
/// `"<start_sec> -- <end_sec> speaker_<NN>"` (see
/// `OfflineSpeakerDiarizationSegment::ToString()` in sherpa-onnx). Speaker
/// count is left to sherpa-onnx's automatic clustering — master prompt §7
/// doesn't require the organizer to know the exact headcount up front.
pub fn diarize(
    paths: &ModelPaths,
    config: &PipelineConfig,
    wav_path: &Path,
) -> Result<Vec<DiarizationSegment>, String> {
    let output = Command::new(paths.sherpa_diarization())
        .arg(format!(
            "--segmentation.pyannote-model={}",
            paths.diarization_segmentation_model.display()
        ))
        .arg(format!(
            "--embedding.model={}",
            paths.diarization_embedding_model.display()
        ))
        .arg(format!(
            "--segmentation.num-threads={}",
            config.diarization_threads
        ))
        .arg(format!(
            "--embedding.num-threads={}",
            config.diarization_threads
        ))
        .arg(format!(
            "--clustering.cluster-threshold={}",
            config.cluster_threshold
        ))
        .arg(wav_path)
        .output()
        .map_err(|e| {
            format!("failed to launch sherpa-onnx-offline-speaker-diarization.exe: {e}")
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "sherpa-onnx diarization exited with {}: {}",
            output.status, stderr
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().filter_map(parse_diarization_line).collect())
}

fn parse_diarization_line(line: &str) -> Option<DiarizationSegment> {
    // "1.234 -- 5.678 speaker_00"
    let (times, rest) = line.split_once(" -- ")?;
    let start_sec: f64 = times.trim().parse().ok()?;
    let mut rest_parts = rest.trim().splitn(2, ' ');
    let end_sec: f64 = rest_parts.next()?.trim().parse().ok()?;
    let speaker_token = rest_parts.next()?.trim();
    let speaker_id = speaker_token
        .strip_prefix("speaker_")
        .unwrap_or(speaker_token)
        .to_string();

    Some(DiarizationSegment {
        speaker_id,
        start_ms: (start_sec * 1000.0).round() as i64,
        end_ms: (end_sec * 1000.0).round() as i64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_valid_line() {
        let seg = parse_diarization_line("1.234 -- 5.678 speaker_00").unwrap();
        assert_eq!(seg.speaker_id, "00");
        assert_eq!(seg.start_ms, 1234);
        assert_eq!(seg.end_ms, 5678);
    }

    #[test]
    fn ignores_garbage_lines() {
        assert!(parse_diarization_line("not a segment line").is_none());
        assert!(parse_diarization_line("").is_none());
    }
}
