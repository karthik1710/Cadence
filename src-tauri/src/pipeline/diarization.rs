use crate::pipeline::models::DiarizationSegment;
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::proc::{self, StageMetrics};
use crate::pipeline::PipelineConfig;
use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

/// Runs sherpa-onnx's offline speaker diarization CLI and parses its
/// stdout, which prints one line per segment in the form
/// `"<start_sec> -- <end_sec> speaker_<NN>"` (see
/// `OfflineSpeakerDiarizationSegment::ToString()` in sherpa-onnx).
///
/// Speaker count is left to sherpa-onnx's automatic clustering unless the
/// organiser gave an expected-speakers hint for this meeting — see
/// [`diarization_args`].
pub fn diarize(
    paths: &ModelPaths,
    config: &PipelineConfig,
    wav_path: &Path,
) -> Result<(Vec<DiarizationSegment>, StageMetrics), String> {
    let mut command = Command::new(paths.sherpa_diarization());
    command.args(diarization_args(
        &paths.diarization_segmentation_model,
        &paths.diarization_embedding_model,
        config,
        wav_path,
    ));

    let measured = proc::run_measured(command)?;
    if !measured.success {
        return Err(format!(
            "sherpa-onnx diarization exited with {}: {}",
            measured.status,
            measured.stderr.trim()
        ));
    }

    let segments = measured
        .stdout
        .lines()
        .filter_map(parse_diarization_line)
        .collect();
    Ok((segments, measured.metrics))
}

/// The sherpa-onnx command line, kept pure so it can be tested without the
/// binary.
///
/// The two clustering modes are mutually exclusive in sherpa-onnx itself
/// (`--help`: "If greater than 0, then cluster threshold is ignored"). With a
/// hint, only `--clustering.num-clusters` is passed, so the command line says
/// what actually decided the speaker count rather than carrying a threshold
/// that was silently overridden. Without one, only the threshold is passed and
/// the run is exactly what it was before the hint existed.
///
/// A fixed count is not a guarantee of that many *voices*: sherpa-onnx cuts
/// its linkage tree at `n` clusters, and stray embeddings can take some of
/// those clusters and then emit almost no speech. Measured 2026-10-06 on the
/// council baseline clip (pipeline's own `processed.wav`): `n = 7` gave 6 ids,
/// only 4 of them holding ≥5% of the talk, against 6 such ids at the 0.5
/// threshold. On the one-speaker TED clip (raw 16 kHz audio) `n = 1` did
/// exactly what it should: one id, against 10 at 0.5.
pub(crate) fn diarization_args(
    segmentation_model: &Path,
    embedding_model: &Path,
    config: &PipelineConfig,
    wav_path: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        format!("--segmentation.pyannote-model={}", segmentation_model.display()).into(),
        format!("--embedding.model={}", embedding_model.display()).into(),
        format!("--segmentation.num-threads={}", config.diarization_threads).into(),
        format!("--embedding.num-threads={}", config.diarization_threads).into(),
    ];
    match config.expected_speakers {
        Some(count) => args.push(format!("--clustering.num-clusters={count}").into()),
        None => args.push(
            format!("--clustering.cluster-threshold={}", config.cluster_threshold).into(),
        ),
    }
    args.push(wav_path.as_os_str().to_owned());
    args
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
    use crate::settings::Settings;
    use crate::system::HostInfo;

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

    fn config(expected_speakers: Option<u32>) -> PipelineConfig {
        let host = HostInfo {
            total_ram_mb: 15_700,
            available_ram_mb: 9_000,
            logical_cores: 8,
        };
        PipelineConfig {
            diarization_threads: 8,
            cluster_threshold: 0.5,
            expected_speakers,
            ..PipelineConfig::from_settings(&Settings::default(), &host)
        }
    }

    fn args(expected_speakers: Option<u32>) -> Vec<String> {
        diarization_args(
            Path::new("seg.onnx"),
            Path::new("emb.onnx"),
            &config(expected_speakers),
            Path::new("processed.wav"),
        )
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
    }

    #[test]
    fn without_a_hint_clusters_by_threshold_exactly_as_before() {
        assert_eq!(
            args(None),
            vec![
                "--segmentation.pyannote-model=seg.onnx",
                "--embedding.model=emb.onnx",
                "--segmentation.num-threads=8",
                "--embedding.num-threads=8",
                "--clustering.cluster-threshold=0.5",
                "processed.wav",
            ]
        );
    }

    #[test]
    fn a_hint_fixes_the_cluster_count_and_drops_the_threshold() {
        let args = args(Some(3));
        assert!(args.contains(&"--clustering.num-clusters=3".to_string()));
        assert!(
            !args.iter().any(|a| a.starts_with("--clustering.cluster-threshold")),
            "sherpa-onnx ignores the threshold when num-clusters is set; passing it would misstate the run: {args:?}"
        );
        // The input stays last — sherpa-onnx takes it as the positional argument.
        assert_eq!(args.last().map(String::as_str), Some("processed.wav"));
    }

    #[test]
    fn a_hint_of_one_is_passed_through() {
        // One speaker is the case automatic clustering handles worst (a single
        // talker over-split into several ids), so it must reach the CLI as-is.
        assert!(args(Some(1)).contains(&"--clustering.num-clusters=1".to_string()));
    }
}
