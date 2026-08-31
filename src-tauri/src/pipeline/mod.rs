pub mod align;
pub mod diarization;
pub mod models;
pub mod paths;
pub mod preprocess;
pub mod stt;

use models::Transcript;
use paths::ModelPaths;
use std::path::Path;

/// Processing knobs the Phase 13 settings screen (§27 "Processing threads")
/// will own. Both native CLIs default to far fewer threads than this machine
/// has — whisper.cpp to 4, sherpa-onnx to 1 — so leaving them alone costs
/// most of the §29 processing-time budget.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub stt_threads: usize,
    pub diarization_threads: usize,
    /// sherpa-onnx clustering distance. Smaller splits one voice across
    /// several speaker ids; larger collapses distinct voices into one.
    pub cluster_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stt_threads: cores_capped_at(MAX_STT_THREADS),
            diarization_threads: cores_capped_at(MAX_DIARIZATION_THREADS),
            cluster_threshold: DEFAULT_CLUSTER_THRESHOLD,
        }
    }
}

/// whisper.cpp keeps gaining out to at least 24 threads on the i9-13950HX;
/// sherpa-onnx peaks around 8 and gets slower past it. See docs/performance.md.
const MAX_STT_THREADS: usize = 24;
const MAX_DIARIZATION_THREADS: usize = 8;

/// Provisional. On the one 57s sample available, sherpa-onnx's own default
/// (0.5) shattered a single dominant voice into 8 speaker ids, while 0.9
/// absorbed the second voice into the first. See docs/models.md — this needs
/// re-testing against labelled 3/5/10-speaker recordings (Phase 3).
const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.8;

fn cores_capped_at(cap: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(cap))
        .unwrap_or(4)
}

/// Runs the full local pipeline (master prompt §11): preprocess → diarize →
/// STT → align. Every stage's raw output is written to `out_dir` before the
/// next stage runs, so a later failure never costs the work already done
/// (§33) — `original.wav` is never touched, and `diarization.json` / `stt.json`
/// stay separate from the derived `transcript.json`.
///
/// `on_stage` receives the §12 state-machine stage name and a rough percentage.
pub fn run(
    paths: &ModelPaths,
    config: &PipelineConfig,
    input_wav: &Path,
    out_dir: &Path,
    meeting_id: &str,
    on_stage: &dyn Fn(&str, i64),
) -> Result<Transcript, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("could not create {out_dir:?}: {e}"))?;

    on_stage("PREPROCESSING", 5);
    let processed = out_dir.join("processed.wav");
    preprocess::resample_to_16k_mono(input_wav, &processed)?;

    on_stage("DIARIZING", 15);
    let diarization = diarization::diarize(paths, config, &processed)?;
    write_json(&out_dir.join("diarization.json"), &diarization)?;

    on_stage("TRANSCRIBING", 40);
    let stt_segments = stt::transcribe(paths, config, &processed)?;
    write_json(&out_dir.join("stt.json"), &stt_segments)?;

    on_stage("ALIGNING", 90);
    let transcript = Transcript {
        meeting_id: meeting_id.to_string(),
        segments: align::align(&stt_segments, &diarization),
    };
    write_json(&out_dir.join("transcript.json"), &transcript)?;

    on_stage("COMPLETED", 100);
    Ok(transcript)
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| format!("could not write {path:?}: {e}"))
}
