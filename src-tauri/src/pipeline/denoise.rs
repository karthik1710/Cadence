//! Optional noise suppression, via sherpa-onnx's offline GTCRN denoiser.
//!
//! Meeting rooms are noisy in ways a high-pass filter cannot fix — keyboards,
//! projector fans, a corridor through an open door. GTCRN is a small ONNX
//! speech-enhancement model that runs on CPU at a fraction of real time, which
//! keeps it inside the budget.
//!
//! It is deliberately **optional** (not in `models_manager::required_ids`):
//! nothing about Cadence stops working without it, and a processing
//! stage must never cost a meeting. So a missing model is a logged skip, not a
//! failure, and even a denoiser that crashes mid-run leaves the pipeline with
//! the perfectly usable un-denoised audio it started from.

use crate::models_manager;
use crate::pipeline::paths;
use crate::pipeline::proc::{self, StageMetrics};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Catalog id of the GTCRN weights. Optional — see `models_manager`.
pub const DENOISE_MODEL_ID: &str = "denoise-gtcrn";

const DENOISE_BINARY: &str = "sherpa-onnx-offline-denoiser.exe";

pub fn binary_path(data_dir: &Path) -> PathBuf {
    paths::models_dir(data_dir).join("bin").join(DENOISE_BINARY)
}

pub fn model_path(data_dir: &Path) -> PathBuf {
    match models_manager::entry(DENOISE_MODEL_ID) {
        Some(entry) => entry.path_in(&paths::models_dir(data_dir)),
        None => paths::models_dir(data_dir)
            .join("denoise")
            .join("gtcrn_simple.onnx"),
    }
}

/// True when both the sherpa-onnx denoiser binary and the GTCRN weights are
/// installed. The pipeline checks this before enabling the stage.
pub fn is_available(data_dir: &Path) -> bool {
    binary_path(data_dir).exists() && model_path(data_dir).exists()
}

/// Explains, in one line, why the stage is being skipped — or `None` when it
/// can run. Kept separate from [`is_available`] so the pipeline log can say
/// which half is missing.
pub fn unavailable_reason(data_dir: &Path) -> Option<String> {
    let binary = binary_path(data_dir);
    if !binary.exists() {
        return Some(format!(
            "the sherpa-onnx denoiser binary is not installed at {}",
            binary.display()
        ));
    }
    let model = model_path(data_dir);
    if !model.exists() {
        return Some(format!(
            "the speech denoiser model is not installed at {} (install \"{}\" from Settings -> Models)",
            model.display(),
            DENOISE_MODEL_ID
        ));
    }
    None
}

/// Runs the denoiser over a 16 kHz mono WAV, writing an enhanced copy.
///
/// GTCRN is a 16 kHz model, so this must run *after* the resample stage and
/// before the gain stage — normalising first would hand it a boosted noise
/// floor to fight.
pub fn denoise(
    data_dir: &Path,
    threads: usize,
    input_wav: &Path,
    output_wav: &Path,
) -> Result<StageMetrics, String> {
    if let Some(reason) = unavailable_reason(data_dir) {
        return Err(reason);
    }

    let mut command = Command::new(binary_path(data_dir));
    command
        .arg(format!(
            "--speech-denoiser-gtcrn-model={}",
            model_path(data_dir).display()
        ))
        .arg(format!("--num-threads={}", threads.max(1)))
        .arg(format!("--input-wav={}", input_wav.display()))
        .arg(format!("--output-wav={}", output_wav.display()));

    let measured = proc::run_measured(command)?;
    if !measured.success {
        return Err(format!(
            "sherpa-onnx denoiser exited with {}: {}",
            measured.status,
            measured.stderr.trim()
        ));
    }
    // The tool reports failures on stderr but still exits 0 in some cases, so
    // the real check is that a non-trivial file exists afterwards.
    let bytes = std::fs::metadata(output_wav).map(|m| m.len()).unwrap_or(0);
    if bytes <= 44 {
        return Err(format!(
            "the denoiser produced no audio at {}",
            output_wav.display()
        ));
    }
    Ok(measured.metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_model_is_optional_and_never_required() {
        assert!(models_manager::entry(DENOISE_MODEL_ID).is_some());
        for profile in [
            crate::system::MemoryProfile::Low,
            crate::system::MemoryProfile::Balanced,
            crate::system::MemoryProfile::High,
        ] {
            assert!(
                !models_manager::required_ids(profile).contains(&DENOISE_MODEL_ID),
                "{profile:?} must not require the optional denoiser"
            );
        }
    }

    #[test]
    fn paths_land_under_the_models_directory() {
        let dir = Path::new("C:\\data");
        assert!(binary_path(dir).ends_with(DENOISE_BINARY));
        assert!(model_path(dir)
            .to_string_lossy()
            .contains("gtcrn_simple.onnx"));
    }

    #[test]
    fn an_uninstalled_denoiser_reports_which_half_is_missing() {
        let dir = empty_dir("cadence-denoise-missing");
        assert!(!is_available(&dir));
        let reason = unavailable_reason(&dir).unwrap();
        assert!(reason.contains("binary is not installed"), "{reason}");

        // With the binary present but no weights, the message changes.
        std::fs::create_dir_all(paths::models_dir(&dir).join("bin")).unwrap();
        std::fs::write(binary_path(&dir), b"stub").unwrap();
        let reason = unavailable_reason(&dir).unwrap();
        assert!(reason.contains("model is not installed"), "{reason}");
        assert!(reason.contains(DENOISE_MODEL_ID));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn denoising_without_the_model_is_an_error_the_pipeline_can_skip_on() {
        let dir = empty_dir("cadence-denoise-skip");
        let err = denoise(&dir, 4, Path::new("in.wav"), Path::new("out.wav")).unwrap_err();
        assert!(err.contains("not installed"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
