use crate::models_manager::{self, ModelKind};
use crate::settings::Settings;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// Resolves where the local AI binaries/models live: `<app_data_dir>/models/...`,
/// the layout the in-app Model Manager (`models_manager.rs`)
/// and `scripts/setup-models.ps1` both provision.
///
/// Which STT/LLM weights those are is not fixed: it comes from the user's
/// settings, which in turn default to the machine's memory profile. A 16 GB
/// laptop resolves to a different LLM than the 32 GB reference machine without
/// anything else in the pipeline knowing or caring (the swappable-provider
/// requirement).
pub struct ModelPaths {
    bin_dir: PathBuf,
    pub stt_model: PathBuf,
    pub llm_model: PathBuf,
    pub diarization_segmentation_model: PathBuf,
    pub diarization_embedding_model: PathBuf,
    /// Set when the user turned MoM generation off — the LLM is then not
    /// required to be installed.
    pub llm_required: bool,
}

pub fn models_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("models")
}

impl ModelPaths {
    /// Resolves paths and confirms every required file is present — missing
    /// models are a clear, recoverable error, never a crash.
    pub fn resolve(app: &AppHandle, settings: &Settings) -> Result<Self, String> {
        let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
        Self::from_data_dir(&data_dir, settings)
    }

    /// Same resolution against an explicit data dir, for the CLI runners,
    /// which have no `AppHandle`.
    pub fn from_data_dir(data_dir: &Path, settings: &Settings) -> Result<Self, String> {
        let paths = Self::unchecked(data_dir, settings);
        let missing = paths.missing(data_dir, settings);
        if missing.is_empty() {
            return Ok(paths);
        }
        let labels: Vec<&str> = missing
            .iter()
            .map(|id| {
                models_manager::entry(id)
                    .map(|e| e.label)
                    .unwrap_or(id.as_str())
            })
            .collect();
        Err(format!(
            "Missing AI models: {}. Install them from Settings → Models (or run scripts/setup-models.ps1).",
            labels.join(", ")
        ))
    }

    /// Resolves paths without checking that anything exists — used by the
    /// Model Manager, which needs to talk about files that are not there yet.
    pub fn unchecked(data_dir: &Path, settings: &Settings) -> Self {
        let models = models_dir(data_dir);
        let stt_id = settings.effective_stt_model();
        let llm_id = settings.effective_llm_model();

        Self {
            bin_dir: models.join("bin"),
            stt_model: catalog_path(&models, &stt_id),
            llm_model: catalog_path(&models, &llm_id),
            diarization_segmentation_model: catalog_path(&models, "diarization-segmentation"),
            diarization_embedding_model: catalog_path(&models, "diarization-embedding"),
            llm_required: settings.generate_mom,
        }
    }

    /// Catalog ids this machine still needs, in install order. Empty means the
    /// pipeline can run end to end offline.
    pub fn missing(&self, data_dir: &Path, settings: &Settings) -> Vec<String> {
        let models = models_dir(data_dir);
        let mut missing = Vec::new();
        for id in [
            "runtime-whisper",
            "runtime-sherpa",
            "diarization-segmentation",
            "diarization-embedding",
        ] {
            if !installed(&models, id) {
                missing.push(id.to_string());
            }
        }
        let stt_id = settings.effective_stt_model();
        if !installed(&models, &stt_id) {
            missing.push(stt_id);
        }
        if settings.generate_mom {
            if !installed(&models, "runtime-llama") {
                missing.push("runtime-llama".to_string());
            }
            let llm_id = settings.effective_llm_model();
            if !installed(&models, &llm_id) {
                missing.push(llm_id);
            }
        }
        missing
    }

    pub fn whisper_cli(&self) -> PathBuf {
        self.bin_dir.join("whisper-cli.exe")
    }

    /// Ships in the same whisper.cpp release as `whisper-cli.exe`, and is what
    /// code-switched STT drives so the weights are loaded once instead of
    /// once per block. Not in `missing()`: an install without it falls back to
    /// a single `whisper-cli.exe` pass rather than refusing to process.
    pub fn whisper_server(&self) -> PathBuf {
        self.bin_dir.join("whisper-server.exe")
    }

    pub fn llama_server(&self) -> PathBuf {
        self.bin_dir.join("llama-server.exe")
    }

    pub fn sherpa_diarization(&self) -> PathBuf {
        self.bin_dir
            .join("sherpa-onnx-offline-speaker-diarization.exe")
    }
}

fn catalog_path(models_dir: &Path, id: &str) -> PathBuf {
    match models_manager::entry(id) {
        Some(entry) => entry.path_in(models_dir),
        // An id that is not in the catalog can only come from a hand-edited
        // settings.json. Point at a path that plainly does not exist rather
        // than silently substituting a different model.
        None => models_dir.join("unknown").join(id),
    }
}

fn installed(models_dir: &Path, id: &str) -> bool {
    models_manager::entry(id)
        .map(|e| e.is_installed(models_dir))
        .unwrap_or(false)
}

/// Catalog entry backing the currently selected LLM, for the pre-flight memory
/// check and for recording provenance in `mom.json`.
pub fn selected_llm_entry(settings: &Settings) -> Option<&'static models_manager::CatalogEntry> {
    let entry = models_manager::entry(&settings.effective_llm_model())?;
    matches!(entry.kind, ModelKind::Llm).then_some(entry)
}

/// The data dir Tauri resolves for this app's identifier, which is also where
/// `scripts/setup-models.ps1` installs — lets the CLI runners reach the very
/// same models the desktop app uses.
pub fn default_data_dir() -> Result<PathBuf, String> {
    std::env::var("APPDATA")
        .map(|appdata| PathBuf::from(appdata).join("com.cadence.app"))
        .map_err(|_| "APPDATA is not set — pass a data dir explicitly".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_16gb_machine_resolves_the_smaller_llm() {
        let settings = Settings {
            profile: "balanced".to_string(),
            ..Settings::default()
        };
        let paths = ModelPaths::unchecked(Path::new("C:\\data"), &settings);
        assert!(paths
            .llm_model
            .to_string_lossy()
            .contains("Qwen3-4B-Q4_K_M.gguf"));
    }

    #[test]
    fn a_32gb_machine_resolves_the_larger_llm() {
        let settings = Settings {
            profile: "high".to_string(),
            ..Settings::default()
        };
        let paths = ModelPaths::unchecked(Path::new("C:\\data"), &settings);
        assert!(paths
            .llm_model
            .to_string_lossy()
            .contains("Qwen3-8B-Q4_K_M.gguf"));
    }

    #[test]
    fn nothing_is_installed_in_an_empty_data_dir() {
        let dir = std::env::temp_dir().join("cadence-paths-empty");
        let _ = std::fs::remove_dir_all(&dir);
        let settings = Settings::default();
        let paths = ModelPaths::unchecked(&dir, &settings);
        let missing = paths.missing(&dir, &settings);
        assert!(missing.contains(&"runtime-whisper".to_string()));
        assert!(missing.contains(&settings.effective_llm_model()));
    }

    #[test]
    fn turning_off_mom_drops_the_llm_requirement() {
        let dir = std::env::temp_dir().join("cadence-paths-nollm");
        let _ = std::fs::remove_dir_all(&dir);
        let settings = Settings {
            generate_mom: false,
            ..Settings::default()
        };
        let paths = ModelPaths::unchecked(&dir, &settings);
        let missing = paths.missing(&dir, &settings);
        assert!(!missing.iter().any(|id| id.starts_with("qwen3")));
        assert!(!missing.contains(&"runtime-llama".to_string()));
    }
}
