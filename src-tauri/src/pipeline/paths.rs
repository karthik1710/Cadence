use crate::models_manager::{self, ModelKind, Runtime};
use crate::settings::Settings;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// Resolves where the local AI binaries/models live. Weights are under
/// `<app_data_dir>/models/...`, the layout the in-app Model Manager
/// (`models_manager.rs`) and `scripts/setup-models.ps1` both provision. Each
/// runtime comes from the copy the installer shipped (`<exe dir>/runtimes/…`)
/// when there is one, else from that same models folder — see
/// `models_manager::runtime_dir`.
///
/// Which STT/LLM weights those are is read through `Settings::effective_*`,
/// which resolve to the one model set every machine runs (`crate::system`).
/// Keeping that lookup in one place is what lets the model change after
/// benchmarking without anything else in the pipeline knowing or caring (the
/// swappable-provider requirement).
pub struct ModelPaths {
    /// Each runtime's folder, resolved once (own folder or legacy flat
    /// `bin/`), so a runtime's executables never mix with another's DLLs.
    whisper_dir: PathBuf,
    llama_dir: PathBuf,
    sherpa_dir: PathBuf,
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
        Self::unchecked_with_bundled(data_dir, settings, models_manager::bundled_runtimes_root())
    }

    /// [`Self::unchecked`] against an explicit bundled-runtimes root (the
    /// `runtimes/` folder an installer puts beside the executable), so tests
    /// never depend on what sits next to the test binary.
    fn unchecked_with_bundled(data_dir: &Path, settings: &Settings, bundled: Option<&Path>) -> Self {
        let models = models_dir(data_dir);
        let stt_id = settings.effective_stt_model();
        let llm_id = settings.effective_llm_model();
        let runtime_dir =
            |runtime| models_manager::runtime_dir_with_bundled(&models, bundled, runtime);

        Self {
            whisper_dir: runtime_dir(Runtime::Whisper),
            llama_dir: runtime_dir(Runtime::Llama),
            sherpa_dir: runtime_dir(Runtime::Sherpa),
            stt_model: catalog_path(&models, &stt_id),
            llm_model: catalog_path(&models, &llm_id),
            diarization_segmentation_model: catalog_path(&models, "diarization-segmentation"),
            diarization_embedding_model: catalog_path(&models, "diarization-embedding"),
            llm_required: settings.generate_mom,
        }
    }

    /// Catalog ids a processing run cannot start without, in install order.
    /// Empty means the pipeline can run end to end offline.
    ///
    /// Narrower than `models_manager::required_ids`: the denoiser and VAD
    /// models are installed with everything else at first run, but a run
    /// without them still succeeds — each stage logs a skip — so their absence
    /// must not refuse a meeting.
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
        self.whisper_dir.join("whisper-cli.exe")
    }

    /// Ships in the same whisper.cpp release as `whisper-cli.exe`, and is what
    /// code-switched STT drives so the weights are loaded once instead of
    /// once per block. Not in `missing()`: an install without it falls back to
    /// a single `whisper-cli.exe` pass rather than refusing to process.
    pub fn whisper_server(&self) -> PathBuf {
        self.whisper_dir.join("whisper-server.exe")
    }

    pub fn llama_server(&self) -> PathBuf {
        self.llama_dir.join("llama-server.exe")
    }

    pub fn sherpa_diarization(&self) -> PathBuf {
        self.sherpa_dir
            .join("sherpa-onnx-offline-speaker-diarization.exe")
    }
}

/// One executable of a native runtime under a data dir — the denoiser and the
/// VAD, which run outside a `ModelPaths`. Resolves exactly as `ModelPaths`
/// does (`models_manager::runtime_dir`): the runtime's own folder under
/// `models/bin`, or the old shared `models/bin` for a runtime an earlier build
/// installed there and nobody has reinstalled since.
pub fn runtime_binary(data_dir: &Path, runtime: Runtime, exe: &str) -> PathBuf {
    models_manager::runtime_binary(&models_dir(data_dir), runtime, exe)
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
    fn every_machine_resolves_the_same_llm() {
        let paths = ModelPaths::unchecked(Path::new("C:\\data"), &Settings::default());
        assert!(paths
            .llm_model
            .to_string_lossy()
            .contains("Qwen3-8B-Q4_K_M.gguf"));
        assert!(paths.stt_model.to_string_lossy().contains("ggml-small.bin"));
    }

    #[test]
    fn each_runtime_runs_from_its_own_folder_or_its_legacy_one() {
        let dir = std::env::temp_dir().join(format!("cadence-paths-runtimes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let bin = models_dir(&dir).join("bin");
        let settings = Settings::default();

        // Fresh machine: every runtime points into its own folder.
        let paths = ModelPaths::unchecked(&dir, &settings);
        assert_eq!(paths.whisper_cli(), bin.join("whisper").join("whisper-cli.exe"));
        assert_eq!(paths.whisper_server(), bin.join("whisper").join("whisper-server.exe"));
        assert_eq!(paths.llama_server(), bin.join("llama").join("llama-server.exe"));
        assert_eq!(
            paths.sherpa_diarization(),
            bin.join("sherpa").join("sherpa-onnx-offline-speaker-diarization.exe")
        );

        // An upgraded machine with only llama.cpp reinstalled: whisper.cpp
        // and sherpa-onnx keep running from the flat folder, untouched.
        std::fs::create_dir_all(bin.join("llama")).unwrap();
        for exe in ["whisper-cli.exe", "sherpa-onnx-offline-speaker-diarization.exe"] {
            std::fs::write(bin.join(exe), b"legacy").unwrap();
        }
        std::fs::write(bin.join("llama").join("llama-server.exe"), b"own").unwrap();
        let paths = ModelPaths::unchecked(&dir, &settings);
        assert_eq!(paths.whisper_cli(), bin.join("whisper-cli.exe"));
        assert_eq!(paths.whisper_server(), bin.join("whisper-server.exe"));
        assert_eq!(paths.llama_server(), bin.join("llama").join("llama-server.exe"));
        assert_eq!(paths.sherpa_diarization(), bin.join("sherpa-onnx-offline-speaker-diarization.exe"));
        assert_eq!(
            runtime_binary(&dir, Runtime::Sherpa, "sherpa-onnx-vad.exe"),
            bin.join("sherpa-onnx-vad.exe")
        );
        let missing = paths.missing(&dir, &settings);
        for id in ["runtime-whisper", "runtime-llama", "runtime-sherpa"] {
            assert!(!missing.contains(&id.to_string()), "{id} reported missing");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bundled_runtime_wins_over_the_downloaded_copies() {
        let dir = std::env::temp_dir().join(format!("cadence-paths-bundled-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let bin = models_dir(&dir).join("bin");
        let bundled = dir.join("install").join("runtimes");
        let settings = Settings::default();
        // Every downloaded layout present: own folder for whisper.cpp, the
        // legacy flat folder for sherpa-onnx.
        std::fs::create_dir_all(bin.join("whisper")).unwrap();
        std::fs::write(bin.join("whisper").join("whisper-cli.exe"), b"own").unwrap();
        std::fs::write(bin.join("sherpa-onnx-offline-speaker-diarization.exe"), b"legacy").unwrap();
        // Bundled: whisper.cpp and sherpa-onnx, not llama.cpp.
        for (folder, exe) in [("whisper", "whisper-cli.exe"), ("sherpa", "sherpa-onnx-offline-speaker-diarization.exe")] {
            std::fs::create_dir_all(bundled.join(folder)).unwrap();
            std::fs::write(bundled.join(folder).join(exe), b"bundled").unwrap();
        }

        let paths = ModelPaths::unchecked_with_bundled(&dir, &settings, Some(&bundled));
        assert_eq!(paths.whisper_cli(), bundled.join("whisper").join("whisper-cli.exe"));
        assert_eq!(paths.whisper_server(), bundled.join("whisper").join("whisper-server.exe"));
        assert_eq!(
            paths.sherpa_diarization(),
            bundled.join("sherpa").join("sherpa-onnx-offline-speaker-diarization.exe")
        );
        // Not bundled: falls back to app data, own folder first.
        assert_eq!(paths.llama_server(), bin.join("llama").join("llama-server.exe"));

        // No bundled root (the process default under `cargo test`): unchanged.
        let paths = ModelPaths::unchecked(&dir, &settings);
        assert_eq!(paths.whisper_cli(), bin.join("whisper").join("whisper-cli.exe"));
        assert_eq!(paths.sherpa_diarization(), bin.join("sherpa-onnx-offline-speaker-diarization.exe"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_audio_quality_models_never_block_a_run() {
        let dir = std::env::temp_dir().join("cadence-paths-audio-optional");
        let _ = std::fs::remove_dir_all(&dir);
        let settings = Settings::default();
        let missing = ModelPaths::unchecked(&dir, &settings).missing(&dir, &settings);
        assert!(!missing.iter().any(|id| id == "denoise-gtcrn" || id == "vad-silero"));
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
