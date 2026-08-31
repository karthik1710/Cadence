use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// Resolves where the local AI binaries/models live: `<app_data_dir>/models/...`,
/// the exact layout `scripts/setup-models.ps1` provisions and the shipped
/// app's future Model Manager (master prompt §27, Phase 13) would use too.
pub struct ModelPaths {
    bin_dir: PathBuf,
    pub stt_model: PathBuf,
    pub llm_model: PathBuf,
    pub diarization_segmentation_model: PathBuf,
    pub diarization_embedding_model: PathBuf,
}

impl ModelPaths {
    /// Resolves paths and confirms every required file is present — missing
    /// models are a clear, recoverable error (§33), never a crash.
    pub fn resolve(app: &AppHandle) -> Result<Self, String> {
        let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
        Self::from_data_dir(&data_dir)
    }

    /// Same resolution against an explicit data dir, for the Phase 5 CLI
    /// runner, which has no `AppHandle`.
    pub fn from_data_dir(data_dir: &Path) -> Result<Self, String> {
        let models_dir = data_dir.join("models");

        let paths = Self {
            bin_dir: models_dir.join("bin"),
            stt_model: models_dir.join("stt").join("ggml-small.bin"),
            llm_model: models_dir.join("llm").join("Qwen3-8B-Q4_K_M.gguf"),
            diarization_segmentation_model: models_dir
                .join("diarization")
                .join("sherpa-onnx-pyannote-segmentation-3-0")
                .join("model.onnx"),
            diarization_embedding_model: models_dir
                .join("diarization")
                .join("wespeaker_en_voxceleb_CAM++.onnx"),
        };
        paths.check()?;
        Ok(paths)
    }

    fn check(&self) -> Result<(), String> {
        let mut missing = Vec::new();
        for (label, p) in [
            ("whisper-cli.exe", self.whisper_cli()),
            ("llama-server.exe", self.llama_server()),
            (
                "sherpa-onnx-offline-speaker-diarization.exe",
                self.sherpa_diarization(),
            ),
        ] {
            if !p.exists() {
                missing.push(label.to_string());
            }
        }
        for (label, p) in [
            ("STT model (ggml-small.bin)", &self.stt_model),
            ("LLM model (Qwen3-8B-Q4_K_M.gguf)", &self.llm_model),
            (
                "diarization segmentation model",
                &self.diarization_segmentation_model,
            ),
            (
                "diarization embedding model",
                &self.diarization_embedding_model,
            ),
        ] {
            if !p.exists() {
                missing.push(label.to_string());
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Missing AI models/binaries: {}. Run scripts/setup-models.ps1 to install them.",
                missing.join(", ")
            ))
        }
    }

    pub fn whisper_cli(&self) -> PathBuf {
        self.bin_dir.join("whisper-cli.exe")
    }

    pub fn llama_server(&self) -> PathBuf {
        self.bin_dir.join("llama-server.exe")
    }

    pub fn sherpa_diarization(&self) -> PathBuf {
        self.bin_dir.join("sherpa-onnx-offline-speaker-diarization.exe")
    }
}

/// The data dir Tauri resolves for this app's identifier, which is also where
/// `scripts/setup-models.ps1` installs — lets the CLI runner reach the very
/// same models the desktop app will use.
pub fn default_data_dir() -> Result<PathBuf, String> {
    std::env::var("APPDATA")
        .map(|appdata| PathBuf::from(appdata).join("com.echomind.app"))
        .map_err(|_| "APPDATA is not set — pass a data dir explicitly".to_string())
}
