//! Tauri commands for Settings and the Model Manager.
//!
//! This is where the machine-awareness surfaces to the user: `get_host_info`
//! reports what Cadence detected and which profile it chose, and everything
//! downstream — which LLM, how much context, how many threads — follows from
//! that unless the user overrides it.

use crate::models_manager::{self, InstallProgress, ModelStatus, VerifyReport};
use crate::pipeline::paths::{self, ModelPaths};
use crate::processing::app_data_dir;
use crate::settings::{self, Settings};
use crate::system::{HostInfo, MemoryProfile};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, State};

/// Live install progress per model id, polled by the Settings screen.
pub type InstallState = Arc<Mutex<HashMap<String, InstallProgress>>>;

#[derive(Serialize)]
pub struct HostReport {
    #[serde(flatten)]
    pub host: HostInfo,
    /// The profile in force, after any user override.
    pub profile: String,
    /// What the hardware alone implies, so the UI can show "auto (balanced)".
    pub detected_profile: String,
    pub llm_ram_budget_mb: u64,
    pub recommended_llm: String,
    pub recommended_stt: String,
    pub recommended_context: usize,
    pub stt_threads: usize,
    pub diarization_threads: usize,
    pub llm_threads: usize,
    pub data_dir: String,
}

#[derive(Serialize)]
pub struct SetupStatus {
    /// True when a meeting can be recorded and fully processed offline.
    pub ready: bool,
    pub missing: Vec<ModelStatus>,
    pub message: Option<String>,
    pub profile: String,
    pub total_ram_mb: u64,
}

#[tauri::command]
pub fn get_host_info(app: AppHandle) -> Result<HostReport, String> {
    let data_dir = app_data_dir(&app)?;
    let user_settings = settings::load(&data_dir);
    let host = HostInfo::detect();
    let profile = user_settings.effective_profile();

    Ok(HostReport {
        host,
        profile: profile.as_str().to_string(),
        detected_profile: MemoryProfile::for_total_ram_mb(host.total_ram_mb)
            .as_str()
            .to_string(),
        llm_ram_budget_mb: profile.llm_ram_budget_mb(),
        recommended_llm: profile.default_llm_model().to_string(),
        recommended_stt: profile.default_stt_model().to_string(),
        recommended_context: profile.default_llm_context(),
        stt_threads: user_settings.effective_stt_threads(&host),
        diarization_threads: user_settings.effective_diarization_threads(&host),
        llm_threads: user_settings.effective_llm_threads(&host),
        data_dir: data_dir.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub fn get_settings(app: AppHandle) -> Result<Settings, String> {
    let data_dir = app_data_dir(&app)?;
    Ok(settings::load(&data_dir))
}

#[tauri::command]
pub fn save_settings(app: AppHandle, new_settings: Settings) -> Result<Settings, String> {
    let data_dir = app_data_dir(&app)?;
    settings::save(&data_dir, &new_settings)?;
    Ok(new_settings)
}

#[tauri::command]
pub fn list_models(app: AppHandle) -> Result<Vec<ModelStatus>, String> {
    let data_dir = app_data_dir(&app)?;
    let user_settings = settings::load(&data_dir);
    Ok(models_manager::status_all(
        &paths::models_dir(&data_dir),
        user_settings.effective_profile(),
    ))
}

/// Everything the app needs before it can process a meeting offline. Called on
/// startup so a missing model is discovered *before* a meeting is recorded,
/// never after.
#[tauri::command]
pub fn check_models(app: AppHandle) -> Result<SetupStatus, String> {
    let data_dir = app_data_dir(&app)?;
    let user_settings = settings::load(&data_dir);
    let profile = user_settings.effective_profile();
    let host = HostInfo::detect();

    let resolved = ModelPaths::unchecked(&data_dir, &user_settings);
    let missing_ids = resolved.missing(&data_dir, &user_settings);
    let all = models_manager::status_all(&paths::models_dir(&data_dir), profile);
    let missing: Vec<ModelStatus> = all
        .into_iter()
        .filter(|status| missing_ids.contains(&status.id))
        .collect();

    let message = if missing.is_empty() {
        None
    } else {
        Some(format!(
            "{} component(s) still need installing before a meeting can be processed.",
            missing.len()
        ))
    };

    Ok(SetupStatus {
        ready: missing.is_empty(),
        missing,
        message,
        profile: profile.as_str().to_string(),
        total_ram_mb: host.total_ram_mb,
    })
}

/// Starts an install in the background and returns immediately; the UI polls
/// `get_install_progress`. Downloads happen only here, only on an explicit
/// user action, and never while a meeting is being recorded or processed.
#[tauri::command]
pub fn install_model(
    app: AppHandle,
    state: State<InstallState>,
    model_id: String,
) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    let models_dir = paths::models_dir(&data_dir);
    if models_manager::entry(&model_id).is_none() {
        return Err(format!("unknown model id: {model_id}"));
    }

    let progress_state = state.inner().clone();
    if let Ok(mut map) = progress_state.lock() {
        if map
            .get(&model_id)
            .map(|p| p.phase == "downloading" || p.phase == "extracting")
            .unwrap_or(false)
        {
            return Err("that download is already running".to_string());
        }
        map.insert(
            model_id.clone(),
            InstallProgress::new(&model_id, "downloading", 0),
        );
    }

    std::thread::spawn(move || {
        let report = |progress: InstallProgress| {
            if let Ok(mut map) = progress_state.lock() {
                map.insert(progress.id.clone(), progress);
            }
        };
        if let Err(err) = models_manager::install(&models_dir, &model_id, &report) {
            report(InstallProgress {
                id: model_id.clone(),
                phase: "failed".to_string(),
                percent: 0,
                downloaded_mb: 0,
                total_mb: 0,
                error: Some(err),
            });
        }
    });

    Ok(())
}

#[tauri::command]
pub fn get_install_progress(state: State<InstallState>, model_id: String) -> Option<InstallProgress> {
    state.lock().ok().and_then(|map| map.get(&model_id).cloned())
}

#[tauri::command]
pub fn remove_model(app: AppHandle, model_id: String) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    let user_settings = settings::load(&data_dir);
    models_manager::remove(
        &paths::models_dir(&data_dir),
        &model_id,
        user_settings.effective_profile(),
    )
}

#[tauri::command]
pub fn verify_model(app: AppHandle, model_id: String) -> Result<VerifyReport, String> {
    let data_dir = app_data_dir(&app)?;
    models_manager::verify(&paths::models_dir(&data_dir), &model_id)
}
