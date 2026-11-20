//! Tauri commands for Settings and the Model Manager.
//!
//! Every machine runs the same model set (see [`crate::system`]), so nothing
//! here chooses a model. `get_host_info` reports what Cadence detected — and
//! whether the machine meets the 16 GB minimum — and `check_models` /
//! `install_required_models` get the one required set onto disk at first run.

use crate::audio::RecordingState;
use crate::models_manager::{self, InstallProgress, ModelStatus, RetiredModel, VerifyReport};
use crate::pipeline::paths;
use crate::processing::{app_data_dir, ProcessingState};
use crate::settings::{self, Settings};
use crate::system::{self, HostInfo};
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
    pub llm_model: String,
    pub stt_model: String,
    pub llm_context: usize,
    pub llm_ram_budget_mb: u64,
    pub minimum_ram_mb: u64,
    pub meets_minimum_ram: bool,
    pub stt_threads: usize,
    pub diarization_threads: usize,
    pub llm_threads: usize,
    pub data_dir: String,
}

#[derive(Serialize)]
pub struct SetupStatus {
    /// True when every required model is installed.
    pub ready: bool,
    pub missing: Vec<ModelStatus>,
    pub message: Option<String>,
    pub total_ram_mb: u64,
    pub minimum_ram_mb: u64,
    /// False on a machine below the minimum. A warning for the UI to show,
    /// never a reason to block recording: a meeting on disk can always be
    /// processed later.
    pub meets_minimum_ram: bool,
    /// Approximate size of everything in `missing`, for the consent screen.
    pub total_download_mb: u64,
    /// Models an earlier build installed (or files nobody recognises) that
    /// the current model set does not use. Informational: never blocks setup,
    /// and only removed on an explicit click (`remove_retired_model`).
    pub retired: Vec<RetiredModel>,
}

#[tauri::command]
pub fn get_host_info(app: AppHandle) -> Result<HostReport, String> {
    let data_dir = app_data_dir(&app)?;
    let user_settings = settings::load(&data_dir);
    let host = HostInfo::detect();

    Ok(HostReport {
        host,
        llm_model: system::LLM_MODEL_ID.to_string(),
        stt_model: system::STT_MODEL_ID.to_string(),
        llm_context: system::LLM_CONTEXT_TOKENS,
        llm_ram_budget_mb: system::LLM_RAM_BUDGET_MB,
        minimum_ram_mb: system::MIN_TOTAL_RAM_MB,
        meets_minimum_ram: host.meets_minimum(),
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
    Ok(models_manager::status_all(&paths::models_dir(&data_dir)))
}

/// Everything the app needs before it can process a meeting offline. Called on
/// startup so a missing model is discovered *before* a meeting is recorded,
/// never after.
#[tauri::command]
pub fn check_models(app: AppHandle) -> Result<SetupStatus, String> {
    let data_dir = app_data_dir(&app)?;
    let models_dir = paths::models_dir(&data_dir);
    let host = HostInfo::detect();

    let missing_ids = models_manager::missing_required_ids(&models_dir);
    let missing: Vec<ModelStatus> = models_manager::status_all(&models_dir)
        .into_iter()
        .filter(|status| missing_ids.contains(&status.id.as_str()))
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
        total_ram_mb: host.total_ram_mb,
        minimum_ram_mb: system::MIN_TOTAL_RAM_MB,
        meets_minimum_ram: host.meets_minimum(),
        total_download_mb: models_manager::required_download_mb(&models_dir),
        retired: models_manager::find_retired(&models_dir),
    })
}

/// A download or unpack is underway for this id.
fn is_active(progress: &InstallProgress) -> bool {
    matches!(
        progress.phase.as_str(),
        "queued" | "downloading" | "extracting" | "verifying"
    )
}

/// Why a download must wait, if it must. Downloads are one at a time across
/// the whole app — the Model Manager and first-run setup share one queue — so
/// a second never goes on the wire beside the first (commit 59ecece). And none
/// starts while a meeting is recording, where disk and CPU contention can drop
/// audio, or processing, where a runtime reinstall would overwrite the very
/// executables and DLLs the pipeline has open.
fn install_blocker(
    installs: &HashMap<String, InstallProgress>,
    recording: bool,
    processing: bool,
) -> Option<&'static str> {
    if installs.values().any(is_active) {
        Some("Another download is already running. Wait for it to finish, then try again.")
    } else if recording {
        Some("A meeting is being recorded. Stop the recording first, then install.")
    } else if processing {
        Some("A meeting is being processed. Wait for it to finish, then install.")
    } else {
        None
    }
}

fn is_recording(recording: &RecordingState) -> bool {
    recording.lock().map(|r| r.is_some()).unwrap_or(false)
}

fn is_processing(processing: &ProcessingState) -> bool {
    processing
        .lock()
        .map(|map| map.values().any(|s| s.status == "processing"))
        .unwrap_or(false)
}

fn failed(id: &str, error: String) -> InstallProgress {
    InstallProgress {
        id: id.to_string(),
        phase: "failed".to_string(),
        percent: 0,
        downloaded_mb: 0,
        total_mb: 0,
        error: Some(error),
    }
}

/// Starts an install in the background and returns immediately; the UI polls
/// `get_install_progress`. Downloads happen only here, only on an explicit
/// user action, and never while a meeting is being recorded or processed.
#[tauri::command]
pub fn install_model(
    app: AppHandle,
    state: State<InstallState>,
    recording: State<RecordingState>,
    processing: State<ProcessingState>,
    model_id: String,
) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    let models_dir = paths::models_dir(&data_dir);
    if models_manager::entry(&model_id).is_none() {
        return Err(format!("unknown model id: {model_id}"));
    }
    // A runtime the installer shipped is never downloaded from the app; say
    // so here rather than as a failed download.
    if let Some(refusal) = models_manager::install_refusal(&model_id) {
        return Err(refusal);
    }
    let recording = is_recording(&recording);
    let processing = is_processing(processing.inner());

    let progress_state = state.inner().clone();
    {
        let mut map = progress_state
            .lock()
            .map_err(|_| "install progress is unavailable".to_string())?;
        if let Some(reason) = install_blocker(&map, recording, processing) {
            return Err(reason.to_string());
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
            report(failed(&model_id, err));
        }
    });

    Ok(())
}

/// First-run setup: installs every missing required model, and returns the
/// queued ids in install order (runtimes, then the small models, the LLM last)
/// so the UI can poll `get_install_progress` for each. On a release build the
/// runtimes ship inside the installer and are never missing, so this queues
/// model data only.
///
/// Commit 59ecece removed an earlier batch installer, because back-to-back
/// curl/tar spawns that no user had agreed to read as dropper-like to endpoint
/// security. This one differs on both counts: it is invoked only from the
/// first-run consent screen, which lists every download and its size before
/// the user agrees, and it installs **sequentially** — one download, one
/// unpack, at a time, on a single background thread, never in parallel. A
/// failure is recorded against that one id and the rest still install, so a
/// flaky multi-gigabyte LLM download cannot leave the runtimes missing too.
#[tauri::command]
pub fn install_required_models(
    app: AppHandle,
    state: State<InstallState>,
    recording: State<RecordingState>,
    processing: State<ProcessingState>,
) -> Result<Vec<String>, String> {
    let recording = is_recording(&recording);
    let processing = is_processing(processing.inner());
    let data_dir = app_data_dir(&app)?;
    let models_dir = paths::models_dir(&data_dir);
    let queued: Vec<String> = models_manager::missing_required_ids(&models_dir)
        .into_iter()
        .map(str::to_string)
        .collect();

    let progress_state = state.inner().clone();
    {
        let mut map = progress_state
            .lock()
            .map_err(|_| "install progress is unavailable".to_string())?;
        // Any in-flight install — a previous setup run, or a single install
        // from the Model Manager — would put two downloads on the wire at once.
        if let Some(reason) = install_blocker(&map, recording, processing) {
            return Err(reason.to_string());
        }
        for id in &queued {
            map.insert(id.clone(), InstallProgress::new(id, "queued", 0));
        }
    }

    let order = queued.clone();
    std::thread::spawn(move || {
        let report = |progress: InstallProgress| {
            if let Ok(mut map) = progress_state.lock() {
                map.insert(progress.id.clone(), progress);
            }
        };
        for id in &order {
            if let Err(err) = models_manager::install(&models_dir, id, &report) {
                eprintln!("[setup] installing {id} failed: {err}");
                report(failed(id, err));
            }
        }
    });

    Ok(queued)
}

#[tauri::command]
pub fn get_install_progress(state: State<InstallState>, model_id: String) -> Option<InstallProgress> {
    state.lock().ok().and_then(|map| map.get(&model_id).cloned())
}

/// Refuses any required id — see `models_manager::remove`.
#[tauri::command]
pub fn remove_model(app: AppHandle, model_id: String) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    models_manager::remove(&paths::models_dir(&data_dir), &model_id)
}

/// Hashes every file of the entry — ~50 files and ~45 MB for a runtime, gigabytes
/// for the LLM — so it is `async` (a sync command runs on the main thread in
/// Tauri 2 and would freeze the window) and does the hashing on a blocking
/// worker, keeping the async runtime's threads free too.
#[tauri::command]
pub async fn verify_model(app: AppHandle, model_id: String) -> Result<VerifyReport, String> {
    let data_dir = app_data_dir(&app)?;
    let models_dir = paths::models_dir(&data_dir);
    tauri::async_runtime::spawn_blocking(move || models_manager::verify(&models_dir, &model_id))
        .await
        .map_err(|e| format!("model check did not finish: {e}"))?
}

/// What `check_models` reports as `retired`, for the Model Manager.
#[tauri::command]
pub fn list_retired_models(app: AppHandle) -> Result<Vec<RetiredModel>, String> {
    let data_dir = app_data_dir(&app)?;
    Ok(models_manager::find_retired(&paths::models_dir(&data_dir)))
}

/// Deletes one model the current set no longer uses. Called only from an
/// explicit, confirmed click; nothing removes these on its own.
///
/// Refused while any download or unpack is running, and the install map stays
/// locked for the (brief) delete, so an install cannot start halfway through
/// one. `models_manager::remove_retired` confines the delete to the models
/// folder and never touches the catalog's files or `models/bench`.
#[tauri::command]
pub fn remove_retired_model(
    app: AppHandle,
    state: State<InstallState>,
    key: String,
) -> Result<RetiredModel, String> {
    let data_dir = app_data_dir(&app)?;
    let map = state
        .lock()
        .map_err(|_| "install progress is unavailable".to_string())?;
    if map.values().any(is_active) {
        return Err(
            "A download is still running. Remove the old model once it has finished.".to_string(),
        );
    }
    let result = models_manager::remove_retired(&paths::models_dir(&data_dir), &key);
    drop(map);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installs(phases: &[(&str, &str)]) -> HashMap<String, InstallProgress> {
        phases
            .iter()
            .map(|(id, phase)| (id.to_string(), InstallProgress::new(id, phase, 0)))
            .collect()
    }

    #[test]
    fn a_second_download_waits_for_the_first_whatever_model_it_is() {
        for phase in ["queued", "downloading", "extracting", "verifying"] {
            let map = installs(&[("runtime-whisper", phase)]);
            assert!(
                install_blocker(&map, false, false).is_some(),
                "a {phase} install must block any other"
            );
        }
    }

    #[test]
    fn finished_or_failed_installs_do_not_block() {
        let map = installs(&[("runtime-whisper", "done"), ("vad-silero", "failed")]);
        assert_eq!(install_blocker(&map, false, false), None);
        assert_eq!(install_blocker(&HashMap::new(), false, false), None);
    }

    #[test]
    fn no_download_starts_while_a_meeting_records_or_processes() {
        let idle = HashMap::new();
        assert!(install_blocker(&idle, true, false).unwrap().contains("recorded"));
        assert!(install_blocker(&idle, false, true).unwrap().contains("processed"));
    }
}
