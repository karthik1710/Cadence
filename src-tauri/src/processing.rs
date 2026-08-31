use crate::pipeline::models::{MeetingStatus, Transcript};
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::{self, PipelineConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};

/// Live processing status per meeting id. In-memory only: a meeting whose
/// processing was interrupted by an app restart still has all its artifacts on
/// disk (§33), and re-running the pipeline is the recovery path.
pub type ProcessingState = Arc<Mutex<HashMap<String, MeetingStatus>>>;

fn meeting_dir(app: &AppHandle, meeting_id: &str) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("meetings")
        .join(meeting_id))
}

fn set_status(statuses: &ProcessingState, meeting_id: &str, status: MeetingStatus) {
    if let Ok(mut map) = statuses.lock() {
        map.insert(meeting_id.to_string(), status);
    }
}

/// Starts the local pipeline for an already-recorded meeting and returns
/// immediately; the UI polls `get_processing_status`. Everything runs on this
/// machine — there is no upload and no remote service involved.
#[tauri::command]
pub fn start_processing(
    app: AppHandle,
    state: State<ProcessingState>,
    meeting_id: String,
) -> Result<(), String> {
    let dir = meeting_dir(&app, &meeting_id)?;
    let original = dir.join("original.wav");
    if !original.exists() {
        return Err(format!(
            "No recording found at {}. The audio file must exist before processing.",
            original.display()
        ));
    }

    // Resolved up front so a missing model is reported to the user immediately
    // rather than after the recording appears to start processing.
    let paths = ModelPaths::resolve(&app)?;
    let config = PipelineConfig::default();
    let statuses = state.inner().clone();

    set_status(&statuses, &meeting_id, MeetingStatus::processing("QUEUED", 0));

    std::thread::spawn(move || {
        let progress_statuses = statuses.clone();
        let progress_id = meeting_id.clone();
        let result = pipeline::run(
            &paths,
            &config,
            &original,
            &dir,
            &meeting_id,
            &move |stage, pct| {
                set_status(
                    &progress_statuses,
                    &progress_id,
                    MeetingStatus::processing(stage, pct),
                );
            },
        );

        match result {
            Ok(_) => set_status(&statuses, &meeting_id, MeetingStatus::completed()),
            Err(err) => {
                // Report the stage that was in flight, so a failure points at
                // the stage that broke rather than just "failed".
                let stage = statuses
                    .lock()
                    .ok()
                    .and_then(|m| m.get(&meeting_id).and_then(|s| s.stage.clone()))
                    .unwrap_or_else(|| "PROCESSING".to_string());
                set_status(&statuses, &meeting_id, MeetingStatus::failed(&stage, err));
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub fn get_processing_status(state: State<ProcessingState>, meeting_id: String) -> MeetingStatus {
    state
        .lock()
        .ok()
        .and_then(|m| m.get(&meeting_id).cloned())
        .unwrap_or_else(|| MeetingStatus {
            status: "idle".to_string(),
            stage: None,
            progress: 0,
            error: None,
        })
}

#[tauri::command]
pub fn get_transcript(app: AppHandle, meeting_id: String) -> Result<Transcript, String> {
    let path = meeting_dir(&app, &meeting_id)?.join("transcript.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("could not parse transcript.json: {e}"))
}

/// Lets the UI tell the user models are missing before they record a meeting,
/// rather than after (§33 — never lose a meeting to a setup problem).
#[tauri::command]
pub fn check_models(app: AppHandle) -> Result<(), String> {
    ModelPaths::resolve(&app).map(|_| ())
}
