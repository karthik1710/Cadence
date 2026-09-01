//! Tauri commands that drive the local pipeline and the §12 meeting state
//! machine.
//!
//! Every one of these runs on this machine. There is no upload, no job queue,
//! and no remote service: `start_processing` spawns a worker thread that
//! shells out to the installed native binaries and returns immediately, and
//! the UI polls `get_processing_status`.

use crate::pipeline::models::{MeetingMetadata, MeetingState, MeetingStatus};
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::{self, PipelineConfig, RunRequest};
use crate::settings;
use crate::storage;
use crate::system::HostInfo;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};

/// Live processing status per meeting id. In-memory only: the durable record
/// is `metadata.json`, which every stage transition also writes, so an
/// interrupted meeting is recognised on the next start (§33).
pub type ProcessingState = Arc<Mutex<HashMap<String, MeetingStatus>>>;

pub fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path().app_data_dir().map_err(|e| e.to_string())
}

fn set_status(statuses: &ProcessingState, meeting_id: &str, status: MeetingStatus) {
    if let Ok(mut map) = statuses.lock() {
        map.insert(meeting_id.to_string(), status);
    }
}

/// Creates the meeting's folder and `metadata.json` *before* recording starts,
/// so a crash mid-recording still leaves an identifiable meeting behind rather
/// than an orphaned WAV.
#[tauri::command]
pub fn create_meeting(app: AppHandle, title: String, notes: String) -> Result<MeetingMetadata, String> {
    let data_dir = app_data_dir(&app)?;
    let id = new_meeting_id();
    let mut meta = MeetingMetadata::new(&id, title.trim(), storage::now_ms());
    meta.notes = notes;
    meta.state = MeetingState::Recording;
    storage::write_metadata(&data_dir, &meta)?;
    Ok(meta)
}

/// Records how the recording actually turned out once it has stopped.
#[tauri::command]
pub fn finalize_recording(
    app: AppHandle,
    meeting_id: String,
    device_name: Option<String>,
    sample_rate: u32,
    channels: u16,
) -> Result<MeetingMetadata, String> {
    let data_dir = app_data_dir(&app)?;
    let wav = storage::meeting_dir(&data_dir, &meeting_id).join("original.wav");
    let duration_ms = storage::wav_duration_ms(&wav).unwrap_or(0);

    storage::update_metadata(&data_dir, &meeting_id, |meta| {
        meta.state = MeetingState::Recorded;
        meta.duration_ms = duration_ms;
        meta.device_name = device_name;
        meta.sample_rate = sample_rate;
        meta.channels = channels;
        meta.error = None;
    })
}

/// Starts the local pipeline for an already-recorded meeting and returns
/// immediately; the UI polls `get_processing_status`.
#[tauri::command]
pub fn start_processing(
    app: AppHandle,
    state: State<ProcessingState>,
    meeting_id: String,
) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    let dir = storage::meeting_dir(&data_dir, &meeting_id);
    let original = dir.join("original.wav");
    if !original.exists() {
        return Err(format!(
            "No recording found at {}. The audio file must exist before processing.",
            original.display()
        ));
    }

    let user_settings = settings::load(&data_dir);
    let host = HostInfo::detect();
    // Resolved up front so a missing model is reported to the user immediately
    // rather than after processing appears to start.
    let paths = ModelPaths::resolve(&app, &user_settings)?;
    let config = PipelineConfig::from_settings(&user_settings, &host);
    let title = storage::read_metadata(&data_dir, &meeting_id)
        .map(|m| m.display_title())
        .unwrap_or_else(|_| "Untitled meeting".to_string());

    let statuses = state.inner().clone();
    set_status(&statuses, &meeting_id, MeetingStatus::processing("QUEUED", 0));

    std::thread::spawn(move || {
        let progress = {
            let statuses = statuses.clone();
            let meeting_id = meeting_id.clone();
            let data_dir = data_dir.clone();
            move |stage: &str, pct: i64| {
                set_status(&statuses, &meeting_id, MeetingStatus::processing(stage, pct));
                // Persist the stage too, so an interruption is recoverable.
                if let Some(state) = MeetingState::parse(stage) {
                    let _ = storage::set_state(&data_dir, &meeting_id, state, None);
                }
            }
        };

        let result = pipeline::run(
            &RunRequest {
                paths: &paths,
                config: &config,
                settings: &user_settings,
                host: &host,
                input_wav: &original,
                out_dir: &dir,
                data_dir: &data_dir,
                meeting_id: &meeting_id,
                title: &title,
            },
            &progress,
        );

        match result {
            Ok(outcome) => {
                let duration_ms = outcome.transcript.duration_ms();
                let mom_error = outcome.mom_error.clone();
                let _ = storage::update_metadata(&data_dir, &meeting_id, |meta| {
                    meta.state = MeetingState::Completed;
                    meta.error = None;
                    meta.mom_error = mom_error.clone();
                    if duration_ms > 0 {
                        meta.duration_ms = meta.duration_ms.max(duration_ms);
                    }
                });
                set_status(
                    &statuses,
                    &meeting_id,
                    MeetingStatus::completed(outcome.mom_error),
                );
            }
            Err(err) => {
                // Report the stage that was in flight, so a failure points at
                // the stage that broke rather than just "failed".
                let stage = statuses
                    .lock()
                    .ok()
                    .and_then(|m| m.get(&meeting_id).and_then(|s| s.stage.clone()))
                    .unwrap_or_else(|| "PROCESSING".to_string());
                let _ = storage::set_state(
                    &data_dir,
                    &meeting_id,
                    MeetingState::Failed,
                    Some(err.clone()),
                );
                set_status(&statuses, &meeting_id, MeetingStatus::failed(&stage, err));
            }
        }
    });

    Ok(())
}

/// Regenerates only the minutes for a meeting that already has a transcript —
/// the retry path after an LLM failure, and what "Regenerate minutes" calls
/// after changing model or output language.
#[tauri::command]
pub fn regenerate_mom(
    app: AppHandle,
    state: State<ProcessingState>,
    meeting_id: String,
) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    let dir = storage::meeting_dir(&data_dir, &meeting_id);
    if !dir.join("transcript.json").exists() {
        return Err(
            "This meeting has no transcript yet. Run processing first.".to_string()
        );
    }

    let mut user_settings = settings::load(&data_dir);
    user_settings.generate_mom = true;
    let host = HostInfo::detect();
    let paths = ModelPaths::resolve(&app, &user_settings)?;
    let config = PipelineConfig::from_settings(&user_settings, &host);
    let title = storage::read_metadata(&data_dir, &meeting_id)
        .map(|m| m.display_title())
        .unwrap_or_else(|_| "Untitled meeting".to_string());

    let statuses = state.inner().clone();
    set_status(
        &statuses,
        &meeting_id,
        MeetingStatus::processing("EXTRACTING", 60),
    );

    std::thread::spawn(move || {
        let progress = {
            let statuses = statuses.clone();
            let meeting_id = meeting_id.clone();
            move |stage: &str, pct: i64| {
                set_status(&statuses, &meeting_id, MeetingStatus::processing(stage, pct));
            }
        };

        let result = pipeline::run_mom_only(
            &RunRequest {
                paths: &paths,
                config: &config,
                settings: &user_settings,
                host: &host,
                input_wav: &dir.join("original.wav"),
                out_dir: &dir,
                data_dir: &data_dir,
                meeting_id: &meeting_id,
                title: &title,
            },
            &progress,
        );

        match result {
            Ok(_) => {
                let _ = storage::update_metadata(&data_dir, &meeting_id, |meta| {
                    meta.state = MeetingState::Completed;
                    meta.mom_error = None;
                    meta.error = None;
                });
                set_status(&statuses, &meeting_id, MeetingStatus::completed(None));
            }
            Err(err) => {
                let _ = storage::update_metadata(&data_dir, &meeting_id, |meta| {
                    // The transcript is intact, so the meeting is not failed —
                    // only its minutes are missing.
                    meta.state = MeetingState::Completed;
                    meta.mom_error = Some(err.clone());
                });
                set_status(
                    &statuses,
                    &meeting_id,
                    MeetingStatus::completed(Some(err)),
                );
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub fn get_processing_status(
    app: AppHandle,
    state: State<ProcessingState>,
    meeting_id: String,
) -> MeetingStatus {
    if let Some(status) = state
        .lock()
        .ok()
        .and_then(|m| m.get(&meeting_id).cloned())
    {
        return status;
    }

    // Nothing in memory: this process did not run it (an app restart, or the
    // history view opening an old meeting). Reconstruct from what is on disk.
    let Ok(data_dir) = app_data_dir(&app) else {
        return MeetingStatus::idle();
    };
    match storage::read_metadata(&data_dir, &meeting_id) {
        Ok(meta) => match meta.state {
            MeetingState::Completed => MeetingStatus::completed(meta.mom_error),
            MeetingState::Failed => MeetingStatus::failed(
                "PROCESSING",
                meta.error
                    .unwrap_or_else(|| "Processing failed.".to_string()),
            ),
            state if state.is_processing() => {
                MeetingStatus::processing(state.as_str(), state.progress())
            }
            _ => MeetingStatus::idle(),
        },
        Err(_) => MeetingStatus::idle(),
    }
}

/// Marks meetings that were interrupted by an app close or crash, so they
/// appear as recoverable rather than stuck (§12/§33). Called once at startup.
pub fn recover_on_startup(app: &AppHandle) {
    let Ok(data_dir) = app_data_dir(app) else {
        return;
    };
    let recovered = storage::recover_interrupted(&data_dir);
    if !recovered.is_empty() {
        eprintln!(
            "[processing] marked {} interrupted meeting(s) as recoverable",
            recovered.len()
        );
    }
}

/// Meeting ids are generated here, on the Rust side, because this is where the
/// folder is created. They double as directory names, so they are hex only.
///
/// Uniqueness needs to hold on one machine, not globally: a millisecond
/// timestamp, a process-lifetime counter, and the hasher seed the standard
/// library takes from the OS at startup are together far more than enough, and
/// the leading timestamp makes the folders sort chronologically.
fn new_meeting_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let millis = storage::now_ms().max(0) as u64;
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let random = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    format!("{millis:011x}{sequence:04x}{random:016x}")
}
