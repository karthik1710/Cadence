//! Tauri commands for the meeting library: history, transcript, minutes,
//! speaker renaming, export.
//!
//! These are thin: they resolve the data directory and delegate to `storage`
//! and `export`. Business logic does not live in commands, and it certainly
//! does not live in React.

use crate::export::{self, Format};
use crate::pipeline::models::{MeetingMetadata, Mom, Transcript};
use crate::processing::app_data_dir;
use crate::storage;
use serde::Serialize;
use tauri::AppHandle;

#[derive(Serialize)]
pub struct MeetingSummary {
    #[serde(flatten)]
    pub meta: MeetingMetadata,
    /// Absolute path to the original recording, if it still exists — the UI
    /// uses it for timestamp-linked playback.
    pub audio_path: Option<String>,
}

#[tauri::command]
pub fn list_meetings(
    app: AppHandle,
    query: Option<String>,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
) -> Result<Vec<MeetingMetadata>, String> {
    let data_dir = app_data_dir(&app)?;
    Ok(storage::list_meetings(
        &data_dir,
        query.as_deref(),
        from_ms,
        to_ms,
    ))
}

#[tauri::command]
pub fn get_meeting(app: AppHandle, meeting_id: String) -> Result<MeetingSummary, String> {
    let data_dir = app_data_dir(&app)?;
    let meta = storage::read_metadata(&data_dir, &meeting_id)?;
    let wav = storage::meeting_dir(&data_dir, &meeting_id).join("original.wav");
    Ok(MeetingSummary {
        meta,
        audio_path: wav
            .exists()
            .then(|| wav.to_string_lossy().to_string()),
    })
}

#[tauri::command]
pub fn update_meeting(
    app: AppHandle,
    meeting_id: String,
    title: Option<String>,
    notes: Option<String>,
) -> Result<MeetingMetadata, String> {
    let data_dir = app_data_dir(&app)?;
    storage::update_metadata(&data_dir, &meeting_id, |meta| {
        if let Some(title) = title {
            meta.title = title.trim().to_string();
        }
        if let Some(notes) = notes {
            meta.notes = notes;
        }
    })
}

/// Renames a speaker for display only. The stable id and the raw
/// `diarization.json` are never rewritten, so the underlying data stays
/// exactly as the model produced it.
#[tauri::command]
pub fn rename_speaker(
    app: AppHandle,
    meeting_id: String,
    speaker: String,
    name: String,
) -> Result<MeetingMetadata, String> {
    let data_dir = app_data_dir(&app)?;
    storage::update_metadata(&data_dir, &meeting_id, |meta| {
        let name = name.trim().to_string();
        if name.is_empty() {
            meta.speaker_names.remove(&speaker);
        } else {
            meta.speaker_names.insert(speaker, name);
        }
    })
}

#[tauri::command]
pub fn delete_meeting(app: AppHandle, meeting_id: String) -> Result<(), String> {
    let data_dir = app_data_dir(&app)?;
    storage::delete_meeting(&data_dir, &meeting_id)
}

#[tauri::command]
pub fn get_transcript(app: AppHandle, meeting_id: String) -> Result<Transcript, String> {
    let data_dir = app_data_dir(&app)?;
    storage::read_transcript(&data_dir, &meeting_id)
}

#[tauri::command]
pub fn get_mom(app: AppHandle, meeting_id: String) -> Result<Mom, String> {
    let data_dir = app_data_dir(&app)?;
    storage::read_mom(&data_dir, &meeting_id)
}

/// Writes an export and returns its path (Markdown and JSON in V1).
#[tauri::command]
pub fn export_meeting(
    app: AppHandle,
    meeting_id: String,
    format: String,
) -> Result<String, String> {
    let data_dir = app_data_dir(&app)?;
    let format = Format::parse(&format)
        .ok_or_else(|| format!("unsupported export format: {format}"))?;
    let path = export::write_export(&data_dir, &meeting_id, format)?;
    Ok(path.to_string_lossy().to_string())
}

/// Opens a meeting's folder in File Explorer, so exports and raw artifacts are
/// reachable without the user hunting through AppData.
#[tauri::command]
pub fn reveal_meeting_folder(app: AppHandle, meeting_id: String) -> Result<String, String> {
    let data_dir = app_data_dir(&app)?;
    let dir = storage::meeting_dir(&data_dir, &meeting_id);
    if !dir.exists() {
        return Err(format!("{} does not exist", dir.display()));
    }
    Ok(dir.to_string_lossy().to_string())
}
