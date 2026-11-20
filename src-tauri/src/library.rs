//! Tauri commands for the meeting library: history, transcript, minutes,
//! speaker renaming, export.
//!
//! These are thin: they resolve the data directory and delegate to `storage`
//! and `export`. Business logic does not live in commands, and it certainly
//! does not live in React.

use crate::export::{self, Format};
use crate::pipeline::models::{MeetingMetadata, Mom, Transcript};
use crate::pipeline::redact::{self, RedactionSummary, ScanOptions};
use crate::pipeline::topic::{self, MeetingClassification};
use crate::processing::app_data_dir;
use crate::settings;
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
    /// The "how many people spoke?" answer processing was last started with —
    /// `None` for "Not sure" or a meeting never processed with one. The UI
    /// pre-selects it so a re-run does not silently drop it.
    pub expected_speakers: Option<u32>,
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
    let wav = storage::meeting_dir(&data_dir, &meeting_id)?.join("original.wav");
    Ok(MeetingSummary {
        meta,
        audio_path: wav
            .exists()
            .then(|| wav.to_string_lossy().to_string()),
        expected_speakers: storage::read_processing_options(&data_dir, &meeting_id)
            .expected_speakers,
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

// ------------------------------------------ derived, recomputable side-cars
//
// Four commands, two pairs, one rule: `get_*` reads what is cached and answers
// `null` when there is nothing there, and the other recomputes from
// `transcript.json` and rewrites the artifact. Nothing here reads
// `original.wav`, `stt.json` or `diarization.json`, and nothing here can fail
// in a way that costs a meeting — the worst case is a stale side-car and an
// error message.

/// Classifies the meeting from its transcript and caches the result.
///
/// Recomputes unconditionally rather than returning the cache, because the
/// caller asking for this is a user who pressed a button — usually because the
/// label on screen is wrong and they want it worked out again.
#[tauri::command]
pub fn classify_meeting(
    app: AppHandle,
    meeting_id: String,
) -> Result<MeetingClassification, String> {
    let data_dir = app_data_dir(&app)?;
    let transcript = storage::read_transcript(&data_dir, &meeting_id).map_err(|_| {
        "This meeting has no transcript yet, so there is nothing to classify.".to_string()
    })?;
    let classification = topic::classify(&transcript);
    // A failed write is logged, not returned: the classification in hand is
    // correct and useful, and losing the cache only costs a recomputation.
    if let Err(err) = storage::write_classification(&data_dir, &meeting_id, &classification) {
        eprintln!("[library] could not cache classification for {meeting_id}: {err}");
    }
    Ok(classification)
}

#[tauri::command]
pub fn get_meeting_classification(
    app: AppHandle,
    meeting_id: String,
) -> Result<Option<MeetingClassification>, String> {
    let data_dir = app_data_dir(&app)?;
    Ok(storage::read_classification(&data_dir, &meeting_id))
}

/// Rescans the transcript for unparliamentary language and **rewrites
/// `transcript.json`** with the flags it found.
///
/// Rewriting the transcript is the point of the command rather than a side
/// effect of it. Flags live on the segments; the transcript view, the export
/// and the summary all read them from there. A rescan that only refreshed
/// `redaction.json` would leave a meeting whose badge says one thing and whose
/// exported file says another — and the export is the copy that leaves the
/// machine.
///
/// The scan runs whatever the mode is set to, including `off`. Flags are
/// annotations, not edits: `redact::apply` at `off` renders the text unchanged
/// anyway, and a user who asked for a rescan asked for the count.
#[tauri::command]
pub fn rescan_redactions(app: AppHandle, meeting_id: String) -> Result<RedactionSummary, String> {
    let data_dir = app_data_dir(&app)?;
    let mut transcript = storage::read_transcript(&data_dir, &meeting_id).map_err(|_| {
        "This meeting has no transcript yet, so there is nothing to scan.".to_string()
    })?;

    let user_settings = settings::load(&data_dir);
    // The user's own lexicon, so a term they added — or removed as a false
    // positive — is what this rescan is actually rescanning against.
    let lexicon = redact::Lexicon::load(&data_dir);
    let turns: Vec<String> = transcript
        .segments
        .iter()
        .map(|segment| segment.text.clone())
        .collect();
    let (per_turn, summary) = redact::scan_turns(
        &turns,
        &lexicon,
        &ScanOptions {
            fuzzy: user_settings.redaction_fuzzy,
        },
    );
    for (segment, flags) in transcript.segments.iter_mut().zip(per_turn) {
        segment.flags = flags;
    }

    storage::write_transcript(&data_dir, &meeting_id, &transcript)?;
    storage::write_redaction_summary(&data_dir, &meeting_id, &summary)?;
    Ok(summary)
}

#[tauri::command]
pub fn get_redaction_summary(
    app: AppHandle,
    meeting_id: String,
) -> Result<Option<RedactionSummary>, String> {
    let data_dir = app_data_dir(&app)?;
    Ok(storage::read_redaction_summary(&data_dir, &meeting_id))
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
    let dir = storage::meeting_dir(&data_dir, &meeting_id)?;
    if !dir.exists() {
        return Err(format!("{} does not exist", dir.display()));
    }
    Ok(dir.to_string_lossy().to_string())
}
