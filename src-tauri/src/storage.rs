//! Local meeting storage and the history index.
//!
//! ```text
//! <app_data>/
//! ├── config/settings.json
//! ├── models/
//! ├── database/meetings.db      SQLite index for history: metadata only
//! ├── logs/
//! └── meetings/<id>/
//!     ├── metadata.json         authoritative record for this meeting
//!     ├── processing_options.json  per-meeting processing input, optional
//!     ├── original.wav          never modified, never deleted automatically
//!     ├── processed.wav
//!     ├── diarization.json      raw stage output
//!     ├── stt.json              raw stage output
//!     ├── transcript.json       derived (turns, plus redaction flags)
//!     ├── mom.json              derived
//!     ├── audio_profile.json    derived, optional
//!     ├── classification.json   derived, optional
//!     ├── redaction.json        derived, optional
//!     └── exports/
//! ```
//!
//! `metadata.json` on disk is the source of truth; SQLite is a rebuildable
//! index over it, holding only what the history list needs to sort and filter
//! ("do not duplicate full transcript data unnecessarily"). If the
//! database is lost or corrupted, [`list_meetings`] rebuilds it by scanning the
//! meetings directory — no meeting can be lost to a broken index.

use crate::pipeline::audio_profile::AudioProfile;
use crate::pipeline::models::{MeetingMetadata, MeetingState, Mom, Transcript};
use crate::pipeline::redact::RedactionSummary;
use crate::pipeline::topic::MeetingClassification;
use rusqlite::Connection;
use std::path::{Path, PathBuf};

pub fn meetings_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("meetings")
}

fn validate_component(value: &str) -> Result<(), String> {
    let stem = value
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    let reserved = matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || (stem.chars().count() == 4
        && (stem.starts_with("COM") || stem.starts_with("LPT"))
        && matches!(stem.chars().last(), Some('1'..='9' | '¹' | '²' | '³')));
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.ends_with(['.', ' '])
        || value
            .chars()
            .any(|c| c.is_control() || "\\/:*?\"<>|".contains(c))
        || reserved
    {
        return Err("invalid meeting path component".to_string());
    }
    Ok(())
}

/// Reject links and Windows junctions before accessing app-owned artifacts.
pub fn reject_link(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            #[cfg(windows)]
            let reparse = {
                use std::os::windows::fs::MetadataExt;
                meta.file_attributes() & 0x400 != 0
            };
            #[cfg(not(windows))]
            let reparse = false;
            if meta.file_type().is_symlink() || reparse {
                return Err(format!(
                    "linked meeting path is not allowed: {}",
                    path.display()
                ));
            }
            Ok(())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!("could not inspect {}: {err}", path.display())),
    }
}

pub fn meeting_dir(data_dir: &Path, meeting_id: &str) -> Result<PathBuf, String> {
    validate_component(meeting_id)?;
    let root = meetings_dir(data_dir);
    reject_link(&root)?;
    let dir = root.join(meeting_id);
    reject_link(&dir)?;
    // Pipeline stages access fixed artifact names directly. Check existing
    // children as well so an artifact link cannot redirect their reads/writes.
    if dir.exists() {
        for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
            reject_link(&entry.map_err(|e| e.to_string())?.path())?;
        }
    }
    Ok(dir)
}

fn metadata_path(data_dir: &Path, meeting_id: &str) -> Result<PathBuf, String> {
    Ok(meeting_dir(data_dir, meeting_id)?.join("metadata.json"))
}

fn database_path(data_dir: &Path) -> PathBuf {
    data_dir.join("database").join("meetings.db")
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ------------------------------------------------------------- metadata

pub fn read_metadata(data_dir: &Path, meeting_id: &str) -> Result<MeetingMetadata, String> {
    let path = metadata_path(data_dir, meeting_id)?;
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let meta: MeetingMetadata = serde_json::from_str(&raw)
        .map_err(|e| format!("could not parse {}: {e}", path.display()))?;
    if meta.id != meeting_id {
        return Err("meeting metadata id does not match its directory".to_string());
    }
    Ok(meta)
}

/// Writes `metadata.json` and mirrors the row into the index. An index write
/// that fails is logged, not propagated: the meeting itself is already safe on
/// disk, and history can always be rebuilt.
pub fn write_metadata(data_dir: &Path, meta: &MeetingMetadata) -> Result<(), String> {
    let path = metadata_path(data_dir, &meta.id)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display()))?;

    if let Err(err) = index_upsert(data_dir, meta) {
        eprintln!(
            "[storage] history index update failed for {}: {err}",
            meta.id
        );
    }
    Ok(())
}

/// Applies `edit` to a meeting's metadata and saves it. Used for every state
/// transition, so the state machine has exactly one write path.
pub fn update_metadata<F: FnOnce(&mut MeetingMetadata)>(
    data_dir: &Path,
    meeting_id: &str,
    edit: F,
) -> Result<MeetingMetadata, String> {
    let mut meta = read_metadata(data_dir, meeting_id)?;
    edit(&mut meta);
    meta.has_transcript = meeting_dir(data_dir, meeting_id)?
        .join("transcript.json")
        .exists();
    meta.has_mom = meeting_dir(data_dir, meeting_id)?.join("mom.json").exists();
    write_metadata(data_dir, &meta)?;
    Ok(meta)
}

pub fn set_state(
    data_dir: &Path,
    meeting_id: &str,
    state: MeetingState,
    error: Option<String>,
) -> Result<(), String> {
    update_metadata(data_dir, meeting_id, |meta| {
        meta.state = state;
        meta.error = error;
    })
    .map(|_| ())
}

// -------------------------------------------------------------- artifacts

pub fn read_transcript(data_dir: &Path, meeting_id: &str) -> Result<Transcript, String> {
    let path = meeting_dir(data_dir, meeting_id)?.join("transcript.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("could not parse transcript.json: {e}"))
}

/// Rewrites `transcript.json`.
///
/// The only write path for it outside the pipeline, and it exists for exactly
/// one caller: a redaction rescan, which recomputes the per-segment flags and
/// has to persist them or the transcript view and the export would disagree
/// about the same meeting. `original.wav`, `stt.json` and `diarization.json`
/// have no equivalent and never will — they are raw stage output.
pub fn write_transcript(
    data_dir: &Path,
    meeting_id: &str,
    transcript: &Transcript,
) -> Result<(), String> {
    write_artifact(data_dir, meeting_id, "transcript.json", transcript)
}

pub fn read_mom(data_dir: &Path, meeting_id: &str) -> Result<Mom, String> {
    let path = meeting_dir(data_dir, meeting_id)?.join("mom.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("could not parse mom.json: {e}"))
}

// ------------------------------------------------ derived side-car artifacts
//
// `audio_profile.json`, `classification.json` and `redaction.json` are all
// derived, all optional, and all recomputable from what is already on disk.
// That is why the read side returns `Option` rather than `Result`: "this
// meeting predates the feature" and "this meeting's file is corrupt" are both
// answered the same way — recompute it — and neither is an error worth
// surfacing to a user opening a meeting.

/// Reads one derived artifact, or `None` when it is absent or unreadable.
pub fn read_artifact<T: serde::de::DeserializeOwned>(
    data_dir: &Path,
    meeting_id: &str,
    file_name: &str,
) -> Option<T> {
    validate_component(file_name).ok()?;
    let path = meeting_dir(data_dir, meeting_id).ok()?.join(file_name);
    let raw = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str(&raw) {
        Ok(value) => Some(value),
        Err(err) => {
            eprintln!("[storage] ignoring unreadable {}: {err}", path.display());
            None
        }
    }
}

/// Writes one derived artifact, creating the meeting directory if needed.
pub fn write_artifact<T: serde::Serialize>(
    data_dir: &Path,
    meeting_id: &str,
    file_name: &str,
    value: &T,
) -> Result<(), String> {
    validate_component(file_name)?;
    let dir = meeting_dir(data_dir, meeting_id)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(file_name);
    let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display()))
}

pub fn read_audio_profile(data_dir: &Path, meeting_id: &str) -> Option<AudioProfile> {
    read_artifact(data_dir, meeting_id, "audio_profile.json")
}

pub fn write_audio_profile(
    data_dir: &Path,
    meeting_id: &str,
    profile: &AudioProfile,
) -> Result<(), String> {
    write_artifact(data_dir, meeting_id, "audio_profile.json", profile)
}

pub fn read_classification(data_dir: &Path, meeting_id: &str) -> Option<MeetingClassification> {
    read_artifact(data_dir, meeting_id, "classification.json")
}

pub fn write_classification(
    data_dir: &Path,
    meeting_id: &str,
    classification: &MeetingClassification,
) -> Result<(), String> {
    write_artifact(data_dir, meeting_id, "classification.json", classification)
}

pub fn read_redaction_summary(data_dir: &Path, meeting_id: &str) -> Option<RedactionSummary> {
    read_artifact(data_dir, meeting_id, "redaction.json")
}

pub fn write_redaction_summary(
    data_dir: &Path,
    meeting_id: &str,
    summary: &RedactionSummary,
) -> Result<(), String> {
    write_artifact(data_dir, meeting_id, "redaction.json", summary)
}

// ------------------------------------------- per-meeting processing input
//
// Unlike the side-cars above, this is not derived from anything: it is what
// the organiser told Cadence about the meeting when processing started, and a
// re-run reads it back so it does not have to be asked again. It lives in its
// own file rather than in `metadata.json` because it is input to the pipeline,
// not part of the meeting's identity or state — and `MeetingMetadata` is
// written on every stage transition, which this never needs to be.

const PROCESSING_OPTIONS_FILE: &str = "processing_options.json";

/// Per-meeting choices that change how the pipeline runs. Every field is
/// optional and `#[serde(default)]`, so a file written by an older or newer
/// build still reads, and an absent file means "all automatic".
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProcessingOptions {
    /// "How many people spoke?" — `None` is "Not sure" (automatic clustering).
    pub expected_speakers: Option<u32>,
}

/// The meeting's processing options, or the all-automatic default when none
/// were ever saved or the file cannot be read. A hint that is out of range on
/// disk (hand-edited, or from a build with different limits) is dropped rather
/// than passed to diarization.
pub fn read_processing_options(data_dir: &Path, meeting_id: &str) -> ProcessingOptions {
    let mut options: ProcessingOptions =
        read_artifact(data_dir, meeting_id, PROCESSING_OPTIONS_FILE).unwrap_or_default();
    options.expected_speakers =
        crate::pipeline::validate_expected_speakers(options.expected_speakers).unwrap_or(None);
    options
}

pub fn write_processing_options(
    data_dir: &Path,
    meeting_id: &str,
    options: &ProcessingOptions,
) -> Result<(), String> {
    write_artifact(data_dir, meeting_id, PROCESSING_OPTIONS_FILE, options)
}

/// Duration of a 16-bit PCM WAV, in milliseconds.
pub fn wav_duration_ms(path: &Path) -> Result<i64, String> {
    let reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let frames = reader.len() as i64 / spec.channels.max(1) as i64;
    Ok(frames * 1000 / spec.sample_rate.max(1) as i64)
}

/// Deletes a meeting and everything in it. The only path that ever removes a
/// recording, and only ever on an explicit user action.
pub fn delete_meeting(data_dir: &Path, meeting_id: &str) -> Result<(), String> {
    let dir = meeting_dir(data_dir, meeting_id)?;
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|e| format!("could not delete {}: {e}", dir.display()))?;
    }
    if let Ok(conn) = open_db(data_dir) {
        let _ = conn.execute("DELETE FROM meetings WHERE id = ?1", [meeting_id]);
    }
    Ok(())
}

/// Marks meetings that were mid-processing when the app stopped as failed, so
/// they show up as recoverable instead of permanently "in progress".
/// Returns the ids it recovered.
pub fn recover_interrupted(data_dir: &Path) -> Vec<String> {
    let mut recovered = Vec::new();
    for meta in scan_meetings(data_dir) {
        let interrupted = meta.state.is_processing() || meta.state == MeetingState::Recording;
        if !interrupted {
            continue;
        }
        let message = if meta.state == MeetingState::Recording {
            "Recording was interrupted (the app closed or the machine slept). The audio captured up to that point is kept."
        } else {
            "Processing was interrupted (the app closed). The recording and any completed stages are kept — retry processing to finish."
        };
        let _ = update_metadata(data_dir, &meta.id, |m| {
            m.state = MeetingState::Failed;
            m.error = Some(message.to_string());
        });
        recovered.push(meta.id);
    }
    recovered
}

// ---------------------------------------------------------------- history

/// Meetings newest first, optionally filtered by a title/notes substring and a
/// created-at range.
pub fn list_meetings(
    data_dir: &Path,
    query: Option<&str>,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
) -> Vec<MeetingMetadata> {
    let mut all = match index_list(data_dir) {
        Ok(ids) if !ids.is_empty() => ids
            .into_iter()
            .filter_map(|id| read_metadata(data_dir, &id).ok())
            .collect::<Vec<_>>(),
        // Empty or unusable index: fall back to the filesystem and rebuild.
        _ => {
            let scanned = scan_meetings(data_dir);
            for meta in &scanned {
                let _ = index_upsert(data_dir, meta);
            }
            scanned
        }
    };

    all.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    let needle = query
        .map(|q| q.trim().to_lowercase())
        .filter(|q| !q.is_empty());

    all.into_iter()
        .filter(|meta| match &needle {
            Some(q) => {
                meta.title.to_lowercase().contains(q) || meta.notes.to_lowercase().contains(q)
            }
            None => true,
        })
        .filter(|meta| {
            from_ms
                .map(|from| meta.created_at_ms >= from)
                .unwrap_or(true)
        })
        .filter(|meta| to_ms.map(|to| meta.created_at_ms <= to).unwrap_or(true))
        .collect()
}

/// Reads every `metadata.json` under `meetings/`. The authoritative view, used
/// to rebuild the index and to recover interrupted meetings.
pub fn scan_meetings(data_dir: &Path) -> Vec<MeetingMetadata> {
    let dir = meetings_dir(data_dir);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        if let Ok(meta) = read_metadata(data_dir, &id) {
            out.push(meta);
        }
    }
    out
}

fn open_db(data_dir: &Path) -> Result<Connection, String> {
    let path = database_path(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let conn = Connection::open(&path).map_err(|e| e.to_string())?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meetings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            created_at_ms INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL,
            has_transcript INTEGER NOT NULL DEFAULT 0,
            has_mom INTEGER NOT NULL DEFAULT 0
        )",
        [],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings (created_at_ms DESC)",
        [],
    )
    .map_err(|e| e.to_string())?;
    Ok(conn)
}

fn index_upsert(data_dir: &Path, meta: &MeetingMetadata) -> Result<(), String> {
    let conn = open_db(data_dir)?;
    conn.execute(
        "INSERT INTO meetings (id, title, notes, created_at_ms, duration_ms, state, has_transcript, has_mom)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            notes = excluded.notes,
            created_at_ms = excluded.created_at_ms,
            duration_ms = excluded.duration_ms,
            state = excluded.state,
            has_transcript = excluded.has_transcript,
            has_mom = excluded.has_mom",
        rusqlite::params![
            meta.id,
            meta.title,
            meta.notes,
            meta.created_at_ms,
            meta.duration_ms,
            meta.state.as_str(),
            meta.has_transcript as i64,
            meta.has_mom as i64,
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn index_list(data_dir: &Path) -> Result<Vec<String>, String> {
    let conn = open_db(data_dir)?;
    let mut stmt = conn
        .prepare("SELECT id FROM meetings ORDER BY created_at_ms DESC")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_every_windows_reserved_device_with_extensions_and_case_variants() {
        let mut devices = vec![
            "CON".to_string(),
            "PRN".into(),
            "AUX".into(),
            "NUL".into(),
            "CONIN$".into(),
            "CONOUT$".into(),
        ];
        for prefix in ["COM", "LPT"] {
            for suffix in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "¹", "²", "³"] {
                devices.push(format!("{prefix}{suffix}"));
            }
        }
        for device in devices {
            for value in [
                device.clone(),
                device.to_lowercase(),
                format!("{device}.json"),
                format!("{device} .json"),
            ] {
                assert!(validate_component(&value).is_err(), "accepted {value:?}");
            }
        }
        // These names are ordinary directories, not Windows devices.
        for value in ["COM0", "COM10", "LPT0", "LPT10", "CONference"] {
            assert!(validate_component(value).is_ok(), "rejected {value:?}");
        }
    }

    #[test]
    fn rejects_control_characters_and_windows_namespace_paths() {
        for control in (0u32..=0x9f)
            .filter_map(char::from_u32)
            .filter(|c| c.is_control())
        {
            let value = format!("meeting{control}id");
            assert!(validate_component(&value).is_err(), "accepted {value:?}");
        }
        for value in [
            r"\\?\C:\outside",
            r"\\.\NUL",
            r"\\?\UNC\server\share",
            r"C:relative",
            "original.wav:payload",
            "file::$DATA",
        ] {
            assert!(validate_component(value).is_err(), "accepted {value:?}");
        }
    }

    #[test]
    fn rejected_artifact_names_preserve_existing_artifacts_and_metadata() {
        let dir = temp_dir("artifact-names");
        meeting(&dir, "safe", "Preserved", 100, MeetingState::Completed);
        let sentinel = serde_json::json!({"sentinel": "keep"});
        write_artifact(&dir, "safe", "sentinel.json", &sentinel).unwrap();
        for name in [
            "../sentinel.json",
            r"..\sentinel.json",
            "sentinel.json:stream",
            "CON.json",
            "sentinel.json.",
            "sentinel.json ",
            "",
            r"\\server\share",
        ] {
            assert!(read_artifact::<serde_json::Value>(&dir, "safe", name).is_none());
            assert!(write_artifact(&dir, "safe", name, &"overwrite").is_err());
        }
        for id in ["../safe", r"..\safe", "safe:stream", "NUL"] {
            assert!(read_artifact::<serde_json::Value>(&dir, id, "sentinel.json").is_none());
            assert!(write_artifact(&dir, id, "sentinel.json", &"overwrite").is_err());
        }
        assert_eq!(
            read_artifact::<serde_json::Value>(&dir, "safe", "sentinel.json"),
            Some(sentinel)
        );
        assert_eq!(read_metadata(&dir, "safe").unwrap().title, "Preserved");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn safe_unicode_ids_and_artifact_names_round_trip_through_history() {
        let dir = temp_dir("unicode-paths");
        let value = serde_json::json!({"text": "தமிழ் meeting ✓"});
        for (index, id) in ["தமிழ்-கூட்டம்", "会議-123", "réunion", "meeting-🙂"]
            .iter()
            .enumerate()
        {
            meeting(&dir, id, id, index as i64, MeetingState::Completed);
            write_artifact(&dir, id, "குறிப்புகள்.json", &value).unwrap();
            assert_eq!(read_metadata(&dir, id).unwrap().id, *id);
            assert_eq!(
                read_artifact::<serde_json::Value>(&dir, id, "குறிப்புகள்.json"),
                Some(value.clone())
            );
        }
        assert_eq!(scan_meetings(&dir).len(), 4);
        assert_eq!(list_meetings(&dir, None, None, None).len(), 4);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    fn junction(link: &Path, target: &Path) {
        let output = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "junction creation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[cfg(windows)]
    #[test]
    fn rejects_junction_meetings_root_for_reads_writes_recovery_and_deletion() {
        let dir = temp_dir("junction-root");
        let target = temp_dir("junction-root-target");
        meeting(&target, "safe", "Outside", 100, MeetingState::Recording);
        write_artifact(&target, "safe", "sentinel.json", &"keep").unwrap();
        let link = meetings_dir(&dir);
        junction(&link, &meetings_dir(&target));
        assert!(read_metadata(&dir, "safe").is_err());
        assert!(write_metadata(&dir, &MeetingMetadata::new("safe", "Overwrite", 0)).is_err());
        assert!(read_artifact::<String>(&dir, "safe", "sentinel.json").is_none());
        assert!(write_artifact(&dir, "safe", "sentinel.json", &"overwrite").is_err());
        assert!(scan_meetings(&dir).is_empty());
        assert!(recover_interrupted(&dir).is_empty());
        assert!(delete_meeting(&dir, "safe").is_err());
        assert_eq!(
            read_metadata(&target, "safe").unwrap().state,
            MeetingState::Recording
        );
        assert_eq!(
            read_artifact::<String>(&target, "safe", "sentinel.json"),
            Some("keep".into())
        );
        std::fs::remove_dir(link).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(target).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn rejects_junction_artifact_directory_without_modifying_either_meeting() {
        let dir = temp_dir("junction-artifact");
        meeting(&dir, "safe", "Preserved", 100, MeetingState::Completed);
        let meeting_path = meeting_dir(&dir, "safe").unwrap();
        let metadata_before = std::fs::read(meeting_path.join("metadata.json")).unwrap();
        let target = dir.join("outside");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("sentinel.json"), "\"keep\"").unwrap();
        let link = meeting_path.join("exports");
        junction(&link, &target);
        assert!(read_metadata(&dir, "safe").is_err());
        assert!(update_metadata(&dir, "safe", |m| m.title = "Overwrite".into()).is_err());
        assert!(write_artifact(&dir, "safe", "sentinel.json", &"overwrite").is_err());
        assert!(read_artifact::<String>(&dir, "safe", "sentinel.json").is_none());
        assert!(delete_meeting(&dir, "safe").is_err());
        assert_eq!(
            std::fs::read(meeting_path.join("metadata.json")).unwrap(),
            metadata_before
        );
        assert_eq!(
            std::fs::read_to_string(target.join("sentinel.json")).unwrap(),
            "\"keep\""
        );
        assert!(!meeting_path.join("sentinel.json").exists());
        std::fs::remove_dir(link).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_traversal_and_windows_special_paths_without_touching_files() {
        let dir = temp_dir("unsafe-paths");
        let sentinel = dir.join("sentinel.txt");
        std::fs::write(&sentinel, "keep").unwrap();
        for id in [
            "",
            ".",
            "..",
            "../outside",
            "a/b",
            "a\\b",
            "C:\\outside",
            "\\\\server\\share",
            "id:stream",
            "id.",
            "id ",
            "CON",
            "nul.txt",
            "COM1",
            "LPT9.txt",
            "COM¹",
            "LPT².txt",
            "NUL .txt",
        ] {
            assert!(meeting_dir(&dir, id).is_err(), "accepted {id:?}");
            assert!(delete_meeting(&dir, id).is_err());
            assert!(write_metadata(&dir, &MeetingMetadata::new(id, "title", 0)).is_err());
        }
        assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "keep");
        for id in [
            "a",
            "b",
            "meeting-123",
            "550e8400-e29b-41d4-a716-446655440000",
        ] {
            assert!(meeting_dir(&dir, id).is_ok());
        }
        assert!(write_artifact(&dir, "a", "../../sentinel.txt", &"overwrite").is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_metadata_redirects_during_reads_and_history_rebuild() {
        let dir = temp_dir("metadata-redirect");
        meeting(&dir, "a", "Safe", 0, MeetingState::Completed);
        let path = meeting_dir(&dir, "a").unwrap().join("metadata.json");
        let forged = MeetingMetadata::new("../outside", "Forged", 0);
        std::fs::write(path, serde_json::to_string(&forged).unwrap()).unwrap();
        assert!(read_metadata(&dir, "a").is_err());
        assert!(update_metadata(&dir, "a", |m| m.title = "Changed".into()).is_err());
        assert!(scan_meetings(&dir).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(windows)]
    #[test]
    fn rejects_windows_junctions_without_deleting_the_target() {
        let dir = temp_dir("junction");
        let outside = dir.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("keep.txt"), "keep").unwrap();
        std::fs::create_dir_all(meetings_dir(&dir)).unwrap();
        let link = meetings_dir(&dir).join("a");
        let status = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(&link)
            .arg(&outside)
            .status()
            .unwrap();
        assert!(status.success());
        assert!(meeting_dir(&dir, "a").is_err());
        assert!(delete_meeting(&dir, "a").is_err());
        assert_eq!(
            std::fs::read_to_string(outside.join("keep.txt")).unwrap(),
            "keep"
        );
        std::fs::remove_dir(link).unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cadence-storage-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn meeting(dir: &Path, id: &str, title: &str, created: i64, state: MeetingState) {
        let mut meta = MeetingMetadata::new(id, title, created);
        meta.state = state;
        write_metadata(dir, &meta).unwrap();
    }

    #[test]
    fn writes_and_reads_metadata() {
        let dir = temp_dir("rw");
        meeting(&dir, "a", "Standup", 1_000, MeetingState::Completed);
        let read = read_metadata(&dir, "a").unwrap();
        assert_eq!(read.title, "Standup");
        assert_eq!(read.state, MeetingState::Completed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lists_meetings_newest_first_and_filters() {
        let dir = temp_dir("list");
        meeting(
            &dir,
            "a",
            "Architecture review",
            1_000,
            MeetingState::Completed,
        );
        meeting(&dir, "b", "Standup", 5_000, MeetingState::Completed);
        meeting(&dir, "c", "Retro", 3_000, MeetingState::Failed);

        let all = list_meetings(&dir, None, None, None);
        assert_eq!(
            all.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["b", "c", "a"]
        );

        let filtered = list_meetings(&dir, Some("arch"), None, None);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "a");

        let ranged = list_meetings(&dir, None, Some(2_000), Some(4_000));
        assert_eq!(ranged.len(), 1);
        assert_eq!(ranged[0].id, "c");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebuilds_the_index_when_the_database_is_deleted() {
        let dir = temp_dir("rebuild");
        meeting(&dir, "a", "Standup", 1_000, MeetingState::Completed);
        std::fs::remove_file(database_path(&dir)).unwrap();
        let all = list_meetings(&dir, None, None, None);
        assert_eq!(all.len(), 1, "history must survive losing the index");
        // ...and the index is back.
        assert_eq!(index_list(&dir).unwrap(), vec!["a".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_removes_both_the_files_and_the_index_row() {
        let dir = temp_dir("delete");
        meeting(&dir, "a", "Standup", 1_000, MeetingState::Completed);
        delete_meeting(&dir, "a").unwrap();
        assert!(!meeting_dir(&dir, "a").unwrap().exists());
        assert!(index_list(&dir).unwrap().is_empty());
        assert!(list_meetings(&dir, None, None, None).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_interrupted_meeting_is_recovered_as_failed() {
        let dir = temp_dir("recover");
        meeting(&dir, "a", "Interrupted", 1_000, MeetingState::Transcribing);
        meeting(&dir, "b", "Fine", 2_000, MeetingState::Completed);

        let recovered = recover_interrupted(&dir);
        assert_eq!(recovered, vec!["a".to_string()]);
        let a = read_metadata(&dir, "a").unwrap();
        assert_eq!(a.state, MeetingState::Failed);
        assert!(a.error.unwrap().contains("retry"));
        assert_eq!(
            read_metadata(&dir, "b").unwrap().state,
            MeetingState::Completed
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_recording_interrupted_by_a_crash_keeps_its_audio_note() {
        let dir = temp_dir("recover-recording");
        meeting(&dir, "a", "Cut short", 1_000, MeetingState::Recording);
        recover_interrupted(&dir);
        let a = read_metadata(&dir, "a").unwrap();
        assert_eq!(a.state, MeetingState::Failed);
        assert!(a.error.unwrap().contains("kept"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn updating_refreshes_the_artifact_flags() {
        let dir = temp_dir("flags");
        meeting(&dir, "a", "Standup", 1_000, MeetingState::Recorded);
        std::fs::write(
            meeting_dir(&dir, "a").unwrap().join("transcript.json"),
            "{}",
        )
        .unwrap();
        let meta = update_metadata(&dir, "a", |m| m.state = MeetingState::Completed).unwrap();
        assert!(meta.has_transcript);
        assert!(!meta.has_mom);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn processing_options_round_trip_and_default_to_automatic() {
        let dir = temp_dir("processing-options");
        meeting(&dir, "a", "Standup", 1_000, MeetingState::Recorded);
        // Never saved: automatic clustering.
        assert_eq!(read_processing_options(&dir, "a"), ProcessingOptions::default());

        let options = ProcessingOptions {
            expected_speakers: Some(4),
        };
        write_processing_options(&dir, "a", &options).unwrap();
        assert_eq!(read_processing_options(&dir, "a"), options);

        // Back to "Not sure" is a real write, not a no-op.
        write_processing_options(&dir, "a", &ProcessingOptions::default()).unwrap();
        assert_eq!(read_processing_options(&dir, "a").expected_speakers, None);

        // The metadata is untouched by any of it.
        assert_eq!(read_metadata(&dir, "a").unwrap().title, "Standup");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_out_of_range_or_corrupt_speaker_hint_on_disk_reads_as_not_sure() {
        let dir = temp_dir("processing-options-bad");
        meeting(&dir, "a", "Standup", 1_000, MeetingState::Recorded);
        let path = meeting_dir(&dir, "a").unwrap().join("processing_options.json");
        for raw in [
            r#"{"expected_speakers":0}"#,
            // Above the small-room cap: automatic, never a fixed count.
            r#"{"expected_speakers":5}"#,
            r#"{"expected_speakers":20}"#,
            "not json",
        ] {
            std::fs::write(&path, raw).unwrap();
            assert_eq!(read_processing_options(&dir, "a").expected_speakers, None, "{raw}");
        }
        // Unknown fields (a newer build) do not cost the hint.
        std::fs::write(&path, r#"{"expected_speakers":3,"future":true}"#).unwrap();
        assert_eq!(read_processing_options(&dir, "a").expected_speakers, Some(3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_the_duration_of_a_wav() {
        let dir = temp_dir("wav");
        let path = dir.join("a.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..32_000 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        assert_eq!(wav_duration_ms(&path).unwrap(), 2_000);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
