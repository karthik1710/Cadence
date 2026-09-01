//! Local meeting storage and the history index (master prompt §21, §25).
//!
//! ```text
//! <app_data>/
//! ├── config/settings.json
//! ├── models/
//! ├── database/meetings.db      SQLite index for history: metadata only
//! ├── logs/
//! └── meetings/<id>/
//!     ├── metadata.json         authoritative record for this meeting
//!     ├── original.wav          never modified, never deleted automatically
//!     ├── processed.wav
//!     ├── diarization.json      raw stage output
//!     ├── stt.json              raw stage output
//!     ├── transcript.json       derived
//!     ├── mom.json              derived
//!     └── exports/
//! ```
//!
//! `metadata.json` on disk is the source of truth; SQLite is a rebuildable
//! index over it, holding only what the history list needs to sort and filter
//! (§21: "do not duplicate full transcript data unnecessarily"). If the
//! database is lost or corrupted, [`list_meetings`] rebuilds it by scanning the
//! meetings directory — no meeting can be lost to a broken index.

use crate::pipeline::models::{MeetingMetadata, MeetingState, Mom, Transcript};
use rusqlite::Connection;
use std::path::{Path, PathBuf};

pub fn meetings_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("meetings")
}

pub fn meeting_dir(data_dir: &Path, meeting_id: &str) -> PathBuf {
    meetings_dir(data_dir).join(meeting_id)
}

fn metadata_path(data_dir: &Path, meeting_id: &str) -> PathBuf {
    meeting_dir(data_dir, meeting_id).join("metadata.json")
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
    let path = metadata_path(data_dir, meeting_id);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("could not parse {}: {e}", path.display()))
}

/// Writes `metadata.json` and mirrors the row into the index. An index write
/// that fails is logged, not propagated: the meeting itself is already safe on
/// disk, and history can always be rebuilt.
pub fn write_metadata(data_dir: &Path, meta: &MeetingMetadata) -> Result<(), String> {
    let path = metadata_path(data_dir, &meta.id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display()))?;

    if let Err(err) = index_upsert(data_dir, meta) {
        eprintln!("[storage] history index update failed for {}: {err}", meta.id);
    }
    Ok(())
}

/// Applies `edit` to a meeting's metadata and saves it. Used for every state
/// transition, so §12's state machine has exactly one write path.
pub fn update_metadata<F: FnOnce(&mut MeetingMetadata)>(
    data_dir: &Path,
    meeting_id: &str,
    edit: F,
) -> Result<MeetingMetadata, String> {
    let mut meta = read_metadata(data_dir, meeting_id)?;
    edit(&mut meta);
    meta.has_transcript = meeting_dir(data_dir, meeting_id)
        .join("transcript.json")
        .exists();
    meta.has_mom = meeting_dir(data_dir, meeting_id).join("mom.json").exists();
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
    let path = meeting_dir(data_dir, meeting_id).join("transcript.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("could not parse transcript.json: {e}"))
}

pub fn read_mom(data_dir: &Path, meeting_id: &str) -> Result<Mom, String> {
    let path = meeting_dir(data_dir, meeting_id).join("mom.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("could not parse mom.json: {e}"))
}

/// Duration of a 16-bit PCM WAV, in milliseconds.
pub fn wav_duration_ms(path: &Path) -> Result<i64, String> {
    let reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let frames = reader.len() as i64 / spec.channels.max(1) as i64;
    Ok(frames * 1000 / spec.sample_rate.max(1) as i64)
}

/// Deletes a meeting and everything in it. The only path that ever removes a
/// recording, and only ever on an explicit user action (§10).
pub fn delete_meeting(data_dir: &Path, meeting_id: &str) -> Result<(), String> {
    let dir = meeting_dir(data_dir, meeting_id);
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
/// they show up as recoverable instead of permanently "in progress" (§12/§33).
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
/// created-at range (§25).
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
    let needle = query.map(|q| q.trim().to_lowercase()).filter(|q| !q.is_empty());

    all.into_iter()
        .filter(|meta| match &needle {
            Some(q) => {
                meta.title.to_lowercase().contains(q) || meta.notes.to_lowercase().contains(q)
            }
            None => true,
        })
        .filter(|meta| from_ms.map(|from| meta.created_at_ms >= from).unwrap_or(true))
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

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("echomind-storage-{name}"));
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
        meeting(&dir, "a", "Architecture review", 1_000, MeetingState::Completed);
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
        assert!(!meeting_dir(&dir, "a").exists());
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
        assert_eq!(read_metadata(&dir, "b").unwrap().state, MeetingState::Completed);
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
        std::fs::write(meeting_dir(&dir, "a").join("transcript.json"), "{}").unwrap();
        let meta = update_metadata(&dir, "a", |m| m.state = MeetingState::Completed).unwrap();
        assert!(meta.has_transcript);
        assert!(!meta.has_mom);
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
