use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone)]
pub struct MeetingStatus {
    pub status: String,
    pub stage: Option<String>,
    pub progress: i64,
    pub error: Option<String>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct TranscriptResponse {
    pub id: String,
    pub segments: Vec<TranscriptSegment>,
}

/// Creates a meeting on the (shared — see meeting-ai-rpi5-PLAN.md §1) Pi
/// backend, uploads the local recording, and kicks off processing. Returns
/// the backend-assigned meeting id, which is authoritative — the caller's
/// local meeting id is a separate concern (multi-client ID safety, see
/// meeting-ai-app-PLAN.md Phase 1 note).
#[tauri::command]
pub fn upload_and_process(
    backend_url: String,
    file_path: String,
    title: Option<String>,
) -> Result<String, String> {
    let client = reqwest::blocking::Client::new();

    let create_resp: serde_json::Value = client
        .post(format!("{backend_url}/api/v1/meetings"))
        .json(&serde_json::json!({ "title": title }))
        .send()
        .map_err(|e| format!("Could not reach backend: {e}"))?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;

    let meeting_id = create_resp["id"]
        .as_str()
        .ok_or_else(|| "Backend did not return a meeting id".to_string())?
        .to_string();

    let form = reqwest::blocking::multipart::Form::new()
        .file("file", &file_path)
        .map_err(|e| format!("Could not read recording file: {e}"))?;

    client
        .post(format!("{backend_url}/api/v1/meetings/{meeting_id}/audio"))
        .multipart(form)
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;

    client
        .post(format!("{backend_url}/api/v1/meetings/{meeting_id}/process"))
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;

    Ok(meeting_id)
}

#[tauri::command]
pub fn get_meeting_status(backend_url: String, meeting_id: String) -> Result<MeetingStatus, String> {
    let client = reqwest::blocking::Client::new();
    client
        .get(format!("{backend_url}/api/v1/meetings/{meeting_id}/status"))
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_transcript(backend_url: String, meeting_id: String) -> Result<TranscriptResponse, String> {
    let client = reqwest::blocking::Client::new();
    client
        .get(format!("{backend_url}/api/v1/meetings/{meeting_id}/transcript"))
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())
}
