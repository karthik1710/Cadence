use serde::{Deserialize, Serialize};

/// Word-level timestamp from whisper.cpp (when `--split-on-word` output is available).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttWord {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    #[serde(default)]
    pub words: Vec<SttWord>,
}

/// Raw diarization interval — `speaker_id` is sherpa-onnx's own numeric label
/// ("0", "1", ...), not yet remapped to a stable `SPEAKER_00` id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegment {
    pub speaker_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// One turn in the final speaker-labelled transcript (master prompt §13).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Stable internal id, e.g. "SPEAKER_00" — never renamed.
    pub speaker: String,
    /// Presentation label, e.g. "Person 1" — first-appearance order (§7).
    pub speaker_label: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub meeting_id: String,
    pub segments: Vec<TranscriptSegment>,
}

/// Matches the master prompt §15 action-item shape exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub deadline: Option<String>,
    #[serde(default)]
    pub timestamp: Option<i64>,
    /// "high" | "medium" | "low" — deliberately lenient at parse time
    /// (defaults to empty rather than failing deserialization); `validate()`
    /// below is what actually enforces the allowed values and triggers the
    /// repair-JSON retry (§16/§17) on a malformed LLM response.
    #[serde(default)]
    pub confidence: String,
}

/// Matches the master prompt §15 MoM schema exactly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Mom {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub participants: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub decisions: Vec<String>,
    #[serde(default, rename = "actionItems")]
    pub action_items: Vec<ActionItem>,
    #[serde(default, rename = "openQuestions")]
    pub open_questions: Vec<String>,
    #[serde(default)]
    pub risks: Vec<String>,
}

impl Mom {
    /// Structural validation only (§16): catches malformed shape so the
    /// caller can retry through the repair-JSON prompt. It cannot verify the
    /// LLM didn't hallucinate a fact — that's a prompting concern, not a
    /// code concern — only that the response is well-formed enough to trust
    /// downstream (non-empty summary, confidence is one of the three allowed
    /// values).
    pub fn validate(&self) -> Result<(), String> {
        if self.summary.trim().is_empty() {
            return Err("MoM summary is empty".to_string());
        }
        for item in &self.action_items {
            if item.description.trim().is_empty() {
                return Err("action item is missing a description".to_string());
            }
            if !matches!(item.confidence.as_str(), "high" | "medium" | "low") {
                return Err(format!(
                    "action item has invalid confidence value: {:?}",
                    item.confidence
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingStatus {
    /// "idle" | "processing" | "completed" | "failed"
    pub status: String,
    /// One of the §12 state-machine stages while processing.
    pub stage: Option<String>,
    pub progress: i64,
    pub error: Option<String>,
}

impl MeetingStatus {
    pub fn processing(stage: &str, progress: i64) -> Self {
        Self {
            status: "processing".to_string(),
            stage: Some(stage.to_string()),
            progress,
            error: None,
        }
    }

    pub fn completed() -> Self {
        Self {
            status: "completed".to_string(),
            stage: Some("COMPLETED".to_string()),
            progress: 100,
            error: None,
        }
    }

    pub fn failed(stage: &str, error: String) -> Self {
        Self {
            status: "failed".to_string(),
            stage: Some(stage.to_string()),
            progress: 0,
            error: Some(error),
        }
    }
}
