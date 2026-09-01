use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// ISO code of the language whisper detected for the block this segment
    /// came from (§8). Only code-switched mode detects per block, so it is
    /// `None` in every other mode — and omitted from `stt.json` entirely, so
    /// a file written before this existed reads back identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
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
    /// Detected language for this turn (§8): an ISO code, or [`MIXED_LANGUAGE`]
    /// when the turn was assembled from fragments whisper detected differently.
    /// `None` outside code-switched mode, and skipped when serializing so an
    /// older `transcript.json` round-trips unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Marks a turn that spans more than one detected language — the thing §8
/// exists for. Not an ISO code, and deliberately not one: no ISO 639 code is
/// `"mixed"`, so nothing downstream can confuse the two.
pub const MIXED_LANGUAGE: &str = "mixed";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub meeting_id: String,
    pub segments: Vec<TranscriptSegment>,
}

impl Transcript {
    pub fn duration_ms(&self) -> i64 {
        self.segments.last().map(|s| s.end_ms).unwrap_or(0)
    }

    /// Presentation labels in first-appearance order — the grounded
    /// participant list (§16: participants are never invented by the LLM).
    pub fn speaker_labels(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for segment in &self.segments {
            if !seen.contains(&segment.speaker_label) {
                seen.push(segment.speaker_label.clone());
            }
        }
        seen
    }
}

// ------------------------------------------------------------------- MoM

/// A generated statement plus the transcript evidence behind it (§15: "every
/// important generated item should reference supporting transcript timestamps
/// when possible"). `timestamp` is milliseconds from the start of the meeting,
/// matching every other timestamp in the app.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Evidenced {
    pub text: String,
    pub timestamp: Option<i64>,
    pub speaker: Option<String>,
}

impl Evidenced {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            timestamp: None,
            speaker: None,
        }
    }
}

/// Accepts either the full object or a bare string, because a local model
/// asked for `{"text": ..., "timestamp": ...}` will sometimes return just the
/// sentence. Losing the evidence is acceptable; failing the whole MoM over it
/// is not.
impl<'de> Deserialize<'de> for Evidenced {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(text) => Ok(Evidenced::new(text)),
            serde_json::Value::Object(map) => {
                let text = map
                    .get("text")
                    .or_else(|| map.get("description"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                Ok(Evidenced {
                    text,
                    timestamp: map.get("timestamp").and_then(number_to_i64),
                    speaker: map
                        .get("speaker")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                })
            }
            other => Err(de::Error::custom(format!(
                "expected a string or object, got {other}"
            ))),
        }
    }
}

/// Models return timestamps as `12`, `12.5`, or `"12"` depending on the day.
fn number_to_i64(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        serde_json::Value::String(s) => s
            .trim()
            .parse::<i64>()
            .ok()
            .or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64)),
        _ => None,
    }
}

/// Matches the master prompt §15 action-item shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionItem {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub deadline: Option<String>,
    #[serde(default, deserialize_with = "lenient_timestamp")]
    pub timestamp: Option<i64>,
    /// "high" | "medium" | "low" — deliberately lenient at parse time
    /// (defaults to empty rather than failing deserialization); `validate()`
    /// below is what actually enforces the allowed values and triggers the
    /// repair-JSON retry (§16/§17) on a malformed LLM response.
    #[serde(default)]
    pub confidence: String,
}

fn lenient_timestamp<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i64>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(number_to_i64(&value))
}

/// Provenance for a generated MoM (§17): which model, which quantization,
/// which prompt versions, and when.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MomMeta {
    pub model: String,
    pub quantization: String,
    pub runtime: String,
    pub context_tokens: usize,
    pub extraction_prompt: String,
    pub synthesis_prompt: String,
    pub language: String,
    pub chunk_count: usize,
    pub generated_at_ms: i64,
    /// True when the model's first answer failed validation and the
    /// repair-JSON prompt was needed.
    pub repaired: bool,
    pub duration_ms: i64,
}

/// Matches the master prompt §15 MoM schema, with evidence attached (§15/§16).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Mom {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub participants: Vec<String>,
    #[serde(default)]
    pub topics: Vec<Evidenced>,
    #[serde(default)]
    pub decisions: Vec<Evidenced>,
    #[serde(default, rename = "actionItems")]
    pub action_items: Vec<ActionItem>,
    #[serde(default, rename = "openQuestions")]
    pub open_questions: Vec<Evidenced>,
    #[serde(default)]
    pub risks: Vec<Evidenced>,
    #[serde(default, rename = "detailedDiscussion")]
    pub detailed_discussion: Vec<Evidenced>,
    #[serde(default)]
    pub meta: Option<MomMeta>,
}

impl Mom {
    /// Structural validation only (§16): catches malformed shape so the
    /// caller can retry through the repair-JSON prompt. It cannot verify the
    /// LLM didn't hallucinate a fact — that's a prompting and grounding
    /// concern, handled in `mom.rs` — only that the response is well-formed
    /// enough to trust downstream.
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
        for (label, list) in [
            ("topic", &self.topics),
            ("decision", &self.decisions),
            ("open question", &self.open_questions),
            ("risk", &self.risks),
        ] {
            if list.iter().any(|e| e.text.trim().is_empty()) {
                return Err(format!("a {label} entry has no text"));
            }
        }
        Ok(())
    }

    /// Shape shown to the model in the repair prompt (§17).
    pub fn schema_hint() -> &'static str {
        r#"{
  "summary": "string, non-empty",
  "participants": ["Person 1"],
  "topics": [{"text": "string", "timestamp": 0}],
  "decisions": [{"text": "string", "timestamp": 0, "speaker": "Person 1"}],
  "actionItems": [{"description": "string", "owner": "Person 1 or null", "deadline": "string or null", "timestamp": 0, "confidence": "high|medium|low"}],
  "openQuestions": [{"text": "string", "timestamp": 0, "speaker": "Person 1"}],
  "risks": [{"text": "string", "timestamp": 0}],
  "detailedDiscussion": [{"text": "string", "timestamp": 0}]
}"#
    }
}

/// What one chunk's extraction pass returns (§14).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkExtraction {
    #[serde(default)]
    pub topics: Vec<Evidenced>,
    #[serde(default)]
    pub decisions: Vec<Evidenced>,
    #[serde(default, rename = "actionItems")]
    pub action_items: Vec<ActionItem>,
    #[serde(default, rename = "openQuestions")]
    pub open_questions: Vec<Evidenced>,
    #[serde(default)]
    pub risks: Vec<Evidenced>,
    #[serde(default)]
    pub notes: Vec<Evidenced>,
}

impl ChunkExtraction {
    pub fn is_empty(&self) -> bool {
        self.topics.is_empty()
            && self.decisions.is_empty()
            && self.action_items.is_empty()
            && self.open_questions.is_empty()
            && self.risks.is_empty()
            && self.notes.is_empty()
    }
}

// --------------------------------------------------------- meeting state

/// The §12 meeting state machine. Persisted in `metadata.json`, so a meeting
/// that was mid-processing when the app closed can be recognised on restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MeetingState {
    Idle,
    Recording,
    Recorded,
    Preprocessing,
    Diarizing,
    Transcribing,
    Aligning,
    Extracting,
    Synthesizing,
    Completed,
    Failed,
}

impl MeetingState {
    pub fn as_str(self) -> &'static str {
        match self {
            MeetingState::Idle => "IDLE",
            MeetingState::Recording => "RECORDING",
            MeetingState::Recorded => "RECORDED",
            MeetingState::Preprocessing => "PREPROCESSING",
            MeetingState::Diarizing => "DIARIZING",
            MeetingState::Transcribing => "TRANSCRIBING",
            MeetingState::Aligning => "ALIGNING",
            MeetingState::Extracting => "EXTRACTING",
            MeetingState::Synthesizing => "SYNTHESIZING",
            MeetingState::Completed => "COMPLETED",
            MeetingState::Failed => "FAILED",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value.trim().to_ascii_uppercase().as_str() {
            "IDLE" => MeetingState::Idle,
            "RECORDING" => MeetingState::Recording,
            "RECORDED" => MeetingState::Recorded,
            "PREPROCESSING" => MeetingState::Preprocessing,
            "DIARIZING" => MeetingState::Diarizing,
            "TRANSCRIBING" => MeetingState::Transcribing,
            "ALIGNING" => MeetingState::Aligning,
            "EXTRACTING" => MeetingState::Extracting,
            "SYNTHESIZING" => MeetingState::Synthesizing,
            "COMPLETED" => MeetingState::Completed,
            "FAILED" => MeetingState::Failed,
            _ => return None,
        })
    }

    /// True while the pipeline should be running. A meeting found in one of
    /// these states at startup was interrupted (crash, power loss, app close)
    /// and must be surfaced as recoverable rather than left looking active.
    pub fn is_processing(self) -> bool {
        matches!(
            self,
            MeetingState::Preprocessing
                | MeetingState::Diarizing
                | MeetingState::Transcribing
                | MeetingState::Aligning
                | MeetingState::Extracting
                | MeetingState::Synthesizing
        )
    }

    /// Rough completion percentage, for the progress UI. The weights follow
    /// measured stage costs (docs/performance.md): transcription and MoM
    /// generation dominate.
    pub fn progress(self) -> i64 {
        match self {
            MeetingState::Idle | MeetingState::Recording => 0,
            MeetingState::Recorded => 2,
            MeetingState::Preprocessing => 5,
            MeetingState::Diarizing => 15,
            MeetingState::Transcribing => 35,
            MeetingState::Aligning => 60,
            MeetingState::Extracting => 65,
            MeetingState::Synthesizing => 90,
            MeetingState::Completed => 100,
            MeetingState::Failed => 0,
        }
    }
}

/// Everything about a meeting that is not audio, transcript, or minutes.
/// Written to `<meeting>/metadata.json` and mirrored into SQLite for history
/// (§21).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingMetadata {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub notes: String,
    pub created_at_ms: i64,
    #[serde(default)]
    pub duration_ms: i64,
    pub state: MeetingState,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub device_name: Option<String>,
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub channels: u16,
    /// User-assigned display names per stable speaker id (§23). Presentation
    /// metadata only — the raw diarization output is never rewritten.
    #[serde(default)]
    pub speaker_names: BTreeMap<String, String>,
    #[serde(default)]
    pub has_transcript: bool,
    #[serde(default)]
    pub has_mom: bool,
    /// Set when the transcript succeeded but minutes generation did not. The
    /// meeting is complete and usable; this is what the UI shows next to the
    /// "Generate minutes" retry (§33).
    #[serde(default)]
    pub mom_error: Option<String>,
}

impl MeetingMetadata {
    pub fn new(id: &str, title: &str, created_at_ms: i64) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            notes: String::new(),
            created_at_ms,
            duration_ms: 0,
            state: MeetingState::Recording,
            error: None,
            device_name: None,
            sample_rate: 0,
            channels: 0,
            speaker_names: BTreeMap::new(),
            has_transcript: false,
            has_mom: false,
            mom_error: None,
        }
    }

    /// Falls back to a date-stamped title so history never shows a blank row.
    pub fn display_title(&self) -> String {
        if self.title.trim().is_empty() {
            "Untitled meeting".to_string()
        } else {
            self.title.clone()
        }
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
    /// A non-fatal problem: the meeting finished, but something in it did not
    /// (in practice, minutes generation failing after a good transcript).
    #[serde(default)]
    pub warning: Option<String>,
}

impl MeetingStatus {
    pub fn processing(stage: &str, progress: i64) -> Self {
        Self {
            status: "processing".to_string(),
            stage: Some(stage.to_string()),
            progress,
            error: None,
            warning: None,
        }
    }

    pub fn idle() -> Self {
        Self {
            status: "idle".to_string(),
            stage: None,
            progress: 0,
            error: None,
            warning: None,
        }
    }

    pub fn completed(warning: Option<String>) -> Self {
        Self {
            status: "completed".to_string(),
            stage: Some("COMPLETED".to_string()),
            progress: 100,
            error: None,
            warning,
        }
    }

    pub fn failed(stage: &str, error: String) -> Self {
        Self {
            status: "failed".to_string(),
            stage: Some(stage.to_string()),
            progress: 0,
            error: Some(error),
            warning: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_an_evidenced_item_as_a_bare_string() {
        let parsed: Vec<Evidenced> = serde_json::from_str(r#"["just a sentence"]"#).unwrap();
        assert_eq!(parsed[0].text, "just a sentence");
        assert_eq!(parsed[0].timestamp, None);
    }

    #[test]
    fn accepts_an_evidenced_item_as_an_object() {
        let parsed: Evidenced =
            serde_json::from_str(r#"{"text":"a decision","timestamp":42,"speaker":"Person 2"}"#)
                .unwrap();
        assert_eq!(parsed.timestamp, Some(42));
        assert_eq!(parsed.speaker.as_deref(), Some("Person 2"));
    }

    #[test]
    fn tolerates_a_float_or_string_timestamp() {
        let parsed: Evidenced = serde_json::from_str(r#"{"text":"x","timestamp":42.9}"#).unwrap();
        assert_eq!(parsed.timestamp, Some(42));
        let parsed: Evidenced = serde_json::from_str(r#"{"text":"x","timestamp":"58"}"#).unwrap();
        assert_eq!(parsed.timestamp, Some(58));
        let parsed: Evidenced = serde_json::from_str(r#"{"text":"x","timestamp":null}"#).unwrap();
        assert_eq!(parsed.timestamp, None);
    }

    #[test]
    fn parses_a_whole_mom_the_way_a_model_would_write_it() {
        let raw = r#"{
            "summary": "We agreed the release slips a week.",
            "participants": ["Person 1", "Person 2"],
            "topics": ["release date"],
            "decisions": [{"text": "slip the release", "timestamp": 120000}],
            "actionItems": [{"description": "update the plan", "owner": null, "deadline": null, "timestamp": 130000, "confidence": "high"}],
            "openQuestions": [],
            "risks": [],
            "detailedDiscussion": []
        }"#;
        let mom: Mom = serde_json::from_str(raw).unwrap();
        assert!(mom.validate().is_ok());
        assert_eq!(mom.topics[0].text, "release date");
        assert_eq!(mom.action_items[0].timestamp, Some(130000));
    }

    #[test]
    fn rejects_a_mom_with_an_empty_summary() {
        let mom = Mom::default();
        assert!(mom.validate().is_err());
    }

    #[test]
    fn rejects_an_invented_confidence_value() {
        let mut mom = Mom {
            summary: "something happened".to_string(),
            ..Mom::default()
        };
        mom.action_items.push(ActionItem {
            description: "do it".to_string(),
            owner: None,
            deadline: None,
            timestamp: None,
            confidence: "very high".to_string(),
        });
        assert!(mom.validate().is_err());
    }

    #[test]
    fn meeting_states_round_trip_and_order_by_progress() {
        let states = [
            MeetingState::Preprocessing,
            MeetingState::Diarizing,
            MeetingState::Transcribing,
            MeetingState::Aligning,
            MeetingState::Extracting,
            MeetingState::Synthesizing,
            MeetingState::Completed,
        ];
        for pair in states.windows(2) {
            assert!(
                pair[0].progress() < pair[1].progress(),
                "{:?} should precede {:?}",
                pair[0],
                pair[1]
            );
        }
        for state in states {
            assert_eq!(MeetingState::parse(state.as_str()), Some(state));
        }
        assert_eq!(MeetingState::parse("nonsense"), None);
    }

    #[test]
    fn only_the_pipeline_stages_count_as_processing() {
        assert!(MeetingState::Transcribing.is_processing());
        assert!(MeetingState::Synthesizing.is_processing());
        assert!(!MeetingState::Recorded.is_processing());
        assert!(!MeetingState::Completed.is_processing());
        assert!(!MeetingState::Failed.is_processing());
    }

    #[test]
    fn metadata_round_trips_through_json() {
        let mut meta = MeetingMetadata::new("abc", "", 1_700_000_000_000);
        meta.speaker_names
            .insert("SPEAKER_00".to_string(), "Priya".to_string());
        let json = serde_json::to_string(&meta).unwrap();
        let back: MeetingMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.display_title(), "Untitled meeting");
        assert_eq!(back.state, MeetingState::Recording);
        assert_eq!(back.speaker_names.get("SPEAKER_00").unwrap(), "Priya");
    }

    #[test]
    fn a_transcript_written_before_language_tags_existed_still_loads() {
        // Exactly the shape `transcript.json` had before §8's per-block
        // detection: a meeting on disk must not become unreadable.
        let raw = r#"{
            "meeting_id": "abc",
            "segments": [
                {"speaker": "SPEAKER_00", "speaker_label": "Person 1",
                 "start_ms": 0, "end_ms": 2000, "text": "hello"}
            ]
        }"#;
        let transcript: Transcript = serde_json::from_str(raw).unwrap();
        assert_eq!(transcript.segments[0].language, None);
        // ...and it round-trips back out without gaining an empty field.
        let out = serde_json::to_string(&transcript).unwrap();
        assert!(!out.contains("language"), "{out}");
    }

    #[test]
    fn an_stt_file_written_before_language_tags_existed_still_loads() {
        let raw = r#"[{"start_ms": 0, "end_ms": 2000, "text": "hello"}]"#;
        let segments: Vec<SttSegment> = serde_json::from_str(raw).unwrap();
        assert_eq!(segments[0].language, None);
        assert!(segments[0].words.is_empty());
        assert!(!serde_json::to_string(&segments).unwrap().contains("language"));
    }

    #[test]
    fn a_language_tag_survives_a_round_trip() {
        let raw = r#"[{"start_ms": 0, "end_ms": 2000, "text": "வணக்கம்", "language": "ta"}]"#;
        let segments: Vec<SttSegment> = serde_json::from_str(raw).unwrap();
        assert_eq!(segments[0].language.as_deref(), Some("ta"));
        assert!(serde_json::to_string(&segments)
            .unwrap()
            .contains("\"language\":\"ta\""));
    }

    #[test]
    fn transcript_reports_its_speakers_in_first_appearance_order() {
        let transcript = Transcript {
            meeting_id: "m".to_string(),
            segments: vec![
                TranscriptSegment {
                    speaker: "SPEAKER_01".to_string(),
                    speaker_label: "Person 1".to_string(),
                    start_ms: 0,
                    end_ms: 1000,
                    text: "hello".to_string(),
                    language: None,
                },
                TranscriptSegment {
                    speaker: "SPEAKER_00".to_string(),
                    speaker_label: "Person 2".to_string(),
                    start_ms: 1000,
                    end_ms: 2500,
                    text: "hi".to_string(),
                    language: None,
                },
                TranscriptSegment {
                    speaker: "SPEAKER_01".to_string(),
                    speaker_label: "Person 1".to_string(),
                    start_ms: 2500,
                    end_ms: 3000,
                    text: "again".to_string(),
                    language: None,
                },
            ],
        };
        assert_eq!(transcript.speaker_labels(), vec!["Person 1", "Person 2"]);
        assert_eq!(transcript.duration_ms(), 3000);
    }
}
