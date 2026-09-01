//! Minutes-of-meeting generation (master prompt §14-§18).
//!
//! ```text
//! transcript → speaker-aware chunks → per-chunk extraction →
//! merged facts → (condense if needed) → synthesis → validate/repair →
//! grounding → mom.json
//! ```
//!
//! The design point that matters: **nothing the model writes is trusted as-is.**
//! Its JSON is validated against the schema and repaired if malformed (§16),
//! and then [`ground`] rewrites the parts it is not allowed to invent —
//! participants come from the transcript, timestamps outside the meeting are
//! dropped rather than kept, and an owner who never spoke becomes `null`.

use crate::pipeline::chunk::{self, Chunk};
use crate::pipeline::llm::{extract_json, LlmProvider};
use crate::pipeline::models::{ActionItem, ChunkExtraction, Evidenced, Mom, MomMeta, Transcript};
use crate::pipeline::prompts;
use std::path::Path;

const EXTRACTION_SYSTEM: &str =
    "You extract structured facts from meeting transcripts. You never invent information. You reply with JSON only.";
const SYNTHESIS_SYSTEM: &str =
    "You write meeting minutes from already-extracted facts. You never add facts that are not given to you. You reply with JSON only.";
const REPAIR_SYSTEM: &str =
    "You fix malformed JSON without changing its meaning. You reply with JSON only.";

/// Overlapping one turn at each chunk boundary is enough to keep a decision
/// that straddles the boundary visible on both sides, without paying for it
/// twice in every chunk.
const CHUNK_OVERLAP_TURNS: usize = 1;

/// How far a model-supplied timestamp may sit past the end of the meeting
/// before it is treated as invented. A few seconds of slack covers rounding
/// against the last turn's end.
const TIMESTAMP_SLACK_MS: i64 = 5_000;

pub struct MomRequest<'a> {
    pub transcript: &'a Transcript,
    pub title: &'a str,
    pub language: &'a str,
    /// App data dir, for prompt overrides. `None` uses the embedded prompts.
    pub data_dir: Option<&'a Path>,
}

/// Runs the full MoM pipeline. `on_progress` receives a §12 stage name and a
/// rough percentage.
pub fn generate(
    llm: &dyn LlmProvider,
    request: &MomRequest,
    on_progress: &dyn Fn(&str, i64),
) -> Result<Mom, String> {
    let started_ms = now_ms();
    let descriptor = llm.describe();
    let context_tokens = descriptor.context_tokens;
    let budget_chars = chunk::max_chunk_chars(context_tokens);
    let max_tokens = (context_tokens / 3).clamp(256, 1536);

    if request.transcript.segments.is_empty() {
        return Err("the transcript is empty, so there is nothing to summarise".to_string());
    }

    let chunks = chunk::chunk_transcript(
        &request.transcript.segments,
        budget_chars,
        CHUNK_OVERLAP_TURNS,
    );
    let extraction_template = prompts::load(request.data_dir, prompts::EXTRACTION_VERSION)?;

    on_progress("EXTRACTING", 65);
    let mut extractions: Vec<ChunkExtraction> = Vec::new();
    let mut failed_chunks = 0usize;
    for chunk in &chunks {
        let prompt = prompts::fill(
            &extraction_template,
            &[
                ("CHUNK_INDEX", &(chunk.index + 1).to_string()),
                ("CHUNK_COUNT", &chunks.len().to_string()),
                ("SPEAKERS", &chunk.speakers().join(", ")),
                ("TRANSCRIPT", &chunk.render()),
            ],
        );

        match extract_chunk(llm, &prompt, max_tokens, request, chunk) {
            Ok(extraction) => extractions.push(extraction),
            Err(err) => {
                // One unreadable chunk must not cost the whole meeting (§33).
                // It is reported, and the rest of the minutes still get made.
                failed_chunks += 1;
                eprintln!("[mom] chunk {} could not be extracted: {err}", chunk.index);
            }
        }

        let done = chunk.index + 1;
        let percent = 65 + ((done * 20) / chunks.len().max(1)) as i64;
        on_progress("EXTRACTING", percent.min(85));
    }

    if extractions.is_empty() {
        return Err(format!(
            "the language model could not extract anything usable from {} transcript chunk(s). \
The transcript itself is saved and unaffected.",
            chunks.len()
        ));
    }

    on_progress("SYNTHESIZING", 88);
    let mut merged = merge(extractions);
    let synthesis_template = prompts::load(request.data_dir, prompts::SYNTHESIS_VERSION)?;
    let duration_ms = request.transcript.duration_ms();
    let speakers = request.transcript.speaker_labels();

    // A long meeting can produce more extracted facts than fit one synthesis
    // call. Condense them in batches until they do (§18's hierarchical step),
    // with a bounded number of rounds so this always terminates.
    let mut facts = serde_json::to_string_pretty(&merged).map_err(|e| e.to_string())?;
    let mut rounds = 0;
    while facts.chars().count() > budget_chars && rounds < 2 {
        merged = condense(
            llm,
            &merged,
            budget_chars,
            max_tokens,
            &synthesis_template,
            request,
            duration_ms,
            &speakers,
        )?;
        facts = serde_json::to_string_pretty(&merged).map_err(|e| e.to_string())?;
        rounds += 1;
    }
    if facts.chars().count() > budget_chars {
        // Still too big after condensing: keep the earliest facts rather than
        // failing, and say so in the log. Truncation is visible in the MoM as
        // missing late-meeting detail, which beats no minutes at all.
        eprintln!("[mom] extracted facts still exceed the context budget; truncating");
        merged = trim_to_budget(merged, budget_chars);
        facts = serde_json::to_string_pretty(&merged).map_err(|e| e.to_string())?;
    }

    let (mut mom, repaired) = synthesize(
        llm,
        &synthesis_template,
        &facts,
        max_tokens,
        request,
        duration_ms,
        &speakers,
    )?;

    ground(&mut mom, request.transcript);
    mom.meta = Some(MomMeta {
        model: descriptor.model,
        quantization: descriptor.quantization,
        runtime: descriptor.runtime,
        context_tokens,
        extraction_prompt: prompts::EXTRACTION_VERSION.to_string(),
        synthesis_prompt: prompts::SYNTHESIS_VERSION.to_string(),
        language: request.language.to_string(),
        chunk_count: chunks.len(),
        generated_at_ms: now_ms(),
        repaired,
        duration_ms: now_ms() - started_ms,
    });

    if failed_chunks > 0 {
        eprintln!(
            "[mom] {failed_chunks} of {} chunks failed extraction; minutes were built from the rest",
            chunks.len()
        );
    }

    Ok(mom)
}

fn extract_chunk(
    llm: &dyn LlmProvider,
    prompt: &str,
    max_tokens: usize,
    request: &MomRequest,
    chunk: &Chunk,
) -> Result<ChunkExtraction, String> {
    let raw = llm.complete_json(EXTRACTION_SYSTEM, prompt, max_tokens)?;
    match parse_extraction(&raw) {
        Ok(extraction) => Ok(extraction),
        Err(first_error) => {
            // §16: invalid JSON is repaired and retried, never accepted.
            let repaired = repair(
                llm,
                request.data_dir,
                &raw,
                &first_error,
                EXTRACTION_SCHEMA_HINT,
                max_tokens,
            )?;
            parse_extraction(&repaired).map_err(|e| {
                format!("chunk {} still unparseable after repair: {e}", chunk.index)
            })
        }
    }
}

const EXTRACTION_SCHEMA_HINT: &str = r#"{
  "topics": [{"text": "string", "timestamp": 0}],
  "decisions": [{"text": "string", "timestamp": 0, "speaker": "Person 1"}],
  "actionItems": [{"description": "string", "owner": "Person 1 or null", "deadline": "string or null", "timestamp": 0, "confidence": "high|medium|low"}],
  "openQuestions": [{"text": "string", "timestamp": 0, "speaker": "Person 1"}],
  "risks": [{"text": "string", "timestamp": 0}],
  "notes": [{"text": "string", "timestamp": 0}]
}"#;

fn parse_extraction(raw: &str) -> Result<ChunkExtraction, String> {
    let json = extract_json(raw).ok_or_else(|| "no JSON object in the response".to_string())?;
    serde_json::from_str::<ChunkExtraction>(&json).map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn synthesize(
    llm: &dyn LlmProvider,
    template: &str,
    facts: &str,
    max_tokens: usize,
    request: &MomRequest,
    duration_ms: i64,
    speakers: &[String],
) -> Result<(Mom, bool), String> {
    let prompt = prompts::fill(
        template,
        &[
            ("LANGUAGE", request.language),
            ("TITLE", request.title),
            ("DURATION", &format_duration(duration_ms)),
            ("SPEAKERS", &speakers.join(", ")),
            ("FACTS", facts),
        ],
    );

    let raw = llm.complete_json(SYNTHESIS_SYSTEM, &prompt, max_tokens)?;
    match parse_mom(&raw) {
        Ok(mom) => Ok((mom, false)),
        Err(first_error) => {
            let repaired = repair(
                llm,
                request.data_dir,
                &raw,
                &first_error,
                Mom::schema_hint(),
                max_tokens,
            )?;
            let mom = parse_mom(&repaired).map_err(|e| {
                format!(
                    "the language model produced minutes that could not be validated ({e}). \
The transcript is saved and unaffected — retry processing to try again."
                )
            })?;
            Ok((mom, true))
        }
    }
}

fn parse_mom(raw: &str) -> Result<Mom, String> {
    let json = extract_json(raw).ok_or_else(|| "no JSON object in the response".to_string())?;
    let mom: Mom = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    mom.validate()?;
    Ok(mom)
}

fn repair(
    llm: &dyn LlmProvider,
    data_dir: Option<&Path>,
    invalid: &str,
    error: &str,
    schema: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let template = prompts::load(data_dir, prompts::REPAIR_VERSION)?;
    let prompt = prompts::fill(
        &template,
        &[
            ("ERROR", error),
            ("SCHEMA", schema),
            ("INVALID", invalid),
        ],
    );
    llm.complete_json(REPAIR_SYSTEM, &prompt, max_tokens)
}

/// Runs synthesis over batches of facts and folds the results back into the
/// same shape, so an over-long meeting reduces in rounds instead of blowing
/// the context window (§18).
#[allow(clippy::too_many_arguments)]
fn condense(
    llm: &dyn LlmProvider,
    merged: &ChunkExtraction,
    budget_chars: usize,
    max_tokens: usize,
    template: &str,
    request: &MomRequest,
    duration_ms: i64,
    speakers: &[String],
) -> Result<ChunkExtraction, String> {
    let batches = split(merged, budget_chars);
    let mut folded: Vec<ChunkExtraction> = Vec::new();
    for batch in &batches {
        let facts = serde_json::to_string_pretty(batch).map_err(|e| e.to_string())?;
        match synthesize(
            llm,
            template,
            &facts,
            max_tokens,
            request,
            duration_ms,
            speakers,
        ) {
            Ok((mom, _)) => folded.push(mom_as_facts(mom)),
            // A batch that fails condensing keeps its raw facts rather than
            // disappearing from the minutes.
            Err(err) => {
                eprintln!("[mom] a condensing pass failed ({err}); keeping its raw facts");
                folded.push(batch.clone());
            }
        }
    }
    Ok(merge(folded))
}

fn mom_as_facts(mom: Mom) -> ChunkExtraction {
    ChunkExtraction {
        topics: mom.topics,
        decisions: mom.decisions,
        action_items: mom.action_items,
        open_questions: mom.open_questions,
        risks: mom.risks,
        notes: mom.detailed_discussion,
    }
}

/// Splits merged facts into contiguous batches that each fit the budget.
fn split(merged: &ChunkExtraction, budget_chars: usize) -> Vec<ChunkExtraction> {
    let total = serde_json::to_string(merged).map(|s| s.chars().count()).unwrap_or(0);
    let parts = (total / budget_chars.max(1)).saturating_add(1).max(2);

    (0..parts)
        .map(|part| ChunkExtraction {
            topics: slice(&merged.topics, part, parts),
            decisions: slice(&merged.decisions, part, parts),
            action_items: slice(&merged.action_items, part, parts),
            open_questions: slice(&merged.open_questions, part, parts),
            risks: slice(&merged.risks, part, parts),
            notes: slice(&merged.notes, part, parts),
        })
        .filter(|batch| !batch.is_empty())
        .collect()
}

fn slice<T: Clone>(items: &[T], part: usize, parts: usize) -> Vec<T> {
    if items.is_empty() || parts == 0 {
        return Vec::new();
    }
    let per = items.len().div_ceil(parts);
    items.iter().skip(part * per).take(per).cloned().collect()
}

/// Last-resort trim: keep whole categories' earliest entries until the
/// serialized facts fit.
fn trim_to_budget(mut merged: ChunkExtraction, budget_chars: usize) -> ChunkExtraction {
    let mut limit = 40usize;
    loop {
        merged.topics.truncate(limit);
        merged.decisions.truncate(limit);
        merged.action_items.truncate(limit);
        merged.open_questions.truncate(limit);
        merged.risks.truncate(limit);
        merged.notes.truncate(limit);
        let size = serde_json::to_string(&merged)
            .map(|s| s.chars().count())
            .unwrap_or(0);
        if size <= budget_chars || limit <= 2 {
            return merged;
        }
        limit /= 2;
    }
}

/// Unions per-chunk extractions, dropping duplicates that the overlap between
/// neighbouring chunks produces and keeping the earliest evidence for each.
pub fn merge(extractions: Vec<ChunkExtraction>) -> ChunkExtraction {
    let mut merged = ChunkExtraction::default();
    for extraction in extractions {
        merged.topics.extend(extraction.topics);
        merged.decisions.extend(extraction.decisions);
        merged.action_items.extend(extraction.action_items);
        merged.open_questions.extend(extraction.open_questions);
        merged.risks.extend(extraction.risks);
        merged.notes.extend(extraction.notes);
    }
    merged.topics = dedupe_evidenced(merged.topics);
    merged.decisions = dedupe_evidenced(merged.decisions);
    merged.action_items = dedupe_actions(merged.action_items);
    merged.open_questions = dedupe_evidenced(merged.open_questions);
    merged.risks = dedupe_evidenced(merged.risks);
    merged.notes = dedupe_evidenced(merged.notes);
    merged
}

fn dedupe_evidenced(items: Vec<Evidenced>) -> Vec<Evidenced> {
    let mut out: Vec<Evidenced> = Vec::new();
    for item in items {
        if item.text.trim().is_empty() {
            continue;
        }
        let key = normalize(&item.text);
        match out.iter_mut().find(|kept| normalize(&kept.text) == key) {
            Some(kept) => keep_earliest(&mut kept.timestamp, item.timestamp),
            None => out.push(item),
        }
    }
    out.sort_by_key(|item| item.timestamp.unwrap_or(i64::MAX));
    out
}

fn dedupe_actions(items: Vec<ActionItem>) -> Vec<ActionItem> {
    let mut out: Vec<ActionItem> = Vec::new();
    for item in items {
        if item.description.trim().is_empty() {
            continue;
        }
        let key = normalize(&item.description);
        match out
            .iter_mut()
            .find(|kept| normalize(&kept.description) == key)
        {
            Some(kept) => {
                keep_earliest(&mut kept.timestamp, item.timestamp);
                if kept.owner.is_none() {
                    kept.owner = item.owner;
                }
                if kept.deadline.is_none() {
                    kept.deadline = item.deadline;
                }
            }
            None => out.push(item),
        }
    }
    out.sort_by_key(|item| item.timestamp.unwrap_or(i64::MAX));
    out
}

fn keep_earliest(kept: &mut Option<i64>, candidate: Option<i64>) {
    if let Some(value) = candidate {
        *kept = Some(kept.map_or(value, |existing| existing.min(value)));
    }
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_string()
}

/// Rewrites every part of the minutes the model is not allowed to decide
/// (§16). This runs after validation and is what makes the evidence in the UI
/// trustworthy: a timestamp shown next to a decision either points at a real
/// moment in the recording or is absent.
pub fn ground(mom: &mut Mom, transcript: &Transcript) {
    let labels = transcript.speaker_labels();
    let duration_ms = transcript.duration_ms();

    // Participants are a fact about the recording, not a model output.
    mom.participants = labels.clone();

    for list in [
        &mut mom.topics,
        &mut mom.decisions,
        &mut mom.open_questions,
        &mut mom.risks,
        &mut mom.detailed_discussion,
    ] {
        list.retain(|item| !item.text.trim().is_empty());
        for item in list.iter_mut() {
            item.timestamp = normalize_timestamp(item.timestamp, duration_ms);
            item.speaker = item
                .speaker
                .as_ref()
                .and_then(|name| match_label(name, &labels));
        }
    }

    mom.action_items
        .retain(|item| !item.description.trim().is_empty());
    for item in &mut mom.action_items {
        item.timestamp = normalize_timestamp(item.timestamp, duration_ms);
        item.owner = item.owner.as_ref().and_then(|name| match_label(name, &labels));
        item.deadline = item
            .deadline
            .as_ref()
            .map(|d| d.trim().to_string())
            .filter(|d| !d.is_empty() && !d.eq_ignore_ascii_case("null"));
        if !matches!(item.confidence.as_str(), "high" | "medium" | "low") {
            // Validation already ran; anything odd left here is downgraded
            // rather than presented as certain.
            item.confidence = "low".to_string();
        }
    }
}

/// Model timestamps are asked for in seconds. A value that only makes sense as
/// milliseconds is accepted as such; one that lands outside the recording is
/// dropped, because a wrong evidence link is worse than none.
fn normalize_timestamp(raw: Option<i64>, duration_ms: i64) -> Option<i64> {
    let value = raw?;
    if value < 0 {
        return None;
    }
    let as_ms = value.saturating_mul(1000);
    if as_ms <= duration_ms + TIMESTAMP_SLACK_MS {
        return Some(as_ms.min(duration_ms.max(0)));
    }
    if value <= duration_ms + TIMESTAMP_SLACK_MS {
        return Some(value.min(duration_ms.max(0)));
    }
    None
}

/// Matches a model-supplied speaker name against the transcript's real
/// presentation labels. Anything else — a real name it guessed, a label from
/// another meeting — becomes `None` (§7: V1 never identifies real people).
fn match_label(name: &str, labels: &[String]) -> Option<String> {
    let candidate = name.trim();
    if candidate.is_empty() || candidate.eq_ignore_ascii_case("null") {
        return None;
    }
    labels
        .iter()
        .find(|label| label.eq_ignore_ascii_case(candidate))
        .cloned()
}

pub fn format_duration(ms: i64) -> String {
    let total_seconds = (ms / 1000).max(0);
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else {
        format!("{minutes}m {seconds}s")
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::llm::MockLlm;
    use crate::pipeline::models::TranscriptSegment;

    fn transcript(turns: usize) -> Transcript {
        Transcript {
            meeting_id: "test".to_string(),
            segments: (0..turns)
                .map(|i| TranscriptSegment {
                    speaker: format!("SPEAKER_0{}", i % 2),
                    speaker_label: format!("Person {}", (i % 2) + 1),
                    start_ms: i as i64 * 10_000,
                    end_ms: i as i64 * 10_000 + 8_000,
                    text: format!("turn {i} of a meeting about the release schedule"),
                })
                .collect(),
        }
    }

    const EXTRACTION_JSON: &str = r#"{
        "topics": [{"text": "release schedule", "timestamp": 10}],
        "decisions": [{"text": "slip the release by a week", "timestamp": 20, "speaker": "Person 1"}],
        "actionItems": [{"description": "update the plan", "owner": "Person 2", "deadline": null, "timestamp": 30, "confidence": "high"}],
        "openQuestions": [{"text": "is QA ready", "timestamp": 40, "speaker": "Person 2"}],
        "risks": [{"text": "hardware may not arrive", "timestamp": 50}],
        "notes": [{"text": "the team discussed staffing", "timestamp": 60}]
    }"#;

    const SYNTHESIS_JSON: &str = r#"{
        "summary": "The team agreed to slip the release by a week.",
        "participants": ["Someone Invented"],
        "topics": [{"text": "release schedule", "timestamp": 10}],
        "decisions": [{"text": "slip the release by a week", "timestamp": 20, "speaker": "Person 1"}],
        "actionItems": [{"description": "update the plan", "owner": "Priya", "deadline": null, "timestamp": 30, "confidence": "high"}],
        "openQuestions": [{"text": "is QA ready", "timestamp": 999999, "speaker": "Person 2"}],
        "risks": [{"text": "hardware may not arrive", "timestamp": 50}],
        "detailedDiscussion": [{"text": "the team discussed staffing", "timestamp": 60}]
    }"#;

    fn request<'a>(transcript: &'a Transcript) -> MomRequest<'a> {
        MomRequest {
            transcript,
            title: "Release planning",
            language: "English",
            data_dir: None,
        }
    }

    #[test]
    fn generates_minutes_end_to_end_against_a_mock_model() {
        let transcript = transcript(6);
        let llm = MockLlm::new(
            vec![EXTRACTION_JSON.to_string(), SYNTHESIS_JSON.to_string()],
            SYNTHESIS_JSON,
        );
        let mom = generate(&llm, &request(&transcript), &|_, _| {}).unwrap();

        assert!(mom.validate().is_ok());
        assert_eq!(mom.decisions.len(), 1);
        let meta = mom.meta.as_ref().unwrap();
        assert_eq!(meta.extraction_prompt, "extraction_v1");
        assert_eq!(meta.synthesis_prompt, "synthesis_v1");
        assert_eq!(meta.model, "mock");
    }

    #[test]
    fn refuses_to_summarise_an_empty_transcript() {
        let empty = Transcript {
            meeting_id: "e".to_string(),
            segments: vec![],
        };
        let llm = MockLlm::new(vec![], SYNTHESIS_JSON);
        assert!(generate(&llm, &request(&empty), &|_, _| {}).is_err());
    }

    #[test]
    fn participants_come_from_the_transcript_not_the_model() {
        let transcript = transcript(6);
        let llm = MockLlm::new(vec![EXTRACTION_JSON.to_string()], SYNTHESIS_JSON);
        let mom = generate(&llm, &request(&transcript), &|_, _| {}).unwrap();
        assert_eq!(mom.participants, vec!["Person 1", "Person 2"]);
    }

    #[test]
    fn an_owner_who_never_spoke_is_dropped() {
        let transcript = transcript(6);
        let llm = MockLlm::new(vec![EXTRACTION_JSON.to_string()], SYNTHESIS_JSON);
        let mom = generate(&llm, &request(&transcript), &|_, _| {}).unwrap();
        // The mock names "Priya", who is not a speaker label in the transcript.
        assert_eq!(mom.action_items[0].owner, None);
    }

    #[test]
    fn a_timestamp_past_the_end_of_the_meeting_is_dropped() {
        let transcript = transcript(6);
        let llm = MockLlm::new(vec![EXTRACTION_JSON.to_string()], SYNTHESIS_JSON);
        let mom = generate(&llm, &request(&transcript), &|_, _| {}).unwrap();
        assert_eq!(mom.open_questions[0].timestamp, None);
        // ...while a valid one survives, converted from seconds to ms.
        assert_eq!(mom.decisions[0].timestamp, Some(20_000));
    }

    #[test]
    fn malformed_json_is_repaired_rather_than_accepted() {
        let transcript = transcript(4);
        let llm = MockLlm::new(
            vec![
                EXTRACTION_JSON.to_string(),
                "I'm afraid I can't do that.".to_string(),
                SYNTHESIS_JSON.to_string(),
            ],
            SYNTHESIS_JSON,
        );
        let mom = generate(&llm, &request(&transcript), &|_, _| {}).unwrap();
        assert!(mom.meta.as_ref().unwrap().repaired, "should record the repair");
        assert!(mom.validate().is_ok());
    }

    #[test]
    fn a_model_that_never_returns_json_fails_loudly() {
        let transcript = transcript(4);
        let llm = MockLlm::new(vec![], "not json at all");
        let err = generate(&llm, &request(&transcript), &|_, _| {}).unwrap_err();
        assert!(err.contains("transcript"), "should reassure about the transcript: {err}");
    }

    #[test]
    fn progress_is_reported_and_never_goes_backwards() {
        let transcript = transcript(30);
        let llm = MockLlm::new(vec![], EXTRACTION_JSON);
        let seen = std::sync::Mutex::new(Vec::new());
        // The mock always returns an extraction, so synthesis parsing fails and
        // the run errors — progress reporting is what is under test here.
        let _ = generate(&llm, &request(&transcript), &|stage, pct| {
            seen.lock().unwrap().push((stage.to_string(), pct));
        });
        let seen = seen.into_inner().unwrap();
        assert!(seen.iter().any(|(stage, _)| stage == "EXTRACTING"));
        for pair in seen.windows(2) {
            assert!(pair[1].1 >= pair[0].1, "progress went backwards: {seen:?}");
        }
    }

    #[test]
    fn merging_drops_duplicates_from_overlapping_chunks() {
        let a: ChunkExtraction = serde_json::from_str(EXTRACTION_JSON).unwrap();
        let b: ChunkExtraction = serde_json::from_str(EXTRACTION_JSON).unwrap();
        let merged = merge(vec![a, b]);
        assert_eq!(merged.decisions.len(), 1);
        assert_eq!(merged.action_items.len(), 1);
    }

    #[test]
    fn merging_keeps_the_earliest_evidence_and_orders_by_it() {
        let early = ChunkExtraction {
            topics: vec![Evidenced {
                text: "Release schedule".to_string(),
                timestamp: Some(90),
                speaker: None,
            }],
            ..ChunkExtraction::default()
        };
        let late = ChunkExtraction {
            topics: vec![
                Evidenced {
                    text: "release   schedule".to_string(),
                    timestamp: Some(20),
                    speaker: None,
                },
                Evidenced {
                    text: "staffing".to_string(),
                    timestamp: Some(10),
                    speaker: None,
                },
            ],
            ..ChunkExtraction::default()
        };
        let merged = merge(vec![early, late]);
        assert_eq!(merged.topics.len(), 2);
        assert_eq!(merged.topics[0].text, "staffing");
        assert_eq!(merged.topics[1].timestamp, Some(20));
    }

    #[test]
    fn grounding_normalizes_timestamps_in_both_units() {
        // 20 seconds into a 60s meeting.
        assert_eq!(normalize_timestamp(Some(20), 60_000), Some(20_000));
        // Already milliseconds.
        assert_eq!(normalize_timestamp(Some(45_000), 60_000), Some(45_000));
        // Beyond the meeting entirely.
        assert_eq!(normalize_timestamp(Some(999_999), 60_000), None);
        assert_eq!(normalize_timestamp(Some(-5), 60_000), None);
        assert_eq!(normalize_timestamp(None, 60_000), None);
    }

    #[test]
    fn grounding_matches_speaker_labels_case_insensitively() {
        let labels = vec!["Person 1".to_string(), "Person 2".to_string()];
        assert_eq!(match_label("person 2", &labels).as_deref(), Some("Person 2"));
        assert_eq!(match_label("Priya", &labels), None);
        assert_eq!(match_label("null", &labels), None);
        assert_eq!(match_label("  ", &labels), None);
    }

    #[test]
    fn grounding_downgrades_an_odd_confidence_value() {
        let transcript = transcript(4);
        let mut mom = Mom {
            summary: "x".to_string(),
            action_items: vec![ActionItem {
                description: "do a thing".to_string(),
                owner: None,
                deadline: Some("  ".to_string()),
                timestamp: None,
                confidence: "certain".to_string(),
            }],
            ..Mom::default()
        };
        ground(&mut mom, &transcript);
        assert_eq!(mom.action_items[0].confidence, "low");
        assert_eq!(mom.action_items[0].deadline, None);
    }

    #[test]
    fn splitting_facts_covers_every_item_exactly_once() {
        let merged = ChunkExtraction {
            topics: (0..10).map(|i| Evidenced::new(format!("topic {i}"))).collect(),
            ..ChunkExtraction::default()
        };
        let batches = split(&merged, 100);
        let total: usize = batches.iter().map(|b| b.topics.len()).sum();
        assert_eq!(total, 10);
        assert!(batches.len() >= 2);
    }

    #[test]
    fn trimming_reduces_an_oversized_fact_set() {
        let merged = ChunkExtraction {
            notes: (0..500)
                .map(|i| Evidenced::new(format!("a fairly wordy note number {i}")))
                .collect(),
            ..ChunkExtraction::default()
        };
        let trimmed = trim_to_budget(merged, 1_000);
        let size = serde_json::to_string(&trimmed).unwrap().chars().count();
        assert!(size <= 1_000 || trimmed.notes.len() <= 2);
    }

    #[test]
    fn formats_durations_for_the_prompt() {
        assert_eq!(format_duration(0), "0m 0s");
        assert_eq!(format_duration(65_000), "1m 5s");
        assert_eq!(format_duration(3_725_000), "1h 2m 5s");
    }
}
