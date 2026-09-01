//! Speaker-aware chunking for MoM generation (master prompt §14, §18).
//!
//! A 60-minute meeting cannot be handed to a local model in one piece — not on
//! a 32 GB machine and certainly not on a 16 GB one, where the context window
//! is 4K tokens. So the transcript is split into chunks that each fit
//! comfortably, extracted from separately, and only then synthesised.
//!
//! Two invariants matter here:
//!
//! * **A turn is never split.** Cutting a speaker mid-sentence is how you get
//!   an action item attributed to the wrong person, or half a decision.
//! * **Timestamps survive chunking** (§18, "never lose original timestamps").
//!   Each rendered turn carries its own second-offset so the model can quote it
//!   back as evidence.

use crate::pipeline::models::TranscriptSegment;

/// One unit of work for the extraction pass.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub index: usize,
    pub start_ms: i64,
    pub end_ms: i64,
    pub segments: Vec<TranscriptSegment>,
}

impl Chunk {
    /// The text handed to the model: `[seconds] Person N: what they said`.
    pub fn render(&self) -> String {
        self.segments
            .iter()
            .map(|s| format!("[{}] {}: {}", s.start_ms / 1000, s.speaker_label, s.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Distinct speaker labels in this chunk, in first-appearance order.
    pub fn speakers(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for segment in &self.segments {
            if !seen.contains(&segment.speaker_label) {
                seen.push(segment.speaker_label.clone());
            }
        }
        seen
    }
}

/// How many characters of transcript fit one extraction call at a given
/// context size.
///
/// Deliberately pessimistic: it assumes ~2 characters per token, because Tamil
/// text in UTF-8 tokenizes far denser than English does, and a code-switched
/// meeting (§8) is a mix of both. Only about a third of the window is spent on
/// transcript — the rest holds the instructions and leaves room for the answer.
pub fn max_chunk_chars(context_tokens: usize) -> usize {
    let budget = (context_tokens as f64 * 0.35 * 2.0) as usize;
    budget.max(1_200)
}

/// Splits a transcript into chunks of at most `max_chars`, repeating
/// `overlap_turns` turns at each boundary so a decision made across a chunk
/// edge is visible on both sides.
pub fn chunk_transcript(
    segments: &[TranscriptSegment],
    max_chars: usize,
    overlap_turns: usize,
) -> Vec<Chunk> {
    let max_chars = max_chars.max(200);
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current: Vec<TranscriptSegment> = Vec::new();
    let mut current_len = 0usize;

    for segment in segments {
        // The rendered length, not the raw text length: the timestamp and
        // speaker prefix are part of what the model has to read.
        let cost = segment.text.chars().count() + segment.speaker_label.chars().count() + 16;

        if !current.is_empty() && current_len + cost > max_chars {
            let carry = tail(&current, overlap_turns, max_chars / 4);
            chunks.push(finish(chunks.len(), std::mem::take(&mut current)));
            current_len = carry
                .iter()
                .map(|s| s.text.chars().count() + s.speaker_label.chars().count() + 16)
                .sum();
            current = carry;
        }

        current.push(segment.clone());
        current_len += cost;
    }

    // A boundary is only ever crossed while placing a new turn, so whatever is
    // left here always contains content that is not just carried-over overlap.
    if !current.is_empty() {
        chunks.push(finish(chunks.len(), current));
    }

    chunks
}

fn finish(index: usize, segments: Vec<TranscriptSegment>) -> Chunk {
    let start_ms = segments.first().map(|s| s.start_ms).unwrap_or(0);
    let end_ms = segments.last().map(|s| s.end_ms).unwrap_or(start_ms);
    Chunk {
        index,
        start_ms,
        end_ms,
        segments,
    }
}

/// The last `count` turns, as long as they stay under `char_budget` — an
/// overlap must never be big enough to crowd out the new content it precedes.
fn tail(
    segments: &[TranscriptSegment],
    count: usize,
    char_budget: usize,
) -> Vec<TranscriptSegment> {
    if count == 0 {
        return Vec::new();
    }
    let mut carry: Vec<TranscriptSegment> = Vec::new();
    let mut used = 0usize;
    for segment in segments.iter().rev().take(count) {
        // Same cost formula as the main loop above — the overlap budget must
        // be measured the same way the chunk budget is, or it can silently
        // under-count what it is carrying over.
        let cost = segment.text.chars().count() + segment.speaker_label.chars().count() + 16;
        if used + cost > char_budget {
            break;
        }
        used += cost;
        carry.push(segment.clone());
    }
    carry.reverse();
    carry
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start_s: i64, speaker: &str, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            speaker: format!("SPEAKER_{speaker}"),
            speaker_label: format!("Person {speaker}"),
            start_ms: start_s * 1000,
            end_ms: start_s * 1000 + 4000,
            text: text.to_string(),
            language: None,
        }
    }

    fn long_meeting(turns: usize) -> Vec<TranscriptSegment> {
        (0..turns)
            .map(|i| {
                turn(
                    i as i64 * 10,
                    if i % 2 == 0 { "0" } else { "1" },
                    "this is a reasonably long sentence from a meeting transcript",
                )
            })
            .collect()
    }

    #[test]
    fn an_empty_transcript_produces_no_chunks() {
        assert!(chunk_transcript(&[], 2000, 1).is_empty());
    }

    #[test]
    fn a_short_meeting_is_one_chunk() {
        let chunks = chunk_transcript(&long_meeting(3), 4000, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].segments.len(), 3);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn a_long_meeting_splits_and_every_turn_survives() {
        let meeting = long_meeting(60);
        let chunks = chunk_transcript(&meeting, 1_200, 0);
        assert!(chunks.len() > 1);
        let seen: usize = chunks.iter().map(|c| c.segments.len()).sum();
        assert_eq!(seen, meeting.len(), "no turn may be dropped");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn chunks_stay_near_the_character_budget() {
        let chunks = chunk_transcript(&long_meeting(60), 1_200, 1);
        for chunk in &chunks {
            // One over-long turn is allowed to exceed the budget alone, but a
            // multi-turn chunk must not run away.
            if chunk.segments.len() > 1 {
                assert!(
                    chunk.render().chars().count() < 1_200 * 2,
                    "chunk {} is {} chars",
                    chunk.index,
                    chunk.render().chars().count()
                );
            }
        }
    }

    #[test]
    fn a_single_over_long_turn_is_never_split() {
        let huge = vec![turn(0, "0", &"word ".repeat(2_000))];
        let chunks = chunk_transcript(&huge, 500, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].segments.len(), 1);
    }

    #[test]
    fn boundaries_overlap_so_context_is_not_lost() {
        let meeting = long_meeting(40);
        let chunks = chunk_transcript(&meeting, 1_200, 2);
        assert!(chunks.len() > 1);
        let first_end = chunks[0].segments.last().unwrap().start_ms;
        let second_start = chunks[1].segments.first().unwrap().start_ms;
        assert!(
            second_start <= first_end,
            "the second chunk should repeat the end of the first"
        );
    }

    #[test]
    fn chunk_bounds_come_from_the_turns_inside_it() {
        let chunks = chunk_transcript(&long_meeting(40), 1_200, 1);
        for chunk in &chunks {
            assert_eq!(chunk.start_ms, chunk.segments.first().unwrap().start_ms);
            assert_eq!(chunk.end_ms, chunk.segments.last().unwrap().end_ms);
            assert!(chunk.end_ms >= chunk.start_ms);
        }
    }

    #[test]
    fn rendering_keeps_timestamps_and_speakers() {
        let chunk = &chunk_transcript(&long_meeting(2), 4000, 0)[0];
        let rendered = chunk.render();
        assert!(rendered.starts_with("[0] Person 0: "));
        assert!(rendered.contains("\n[10] Person 1: "));
        assert_eq!(chunk.speakers(), vec!["Person 0", "Person 1"]);
    }

    #[test]
    fn context_size_drives_the_chunk_budget() {
        assert!(max_chunk_chars(8192) > max_chunk_chars(4096));
        assert!(max_chunk_chars(4096) > max_chunk_chars(2048));
        // Even a tiny context still gets a workable chunk.
        assert_eq!(max_chunk_chars(256), 1_200);
    }
}
