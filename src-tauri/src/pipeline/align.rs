use crate::pipeline::models::{DiarizationSegment, SttSegment, TranscriptSegment};
use std::collections::HashMap;

/// Consecutive same-speaker STT fragments closer than this are one turn.
/// whisper.cpp runs with `--max-len 80` (docs/models.md), so it emits short
/// fragments that would otherwise render as a wall of one-line speaker blocks.
const MERGE_GAP_MS: i64 = 1_000;

const UNKNOWN_SPEAKER: &str = "SPEAKER_UNKNOWN";

/// Merges STT segments with diarization intervals into the canonical
/// speaker-labelled transcript (master prompt §13).
///
/// Speaker attribution is per STT segment by maximum temporal overlap: when a
/// speaker change falls inside one segment, the segment goes wholly to
/// whoever holds most of it. Splitting mid-segment would need word-level
/// timestamps, which the current whisper.cpp JSON output doesn't carry.
pub fn align(stt: &[SttSegment], diarization: &[DiarizationSegment]) -> Vec<TranscriptSegment> {
    let mut turns: Vec<TranscriptSegment> = Vec::new();

    for seg in stt {
        let speaker = speaker_for(seg, diarization);
        match turns.last_mut() {
            Some(prev) if prev.speaker == speaker && seg.start_ms - prev.end_ms <= MERGE_GAP_MS => {
                prev.end_ms = prev.end_ms.max(seg.end_ms);
                prev.text.push(' ');
                prev.text.push_str(&seg.text);
            }
            _ => turns.push(TranscriptSegment {
                speaker,
                speaker_label: String::new(),
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                text: seg.text.clone(),
            }),
        }
    }

    let mut labels: HashMap<String, String> = HashMap::new();
    for turn in &mut turns {
        let next_label = format!("Person {}", labels.len() + 1);
        turn.speaker_label = labels
            .entry(turn.speaker.clone())
            .or_insert_with(|| {
                if turn.speaker == UNKNOWN_SPEAKER {
                    "Unknown speaker".to_string()
                } else {
                    next_label
                }
            })
            .clone();
    }

    turns
}

fn speaker_for(seg: &SttSegment, diarization: &[DiarizationSegment]) -> String {
    let best_overlap = diarization
        .iter()
        .map(|d| (d, seg.end_ms.min(d.end_ms) - seg.start_ms.max(d.start_ms)))
        .filter(|(_, overlap)| *overlap > 0)
        .max_by_key(|(_, overlap)| *overlap);

    if let Some((d, _)) = best_overlap {
        return stable_id(&d.speaker_id);
    }

    // Nothing overlaps: diarization skips non-speech, so a segment landing in
    // a gap (or in a stretch the clusterer dropped) still belongs to whoever
    // spoke nearest it rather than to nobody.
    diarization
        .iter()
        .min_by_key(|d| gap_to(seg, d))
        .map(|d| stable_id(&d.speaker_id))
        .unwrap_or_else(|| UNKNOWN_SPEAKER.to_string())
}

fn gap_to(seg: &SttSegment, d: &DiarizationSegment) -> i64 {
    if d.end_ms < seg.start_ms {
        seg.start_ms - d.end_ms
    } else if seg.end_ms < d.start_ms {
        d.start_ms - seg.end_ms
    } else {
        0
    }
}

/// sherpa-onnx's numeric cluster id ("0", "1", …) to the stable internal id
/// the rest of the app uses (master prompt §7).
fn stable_id(raw: &str) -> String {
    match raw.trim().parse::<u32>() {
        Ok(n) => format!("SPEAKER_{n:02}"),
        Err(_) => format!("SPEAKER_{}", raw.trim()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stt(start_ms: i64, end_ms: i64, text: &str) -> SttSegment {
        SttSegment {
            start_ms,
            end_ms,
            text: text.to_string(),
            words: Vec::new(),
        }
    }

    fn diar(speaker_id: &str, start_ms: i64, end_ms: i64) -> DiarizationSegment {
        DiarizationSegment {
            speaker_id: speaker_id.to_string(),
            start_ms,
            end_ms,
        }
    }

    #[test]
    fn assigns_segment_to_the_dominant_speaker() {
        let segments = align(
            &[stt(1000, 5000, "mostly speaker one")],
            &[diar("0", 0, 4500), diar("1", 4500, 9000)],
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
    }

    #[test]
    fn merges_consecutive_turns_by_the_same_speaker() {
        let segments = align(
            &[stt(0, 2000, "first part"), stt(2200, 4000, "second part")],
            &[diar("0", 0, 4000)],
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "first part second part");
        assert_eq!(segments[0].end_ms, 4000);
    }

    #[test]
    fn keeps_same_speaker_turns_separate_across_a_long_silence() {
        let segments = align(
            &[stt(0, 2000, "before"), stt(9000, 11000, "after")],
            &[diar("0", 0, 12000)],
        );
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn splits_on_speaker_change() {
        let segments = align(
            &[stt(0, 2000, "question"), stt(2100, 4000, "answer")],
            &[diar("0", 0, 2000), diar("1", 2100, 4000)],
        );
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn falls_back_to_the_nearest_speaker_when_nothing_overlaps() {
        let segments = align(
            &[stt(5000, 5500, "in a diarization gap")],
            &[diar("0", 0, 1000), diar("1", 5600, 8000)],
        );
        assert_eq!(segments[0].speaker, "SPEAKER_01");
    }

    #[test]
    fn marks_speech_unknown_when_diarization_produced_nothing() {
        let segments = align(&[stt(0, 2000, "orphaned speech")], &[]);
        assert_eq!(segments[0].speaker, "SPEAKER_UNKNOWN");
        assert_eq!(segments[0].speaker_label, "Unknown speaker");
    }

    #[test]
    fn labels_speakers_in_first_appearance_order() {
        let segments = align(
            &[
                stt(0, 2000, "second cluster speaks first"),
                stt(3000, 5000, "then the first cluster"),
                stt(6000, 8000, "and back again"),
            ],
            &[diar("1", 0, 2000), diar("0", 3000, 5000), diar("1", 6000, 8000)],
        );
        assert_eq!(segments[0].speaker_label, "Person 1");
        assert_eq!(segments[0].speaker, "SPEAKER_01");
        assert_eq!(segments[1].speaker_label, "Person 2");
        assert_eq!(segments[2].speaker_label, "Person 1");
    }
}
