use crate::pipeline::models::{
    DiarizationSegment, SttSegment, TranscriptSegment, MIXED_LANGUAGE,
};
#[cfg(test)]
use crate::pipeline::models::SttWord;
use std::collections::HashMap;

/// Consecutive same-speaker STT fragments closer than this are one turn.
/// whisper.cpp runs with `--max-len 80` (docs/benchmarks.md), so it emits short
/// fragments that would otherwise render as a wall of one-line speaker blocks.
const MERGE_GAP_MS: i64 = 1_000;

const UNKNOWN_SPEAKER: &str = "SPEAKER_UNKNOWN";

/// A diarization segment shorter than this, touching a neighbour on both
/// sides, is treated as a clustering artifact rather than a turn — see
/// [`absorb_islands`].
const ISLAND_MAX_MS: i64 = 1_500;

/// Two diarization segments this close are "touching": no pause between them.
const TOUCH_GAP_MS: i64 = 250;

/// A cluster holding less talk time than this, in total, can be a minor
/// cluster — see [`merge_minor_clusters`]. The guard against merging someone
/// who speaks often but briefly; on the 2026-10-06 baseline runs the largest
/// cluster made only of short fragments held 6.9 s (council clip, 0.5 and
/// 0.6), and no cluster between that and 8 s existed on either clip.
const MINOR_TOTAL_MS: i64 = 8_000;

/// A cluster with any segment this long is a voice in its own right and is
/// never a minor cluster. The separating measurement, from the same runs: the
/// longest segment of every cluster read as spurious was at most 2.92 s (the
/// council clip's anchor and mayor split off for their first words, and a
/// scattered 2 s-fragment cluster), while the quietest genuine voice — a
/// resident with two turns, a stable cluster at every threshold from 0.5 to
/// 0.8 — had a 4.91 s segment. 3.5 s sits between with margin both ways, and
/// is also where speaker embeddings conventionally stop being trusted
/// (~3 s is the usual "short utterance" bound). It deliberately does not
/// reach the TED clip's 4.44 s spurious cluster: that close to 4.91 s it
/// would be a guess, not a measurement.
const MINOR_SEGMENT_MS: i64 = 3_500;

/// A segment at least this long with a pause on both sides (more than
/// [`TOUCH_GAP_MS`] to any other speech) is a standalone utterance — someone
/// spoke on their own, not a window that broke off another voice — and keeps
/// its cluster. Every standalone segment in a spurious cluster on the
/// baseline runs was shorter (0.66, 0.93 and 1.27 s, all on the single-talker
/// TED clip), so this protects every isolated remark longer than those.
const STANDALONE_MIN_MS: i64 = 1_500;

/// Merges STT segments with diarization intervals into the canonical
/// speaker-labelled transcript.
///
/// Speaker attribution is per STT segment by maximum temporal overlap. When
/// the segment carries word timestamps ([`SttSegment::words`], populated by
/// `--output-json-full`) it can be cut at a word boundary instead, so a fast
/// two-way exchange inside one whisper segment keeps both speakers rather than
/// being absorbed wholesale by whoever held most of it — see
/// [`speaker_pieces`] for the three conditions that has to satisfy first.
///
/// Word timestamps are an *addition*, never a change: a segment without them
/// takes exactly the path it always did.
pub fn align(stt: &[SttSegment], diarization: &[DiarizationSegment]) -> Vec<TranscriptSegment> {
    // The last moment anything was heard, so a segment ending the recording
    // is not mistaken for one followed by silence — see `merge_minor_clusters`.
    let audio_end_ms = stt
        .iter()
        .map(|s| s.end_ms)
        .chain(diarization.iter().map(|d| d.end_ms))
        .max()
        .unwrap_or(0);
    let diarization = merge_minor_clusters(absorb_islands(diarization), audio_end_ms);
    let diarization = diarization.as_slice();
    let mut turns: Vec<TranscriptSegment> = Vec::new();

    for seg in stt {
        for piece in speaker_pieces(seg, diarization) {
            match turns.last_mut() {
                Some(prev)
                    if prev.speaker == piece.speaker
                        && piece.start_ms - prev.end_ms <= MERGE_GAP_MS =>
                {
                    prev.end_ms = prev.end_ms.max(piece.end_ms);
                    prev.text.push(' ');
                    prev.text.push_str(&piece.text);
                    prev.language =
                        merge_language(prev.language.as_deref(), seg.language.as_deref());
                }
                _ => turns.push(TranscriptSegment {
                    speaker: piece.speaker,
                    speaker_label: String::new(),
                    start_ms: piece.start_ms,
                    end_ms: piece.end_ms,
                    text: piece.text,
                    language: seg.language.clone(),
                    // Alignment does not scan; the redaction stage runs after
                    // it, once the turns are final and offsets are stable.
                    flags: Vec::new(),
                }),
            }
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

/// The language of a turn built from two fragments.
///
/// Two fragments detected differently is the code switch itself, and the turn
/// is honestly [`MIXED_LANGUAGE`] rather than whichever half came first. An
/// untagged fragment (a single pass, which detects per file rather than per
/// block) carries no information and never overrides one that does.
fn merge_language(current: Option<&str>, incoming: Option<&str>) -> Option<String> {
    match (current, incoming) {
        (None, other) | (other, None) => other.map(|l| l.to_string()),
        (Some(a), Some(b)) if a == b => Some(a.to_string()),
        _ => Some(MIXED_LANGUAGE.to_string()),
    }
}

// ------------------------------------------------ diarization islands

/// Relabels the short, gapless speaker "islands" sherpa-onnx emits inside one
/// continuous voice.
///
/// On real meeting audio (the council clip in `tests/baseline/clips.json`) its
/// pyannote segmentation cuts one person's sentence into back-to-back ~1 s
/// windows and the clusterer gives several of them different ids, so an
/// anchor's opening line rendered as Person 1 → 2 → 3 → 4 → 2 mid-sentence and
/// word-level splitting then cut the text at every flip.
///
/// An island is a segment shorter than [`ISLAND_MAX_MS`] that touches a
/// neighbour on *both* sides (within [`TOUCH_GAP_MS`], overlap included). A
/// short segment with a pause on either side — a genuine "yes" or "okay" — is
/// left alone. A short segment lying wholly inside another is the same
/// artifact in overlapped form and is dropped first, by [`drop_nested`]. This
/// shapes only the derived transcript: `diarization.json` keeps what
/// sherpa-onnx actually said.
///
/// The known cost: a genuine back-channel under 1.5 s ("yeah") butted between
/// two stretches of one speaker, or spoken over them, is absorbed into that
/// speaker's turn. sherpa-onnx emits real turn changes with zero gap too, so
/// nothing in the diarization alone separates the two; losing a short
/// interjection's attribution was judged cheaper than flipping speakers
/// mid-sentence.
///
/// Each pass decides from the segments as they stood at its start, so the
/// result never depends on scan order. Islands whose two neighbours agree are
/// the strongest evidence and are all resolved first; only when none remain is
/// the single shortest island handed to its longer neighbour, and the pass
/// repeats, since every merge can change what is short and what touches.
fn absorb_islands(diarization: &[DiarizationSegment]) -> Vec<DiarizationSegment> {
    let mut sorted = diarization.to_vec();
    // Earliest start first and, of equal starts, the longest first, so a
    // segment always comes after any segment containing it. Speaker last, so
    // even identical intervals order the same way whatever order they came in.
    sorted.sort_by(|a, b| {
        (a.start_ms, b.end_ms, &a.speaker_id).cmp(&(b.start_ms, a.end_ms, &b.speaker_id))
    });
    let mut segs = merge_touching(drop_nested(sorted));
    let len_of = |d: &DiarizationSegment| d.end_ms - d.start_ms;

    // Every productive pass removes at least one segment, which bounds the
    // passes: a relabelled island merges into the neighbour whose label it
    // took. In an agreed batch at least the first island of each adjacent run
    // does, since its previous neighbour is not itself relabelled that pass.
    for _ in 0..segs.len() {
        let islands: Vec<usize> = (1..segs.len().saturating_sub(1))
            .filter(|&i| {
                let (prev, cur, next) = (&segs[i - 1], &segs[i], &segs[i + 1]);
                len_of(cur) < ISLAND_MAX_MS
                    && cur.start_ms - prev.end_ms <= TOUCH_GAP_MS
                    && next.start_ms - cur.end_ms <= TOUCH_GAP_MS
            })
            .collect();

        let agreed: Vec<usize> = islands
            .iter()
            .copied()
            .filter(|&i| segs[i - 1].speaker_id == segs[i + 1].speaker_id)
            .collect();
        // Adjacent agreed islands all move at once, each to its *previous*
        // neighbour's label as it stood. On an alternating run of windows that
        // erodes the run from both ends at once: between two anchors of one
        // voice (A [B A B] A) it collapses entirely into A, and between two
        // voices (A [B A] B) it splits down the middle (A A B B). Resolving
        // the shortest first instead was tried on the council clip and is
        // worse: relabelling a middle window merges three windows into one
        // segment too long to be an island, which then survives as a turn.
        let relabels: Vec<(usize, String)> = if !agreed.is_empty() {
            agreed
                .into_iter()
                .map(|i| (i, segs[i - 1].speaker_id.clone()))
                .collect()
        } else if let Some(i) = islands.into_iter().min_by_key(|&i| (len_of(&segs[i]), i)) {
            let (prev, next) = (&segs[i - 1], &segs[i + 1]);
            let winner = if len_of(prev) >= len_of(next) { prev } else { next };
            vec![(i, winner.speaker_id.clone())]
        } else {
            break;
        };

        for (i, speaker) in relabels {
            segs[i].speaker_id = speaker;
        }
        segs = merge_touching(segs);
    }
    segs
}

/// Drops every segment shorter than [`ISLAND_MAX_MS`] that lies inside an
/// earlier one (to within [`TOUCH_GAP_MS`]), plus any that ends before it
/// starts. `sorted` must be in [`absorb_islands`]'s order.
///
/// sherpa-onnx reports overlapped speech as a second segment nested in the
/// first; on the council clip these were 0.3–0.5 s slivers inside one
/// sentence, rendering as a one-word turn mid-sentence. Their neighbours in
/// start order are not their neighbours in time, so the island test cannot
/// see them — but whoever contains one holds the audio on both its sides,
/// which is exactly an island whose neighbours agree. Dropping it hands its
/// span to the container, as relabelling would.
fn drop_nested(sorted: Vec<DiarizationSegment>) -> Vec<DiarizationSegment> {
    let mut kept: Vec<DiarizationSegment> = Vec::with_capacity(sorted.len());
    // Latest end among the *kept* segments, so a dropped sliver can never
    // stretch the reach for the next one.
    let mut reach = i64::MIN;
    for seg in sorted {
        let malformed = seg.end_ms < seg.start_ms;
        let nested = seg.end_ms - seg.start_ms < ISLAND_MAX_MS
            && seg.end_ms <= reach.saturating_add(TOUCH_GAP_MS);
        if !malformed && !nested {
            reach = reach.max(seg.end_ms);
            kept.push(seg);
        }
    }
    kept
}

/// Joins consecutive same-speaker segments with no pause between them, so a
/// relabelled island lengthens its neighbour and stops looking short.
fn merge_touching(segs: Vec<DiarizationSegment>) -> Vec<DiarizationSegment> {
    let mut merged: Vec<DiarizationSegment> = Vec::with_capacity(segs.len());
    for seg in segs {
        match merged.last_mut() {
            Some(prev)
                if prev.speaker_id == seg.speaker_id
                    && seg.start_ms - prev.end_ms <= TOUCH_GAP_MS =>
            {
                prev.end_ms = prev.end_ms.max(seg.end_ms);
            }
            _ => merged.push(seg),
        }
    }
    merged
}

// ------------------------------------------------ minor clusters

/// Folds whole spurious clusters into the voices around them.
///
/// [`absorb_islands`] fixes a short window inside one voice; it cannot fix a
/// cluster that holds a few seconds scattered across the recording. sherpa-
/// onnx makes those at the shipped threshold: on the council clip it split
/// the anchor's and the mayor's first words off as clusters of their own
/// (2.9 s and 2.0 s), and built a third out of four ~2 s fragments of three
/// different people. Each one added a "Person N" to the transcript.
///
/// A cluster is minor only when all of these hold, and each is a guard
/// against erasing a real participant who says little, which is the
/// expensive failure here:
///
/// 1. **Little talk time** — under [`MINOR_TOTAL_MS`] in total.
/// 2. **Only short segments** — none reaching [`MINOR_SEGMENT_MS`]. One real
///    sentence keeps the cluster.
/// 3. **Nothing said on its own** — no segment of [`STANDALONE_MIN_MS`] or
///    more with a pause on both sides. A short remark in silence is what a
///    quiet participant sounds like; a fragment butted against someone else's
///    speech is what a split-off window looks like.
///
/// Each segment of a minor cluster then takes the speaker of the nearest
/// segment of a cluster that fails 1 or 2 — a substantial voice (overlap
/// counts as nearest, then the longer segment) — so a fragment of three
/// people goes back to whichever of them it sits against. A small cluster
/// kept only by 3 is never a target. When no cluster is substantial nothing
/// happens: there is no voice to fold into, and a short recording is not
/// evidence of a spurious cluster.
///
/// It is deliberately not a share-of-talk-time rule. A share that flags a
/// 6 s cluster in a 2-minute clip flags a 60 s one in an hour-long meeting,
/// and the measurements above are absolute.
///
/// What it does not do: the single-talker TED clip stays over-split, because
/// its spurious clusters are large — six of them hold 4–32% of the talk
/// time each — and nothing in the timeline separates them from real voices
/// holding 11–14% on the council clip. That needs speaker information
/// (a headcount hint, or the embeddings), not alignment.
///
/// The known cost: a participant whose whole contribution is under 8 s, made
/// of short remarks that each butt against someone else's speech (or one
/// isolated remark under 1.5 s), is folded into a neighbour. Their words stay
/// in the transcript under the wrong name. `diarization.json` is untouched.
fn merge_minor_clusters(segs: Vec<DiarizationSegment>, audio_end_ms: i64) -> Vec<DiarizationSegment> {
    let len_of = |d: &DiarizationSegment| d.end_ms - d.start_ms;

    let mut clusters: HashMap<&str, (i64, i64)> = HashMap::new();
    for seg in &segs {
        let (total, longest) = clusters.entry(seg.speaker_id.as_str()).or_insert((0, 0));
        *total += len_of(seg);
        *longest = (*longest).max(len_of(seg));
    }

    let standalone = |i: usize| -> bool {
        let seg = &segs[i];
        if len_of(seg) < STANDALONE_MIN_MS {
            return false;
        }
        // Silence before and after, measured to every other segment of any
        // speaker; the recording's own edges count as speech, so a fragment
        // the recording cut into is not mistaken for a remark in silence.
        let mut before = seg.start_ms;
        let mut after = audio_end_ms - seg.end_ms;
        for (j, other) in segs.iter().enumerate() {
            if j == i {
                continue;
            }
            if other.end_ms <= seg.start_ms {
                before = before.min(seg.start_ms - other.end_ms);
            } else if other.start_ms >= seg.end_ms {
                after = after.min(other.start_ms - seg.end_ms);
            } else {
                return false; // overlapped by someone
            }
        }
        before > TOUCH_GAP_MS && after > TOUCH_GAP_MS
    };

    // Small by conditions 1 and 2. Only a cluster that is not small is a
    // voice to fold into: a small cluster kept by condition 3 is someone's
    // own remark, not somewhere to put another cluster's fragments.
    let small: Vec<&str> = clusters
        .iter()
        .filter(|(_, &(total, longest))| total < MINOR_TOTAL_MS && longest < MINOR_SEGMENT_MS)
        .map(|(&id, _)| id)
        .collect();
    if small.len() == clusters.len() {
        return segs;
    }
    let protected: Vec<&str> = (0..segs.len())
        .filter(|&i| standalone(i))
        .map(|i| segs[i].speaker_id.as_str())
        .collect();
    let minor: Vec<&str> = small
        .iter()
        .copied()
        .filter(|id| !protected.contains(id))
        .collect();
    if minor.is_empty() {
        return segs;
    }

    let relabels: Vec<(usize, String)> = (0..segs.len())
        .filter(|&i| minor.contains(&segs[i].speaker_id.as_str()))
        .filter_map(|i| {
            let seg = &segs[i];
            segs.iter()
                .filter(|d| !small.contains(&d.speaker_id.as_str()))
                .min_by_key(|d| {
                    let overlap = seg.end_ms.min(d.end_ms) - seg.start_ms.max(d.start_ms);
                    (
                        gap_to(seg.start_ms, seg.end_ms, d),
                        -overlap.max(0),
                        -len_of(d),
                        d.speaker_id.clone(),
                    )
                })
                .map(|d| (i, d.speaker_id.clone()))
        })
        .collect();

    let mut segs = segs;
    for (i, speaker) in relabels {
        segs[i].speaker_id = speaker;
    }
    merge_touching(segs)
}

// ------------------------------------------------ splitting on word times

/// A run of one STT segment that belongs to a single speaker. A segment
/// without usable word timestamps yields exactly one of these, holding the
/// whole segment — which is what keeps the no-words path identical.
struct Piece {
    speaker: String,
    start_ms: i64,
    end_ms: i64,
    text: String,
}

/// Cuts one STT segment into per-speaker runs.
///
/// Whisper decides where a segment ends from the audio and `--max-len`, not
/// from who is talking, so a quick "yes — no, we can't" exchange routinely
/// lands inside a single segment. Whole-segment attribution then hands all of
/// it to whoever overlapped it most and the other speaker's words vanish from
/// the record entirely; that is the known limitation this exists to fix.
///
/// It only splits when all three of these hold, because a wrong split is worse
/// than the coarse attribution it replaces:
///
/// 1. **The segment has word timestamps.** Without them there is no boundary
///    to cut on, and the function returns the single whole-segment piece the
///    old code built — the server path, an `stt.json` written before word
///    timestamps existed, and any parse that gave up all land here.
/// 2. **The words disagree about the speaker.** If they all agree, the piece
///    is attributed by [`speaker_for`] — the *segment-level* answer, not the
///    word-level one. So word timestamps can only ever split a segment; they
///    can never quietly re-attribute one that stays whole.
/// 3. **The words reconstruct the segment's own text.** Splitting builds each
///    piece's text from its words, so unless they spell out exactly what the
///    segment says, a split would drop or invent transcript text. Whisper
///    emits tokens this does not survive (a dropped undated token, a script
///    without spaces between words); when it does not hold, the segment stays
///    whole and only its speaker labelling is coarser.
fn speaker_pieces(seg: &SttSegment, diarization: &[DiarizationSegment]) -> Vec<Piece> {
    let whole = || {
        vec![Piece {
            speaker: speaker_for(seg, diarization),
            start_ms: seg.start_ms,
            end_ms: seg.end_ms,
            text: seg.text.clone(),
        }]
    };

    if seg.words.is_empty() || !words_reconstruct_text(seg) {
        return whole();
    }

    let speakers: Vec<String> = seg
        .words
        .iter()
        .map(|word| speaker_for_span(word.start_ms, word.end_ms, diarization))
        .collect();
    if speakers.windows(2).all(|pair| pair[0] == pair[1]) {
        return whole();
    }

    // Maximal runs of consecutive words sharing a speaker. The cut therefore
    // falls on the word boundary nearest the diarization change by
    // construction: the first word whose own span leans to the new speaker
    // starts the new run.
    let mut runs: Vec<(String, usize, usize)> = Vec::new();
    for (index, speaker) in speakers.into_iter().enumerate() {
        match runs.last_mut() {
            Some((current, _, last)) if *current == speaker => *last = index,
            _ => runs.push((speaker, index, index)),
        }
    }

    let final_run = runs.len() - 1;
    runs.into_iter()
        .enumerate()
        .map(|(position, (speaker, first, last))| {
            // The segment's own bounds win at the two ends, so the audio
            // before the first word and after the last is not orphaned.
            let start_ms = if position == 0 {
                seg.start_ms
            } else {
                seg.words[first].start_ms
            };
            let end_ms = if position == final_run {
                seg.end_ms
            } else {
                seg.words[last].end_ms
            };
            let start_ms = start_ms.clamp(seg.start_ms, seg.end_ms);
            Piece {
                speaker,
                start_ms,
                end_ms: end_ms.clamp(start_ms, seg.end_ms),
                text: seg.words[first..=last]
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            }
        })
        .collect()
}

/// Whether the words spell out the segment, ignoring how whitespace was laid
/// out — the guard behind condition 3 of [`speaker_pieces`]. Cheap, and the
/// only check that can actually promise a split loses no transcript text.
fn words_reconstruct_text(seg: &SttSegment) -> bool {
    let joined = seg
        .words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    collapse_whitespace(&joined) == collapse_whitespace(&seg.text)
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn speaker_for(seg: &SttSegment, diarization: &[DiarizationSegment]) -> String {
    speaker_for_span(seg.start_ms, seg.end_ms, diarization)
}

fn speaker_for_span(start_ms: i64, end_ms: i64, diarization: &[DiarizationSegment]) -> String {
    let best_overlap = diarization
        .iter()
        .map(|d| (d, end_ms.min(d.end_ms) - start_ms.max(d.start_ms)))
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
        .min_by_key(|d| gap_to(start_ms, end_ms, d))
        .map(|d| stable_id(&d.speaker_id))
        .unwrap_or_else(|| UNKNOWN_SPEAKER.to_string())
}

fn gap_to(start_ms: i64, end_ms: i64, d: &DiarizationSegment) -> i64 {
    if d.end_ms < start_ms {
        start_ms - d.end_ms
    } else if end_ms < d.start_ms {
        d.start_ms - end_ms
    } else {
        0
    }
}

/// sherpa-onnx's numeric cluster id ("0", "1", …) to the stable internal id
/// the rest of the app uses.
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
            language: None,
        }
    }

    fn tagged(start_ms: i64, end_ms: i64, text: &str, language: &str) -> SttSegment {
        SttSegment {
            language: Some(language.to_string()),
            ..stt(start_ms, end_ms, text)
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

    // ------------------------------------------------------- language

    #[test]
    fn a_turn_in_one_language_keeps_that_language() {
        let segments = align(
            &[tagged(0, 2000, "one", "ta"), tagged(2100, 4000, "two", "ta")],
            &[diar("0", 0, 4000)],
        );
        assert_eq!(segments.len(), 1, "same speaker, one turn");
        assert_eq!(segments[0].language.as_deref(), Some("ta"));
    }

    #[test]
    fn a_turn_that_switches_language_is_marked_mixed() {
        let segments = align(
            &[tagged(0, 2000, "நாம", "ta"), tagged(2100, 4000, "ship it", "en")],
            &[diar("0", 0, 4000)],
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].language.as_deref(), Some(MIXED_LANGUAGE));
        assert_eq!(segments[0].text, "நாம ship it");
    }

    #[test]
    fn separate_turns_keep_their_own_languages() {
        let segments = align(
            &[tagged(0, 2000, "one", "ta"), tagged(2100, 4000, "two", "en")],
            &[diar("0", 0, 2000), diar("1", 2100, 4000)],
        );
        assert_eq!(segments[0].language.as_deref(), Some("ta"));
        assert_eq!(segments[1].language.as_deref(), Some("en"));
    }

    #[test]
    fn an_untagged_single_pass_transcript_gains_no_language() {
        let segments = align(
            &[stt(0, 2000, "one"), stt(2100, 4000, "two")],
            &[diar("0", 0, 4000)],
        );
        assert_eq!(segments[0].language, None);
    }

    // --------------------------------------------------- word-level splits

    fn word(text: &str, start_ms: i64, end_ms: i64) -> SttWord {
        SttWord {
            start_ms,
            end_ms,
            text: text.to_string(),
        }
    }

    /// The motivating case: whisper packs a quick two-way exchange into one
    /// segment, because it ends segments on audio and `--max-len`, not on who
    /// is speaking.
    fn fast_exchange() -> SttSegment {
        SttSegment {
            words: vec![
                word("Yes", 0, 400),
                word("exactly.", 400, 900),
                word("No,", 2000, 2300),
                word("we", 2300, 2600),
                word("cannot.", 2600, 4000),
            ],
            ..stt(0, 4000, "Yes exactly. No, we cannot.")
        }
    }

    fn two_speaker_exchange() -> Vec<DiarizationSegment> {
        vec![diar("0", 0, 1000), diar("1", 1900, 4000)]
    }

    fn speakers_of(segs: &[DiarizationSegment]) -> Vec<(&str, i64, i64)> {
        segs.iter()
            .map(|d| (d.speaker_id.as_str(), d.start_ms, d.end_ms))
            .collect()
    }

    #[test]
    fn a_gapless_island_inside_one_voice_takes_that_voice() {
        let segs = absorb_islands(&[diar("2", 0, 4000), diar("0", 4000, 5000), diar("2", 5000, 9000)]);
        assert_eq!(speakers_of(&segs), vec![("2", 0, 9000)]);
    }

    #[test]
    fn a_chain_of_window_islands_collapses_like_the_council_clip() {
        // The shape sherpa-onnx produced on real audio: ~1 s windows with
        // different ids, back to back, inside one continuous voice.
        let segs = absorb_islands(&[
            diar("0", 3777, 6038),
            diar("5", 6038, 7034),
            diar("2", 7034, 8030),
            diar("0", 8030, 9025),
            diar("2", 9025, 14037),
        ]);
        assert_eq!(speakers_of(&segs), vec![("0", 3777, 6038), ("2", 6038, 14037)]);
    }

    #[test]
    fn a_short_interjection_with_a_pause_is_kept() {
        let input = [diar("0", 0, 4000), diar("1", 4400, 5000), diar("0", 5400, 9000)];
        let segs = absorb_islands(&input);
        assert_eq!(speakers_of(&segs), speakers_of(&input));
    }

    #[test]
    fn a_long_segment_between_touching_neighbours_is_kept() {
        let input = [diar("0", 0, 4000), diar("1", 4000, 6000), diar("0", 6000, 9000)];
        assert_eq!(speakers_of(&absorb_islands(&input)), speakers_of(&input));
    }

    #[test]
    fn an_island_between_two_voices_joins_the_longer_one() {
        let segs = absorb_islands(&[diar("0", 0, 2000), diar("7", 2000, 3000), diar("1", 3000, 8000)]);
        assert_eq!(speakers_of(&segs), vec![("0", 0, 2000), ("1", 2000, 8000)]);
    }

    #[test]
    fn a_short_segment_nested_inside_another_is_dropped() {
        let segs = absorb_islands(&[diar("0", 0, 10000), diar("1", 3000, 3500), diar("0", 10000, 12000)]);
        assert_eq!(speakers_of(&segs), vec![("0", 0, 12000)]);
    }

    #[test]
    fn a_nested_sliver_no_longer_hides_the_island_after_it() {
        // The council clip's tail: in start order the nested sliver sat
        // between the long segment and the island, so the island's "previous
        // neighbour" ended 9 s before it and it never counted as touching.
        let segs = absorb_islands(&[
            diar("6", 121970, 132888),
            diar("9", 123657, 123978),
            diar("4", 132938, 133445),
            diar("6", 133445, 134896),
        ]);
        assert_eq!(speakers_of(&segs), vec![("6", 121970, 134896)]);
    }

    #[test]
    fn sustained_overlapped_speech_is_kept() {
        let input = [diar("0", 0, 10000), diar("1", 3000, 6000)];
        assert_eq!(speakers_of(&absorb_islands(&input)), speakers_of(&input));
    }

    #[test]
    fn a_partly_overlapped_island_is_still_absorbed() {
        // From the unenhanced council run: a 1 s window overlapping the voice
        // before it by 287 ms, in the middle of that voice's sentence.
        let segs = absorb_islands(&[
            diar("1", 94497, 107727),
            diar("11", 107440, 108487),
            diar("1", 108487, 110950),
        ]);
        assert_eq!(speakers_of(&segs), vec![("1", 94497, 110950)]);
    }

    #[test]
    fn degenerate_segments_neither_panic_nor_survive() {
        let segs = absorb_islands(&[
            diar("0", 0, 4000),
            diar("3", 4000, 4000),
            diar("0", 4000, 9000),
            diar("5", 9500, 9000),
        ]);
        assert_eq!(speakers_of(&segs), vec![("0", 0, 9000)]);
        assert!(absorb_islands(&[]).is_empty());
        assert_eq!(speakers_of(&absorb_islands(&[diar("2", 0, 500)])), vec![("2", 0, 500)]);
    }

    #[test]
    fn input_order_never_changes_the_result() {
        // Two identical intervals: which label survives must not depend on
        // the order sherpa-onnx happened to list them in.
        let forward = [diar("5", 0, 1000), diar("0", 0, 1000), diar("1", 3000, 8000)];
        let mut reversed = forward.clone();
        reversed.reverse();
        let expected = vec![("0", 0, 1000), ("1", 3000, 8000)];
        assert_eq!(speakers_of(&absorb_islands(&forward)), expected);
        assert_eq!(speakers_of(&absorb_islands(&reversed)), expected);
    }

    #[test]
    fn alternating_windows_inside_one_voice_collapse_into_it() {
        let segs = absorb_islands(&[
            diar("0", 0, 3000),
            diar("1", 3000, 4000),
            diar("0", 4000, 5000),
            diar("1", 5000, 6000),
            diar("0", 6000, 9000),
        ]);
        assert_eq!(speakers_of(&segs), vec![("0", 0, 9000)]);
    }

    #[test]
    fn alternating_windows_between_two_voices_split_down_the_middle() {
        let segs = absorb_islands(&[
            diar("0", 0, 3000),
            diar("1", 3000, 4000),
            diar("0", 4000, 5000),
            diar("1", 5000, 8000),
        ]);
        assert_eq!(speakers_of(&segs), vec![("0", 0, 4000), ("1", 4000, 8000)]);
    }

    #[test]
    fn short_segments_at_either_end_are_kept() {
        let input = [diar("3", 0, 800), diar("0", 800, 5000), diar("4", 5000, 5600)];
        assert_eq!(speakers_of(&absorb_islands(&input)), speakers_of(&input));
    }

    #[test]
    fn a_long_alternating_chain_terminates() {
        let input: Vec<DiarizationSegment> = (0..400)
            .map(|i| diar(if i % 2 == 0 { "0" } else { "1" }, i * 900, (i + 1) * 900))
            .collect();
        let segs = absorb_islands(&input);
        assert_eq!(speakers_of(&segs), vec![("0", 0, 180000), ("1", 180000, 360000)]);
    }

    #[test]
    fn a_gapless_back_channel_is_absorbed_which_is_the_known_cost() {
        // Pinned on purpose: nothing in the diarization separates this from a
        // window flip, and the doc comment on `absorb_islands` says why losing
        // it was judged the cheaper error. Changing this is a policy change.
        let segs = align(
            &[
                stt(0, 4000, "So the budget is final."),
                stt(4000, 4600, "Yeah."),
                stt(4600, 9000, "Next item."),
            ],
            &[diar("0", 0, 4000), diar("1", 4000, 4600), diar("0", 4600, 9000)],
        );
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker, "SPEAKER_00");
    }

    #[test]
    fn align_no_longer_flips_speaker_mid_sentence_on_an_island() {
        let segs = align(
            &[stt(0, 9000, "one continuous sentence from a single speaker")],
            &[diar("2", 0, 4000), diar("0", 4000, 5000), diar("2", 5000, 9000)],
        );
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker, "SPEAKER_02");
    }

    #[test]
    fn a_fast_exchange_inside_one_segment_splits_at_the_word_boundary() {
        let segments = align(&[fast_exchange()], &two_speaker_exchange());
        assert_eq!(segments.len(), 2);

        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[0].text, "Yes exactly.");
        // The segment's own start is kept; the cut is the last word's end.
        assert_eq!((segments[0].start_ms, segments[0].end_ms), (0, 900));

        assert_eq!(segments[1].speaker, "SPEAKER_01");
        assert_eq!(segments[1].text, "No, we cannot.");
        // The cut is the first word of the new run; the segment's own end is
        // kept, so no audio falls between the two turns.
        assert_eq!((segments[1].start_ms, segments[1].end_ms), (2000, 4000));
    }

    #[test]
    fn the_same_exchange_without_word_timestamps_collapses_onto_one_speaker() {
        // What the pipeline did before word timestamps, and still does for the
        // server path and every `stt.json` written earlier. Speaker 1 holds
        // 2100 ms of the segment against speaker 0's 1000, so it takes all of
        // it — "Yes exactly." is attributed to the wrong person, which is the
        // limitation the test above fixes.
        let segments = align(
            &[SttSegment {
                words: Vec::new(),
                ..fast_exchange()
            }],
            &two_speaker_exchange(),
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "SPEAKER_01");
        assert_eq!(segments[0].text, "Yes exactly. No, we cannot.");
        assert_eq!((segments[0].start_ms, segments[0].end_ms), (0, 4000));
    }

    #[test]
    fn word_timestamps_never_reattribute_a_segment_they_do_not_split() {
        // Every word lands in speaker 0's interval, but the segment as a whole
        // overlaps speaker 1 more. An unsplit segment keeps the whole-segment
        // answer, so this can only ever add turns, never silently move one.
        let segment = SttSegment {
            words: vec![word("Just", 0, 200), word("checking.", 200, 500)],
            ..stt(0, 4000, "Just checking.")
        };
        let diarization = [diar("0", 0, 600), diar("1", 600, 4000)];
        assert_eq!(align(&[segment], &diarization)[0].speaker, "SPEAKER_01");
    }

    #[test]
    fn declines_to_split_when_the_words_do_not_spell_out_the_segment() {
        // A dropped token (whisper omits `offsets` on a token it cannot time)
        // leaves words that no longer reconstruct the text. Splitting here
        // would delete "we cannot." from the transcript, so it does not.
        let segment = SttSegment {
            words: vec![
                word("Yes", 0, 400),
                word("exactly.", 400, 900),
                word("No,", 2000, 2300),
            ],
            ..stt(0, 4000, "Yes exactly. No, we cannot.")
        };
        let segments = align(&[segment], &two_speaker_exchange());
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Yes exactly. No, we cannot.");
    }

    #[test]
    fn a_split_piece_still_merges_into_the_turn_before_it() {
        // The first half of the exchange belongs to the speaker already
        // talking, so it must extend that turn rather than start a new one —
        // splitting must not defeat turn merging.
        let segments = align(
            &[stt(0, 1500, "So anyway"), fast_exchange_at(1800)],
            &[diar("0", 0, 2800), diar("1", 3700, 5800)],
        );
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[0].text, "So anyway Yes exactly.");
        assert_eq!(segments[1].speaker, "SPEAKER_01");
        assert_eq!(segments[1].text, "No, we cannot.");
    }

    fn fast_exchange_at(offset_ms: i64) -> SttSegment {
        let base = fast_exchange();
        SttSegment {
            start_ms: base.start_ms + offset_ms,
            end_ms: base.end_ms + offset_ms,
            words: base
                .words
                .iter()
                .map(|w| word(&w.text, w.start_ms + offset_ms, w.end_ms + offset_ms))
                .collect(),
            ..base
        }
    }

    #[test]
    fn a_split_turn_keeps_the_language_of_the_segment_it_came_from() {
        let segment = SttSegment {
            language: Some("ta".to_string()),
            ..fast_exchange()
        };
        let segments = align(&[segment], &two_speaker_exchange());
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().all(|s| s.language.as_deref() == Some("ta")));
    }

    #[test]
    fn merging_language_never_invents_or_loses_one() {
        assert_eq!(merge_language(Some("ta"), Some("ta")).as_deref(), Some("ta"));
        assert_eq!(
            merge_language(Some("ta"), Some("en")).as_deref(),
            Some(MIXED_LANGUAGE)
        );
        // An untagged fragment says nothing, so it overrides nothing.
        assert_eq!(merge_language(Some("ta"), None).as_deref(), Some("ta"));
        assert_eq!(merge_language(None, Some("en")).as_deref(), Some("en"));
        assert_eq!(merge_language(None, None), None);
        // Already mixed stays mixed rather than reverting to the newcomer.
        assert_eq!(
            merge_language(Some(MIXED_LANGUAGE), Some("en")).as_deref(),
            Some(MIXED_LANGUAGE)
        );
    }

    // ------------------------------------------------------ minor clusters

    fn minor_merged(input: &[DiarizationSegment], audio_end_ms: i64) -> Vec<(String, i64, i64)> {
        merge_minor_clusters(absorb_islands(input), audio_end_ms)
            .into_iter()
            .map(|d| (d.speaker_id, d.start_ms, d.end_ms))
            .collect()
    }

    fn owned(expected: &[(&str, i64, i64)]) -> Vec<(String, i64, i64)> {
        expected.iter().map(|&(s, a, b)| (s.to_string(), a, b)).collect()
    }

    #[test]
    fn a_quiet_genuine_participant_is_kept() {
        // The council resident's shape: two turns, 4.9 s and 2.2 s, each
        // butted against the dominant voice with no pause. Little talk time
        // and no pause around it, but one real sentence is enough.
        let input = [
            diar("0", 0, 60_000),
            diar("1", 60_000, 64_900),
            diar("1", 65_700, 67_900),
            diar("0", 67_900, 90_000),
        ];
        assert_eq!(minor_merged(&input, 90_000), owned(&speakers_of(&input)));
    }

    #[test]
    fn a_short_remark_said_in_silence_is_kept() {
        // One 2 s "I'd rather we waited." with a pause either side: the only
        // thing this person says all meeting, and still their own.
        let input = [
            diar("0", 0, 30_000),
            diar("1", 31_000, 33_000),
            diar("0", 34_000, 60_000),
        ];
        assert_eq!(minor_merged(&input, 60_000), owned(&speakers_of(&input)));
    }

    #[test]
    fn someone_who_speaks_often_but_briefly_is_kept() {
        // Five 2 s replies, each butted against the main voice: every segment
        // is short, but 10 s in total is a participant, not a fragment.
        let mut input = vec![diar("0", 0, 10_000)];
        for k in 0..5 {
            let at = 10_000 + k * 12_000;
            input.push(diar("1", at, at + 2_000));
            input.push(diar("0", at + 2_000, at + 12_000));
        }
        assert_eq!(minor_merged(&input, 70_000), owned(&speakers_of(&input)));
    }

    #[test]
    fn first_words_split_off_at_the_start_of_the_recording_rejoin_the_voice() {
        // The council anchor: the clip opens mid-sentence, sherpa-onnx gives
        // the first 2.9 s a cluster of their own, and the same voice carries
        // on after a short pause. The recording's edge is not a pause.
        let segs = minor_merged(&[diar("5", 0, 2_900), diar("0", 3_700, 20_000)], 20_000);
        assert_eq!(segs, owned(&[("0", 0, 2_900), ("0", 3_700, 20_000)]));
    }

    #[test]
    fn a_scattered_fragment_cluster_returns_each_piece_to_its_neighbour() {
        // Four ~2 s fragments of different people sharing one id. Each goes
        // back to the voice it is butted against, not all to one of them.
        let segs = minor_merged(
            &[
                diar("0", 0, 10_000),
                diar("9", 10_000, 12_000),
                diar("1", 14_000, 30_000),
                diar("9", 30_000, 32_000),
                diar("2", 32_000, 45_000),
            ],
            45_000,
        );
        assert_eq!(
            segs,
            owned(&[("0", 0, 12_000), ("1", 14_000, 32_000), ("2", 32_000, 45_000)])
        );
    }

    #[test]
    fn a_short_isolated_singleton_is_folded_into_the_nearest_voice() {
        // The TED clip's opening 1.3 s, alone and followed by a long silence:
        // shorter than a remark the rule protects, so it joins the talker.
        let segs = minor_merged(&[diar("4", 3_800, 5_100), diar("0", 12_900, 40_000)], 40_000);
        assert_eq!(segs, owned(&[("0", 3_800, 5_100), ("0", 12_900, 40_000)]));
    }

    #[test]
    fn a_fragment_never_folds_into_someone_elses_short_remark() {
        // The nearest speech to the 1 s fragment is a quiet participant's
        // standalone remark. That remark is kept, but it is not a voice to
        // fold fragments into — the fragment goes to the main talker.
        let segs = minor_merged(
            &[
                diar("0", 0, 30_000),
                diar("1", 31_000, 33_000),
                diar("9", 33_400, 34_400),
                diar("0", 36_000, 60_000),
            ],
            60_000,
        );
        assert_eq!(
            segs,
            owned(&[
                ("0", 0, 30_000),
                ("1", 31_000, 33_000),
                ("0", 33_400, 34_400),
                ("0", 36_000, 60_000),
            ])
        );
    }

    #[test]
    fn nothing_is_merged_when_every_cluster_is_minor() {
        // A short recording of two people: no voice to fold into, and brevity
        // alone is not evidence that either cluster is spurious.
        let input = [diar("0", 0, 2_000), diar("1", 2_000, 4_000), diar("0", 4_000, 6_000)];
        assert_eq!(minor_merged(&input, 6_000), owned(&speakers_of(&absorb_islands(&input))));
    }

    #[test]
    fn a_fragment_overlapping_a_voice_joins_the_one_it_overlaps() {
        let segs = minor_merged(
            &[
                diar("0", 0, 20_000),
                diar("7", 19_000, 21_500),
                diar("1", 23_000, 40_000),
            ],
            40_000,
        );
        assert_eq!(segs, owned(&[("0", 0, 21_500), ("1", 23_000, 40_000)]));
    }

    #[test]
    fn align_removes_a_split_off_speaker_from_the_transcript() {
        let segments = align(
            &[stt(0, 2_800, "And tonight the"), stt(3_700, 12_000, "council voted.")],
            &[diar("5", 0, 2_900), diar("0", 3_700, 12_000)],
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[0].speaker_label, "Person 1");
    }
}

