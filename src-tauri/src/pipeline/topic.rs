//! What kind of meeting was this, and what was it about.
//!
//! Two answers, from the transcript only: a **meeting type** (a standup, a
//! planning session, a retrospective) and a set of **topic tags** (the terms
//! the meeting actually kept returning to). Both feed the history list, where
//! twenty meetings called "Weekly sync" are otherwise indistinguishable.
//!
//! Like the noise classifier, this is deterministic and lexical rather than
//! learned, and for a stronger reason here: the app already has a local LLM
//! loaded during MoM generation, and asking it "what kind of meeting is this"
//! would be one more prompt. That was rejected. An LLM answer is
//! non-deterministic, costs seconds of the processing budget, varies with
//! whichever model the memory tier picked — so the same meeting would classify
//! differently on a 16 GB laptop and a 32 GB one — and, worst of all, cannot
//! be reproduced when someone asks why a meeting was labelled the way it was.
//! Counting words is reproducible.
//!
//! **The honest limitation: type classification is cued in English.** The
//! keyword profiles below are English phrases, so a meeting conducted mostly
//! in Tamil will fall through to `General` with low confidence rather than
//! being mislabelled. Topic extraction has no such problem — it is pure term
//! frequency and works on whatever language the transcript is in. That split
//! is deliberate: a wrong label is worse than no label, and no Tamil audio has
//! been through this pipeline yet to build Tamil profiles against.

use crate::pipeline::models::Transcript;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A meeting shorter than this is a quick sync whatever else it is.
const SHORT_MEETING_MS: i64 = 20 * 60 * 1000;

/// Below this score, the evidence is too thin to name a type.
const MIN_TYPE_SCORE: f32 = 1.5;

/// Most topic tags to return. More than this and it stops being a summary.
const MAX_TOPICS: usize = 8;

/// A term has to appear in at least this many separate turns to be a topic.
/// One person saying "migration" nine times in one answer is not what the
/// meeting was about; four people mentioning it once each is.
const MIN_TOPIC_SPREAD: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeetingType {
    Standup,
    Planning,
    Review,
    Retrospective,
    Decision,
    Brainstorm,
    Interview,
    OneOnOne,
    Training,
    Troubleshooting,
    /// The honest answer when nothing else fits.
    General,
}

impl MeetingType {
    pub fn as_str(self) -> &'static str {
        match self {
            MeetingType::Standup => "standup",
            MeetingType::Planning => "planning",
            MeetingType::Review => "review",
            MeetingType::Retrospective => "retrospective",
            MeetingType::Decision => "decision",
            MeetingType::Brainstorm => "brainstorm",
            MeetingType::Interview => "interview",
            MeetingType::OneOnOne => "one_on_one",
            MeetingType::Training => "training",
            MeetingType::Troubleshooting => "troubleshooting",
            MeetingType::General => "general",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MeetingType::Standup => "Standup",
            MeetingType::Planning => "Planning",
            MeetingType::Review => "Review",
            MeetingType::Retrospective => "Retrospective",
            MeetingType::Decision => "Decision-making",
            MeetingType::Brainstorm => "Brainstorm",
            MeetingType::Interview => "Interview",
            MeetingType::OneOnOne => "One-on-one",
            MeetingType::Training => "Training or walkthrough",
            MeetingType::Troubleshooting => "Troubleshooting",
            MeetingType::General => "General discussion",
        }
    }

    /// Phrases that suggest this type, with how much each is worth. Longer
    /// phrases score higher because they are far less likely to be incidental.
    fn cues(self) -> &'static [(&'static str, f32)] {
        match self {
            MeetingType::Standup => &[
                ("yesterday i", 3.0),
                ("today i", 3.0),
                ("working on", 1.5),
                ("blocked on", 3.0),
                ("blocker", 2.5),
                ("no blockers", 3.5),
                ("stand up", 2.0),
                ("standup", 3.0),
                ("daily sync", 2.5),
                ("what i did", 2.5),
            ],
            MeetingType::Planning => &[
                ("next sprint", 3.0),
                ("sprint planning", 4.0),
                ("story points", 3.5),
                ("estimate", 1.5),
                ("estimation", 2.0),
                ("backlog", 2.5),
                ("roadmap", 2.0),
                ("milestone", 2.0),
                ("deadline", 1.5),
                ("timeline", 1.5),
                ("scope", 1.5),
                ("prioritise", 2.0),
                ("prioritize", 2.0),
                ("next quarter", 2.5),
            ],
            MeetingType::Review => &[
                ("code review", 3.5),
                ("walk through the", 2.0),
                ("looks good to me", 2.5),
                ("feedback on", 2.0),
                ("review the", 2.0),
                ("demo", 2.0),
                ("pull request", 3.0),
                ("this pr", 2.5),
                ("sign off", 2.0),
                ("approve", 1.5),
            ],
            MeetingType::Retrospective => &[
                ("went well", 3.5),
                ("did not go well", 3.5),
                ("what went wrong", 3.5),
                ("retro", 3.0),
                ("retrospective", 4.0),
                ("lessons learned", 3.5),
                ("next time we", 2.5),
                ("improve on", 2.0),
                ("action for next", 2.0),
            ],
            MeetingType::Decision => &[
                ("we decided", 3.5),
                ("decision is", 3.0),
                ("let us go with", 2.5),
                ("lets go with", 2.5),
                ("option a", 2.0),
                ("option b", 2.0),
                ("pros and cons", 3.0),
                ("i propose", 2.5),
                ("vote", 2.0),
                ("agreed", 1.5),
                ("final call", 2.5),
                ("approval", 1.5),
            ],
            MeetingType::Brainstorm => &[
                ("what if we", 2.5),
                ("idea", 1.5),
                ("ideas", 1.5),
                ("brainstorm", 4.0),
                ("throw out", 1.5),
                ("crazy idea", 3.0),
                ("another option", 2.0),
                ("we could try", 2.0),
            ],
            MeetingType::Interview => &[
                ("tell me about yourself", 4.0),
                ("your experience", 2.5),
                ("previous role", 3.0),
                ("why do you want", 3.0),
                ("do you have any questions", 3.0),
                ("your background", 2.5),
                ("cv", 1.5),
                ("resume", 1.5),
                ("candidate", 2.5),
            ],
            MeetingType::OneOnOne => &[
                ("how are you finding", 3.0),
                ("career", 2.0),
                ("your goals", 2.5),
                ("feedback for you", 3.0),
                ("one on one", 3.5),
                ("check in on you", 3.0),
                ("promotion", 2.0),
            ],
            MeetingType::Training => &[
                ("as you can see", 2.0),
                ("let me show you", 3.0),
                ("first step", 2.0),
                ("next step is", 2.0),
                ("walkthrough", 2.5),
                ("training", 2.5),
                ("onboarding", 3.0),
                ("documentation", 1.5),
                ("any questions so far", 3.0),
            ],
            MeetingType::Troubleshooting => &[
                ("root cause", 3.5),
                ("the error", 2.0),
                ("stack trace", 3.0),
                ("reproduce", 2.5),
                ("outage", 3.0),
                ("incident", 2.5),
                ("failing", 1.5),
                ("bug", 1.5),
                ("logs", 1.5),
                ("rollback", 2.5),
                ("hotfix", 3.0),
            ],
            MeetingType::General => &[],
        }
    }

    fn all() -> [MeetingType; 10] {
        [
            MeetingType::Standup,
            MeetingType::Planning,
            MeetingType::Review,
            MeetingType::Retrospective,
            MeetingType::Decision,
            MeetingType::Brainstorm,
            MeetingType::Interview,
            MeetingType::OneOnOne,
            MeetingType::Training,
            MeetingType::Troubleshooting,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeScore {
    #[serde(rename = "type")]
    pub meeting_type: MeetingType,
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicTag {
    pub term: String,
    /// Times the term appears across the whole transcript.
    pub count: usize,
    /// How many separate turns it appears in — the measure that stops one
    /// long monologue from setting the agenda.
    pub spread: usize,
    pub score: f32,
}

/// The structural facts the classifier used, reported so a label can be
/// argued with rather than just believed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeetingSignals {
    pub duration_ms: i64,
    pub speaker_count: usize,
    pub turn_count: usize,
    pub word_count: usize,
    pub mean_turn_words: f32,
    /// Share of turns that are questions, 0.0..1.0.
    pub question_ratio: f32,
    /// Share of all words spoken by the most talkative speaker.
    pub dominant_speaker_share: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeetingClassification {
    #[serde(rename = "type")]
    pub meeting_type: String,
    pub label: String,
    /// 0.0..1.0, from the winning score and its margin over the runner-up.
    /// Low confidence with a named type means "probably, on thin evidence".
    pub confidence: f32,
    /// Every type that scored anything, best first — including the winner.
    pub alternatives: Vec<TypeScore>,
    pub topics: Vec<TopicTag>,
    pub signals: MeetingSignals,
}

impl MeetingClassification {
    /// The type as an enum again, for callers that want to match on it.
    pub fn parsed_type(&self) -> MeetingType {
        MeetingType::all()
            .into_iter()
            .find(|t| t.as_str() == self.meeting_type)
            .unwrap_or(MeetingType::General)
    }
}

/// Words that carry no topic. English, plus the romanized Tamil discourse
/// particles that dominate a Tanglish transcript and would otherwise be the
/// top "topic" of every meeting.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because", "as", "of", "at", "by",
    "for", "with", "about", "into", "through", "to", "from", "in", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "than", "too", "very", "can", "will", "just", "should", "now", "i", "me", "my",
    "we", "our", "us", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "shall", "may", "might", "must", "get", "got", "go", "going", "went", "know", "think",
    "like", "want", "need", "see", "look", "say", "said", "says", "come", "came", "take", "make",
    "made", "one", "two", "three", "first", "last", "next", "good", "right", "well", "okay", "ok",
    "yeah", "yes", "no", "actually", "basically", "really", "maybe", "something", "anything",
    "everything", "nothing", "thing", "things", "lot", "bit", "kind", "sort", "sure", "let",
    "lets", "thanks", "thank", "please", "sorry", "hello", "hi", "bye", "um", "uh", "hmm", "mm",
    // Romanized Tamil discourse words.
    "naan", "nee", "neenga", "avan", "aval", "avanga", "naanga", "namma", "enna", "epdi", "eppadi",
    "appo", "appuram", "aprm", "seri", "sari", "illa", "illai", "irukku", "irukkum", "iruku",
    "pannunga", "panna", "pannu", "solli", "sollu", "sollunga", "vandhu", "vanthu", "than",
    "thaan", "dhan", "da", "di", "ah", "aa", "oru", "onnu", "rendu", "romba", "konjam", "pola",
    "mathiri", "innum", "ippo", "ipdi", "athu", "idhu", "ithu", "adhu", "enaku", "unaku", "ungaluku",
];

/// Words that make a turn a question even without a question mark, which
/// whisper.cpp often does not produce.
const QUESTION_OPENERS: &[&str] = &[
    "what", "why", "how", "when", "where", "who", "which", "did", "do", "does", "can", "could",
    "should", "would", "will", "is", "are", "was", "were", "have", "has", "any",
];

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.trim_matches('\'').to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

fn is_question(text: &str, tokens: &[String]) -> bool {
    if text.contains('?') {
        return true;
    }
    tokens
        .first()
        .map(|first| QUESTION_OPENERS.contains(&first.as_str()))
        .unwrap_or(false)
}

/// Scores each meeting type from its cue phrases.
///
/// Cues are counted over the whole transcript as one lowercased string, then
/// divided by the square root of the turn count. Without that, a two-hour
/// meeting outscores a ten-minute one on every type simply by having more
/// words in it; the square root damps length without erasing the fact that
/// more evidence is more evidence.
fn score_cues(joined: &str, turn_count: usize) -> Vec<TypeScore> {
    let damping = (turn_count.max(1) as f32).sqrt();
    let mut scores: Vec<TypeScore> = MeetingType::all()
        .into_iter()
        .map(|meeting_type| {
            let raw: f32 = meeting_type
                .cues()
                .iter()
                .map(|(phrase, weight)| joined.matches(phrase).count() as f32 * weight)
                .sum();
            TypeScore {
                meeting_type,
                label: meeting_type.label().to_string(),
                score: raw / damping,
            }
        })
        .filter(|entry| entry.score > 0.0)
        .collect();
    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

/// Nudges the cue scores with the shape of the conversation.
///
/// These are multipliers rather than additions, so structure can only support
/// or undercut what the words already said — never invent a type on its own.
/// A two-person meeting is not a one-on-one just because it has two people in
/// it; it is a one-on-one if it *also* sounds like one.
fn apply_structure(scores: &mut [TypeScore], signals: &MeetingSignals) {
    for entry in scores.iter_mut() {
        let multiplier = match entry.meeting_type {
            MeetingType::Standup => {
                let mut m = 1.0;
                if signals.duration_ms > 0 && signals.duration_ms <= SHORT_MEETING_MS {
                    m *= 1.4;
                }
                if signals.speaker_count >= 3 {
                    m *= 1.2;
                }
                if signals.mean_turn_words > 60.0 {
                    m *= 0.6; // standups are not monologues
                }
                m
            }
            MeetingType::OneOnOne => {
                if signals.speaker_count == 2 {
                    1.5
                } else {
                    0.4
                }
            }
            MeetingType::Interview => {
                let mut m = if signals.speaker_count == 2 { 1.4 } else { 0.6 };
                if signals.question_ratio > 0.25 {
                    m *= 1.3;
                }
                m
            }
            MeetingType::Training => {
                // One voice doing most of the talking.
                if signals.dominant_speaker_share > 0.65 {
                    1.5
                } else {
                    0.7
                }
            }
            MeetingType::Brainstorm => {
                if signals.speaker_count >= 3 {
                    1.2
                } else {
                    0.8
                }
            }
            _ => 1.0,
        };
        entry.score *= multiplier;
    }
    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Term frequency crossed with how many turns a term appears in.
///
/// Bigrams are scored alongside single words and given a bonus, because
/// "release train" is a better topic tag than "release" and "train"
/// separately. A bigram that survives suppresses its own halves.
fn extract_topics(turns: &[Vec<String>]) -> Vec<TopicTag> {
    let stop: HashSet<&str> = STOPWORDS.iter().copied().collect();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut spread: HashMap<String, usize> = HashMap::new();

    for tokens in turns {
        let mut seen: HashSet<String> = HashSet::new();
        for (index, word) in tokens.iter().enumerate() {
            // Single characters and pure numbers are never topics.
            if word.chars().count() < 3 || stop.contains(word.as_str()) {
                continue;
            }
            if word.chars().all(|c| c.is_numeric()) {
                continue;
            }
            *counts.entry(word.clone()).or_default() += 1;
            seen.insert(word.clone());

            if let Some(following) = tokens.get(index + 1) {
                if following.chars().count() >= 3
                    && !stop.contains(following.as_str())
                    && !following.chars().all(|c| c.is_numeric())
                {
                    let bigram = format!("{word} {following}");
                    *counts.entry(bigram.clone()).or_default() += 1;
                    seen.insert(bigram);
                }
            }
        }
        for term in seen {
            *spread.entry(term).or_default() += 1;
        }
    }

    let mut tags: Vec<TopicTag> = counts
        .into_iter()
        .filter_map(|(term, count)| {
            let spread = spread.get(&term).copied().unwrap_or(0);
            if spread < MIN_TOPIC_SPREAD {
                return None;
            }
            // Two-word terms are worth more per occurrence: they are much
            // less likely to be coincidence.
            let bonus = if term.contains(' ') { 1.6 } else { 1.0 };
            Some(TopicTag {
                score: count as f32 * (spread as f32).sqrt() * bonus,
                term,
                count,
                spread,
            })
        })
        .collect();

    tags.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.term.cmp(&b.term))
    });

    // Drop a single word that a surviving bigram already contains, so the
    // tag list is not "deployment, pipeline, deployment pipeline".
    let mut kept: Vec<TopicTag> = Vec::new();
    for tag in tags {
        let covered = kept.iter().any(|existing| {
            existing.term.contains(' ')
                && !tag.term.contains(' ')
                && existing
                    .term
                    .split(' ')
                    .any(|part| part == tag.term)
        });
        if !covered {
            kept.push(tag);
        }
        if kept.len() >= MAX_TOPICS {
            break;
        }
    }
    kept
}

/// Classifies a transcript.
pub fn classify(transcript: &Transcript) -> MeetingClassification {
    let turns: Vec<Vec<String>> = transcript
        .segments
        .iter()
        .map(|segment| words(&segment.text))
        .collect();

    let word_count: usize = turns.iter().map(|t| t.len()).sum();
    let mut per_speaker: BTreeMap<&str, usize> = BTreeMap::new();
    let mut questions = 0usize;
    for (segment, tokens) in transcript.segments.iter().zip(turns.iter()) {
        *per_speaker.entry(segment.speaker.as_str()).or_default() += tokens.len();
        if is_question(&segment.text, tokens) {
            questions += 1;
        }
    }

    let turn_count = transcript.segments.len();
    let signals = MeetingSignals {
        duration_ms: transcript.duration_ms(),
        speaker_count: per_speaker.len(),
        turn_count,
        word_count,
        mean_turn_words: if turn_count == 0 {
            0.0
        } else {
            word_count as f32 / turn_count as f32
        },
        question_ratio: if turn_count == 0 {
            0.0
        } else {
            questions as f32 / turn_count as f32
        },
        dominant_speaker_share: if word_count == 0 {
            0.0
        } else {
            per_speaker.values().max().copied().unwrap_or(0) as f32 / word_count as f32
        },
    };

    let topics = extract_topics(&turns);

    // Too little was said to classify anything. Reporting General with zero
    // confidence is the correct answer here, not a guess.
    if word_count < 20 {
        return MeetingClassification {
            meeting_type: MeetingType::General.as_str().to_string(),
            label: MeetingType::General.label().to_string(),
            confidence: 0.0,
            alternatives: Vec::new(),
            topics,
            signals,
        };
    }

    let joined: String = transcript
        .segments
        .iter()
        .map(|s| s.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    let mut scores = score_cues(&joined, turn_count);
    apply_structure(&mut scores, &signals);

    let best = scores.first();
    let (meeting_type, confidence) = match best {
        Some(entry) if entry.score >= MIN_TYPE_SCORE => {
            let runner_up = scores.get(1).map(|e| e.score).unwrap_or(0.0);
            // Confidence is how far past the bar it got, tempered by how
            // clearly it beat the next candidate. A meeting that scores high
            // on two types at once is genuinely ambiguous and should say so.
            let strength = ((entry.score - MIN_TYPE_SCORE) / (MIN_TYPE_SCORE * 3.0)).clamp(0.0, 1.0);
            let margin = if entry.score > 0.0 {
                ((entry.score - runner_up) / entry.score).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (entry.meeting_type, (0.4 + 0.6 * strength) * (0.5 + 0.5 * margin))
        }
        _ => (MeetingType::General, 0.0),
    };

    MeetingClassification {
        meeting_type: meeting_type.as_str().to_string(),
        label: meeting_type.label().to_string(),
        confidence,
        alternatives: scores,
        topics,
        signals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::models::TranscriptSegment;

    fn transcript(turns: &[(&str, &str)]) -> Transcript {
        Transcript {
            meeting_id: "test".to_string(),
            segments: turns
                .iter()
                .enumerate()
                .map(|(index, (speaker, text))| TranscriptSegment {
                    speaker: speaker.to_string(),
                    speaker_label: speaker.to_string(),
                    start_ms: index as i64 * 10_000,
                    end_ms: index as i64 * 10_000 + 9_000,
                    text: text.to_string(),
                    language: None,
                    flags: Vec::new(),
                })
                .collect(),
        }
    }

    fn classify_type(turns: &[(&str, &str)]) -> String {
        classify(&transcript(turns)).meeting_type
    }

    #[test]
    fn a_standup_is_recognised() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Yesterday I finished the login screen. Today I am working on the export dialog. No blockers."),
            ("SPEAKER_01", "Yesterday I was working on the migration script. Today I will continue. I am blocked on the staging database."),
            ("SPEAKER_02", "Today I am picking up the search bug. No blockers from me."),
        ]));
        assert_eq!(result.meeting_type, "standup");
        assert!(result.confidence > 0.3, "confidence {}", result.confidence);
    }

    #[test]
    fn a_retrospective_is_not_confused_with_a_review() {
        let result = classify_type(&[
            ("SPEAKER_00", "Let us start the retrospective. What went well this sprint?"),
            ("SPEAKER_01", "The deployment went well. Testing did not go well at all."),
            ("SPEAKER_02", "Lessons learned: next time we should freeze the scope earlier."),
        ]);
        assert_eq!(result, "retrospective");
    }

    #[test]
    fn a_planning_session_is_recognised() {
        let result = classify_type(&[
            ("SPEAKER_00", "This is sprint planning for next sprint. Let us look at the backlog."),
            ("SPEAKER_01", "I would put story points of five on the migration. The estimate for the rest is smaller."),
            ("SPEAKER_00", "We should prioritise the roadmap milestone before the deadline."),
        ]);
        assert_eq!(result, "planning");
    }

    #[test]
    fn an_incident_call_is_recognised_as_troubleshooting() {
        let result = classify_type(&[
            ("SPEAKER_00", "We have an outage. The error started at nine and the logs are full of failures."),
            ("SPEAKER_01", "I can reproduce it. The stack trace points at the cache layer."),
            ("SPEAKER_00", "Root cause looks like the config change. Let us do a rollback and then a hotfix."),
        ]);
        assert_eq!(result, "troubleshooting");
    }

    #[test]
    fn a_lecture_by_one_person_is_training_not_a_standup() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Let me show you how the onboarding flow works. As you can see the first step is authentication, and the next step is the profile screen. I will walk through the documentation for each part of this and show you where the walkthrough lives. Any questions so far?"),
            ("SPEAKER_01", "No, that is clear."),
        ]));
        assert_eq!(result.meeting_type, "training");
        assert!(result.signals.dominant_speaker_share > 0.65);
    }

    #[test]
    fn structure_alone_never_invents_a_type() {
        // Two speakers and nothing that sounds like a one-on-one. The
        // structural multiplier can support a cue score, never create one.
        let result = classify_type(&[
            ("SPEAKER_00", "The build server is in the rack behind the second door on the left of the corridor."),
            ("SPEAKER_01", "Understood, I will find it later this afternoon when the room is free again."),
        ]);
        assert_eq!(result, "general");
    }

    #[test]
    fn an_unclassifiable_meeting_says_general_with_no_confidence() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "The weather outside is quite pleasant today and the coffee machine on our floor is finally working again after several weeks."),
            ("SPEAKER_01", "That is good to hear about the coffee machine because the queue downstairs was getting rather long every morning."),
        ]));
        assert_eq!(result.meeting_type, "general");
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn a_transcript_with_almost_no_speech_is_not_classified() {
        let result = classify(&transcript(&[("SPEAKER_00", "Hello. Yes.")]));
        assert_eq!(result.meeting_type, "general");
        assert_eq!(result.confidence, 0.0);
        assert!(result.alternatives.is_empty());
    }

    #[test]
    fn an_empty_transcript_is_handled() {
        let result = classify(&transcript(&[]));
        assert_eq!(result.meeting_type, "general");
        assert_eq!(result.signals.turn_count, 0);
        assert_eq!(result.signals.speaker_count, 0);
        assert!(result.topics.is_empty());
    }

    #[test]
    fn topics_come_from_what_the_meeting_kept_returning_to() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "The database migration is the main risk this week."),
            ("SPEAKER_01", "Agreed, the database migration needs a rehearsal first."),
            ("SPEAKER_02", "I will schedule the database migration rehearsal for Thursday."),
        ]));
        let terms: Vec<&str> = result.topics.iter().map(|t| t.term.as_str()).collect();
        assert!(
            terms.contains(&"database migration"),
            "topics were {terms:?}"
        );
    }

    #[test]
    fn a_bigram_suppresses_its_own_halves() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "The deployment pipeline is broken."),
            ("SPEAKER_01", "The deployment pipeline was broken yesterday too."),
            ("SPEAKER_02", "We should fix the deployment pipeline."),
        ]));
        let terms: Vec<&str> = result.topics.iter().map(|t| t.term.as_str()).collect();
        assert!(terms.contains(&"deployment pipeline"), "{terms:?}");
        assert!(!terms.contains(&"deployment"), "{terms:?}");
        assert!(!terms.contains(&"pipeline"), "{terms:?}");
    }

    #[test]
    fn one_persons_repetition_does_not_become_the_topic() {
        // Said nine times, but only ever in one turn: not what the meeting
        // was about.
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Kubernetes kubernetes kubernetes kubernetes kubernetes kubernetes kubernetes kubernetes kubernetes."),
            ("SPEAKER_01", "The invoice needs approval before Friday."),
            ("SPEAKER_02", "I will send the invoice today for approval."),
        ]));
        let terms: Vec<&str> = result.topics.iter().map(|t| t.term.as_str()).collect();
        assert!(!terms.contains(&"kubernetes"), "{terms:?}");
        assert!(terms.contains(&"invoice"), "{terms:?}");
    }

    #[test]
    fn stopwords_and_filler_never_become_topics() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Yeah okay so basically the thing is that we should actually just do it."),
            ("SPEAKER_01", "Yeah okay basically the thing is really that we could actually just do it."),
        ]));
        let terms: Vec<&str> = result.topics.iter().map(|t| t.term.as_str()).collect();
        for filler in ["yeah", "okay", "basically", "thing", "actually", "just"] {
            assert!(!terms.contains(&filler), "{filler} became a topic: {terms:?}");
        }
    }

    #[test]
    fn tamil_discourse_particles_are_not_topics() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Naan enna panna, appo seri, invoice approval venum."),
            ("SPEAKER_01", "Seri naan appo invoice approval pannuren."),
        ]));
        let terms: Vec<&str> = result.topics.iter().map(|t| t.term.as_str()).collect();
        for particle in ["naan", "enna", "appo", "seri", "panna"] {
            assert!(!terms.contains(&particle), "{particle} became a topic: {terms:?}");
        }
        assert!(terms.iter().any(|t| t.contains("invoice")), "{terms:?}");
    }

    #[test]
    fn a_tamil_meeting_falls_through_to_general_rather_than_being_mislabelled() {
        // The stated limitation: the type cues are English, so a Tamil
        // meeting must decline to guess rather than pick a type at random.
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Naanga inniki project status pathi pesanum, ellarum ready ah irukeengala nu paapom."),
            ("SPEAKER_01", "Seri, naan en pakkam ellam mudichiten, innum konjam vela iruku."),
            ("SPEAKER_02", "Enakku innum rendu naal aagum, appuram project status update pannuren ungaluku."),
        ]));
        assert_eq!(result.meeting_type, "general");
        // Topics still work: they are pure term frequency, so the one thing
        // this meeting kept coming back to surfaces regardless of language.
        let terms: Vec<&str> = result.topics.iter().map(|t| t.term.as_str()).collect();
        assert!(terms.contains(&"project status"), "{terms:?}");
    }

    #[test]
    fn signals_describe_the_conversation() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "What do you think about the plan for the release?"),
            ("SPEAKER_01", "It looks reasonable to me overall."),
            ("SPEAKER_00", "How long will the release take to finish?"),
        ]));
        assert_eq!(result.signals.speaker_count, 2);
        assert_eq!(result.signals.turn_count, 3);
        assert!(result.signals.question_ratio > 0.6);
        assert!(result.signals.mean_turn_words > 5.0);
        assert!(result.signals.duration_ms > 0);
    }

    #[test]
    fn a_question_without_a_question_mark_still_counts() {
        // whisper.cpp often does not punctuate questions.
        let result = classify(&transcript(&[
            ("SPEAKER_00", "What did we agree on last week"),
            ("SPEAKER_01", "The migration plan was agreed"),
        ]));
        assert!(result.signals.question_ratio >= 0.5);
    }

    #[test]
    fn alternatives_are_ordered_and_include_the_winner() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "This is sprint planning. Let us estimate the backlog and set the milestone."),
            ("SPEAKER_01", "We decided to go with option a, and the decision is final."),
        ]));
        assert!(result.alternatives.len() >= 2);
        assert_eq!(result.alternatives[0].meeting_type.as_str(), result.meeting_type);
        for pair in result.alternatives.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn an_ambiguous_meeting_reports_lower_confidence_than_a_clear_one() {
        let clear = classify(&transcript(&[
            ("SPEAKER_00", "Retrospective time. What went well, and what went wrong?"),
            ("SPEAKER_01", "The release went well. Lessons learned: next time we test earlier."),
            ("SPEAKER_02", "Retro action: improve on the handover."),
        ]));
        let ambiguous = classify(&transcript(&[
            ("SPEAKER_00", "Sprint planning and also a code review of this pr, plus we decided on option a."),
            ("SPEAKER_01", "The estimate is fine and the pull request looks good to me."),
        ]));
        assert!(
            clear.confidence > ambiguous.confidence,
            "clear {} vs ambiguous {}",
            clear.confidence,
            ambiguous.confidence
        );
    }

    #[test]
    fn the_classification_round_trips_through_json() {
        let result = classify(&transcript(&[
            ("SPEAKER_00", "Yesterday I finished the import. Today I am on the export. No blockers."),
            ("SPEAKER_01", "Today I am working on the parser. I am blocked on the schema."),
        ]));
        let json = serde_json::to_string(&result).unwrap();
        let back: MeetingClassification = serde_json::from_str(&json).unwrap();
        assert_eq!(back, result);
        assert_eq!(back.parsed_type(), MeetingType::Standup);
        // The serialised key is "type", not "meeting_type".
        assert!(json.contains("\"type\":\"standup\""));
    }

    #[test]
    fn every_type_has_a_name_and_a_label() {
        for meeting_type in MeetingType::all() {
            assert!(!meeting_type.as_str().is_empty());
            assert!(!meeting_type.label().is_empty());
            assert!(!meeting_type.cues().is_empty());
        }
        assert!(MeetingType::General.cues().is_empty());
    }
}
