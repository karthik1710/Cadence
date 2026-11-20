//! Export: Markdown and JSON in V1.
//!
//! Exports are written into the meeting's own `exports/` directory rather than
//! anywhere outside it, so everything about a meeting stays in one place and
//! deleting the meeting deletes its exports too.
//!
//! This is also the file where getting redaction wrong actually costs
//! something. Everywhere else a mask is cosmetic — the original text is one
//! setting away, on the same machine, for the same user. An export is a file
//! that leaves: it gets mailed, pasted into a ticket, dropped in a shared
//! folder. So the rule here is stricter than "render through
//! [`redact::apply`]": at `mask` or `redact`, the original word must not appear
//! **anywhere** in the output — not in the transcript, not in a minutes
//! evidence quote, and not in a JSON field a reader would not think to check.
//! [`Redactor`] below is the single choke point that enforces it, and
//! `masking_leaves_no_trace_of_the_original_word` is the test that holds it.

use crate::pipeline::audio_profile::AudioProfile;
use crate::pipeline::models::{Evidenced, MeetingMetadata, Mom, Transcript};
use crate::pipeline::redact::{self, Lexicon, RedactionMode, ScanOptions};
use crate::pipeline::topic::MeetingClassification;
use crate::settings;
use crate::storage;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Markdown,
    Json,
}

impl Format {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "markdown" | "md" => Some(Format::Markdown),
            "json" => Some(Format::Json),
            _ => None,
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Format::Markdown => "md",
            Format::Json => "json",
        }
    }
}

/// The redaction policy an export renders under, plus everything needed to
/// enforce it.
///
/// It carries the lexicon rather than trusting the flags already stored on the
/// transcript, because those flags were written by whichever scan last ran and
/// an export may be taken long after — after the user edited the lexicon, after
/// a rescan that never happened, or from a `transcript.json` written before
/// flags existed at all. Stale flags on a *cosmetic* surface are a nuisance;
/// stale flags on a file that leaves the machine are the leak. So at `mask` and
/// `redact` every string is rescanned here and the two flag sets are unioned:
/// `redact::apply` skips overlaps, so a span found twice is masked once.
pub struct Redactor {
    mode: RedactionMode,
    lexicon: Lexicon,
    options: ScanOptions,
}

impl Redactor {
    /// Renders everything verbatim. What a caller with no settings to consult
    /// gets, and what `off` and `flag` both amount to at export time — `flag`
    /// is a highlight in the UI, and a Markdown file has nowhere to put one.
    pub fn none() -> Self {
        Self {
            mode: RedactionMode::Off,
            lexicon: Lexicon::from_sources(&[]),
            options: ScanOptions::default(),
        }
    }

    pub fn new(mode: RedactionMode, fuzzy: bool) -> Self {
        if !mode.hides_text() {
            return Self::none();
        }
        Self {
            mode,
            lexicon: Lexicon::builtin(),
            options: ScanOptions { fuzzy },
        }
    }

    /// The policy in force for this machine, including the user's own lexicon
    /// edits — a term they removed as a false positive must not come back at
    /// export time.
    pub fn load(data_dir: &Path) -> Self {
        let settings = settings::load(data_dir);
        let mode = settings.effective_redaction_mode();
        if !mode.hides_text() {
            return Self::none();
        }
        Self {
            mode,
            lexicon: Lexicon::load(data_dir),
            options: ScanOptions {
                fuzzy: settings.redaction_fuzzy,
            },
        }
    }

    fn hides_text(&self) -> bool {
        self.mode.hides_text()
    }

    /// Text that already carries flags — a transcript segment.
    fn segment(&self, text: &str, flags: &[redact::Flag]) -> String {
        if !self.hides_text() {
            return text.to_string();
        }
        let mut all = flags.to_vec();
        all.extend(redact::scan(text, &self.lexicon, &self.options));
        redact::apply(text, &all, self.mode)
    }

    /// Text with no flags of its own — a minutes summary, an action item, an
    /// evidence quote. Evidence quotes in particular *are* transcript text,
    /// lifted by the LLM, so they need exactly the same treatment.
    fn text(&self, text: &str) -> String {
        if !self.hides_text() {
            return text.to_string();
        }
        let flags = redact::scan(text, &self.lexicon, &self.options);
        redact::apply(text, &flags, self.mode)
    }

    /// Flags survive a masked export — their offsets, categories, severities
    /// and match methods are metadata about *where* something was hidden, and
    /// dropping them would cost the reader any sign that anything was.
    ///
    /// `term_id` is the exception, and it is the trap. It is documented on
    /// [`redact::Flag`] as "the lexicon entry that matched, spelled as the
    /// policy author wrote it" — which is to say it *is* the word. Shipping it
    /// beside a masked turn would hand the reader back exactly what the mask
    /// took away, from a field nobody thinks to look at. So it is masked by the
    /// same rule as the text: `d***` still identifies the entry to anyone
    /// holding the lexicon, and reprints it to nobody who is not.
    fn hidden_flags(&self, flags: &[redact::Flag]) -> Vec<redact::Flag> {
        flags
            .iter()
            .map(|flag| {
                let whole = redact::Flag {
                    start_char: 0,
                    end_char: flag.term_id.chars().count(),
                    ..flag.clone()
                };
                redact::Flag {
                    term_id: redact::apply(&flag.term_id, &[whole], self.mode),
                    ..flag.clone()
                }
            })
            .collect()
    }

    fn transcript(&self, transcript: &Transcript) -> Transcript {
        if !self.hides_text() {
            return transcript.clone();
        }
        Transcript {
            meeting_id: transcript.meeting_id.clone(),
            segments: transcript
                .segments
                .iter()
                .map(|segment| crate::pipeline::models::TranscriptSegment {
                    text: self.segment(&segment.text, &segment.flags),
                    flags: self.hidden_flags(&segment.flags),
                    ..segment.clone()
                })
                .collect(),
        }
    }

    fn mom(&self, mom: &Mom) -> Mom {
        if !self.hides_text() {
            return mom.clone();
        }
        let evidenced = |items: &[Evidenced]| -> Vec<Evidenced> {
            items
                .iter()
                .map(|item| Evidenced {
                    text: self.text(&item.text),
                    ..item.clone()
                })
                .collect()
        };
        Mom {
            // The summary and the detailed discussion are model prose, but the
            // model is quoting a transcript it was given verbatim, so they can
            // and do echo it. Scanned for the same reason the quotes are.
            summary: self.text(&mom.summary),
            participants: mom.participants.clone(),
            topics: evidenced(&mom.topics),
            decisions: evidenced(&mom.decisions),
            action_items: mom
                .action_items
                .iter()
                .map(|item| crate::pipeline::models::ActionItem {
                    description: self.text(&item.description),
                    ..item.clone()
                })
                .collect(),
            open_questions: evidenced(&mom.open_questions),
            risks: evidenced(&mom.risks),
            detailed_discussion: evidenced(&mom.detailed_discussion),
            meta: mom.meta.clone(),
        }
    }
}

/// The optional derived artifacts a Markdown export folds in as sections.
///
/// All three are `Option` because all three are recomputable side-cars: a
/// meeting recorded before these stages existed simply has none of them, and
/// its export is the export it always was rather than an error.
#[derive(Debug, Clone, Default)]
pub struct Extras {
    pub classification: Option<MeetingClassification>,
    pub audio_profile: Option<AudioProfile>,
    pub redaction: Option<crate::pipeline::redact::RedactionSummary>,
}

impl Extras {
    pub fn load(data_dir: &Path, meeting_id: &str) -> Self {
        Self {
            classification: storage::read_classification(data_dir, meeting_id),
            audio_profile: storage::read_audio_profile(data_dir, meeting_id),
            redaction: storage::read_redaction_summary(data_dir, meeting_id),
        }
    }
}

/// Renders and writes an export, returning the file it wrote.
pub fn write_export(
    data_dir: &Path,
    meeting_id: &str,
    format: Format,
) -> Result<PathBuf, String> {
    let meta = storage::read_metadata(data_dir, meeting_id)?;
    let transcript = storage::read_transcript(data_dir, meeting_id).ok();
    let mom = storage::read_mom(data_dir, meeting_id).ok();

    if transcript.is_none() && mom.is_none() {
        return Err(
            "this meeting has no transcript or minutes yet, so there is nothing to export"
                .to_string(),
        );
    }

    let redactor = Redactor::load(data_dir);
    let extras = Extras::load(data_dir, meeting_id);
    let contents = match format {
        Format::Markdown => render_markdown(
            &meta,
            mom.as_ref(),
            transcript.as_ref(),
            &extras,
            &redactor,
        ),
        Format::Json => render_json(&meta, mom.as_ref(), transcript.as_ref(), &redactor)?,
    };

    let dir = storage::meeting_dir(data_dir, meeting_id)?.join("exports");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!(
        "{}-{}.{}",
        slug(&meta.display_title()),
        &meta.id.chars().take(8).collect::<String>(),
        format.extension()
    ));
    storage::reject_link(&path)?;
    std::fs::write(&path, contents)
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path)
}

pub fn render_json(
    meta: &MeetingMetadata,
    mom: Option<&Mom>,
    transcript: Option<&Transcript>,
    redactor: &Redactor,
) -> Result<String, String> {
    // Redacted *before* serialisation, not after: there is no field in this
    // document that may carry the original text, so the safe construction is
    // one where the original never enters the value at all.
    let transcript = transcript.map(|t| redactor.transcript(t));
    let mom = mom.map(|m| redactor.mom(m));
    let value = serde_json::json!({
        "meeting": {
            "id": meta.id,
            "title": meta.display_title(),
            "notes": meta.notes,
            "createdAt": format_utc(meta.created_at_ms),
            "createdAtMs": meta.created_at_ms,
            "durationMs": meta.duration_ms,
            "state": meta.state.as_str(),
            "speakerNames": meta.speaker_names,
        },
        // The policy *this export* rendered under. Deliberately not spelled
        // `redactionMode`: `mom.meta.redaction_mode` beside it is the policy
        // the minutes were *generated* under, and the two are different facts
        // that can legitimately disagree — minutes generated with redaction
        // off and exported under `mask` have evidence quotes the model wrote
        // verbatim and that were masked on the way out. One name for both
        // would invite a reader to treat either as the whole story.
        "exportRedactionMode": redactor.mode.as_str(),
        "mom": mom,
        "transcript": transcript,
        "generatedBy": "Cadence (local processing, no cloud services)",
    });
    serde_json::to_string_pretty(&value).map_err(|e| e.to_string())
}

pub fn render_markdown(
    meta: &MeetingMetadata,
    mom: Option<&Mom>,
    transcript: Option<&Transcript>,
    extras: &Extras,
    redactor: &Redactor,
) -> String {
    let mut out = String::new();
    // `speaker_names` is keyed by stable id (SPEAKER_00); the minutes talk in
    // presentation labels (Person 1). The transcript is what maps one to the
    // other, so renames apply to both views or to neither.
    let renamed = label_renames(meta, transcript);
    let display = |label: &str| -> String {
        renamed.get(label).cloned().unwrap_or_else(|| label.to_string())
    };

    out.push_str(&format!("# {}\n\n", meta.display_title()));
    out.push_str("## Meeting information\n\n");
    out.push_str(&format!("- **Date:** {}\n", format_utc(meta.created_at_ms)));
    out.push_str(&format!(
        "- **Duration:** {}\n",
        crate::pipeline::mom::format_duration(meta.duration_ms)
    ));
    out.push_str(&format!("- **Status:** {}\n", meta.state.as_str()));
    if let Some(mom) = mom {
        if let Some(m) = &mom.meta {
            // The redaction mode belongs on this line and not with the
            // masking notice below it: this is the policy the *model* ran
            // under, which is provenance for how the minutes were written,
            // not a statement about what this document shows.
            out.push_str(&format!(
                "- **Minutes generated by:** {} ({}, {}) · prompts {} / {} · redaction {}\n",
                m.model,
                m.quantization,
                m.runtime,
                m.extraction_prompt,
                m.synthesis_prompt,
                m.redaction_mode
            ));
        }
    }
    if redactor.hides_text() {
        // Stated on the document itself, because a reader of an exported file
        // has no other way to tell a masked quote from a mis-transcribed one.
        out.push_str(&format!(
            "- **Redaction:** {} — flagged language is hidden in this export\n",
            redactor.mode.as_str()
        ));
    }
    if let Some(summary) = &extras.redaction {
        if summary.total > 0 {
            out.push_str(&format!(
                "- **Flagged language:** {} span(s) across {} turn(s)\n",
                summary.total, summary.turns_affected
            ));
        }
    }
    if !meta.notes.trim().is_empty() {
        out.push_str(&format!("\n**Notes:** {}\n", meta.notes.trim()));
    }

    push_classification(&mut out, extras.classification.as_ref());
    push_audio_quality(&mut out, extras.audio_profile.as_ref());

    if let Some(mom) = mom {
        out.push_str("\n## Participants\n\n");
        if mom.participants.is_empty() {
            out.push_str("_None detected._\n");
        }
        for participant in &mom.participants {
            out.push_str(&format!("- {}\n", display(participant)));
        }

        out.push_str("\n## Executive summary\n\n");
        out.push_str(&format!("{}\n", redactor.text(mom.summary.trim())));

        push_evidenced(&mut out, "Topics discussed", &mom.topics, redactor);
        push_evidenced(&mut out, "Key decisions", &mom.decisions, redactor);

        out.push_str("\n## Action items\n\n");
        if mom.action_items.is_empty() {
            out.push_str("_None recorded._\n");
        }
        for item in &mom.action_items {
            out.push_str(&format!("- {}\n", redactor.text(item.description.trim())));
            out.push_str(&format!(
                "  - Owner: {} · Deadline: {} · Confidence: {}{}\n",
                item.owner.clone().unwrap_or_else(|| "unassigned".to_string()),
                item.deadline.clone().unwrap_or_else(|| "none".to_string()),
                if item.confidence.is_empty() {
                    "low"
                } else {
                    item.confidence.as_str()
                },
                item.timestamp
                    .map(|t| format!(" · at {}", timecode(t)))
                    .unwrap_or_default()
            ));
        }

        push_evidenced(&mut out, "Open questions", &mom.open_questions, redactor);
        push_evidenced(&mut out, "Risks and concerns", &mom.risks, redactor);
        push_evidenced(
            &mut out,
            "Detailed discussion",
            &mom.detailed_discussion,
            redactor,
        );
    } else {
        out.push_str("\n## Minutes\n\n_Not generated for this meeting._\n");
    }

    if let Some(transcript) = transcript {
        out.push_str("\n## Transcript\n\n");
        for segment in &transcript.segments {
            out.push_str(&format!(
                "**[{}] {}**\n\n{}\n\n",
                timecode(segment.start_ms),
                display(&segment.speaker_label),
                // Masked before trimming, never after: `flags` hold code-point
                // offsets into `text` exactly as stored, and trimming first
                // would slide every one of them.
                redactor.segment(&segment.text, &segment.flags).trim()
            ));
        }
    }

    out.push_str("\n---\n\nGenerated locally by Cadence. No cloud services were used.\n");
    out
}

/// Presentation label → user-assigned name, for meetings where speakers were
/// renamed. Empty when there is no transcript to map ids to labels.
fn label_renames(
    meta: &MeetingMetadata,
    transcript: Option<&Transcript>,
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Some(transcript) = transcript else {
        return map;
    };
    for segment in &transcript.segments {
        if let Some(name) = meta.speaker_names.get(&segment.speaker) {
            map.insert(segment.speaker_label.clone(), name.clone());
        }
    }
    map
}

fn push_evidenced(out: &mut String, heading: &str, items: &[Evidenced], redactor: &Redactor) {
    out.push_str(&format!("\n## {heading}\n\n"));
    if items.is_empty() {
        out.push_str("_None recorded._\n");
        return;
    }
    for item in items {
        let evidence = match (&item.timestamp, &item.speaker) {
            (Some(t), Some(s)) => format!(" _({s}, {})_", timecode(*t)),
            (Some(t), None) => format!(" _({})_", timecode(*t)),
            (None, Some(s)) => format!(" _({s})_"),
            (None, None) => String::new(),
        };
        // An evidence line is a quotation lifted straight out of the
        // transcript, so it is transcript text and gets scanned like it.
        out.push_str(&format!(
            "- {}{evidence}\n",
            redactor.text(item.text.trim())
        ));
    }
}

/// What the meeting looks like it was, and what it was about.
///
/// Rendered with its confidence and its runners-up rather than as a bare
/// label: this is a heuristic over turn shape and vocabulary, not a fact, and
/// an export that presents it as one invites an argument the data cannot win.
fn push_classification(out: &mut String, classification: Option<&MeetingClassification>) {
    let Some(classification) = classification else {
        return;
    };
    out.push_str("\n## Meeting type\n\n");
    out.push_str(&format!(
        "- **Looks like:** {} ({:.0}% confidence)\n",
        classification.label,
        classification.confidence * 100.0
    ));
    let runners_up: Vec<String> = classification
        .alternatives
        .iter()
        .filter(|alt| alt.label != classification.label)
        .take(2)
        .map(|alt| alt.label.clone())
        .collect();
    if !runners_up.is_empty() {
        out.push_str(&format!("- **Also possible:** {}\n", runners_up.join(", ")));
    }
    if !classification.topics.is_empty() {
        let topics: Vec<&str> = classification
            .topics
            .iter()
            .take(8)
            .map(|topic| topic.term.as_str())
            .collect();
        out.push_str(&format!("- **Topics:** {}\n", topics.join(", ")));
    }
    let signals = &classification.signals;
    out.push_str(&format!(
        "- **Shape:** {} turns, {} speakers, {} words, {:.0}% questions\n",
        signals.turn_count,
        signals.speaker_count,
        signals.word_count,
        signals.question_ratio * 100.0
    ));
}

/// The recording's own quality, so a thin transcript can be read against the
/// audio it came from instead of blamed on the model.
fn push_audio_quality(out: &mut String, profile: Option<&AudioProfile>) {
    let Some(profile) = profile else {
        return;
    };
    out.push_str("\n## Audio quality\n\n");
    out.push_str(&format!(
        "- **Overall:** {} ({}/100)\n",
        profile.quality_grade, profile.quality_score
    ));
    out.push_str(&format!(
        "- **Levels:** speech {:.1} dBFS, peak {:.1} dBFS, noise floor {:.1} dBFS\n",
        profile.speech_dbfs, profile.peak_dbfs, profile.noise_floor_dbfs
    ));
    out.push_str(&format!(
        "- **Signal to noise:** {:.1} dB ({:.1} dB in the speech band)\n",
        profile.estimated_snr_db, profile.speech_band_snr_db
    ));
    if profile.clipped_percent > 0.0 {
        out.push_str(&format!(
            "- **Clipping:** {:.2}% of samples\n",
            profile.clipped_percent
        ));
    }
    if !profile.noise.findings.is_empty() {
        out.push_str("\n**What was heard in the room:**\n\n");
        for finding in &profile.noise.findings {
            out.push_str(&format!(
                "- **{}** ({:.0}% confidence) — {} {}\n",
                finding.label,
                finding.confidence * 100.0,
                finding.detail,
                finding.advice
            ));
        }
    }
}

/// `hh:mm:ss` from milliseconds.
pub fn timecode(ms: i64) -> String {
    let total = (ms / 1000).max(0);
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

/// `YYYY-MM-DD HH:MM UTC` from epoch milliseconds.
///
/// Written out rather than pulled from a date crate: this is the only place the
/// app needs calendar arithmetic, and the algorithm (Howard Hinnant's
/// `civil_from_days`) is short and exactly testable.
pub fn format_utc(ms: i64) -> String {
    if ms <= 0 {
        return "unknown".to_string();
    }
    let seconds = ms / 1000;
    let days = seconds.div_euclid(86_400);
    let time_of_day = seconds.rem_euclid(86_400);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02} UTC",
        time_of_day / 3600,
        (time_of_day % 3600) / 60
    )
}

/// Filesystem-safe, human-readable file-name stem.
fn slug(title: &str) -> String {
    let mut out = String::new();
    let mut last_was_dash = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash && !out.is_empty() {
            out.push('-');
            last_was_dash = true;
        }
    }
    let trimmed = out.trim_end_matches('-').to_string();
    if trimmed.is_empty() {
        "meeting".to_string()
    } else {
        trimmed.chars().take(48).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::models::{ActionItem, MeetingState, TranscriptSegment};
    use std::collections::BTreeMap;

    fn meta() -> MeetingMetadata {
        let mut meta = MeetingMetadata::new("abcdef123456", "Architecture Review", 1_756_600_000_000);
        meta.duration_ms = 1_845_000;
        meta.state = MeetingState::Completed;
        meta.notes = "Weekly sync".to_string();
        let mut names = BTreeMap::new();
        names.insert("SPEAKER_00".to_string(), "Karthik".to_string());
        meta.speaker_names = names;
        meta
    }

    fn transcript() -> Transcript {
        Transcript {
            meeting_id: "abcdef123456".to_string(),
            segments: vec![
                TranscriptSegment {
                    speaker: "SPEAKER_00".to_string(),
                    speaker_label: "Person 1".to_string(),
                    start_ms: 133_000,
                    end_ms: 145_000,
                    text: "இந்த feature ready ஆச்சா? We need to test it before Friday."
                        .to_string(),
                    language: Some("mixed".to_string()),
                    flags: Vec::new(),
                },
                TranscriptSegment {
                    speaker: "SPEAKER_01".to_string(),
                    speaker_label: "Person 2".to_string(),
                    start_ms: 145_000,
                    end_ms: 152_000,
                    text: "Yes, testing இன்னும் pending இருக்கு.".to_string(),
                    language: Some("mixed".to_string()),
                    flags: Vec::new(),
                },
            ],
        }
    }

    fn mom() -> Mom {
        Mom {
            summary: "The team reviewed the release plan.".to_string(),
            participants: vec!["Person 1".to_string(), "Person 2".to_string()],
            topics: vec![Evidenced {
                text: "release plan".to_string(),
                timestamp: Some(133_000),
                speaker: None,
            }],
            decisions: vec![Evidenced {
                text: "test before Friday".to_string(),
                timestamp: Some(140_000),
                speaker: Some("Person 1".to_string()),
            }],
            action_items: vec![ActionItem {
                description: "finish testing".to_string(),
                owner: Some("Person 2".to_string()),
                deadline: None,
                timestamp: Some(145_000),
                confidence: "medium".to_string(),
            }],
            ..Mom::default()
        }
    }

    #[test]
    fn markdown_contains_every_required_section() {
        let out = render_markdown(&meta(), Some(&mom()), Some(&transcript()), &Extras::default(), &Redactor::none());
        for heading in [
            "# Architecture Review",
            "## Meeting information",
            "## Participants",
            "## Executive summary",
            "## Topics discussed",
            "## Key decisions",
            "## Action items",
            "## Open questions",
            "## Risks and concerns",
            "## Transcript",
        ] {
            assert!(out.contains(heading), "missing {heading}");
        }
    }

    #[test]
    fn markdown_preserves_code_switched_text_verbatim() {
        let out = render_markdown(&meta(), Some(&mom()), Some(&transcript()), &Extras::default(), &Redactor::none());
        assert!(out.contains("இந்த feature ready ஆச்சா? We need to test it before Friday."));
    }

    #[test]
    fn markdown_uses_renamed_speakers_in_the_transcript() {
        let out = render_markdown(&meta(), Some(&mom()), Some(&transcript()), &Extras::default(), &Redactor::none());
        assert!(out.contains("Karthik"), "a renamed speaker should be used");
        assert!(out.contains("Person 2"), "an unrenamed speaker keeps its label");
    }

    #[test]
    fn markdown_marks_empty_sections_rather_than_omitting_them() {
        let out = render_markdown(
            &meta(),
            Some(&Mom {
                summary: "Nothing much happened.".to_string(),
                ..Mom::default()
            }),
            None,
            &Extras::default(),
            &Redactor::none(),
        );
        assert!(out.contains("_None recorded._"));
    }

    #[test]
    fn markdown_says_so_when_there_are_no_minutes() {
        let out = render_markdown(&meta(), None, Some(&transcript()), &Extras::default(), &Redactor::none());
        assert!(out.contains("_Not generated for this meeting._"));
    }

    #[test]
    fn markdown_shows_action_item_evidence() {
        let out = render_markdown(&meta(), Some(&mom()), None, &Extras::default(), &Redactor::none());
        assert!(out.contains("Owner: Person 2"));
        assert!(out.contains("Confidence: medium"));
        assert!(out.contains("at 00:02:25"));
    }

    #[test]
    fn json_export_round_trips() {
        let raw = render_json(&meta(), Some(&mom()), Some(&transcript()), &Redactor::none()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(value["meeting"]["title"], "Architecture Review");
        assert_eq!(value["mom"]["summary"], "The team reviewed the release plan.");
        assert_eq!(value["transcript"]["segments"].as_array().unwrap().len(), 2);
    }

    /// A turn with a real lexicon term in it, flagged exactly the way the
    /// pipeline would have flagged it.
    fn flagged_transcript() -> Transcript {
        let lexicon = redact::Lexicon::builtin();
        let text = "This damn build broke again.".to_string();
        let flags = redact::scan(&text, &lexicon, &ScanOptions::default());
        assert!(!flags.is_empty(), "fixture must actually match the lexicon");
        Transcript {
            meeting_id: "abcdef123456".to_string(),
            segments: vec![TranscriptSegment {
                speaker: "SPEAKER_00".to_string(),
                speaker_label: "Person 1".to_string(),
                start_ms: 1_000,
                end_ms: 5_000,
                text,
                language: None,
                flags,
            }],
        }
    }

    fn flagged_mom() -> Mom {
        Mom {
            summary: "The damn build broke again.".to_string(),
            decisions: vec![Evidenced {
                // An evidence quote is transcript text lifted verbatim, which
                // is the whole reason it has to be scanned at render time.
                text: "This damn build broke again.".to_string(),
                timestamp: Some(1_000),
                speaker: Some("Person 1".to_string()),
            }],
            action_items: vec![ActionItem {
                description: "fix the damn build".to_string(),
                owner: None,
                deadline: None,
                timestamp: None,
                confidence: "low".to_string(),
            }],
            ..Mom::default()
        }
    }

    #[test]
    fn masking_leaves_no_trace_of_the_original_word_in_a_json_export() {
        let redactor = Redactor::new(RedactionMode::Mask, false);
        let raw = render_json(
            &meta(),
            Some(&flagged_mom()),
            Some(&flagged_transcript()),
            &redactor,
        )
        .unwrap();

        // The point of the whole exercise: the word is gone from the file, not
        // merely from the field a reader happens to look at.
        assert!(
            !raw.to_lowercase().contains("damn"),
            "a masked JSON export still contained the original term:\n{raw}"
        );
        assert!(raw.contains("d***"), "the mask itself should be visible");
        // ...and the annotations survive, so the reader can see what happened.
        assert!(raw.contains("\"start_char\""));
        assert!(raw.contains("\"profanity\""));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&raw).unwrap()["exportRedactionMode"],
            "mask"
        );
    }

    #[test]
    fn the_export_policy_and_the_generation_policy_are_reported_separately() {
        // These are different facts and are allowed to disagree: minutes
        // written from a verbatim transcript, exported masked. A reader who
        // saw only one of them would draw the wrong conclusion about what the
        // model was shown, so both have to be on the document under names
        // that cannot be mistaken for each other.
        let mut minutes = flagged_mom();
        // Set outright rather than mutated through an `if let`: the fixture
        // carries no meta, so a conditional edit would leave both assertions
        // below reading a null that happened to look like a pass.
        minutes.meta = Some(crate::pipeline::models::MomMeta {
            redaction_mode: "off".to_string(),
            ..crate::pipeline::models::MomMeta::default()
        });
        let transcript = flagged_transcript();
        let redactor = Redactor::new(RedactionMode::Mask, false);

        let json = render_json(&meta(), Some(&minutes), Some(&transcript), &redactor).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["exportRedactionMode"], "mask");
        assert_eq!(parsed["mom"]["meta"]["redaction_mode"], "off");

        let markdown = render_markdown(
            &meta(),
            Some(&minutes),
            Some(&transcript),
            &Extras::default(),
            &redactor,
        );
        assert!(markdown.contains("redaction off"), "{markdown}");
    }

    #[test]
    fn masking_leaves_no_trace_of_the_original_word_in_a_markdown_export() {
        let redactor = Redactor::new(RedactionMode::Mask, false);
        let out = render_markdown(
            &meta(),
            Some(&flagged_mom()),
            Some(&flagged_transcript()),
            &Extras::default(),
            &redactor,
        );
        assert!(
            !out.to_lowercase().contains("damn"),
            "a masked Markdown export still contained the original term:\n{out}"
        );
        // Every surface that carries transcript text, not just the transcript.
        assert_eq!(out.matches("d***").count(), 4, "summary, quote, action item, turn");
        assert!(out.contains("**Redaction:** mask"));
    }

    #[test]
    fn redact_mode_hides_the_length_as_well_as_the_word() {
        let redactor = Redactor::new(RedactionMode::Redact, false);
        let out = render_markdown(
            &meta(),
            None,
            Some(&flagged_transcript()),
            &Extras::default(),
            &redactor,
        );
        assert!(!out.to_lowercase().contains("damn"));
        assert!(out.contains("[expunged]"));
    }

    #[test]
    fn flag_mode_exports_the_original_text_unchanged() {
        // `flag` is a review mode: it decorates, it does not hide, and an
        // export has nowhere to put a decoration. Anything else would be a
        // silent edit of what someone said.
        let redactor = Redactor::new(RedactionMode::Flag, false);
        let out = render_markdown(
            &meta(),
            Some(&flagged_mom()),
            Some(&flagged_transcript()),
            &Extras::default(),
            &redactor,
        );
        assert!(out.contains("This damn build broke again."));
        assert!(!out.contains("**Redaction:**"));
    }

    #[test]
    fn masking_survives_a_transcript_that_was_never_scanned() {
        // A meeting processed before flags existed, or with redaction off at
        // the time, has no flags at all. The export must still not leak.
        let mut transcript = flagged_transcript();
        transcript.segments[0].flags.clear();
        let out = render_markdown(
            &meta(),
            None,
            Some(&transcript),
            &Extras::default(),
            &Redactor::new(RedactionMode::Mask, false),
        );
        assert!(!out.to_lowercase().contains("damn"));
    }

    /// A real profile of a real (if boring) signal, rather than a hand-filled
    /// struct — it keeps the fixture honest about the field names and about
    /// what `analyze` actually produces.
    fn audio_profile() -> AudioProfile {
        let samples: Vec<f32> = (0..16_000)
            .map(|i| 0.3 * (i as f32 * 0.05).sin())
            .collect();
        let mut profile = crate::pipeline::audio_profile::analyze(&samples, 16_000, 1);
        profile.quality_score = 61;
        profile.quality_grade = "fair".to_string();
        profile
    }

    #[test]
    fn markdown_adds_the_classification_and_audio_sections_when_they_exist() {
        let extras = Extras {
            classification: Some(MeetingClassification {
                meeting_type: "standup".to_string(),
                label: "Stand-up".to_string(),
                confidence: 0.72,
                topics: vec![crate::pipeline::topic::TopicTag {
                    term: "deployment".to_string(),
                    count: 9,
                    spread: 4,
                    score: 1.5,
                }],
                ..MeetingClassification::default()
            }),
            audio_profile: Some(audio_profile()),
            redaction: None,
        };
        let out = render_markdown(
            &meta(),
            None,
            Some(&transcript()),
            &extras,
            &Redactor::none(),
        );
        assert!(out.contains("## Meeting type"));
        assert!(out.contains("Stand-up (72% confidence)"));
        assert!(out.contains("deployment"));
        assert!(out.contains("## Audio quality"));
        assert!(out.contains("fair (61/100)"));
    }

    #[test]
    fn markdown_omits_the_optional_sections_when_the_artifacts_are_missing() {
        let out = render_markdown(
            &meta(),
            None,
            Some(&transcript()),
            &Extras::default(),
            &Redactor::none(),
        );
        assert!(!out.contains("## Meeting type"));
        assert!(!out.contains("## Audio quality"));
    }

    #[test]
    fn writes_both_formats_into_the_meetings_exports_dir() {
        let dir = std::env::temp_dir().join("cadence-export-test");
        let _ = std::fs::remove_dir_all(&dir);
        let meta = meta();
        storage::write_metadata(&dir, &meta).unwrap();
        let meeting_dir = storage::meeting_dir(&dir, &meta.id).unwrap();
        std::fs::write(
            meeting_dir.join("transcript.json"),
            serde_json::to_string(&transcript()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            meeting_dir.join("mom.json"),
            serde_json::to_string(&mom()).unwrap(),
        )
        .unwrap();

        let md = write_export(&dir, &meta.id, Format::Markdown).unwrap();
        let json = write_export(&dir, &meta.id, Format::Json).unwrap();
        assert!(md.ends_with("architecture-review-abcdef12.md"));
        assert!(json.exists());
        assert!(std::fs::read_to_string(&md).unwrap().contains("# Architecture Review"));
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        let exported: Transcript = serde_json::from_value(parsed["transcript"].clone()).unwrap();
        assert_eq!(exported.meeting_id, meta.id);
        assert_eq!(exported.segments[0].text, transcript().segments[0].text);
        // Exporting again may replace its own regular file, while preserving
        // the source artifacts and the existing filename.
        let original = std::fs::read(meeting_dir.join("transcript.json")).unwrap();
        assert_eq!(write_export(&dir, &meta.id, Format::Json).unwrap(), json);
        assert_eq!(std::fs::read(meeting_dir.join("transcript.json")).unwrap(), original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_export_a_meeting_with_nothing_in_it() {
        let dir = std::env::temp_dir().join("cadence-export-empty");
        let _ = std::fs::remove_dir_all(&dir);
        storage::write_metadata(&dir, &meta()).unwrap();
        assert!(write_export(&dir, "abcdef123456", Format::Markdown).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_meeting_ids_cannot_export_outside_meeting_storage() {
        let dir = std::env::temp_dir().join("cadence-export-invalid-ids");
        let outside = dir.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let sentinel = outside.join("keep.txt");
        std::fs::write(&sentinel, "untouched").unwrap();
        for id in ["../outside", "..\\outside", "C:\\outside", "a:stream", "NUL", "a.", ""] {
            for format in [Format::Markdown, Format::Json] {
                let err = write_export(&dir, id, format).unwrap_err();
                assert!(err.contains("invalid meeting path component"), "{id:?}: {err}");
            }
        }
        assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "untouched");
        assert!(!outside.join("exports").exists());
        assert!(!dir.join("meetings").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn export_rejects_junctions_at_directory_and_output_paths() {
        // Junctions require no symlink privilege on Windows. Test both the
        // exports directory and the final output, without following either.
        for final_output in [false, true] {
            let dir = std::env::temp_dir().join(format!("cadence-export-junction-{final_output}"));
            let outside = dir.join("outside");
            std::fs::create_dir_all(&outside).unwrap();
            let sentinel = outside.join("keep.txt");
            std::fs::write(&sentinel, "untouched").unwrap();
            let meta = meta();
            storage::write_metadata(&dir, &meta).unwrap();
            let meeting = storage::meeting_dir(&dir, &meta.id).unwrap();
            std::fs::write(meeting.join("transcript.json"), serde_json::to_string(&transcript()).unwrap()).unwrap();
            let exports = meeting.join("exports");
            let link = if final_output {
                std::fs::create_dir_all(&exports).unwrap();
                exports.join("architecture-review-abcdef12.md")
            } else {
                exports.clone()
            };
            let output = std::process::Command::new("cmd")
                .args(["/c", "mklink", "/J"])
                .arg(&link)
                .arg(&outside)
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            let err = write_export(&dir, &meta.id, Format::Markdown).unwrap_err();
            assert!(err.contains("linked meeting path"), "{err}");
            assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "untouched");
            assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 1);
            std::fs::remove_dir(link).unwrap();
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn formats_timecodes() {
        assert_eq!(timecode(0), "00:00:00");
        assert_eq!(timecode(145_000), "00:02:25");
        assert_eq!(timecode(3_725_000), "01:02:05");
        assert_eq!(timecode(-1), "00:00:00");
    }

    #[test]
    fn formats_utc_dates() {
        assert_eq!(format_utc(0), "unknown");
        assert_eq!(format_utc(1), "1970-01-01 00:00 UTC");
        // 2026-08-31T00:00:00Z
        assert_eq!(format_utc(1_788_134_400_000), "2026-08-31 00:00 UTC");
        // 2000-02-29T12:30:00Z — a leap day, the case this algorithm exists for.
        assert_eq!(format_utc(951_827_400_000), "2000-02-29 12:30 UTC");
    }

    #[test]
    fn slugs_titles_safely() {
        assert_eq!(slug("Architecture Review"), "architecture-review");
        assert_eq!(slug("  Q3 / planning!! "), "q3-planning");
        assert_eq!(slug("***"), "meeting");
        assert_eq!(slug(""), "meeting");
        assert!(slug(&"x".repeat(200)).chars().count() <= 48);
    }

    #[test]
    fn parses_format_names() {
        assert_eq!(Format::parse("Markdown"), Some(Format::Markdown));
        assert_eq!(Format::parse("md"), Some(Format::Markdown));
        assert_eq!(Format::parse("json"), Some(Format::Json));
        assert_eq!(Format::parse("pdf"), None);
    }
}
