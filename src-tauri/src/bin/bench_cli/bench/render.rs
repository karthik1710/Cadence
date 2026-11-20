//! Human-readable renderings of bench outputs — the files a person actually
//! opens side by side.

use super::text::clock;
use app_lib::pipeline::models::{Evidenced, Mom, Transcript};

/// `[mm:ss] Person N: text`, one turn per line.
pub fn transcript_text(transcript: &Transcript) -> String {
    let mut out = String::new();
    for segment in &transcript.segments {
        out.push_str(&format!(
            "[{}] {}: {}\n",
            clock(segment.start_ms),
            segment.speaker_label,
            segment.text.trim()
        ));
    }
    out
}

pub fn word_count(transcript: &Transcript) -> usize {
    transcript
        .segments
        .iter()
        .map(|s| s.text.split_whitespace().count())
        .sum()
}

pub fn speaker_count(transcript: &Transcript) -> usize {
    transcript.speaker_labels().len()
}

/// The minutes as Markdown, in the order a reader checks them: decisions and
/// action items first, since those are what the recording guide plants
/// deliberately and what a MoM comparison is scored on.
pub fn mom_markdown(mom: &Mom, heading: &str) -> String {
    let mut out = format!("# {heading}\n\n");
    if let Some(meta) = &mom.meta {
        out.push_str(&format!(
            "_{} ({}), {} context, {} chunk(s), {} failed, repaired: {}, prompts {}/{}, redaction {}_\n\n",
            meta.model,
            meta.quantization,
            meta.context_tokens,
            meta.chunk_count,
            meta.failed_chunks,
            if meta.repaired { "yes" } else { "no" },
            meta.extraction_prompt,
            meta.synthesis_prompt,
            meta.redaction_mode
        ));
    }
    out.push_str("## Summary\n\n");
    out.push_str(mom.summary.trim());
    out.push_str("\n\n");
    out.push_str(&format!("**Participants:** {}\n\n", mom.participants.join(", ")));

    out.push_str("## Decisions\n\n");
    list(&mut out, &mom.decisions);

    out.push_str("## Action items\n\n");
    if mom.action_items.is_empty() {
        out.push_str("_None._\n\n");
    } else {
        out.push_str("| # | action | owner | deadline | at | confidence |\n|---|---|---|---|---|---|\n");
        for (i, item) in mom.action_items.iter().enumerate() {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                i + 1,
                cell(&item.description),
                item.owner.as_deref().unwrap_or("—"),
                item.deadline.as_deref().map(cell).unwrap_or_else(|| "—".to_string()),
                item.timestamp.map(clock).unwrap_or_else(|| "—".to_string()),
                item.confidence
            ));
        }
        out.push('\n');
    }

    out.push_str("## Open questions\n\n");
    list(&mut out, &mom.open_questions);
    out.push_str("## Risks\n\n");
    list(&mut out, &mom.risks);
    out.push_str("## Topics\n\n");
    list(&mut out, &mom.topics);
    out.push_str("## Detailed discussion\n\n");
    list(&mut out, &mom.detailed_discussion);
    out
}

fn list(out: &mut String, items: &[Evidenced]) {
    if items.is_empty() {
        out.push_str("_None._\n\n");
        return;
    }
    for item in items {
        let mut evidence = Vec::new();
        if let Some(ts) = item.timestamp {
            evidence.push(clock(ts));
        }
        if let Some(speaker) = &item.speaker {
            evidence.push(speaker.clone());
        }
        if evidence.is_empty() {
            out.push_str(&format!("- {}\n", item.text.trim()));
        } else {
            out.push_str(&format!("- {} _({})_\n", item.text.trim(), evidence.join(", ")));
        }
    }
    out.push('\n');
}

/// Makes free text safe inside a Markdown table cell.
pub fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\n', '\r'], " ")
}
