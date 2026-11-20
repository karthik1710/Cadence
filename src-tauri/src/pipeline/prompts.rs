//! Versioned prompt templates.
//!
//! Prompts live as plain text under `prompts/` at the repo root so they can be
//! reviewed and diffed like any other source, and are compiled into the binary
//! with `include_str!` so a packaged install has no loose files to lose. A
//! matching file dropped in `<app_data>/prompts/` overrides the embedded one,
//! which is how a prompt can be iterated on without a rebuild.
//!
//! The version string is recorded in every generated `mom.json` alongside the
//! model and quantization, so any set of minutes can be traced back to exactly
//! what produced it. That is also why a superseded template is never deleted:
//! minutes already on disk name the version that produced them, and a set of
//! minutes nobody can explain afterwards is worse than one nobody regenerated.
//! New runs use the `*_VERSION` constants below; [`load`] resolves every
//! version that has ever shipped.

use std::path::Path;

pub const EXTRACTION_VERSION: &str = "extraction_v2";
pub const SYNTHESIS_VERSION: &str = "synthesis_v2";
pub const REPAIR_VERSION: &str = "repair_json_v1";
/// Bench-only: the experimental transcript-correction pass `bench_cli compare
/// --llm-cleanup` runs. Nothing in the app pipeline loads it — whether a
/// cleanup pass helps or quietly rewrites what was said is exactly what that
/// comparison exists to find out before anything ships.
pub const CLEANUP_VERSION: &str = "transcript_cleanup_v1";

/// Every version this binary can resolve, current and retired alike. The tests
/// sweep it, so a retired template cannot quietly rot into something that no
/// longer loads or that grew a placeholder nobody fills.
pub const KNOWN_VERSIONS: [&str; 6] = [
    "extraction_v1",
    "extraction_v2",
    "synthesis_v1",
    "synthesis_v2",
    "repair_json_v1",
    "transcript_cleanup_v1",
];

const EXTRACTION_V1: &str = include_str!("../../../prompts/extraction_v1.txt");
/// v2 carries the redaction contract, in two halves: do not reproduce
/// offensive language in a summary even where the transcript shows it, and
/// treat runs of `*` and `[expunged]` as deliberate masking to be preserved
/// rather than guessed at or "corrected". It is the belt to `mom.rs`'s braces —
/// that module masks the transcript *before* it is chunked, so the model
/// normally never sees the words at all, but a prompt costs nothing and covers
/// the case where a term is not in the lexicon.
const EXTRACTION_V2: &str = include_str!("../../../prompts/extraction_v2.txt");
const SYNTHESIS_V1: &str = include_str!("../../../prompts/synthesis_v1.txt");
const SYNTHESIS_V2: &str = include_str!("../../../prompts/synthesis_v2.txt");
const REPAIR_V1: &str = include_str!("../../../prompts/repair_json_v1.txt");
const CLEANUP_V1: &str = include_str!("../../../prompts/transcript_cleanup_v1.txt");

/// Loads a prompt template by version name, preferring a user override.
pub fn load(data_dir: Option<&Path>, version: &str) -> Result<String, String> {
    if let Some(dir) = data_dir {
        let override_path = dir.join("prompts").join(format!("{version}.txt"));
        if override_path.exists() {
            return std::fs::read_to_string(&override_path)
                .map_err(|e| format!("could not read {}: {e}", override_path.display()));
        }
    }
    // Matched on literal names rather than on the `*_VERSION` constants: those
    // move whenever a prompt is revised, and a `mom.json` recording the old
    // name has to keep resolving to the template that actually produced it.
    match version {
        "extraction_v1" => Ok(EXTRACTION_V1.to_string()),
        "extraction_v2" => Ok(EXTRACTION_V2.to_string()),
        "synthesis_v1" => Ok(SYNTHESIS_V1.to_string()),
        "synthesis_v2" => Ok(SYNTHESIS_V2.to_string()),
        "repair_json_v1" => Ok(REPAIR_V1.to_string()),
        "transcript_cleanup_v1" => Ok(CLEANUP_V1.to_string()),
        other => Err(format!("unknown prompt version: {other}")),
    }
}

/// Substitutes `{{NAME}}` placeholders. Unknown placeholders are left in
/// place rather than blanked, so a typo in a template shows up in the prompt
/// instead of silently dropping an instruction.
pub fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (key, value) in values {
        out = out.replace(&format!("{{{{{key}}}}}"), value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_prompts_are_present_and_substantial() {
        for v in KNOWN_VERSIONS {
            let text = load(None, v).unwrap();
            assert!(text.len() > 200, "{v} looks empty");
        }
    }

    #[test]
    fn every_placeholder_in_a_template_is_documented_here() {
        // Guards against a template gaining a `{{VAR}}` that no caller fills,
        // which would otherwise ship a literal `{{VAR}}` to the model.
        let known = [
            "CHUNK_INDEX",
            "CHUNK_COUNT",
            "SPEAKERS",
            "TRANSCRIPT",
            "LANGUAGE",
            "TITLE",
            "DURATION",
            "FACTS",
            "ERROR",
            "SCHEMA",
            "INVALID",
        ];
        for v in KNOWN_VERSIONS {
            let text = load(None, v).unwrap();
            for (index, _) in text.match_indices("{{") {
                let rest = &text[index + 2..];
                let end = rest.find("}}").expect("unterminated placeholder");
                let name = &rest[..end];
                assert!(known.contains(&name), "{v} uses unknown placeholder {name}");
            }
        }
    }

    #[test]
    fn fills_placeholders() {
        let filled = fill("a {{X}} b {{Y}} {{X}}", &[("X", "1"), ("Y", "2")]);
        assert_eq!(filled, "a 1 b 2 1");
    }

    #[test]
    fn leaves_unknown_placeholders_visible() {
        assert_eq!(fill("{{MISSING}}", &[("X", "1")]), "{{MISSING}}");
    }

    #[test]
    fn the_retired_v1_prompts_still_resolve_by_name() {
        // Every `mom.json` on disk names the version that produced it. If a
        // retired template stopped loading, those minutes would become
        // unexplainable and could never be regenerated under their own rules.
        assert!(load(None, "extraction_v1").unwrap().contains("{{TRANSCRIPT}}"));
        assert!(load(None, "synthesis_v1").unwrap().contains("{{FACTS}}"));
        assert_ne!(
            load(None, "extraction_v1").unwrap(),
            load(None, "extraction_v2").unwrap()
        );
        assert_ne!(
            load(None, "synthesis_v1").unwrap(),
            load(None, "synthesis_v2").unwrap()
        );
    }

    #[test]
    fn the_current_templates_carry_the_redaction_contract() {
        assert_eq!(EXTRACTION_VERSION, "extraction_v2");
        assert_eq!(SYNTHESIS_VERSION, "synthesis_v2");
        for v in [EXTRACTION_VERSION, SYNTHESIS_VERSION] {
            let text = load(None, v).unwrap();
            assert!(
                text.contains("[expunged]"),
                "{v} never explains what a redaction looks like"
            );
            assert!(
                text.to_lowercase().contains("offensive"),
                "{v} does not forbid reproducing flagged language"
            );
        }
        // ...and the retired ones do not, which is the whole reason for the
        // bump: minutes generated under v1 were not held to this.
        assert!(!load(None, "extraction_v1").unwrap().contains("[expunged]"));
    }

    #[test]
    fn rejects_an_unknown_version() {
        assert!(load(None, "extraction_v99").is_err());
    }
}
