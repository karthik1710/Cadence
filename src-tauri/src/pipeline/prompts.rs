//! Versioned prompt templates (master prompt §17).
//!
//! Prompts live as plain text under `prompts/` at the repo root so they can be
//! reviewed and diffed like any other source, and are compiled into the binary
//! with `include_str!` so a packaged install has no loose files to lose. A
//! matching file dropped in `<app_data>/prompts/` overrides the embedded one,
//! which is how a prompt can be iterated on without a rebuild.
//!
//! The version string is recorded in every generated `mom.json` alongside the
//! model and quantization, so any set of minutes can be traced back to exactly
//! what produced it.

use std::path::Path;

pub const EXTRACTION_VERSION: &str = "extraction_v1";
pub const SYNTHESIS_VERSION: &str = "synthesis_v1";
pub const REPAIR_VERSION: &str = "repair_json_v1";

const EXTRACTION: &str = include_str!("../../../prompts/extraction_v1.txt");
const SYNTHESIS: &str = include_str!("../../../prompts/synthesis_v1.txt");
const REPAIR: &str = include_str!("../../../prompts/repair_json_v1.txt");

/// Loads a prompt template by version name, preferring a user override.
pub fn load(data_dir: Option<&Path>, version: &str) -> Result<String, String> {
    if let Some(dir) = data_dir {
        let override_path = dir.join("prompts").join(format!("{version}.txt"));
        if override_path.exists() {
            return std::fs::read_to_string(&override_path)
                .map_err(|e| format!("could not read {}: {e}", override_path.display()));
        }
    }
    match version {
        EXTRACTION_VERSION => Ok(EXTRACTION.to_string()),
        SYNTHESIS_VERSION => Ok(SYNTHESIS.to_string()),
        REPAIR_VERSION => Ok(REPAIR.to_string()),
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
        for v in [EXTRACTION_VERSION, SYNTHESIS_VERSION, REPAIR_VERSION] {
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
        for v in [EXTRACTION_VERSION, SYNTHESIS_VERSION, REPAIR_VERSION] {
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
    fn rejects_an_unknown_version() {
        assert!(load(None, "extraction_v99").is_err());
    }
}
