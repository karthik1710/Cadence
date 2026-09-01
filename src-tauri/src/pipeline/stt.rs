//! Speech-to-text, and the three ways a user can ask for a language.
//!
//! whisper.cpp's `--language auto` detects **once**, from the opening window,
//! and decodes the whole file in whatever it found. That is fine for a
//! monolingual meeting and wrong for the meetings this app exists for, which
//! switch between English and Tamil mid-sentence — see
//! [`crate::pipeline::codeswitch`], which is where the code-switched mode
//! actually lives.

use crate::pipeline::codeswitch;
use crate::pipeline::models::SttSegment;
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::proc::{self, StageMetrics};
use crate::pipeline::vad::SpeechSegment;
use crate::pipeline::PipelineConfig;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

/// How the user asked STT to handle language (`Settings::stt_language`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SttLanguage {
    /// whisper.cpp's own `auto`: one detection for the whole recording.
    Auto,
    /// One detection per ~30 s block. The default, because these
    /// meetings are code-switched.
    CodeSwitched,
    /// An ISO code the user forced, passed straight through to whisper.
    Forced(String),
}

impl SttLanguage {
    /// Anything unrecognisable resolves to [`SttLanguage::Auto`] rather than
    /// being handed to whisper to reject: a hand-edited settings file must not
    /// cost someone every meeting they record afterwards.
    pub fn parse(raw: &str) -> Self {
        let value = raw.trim().to_ascii_lowercase();
        match value.as_str() {
            "" | "auto" => SttLanguage::Auto,
            "code-switched" | "code_switched" | "codeswitched" | "mixed" => {
                SttLanguage::CodeSwitched
            }
            code if is_language_code(code) => SttLanguage::Forced(code.to_string()),
            _ => SttLanguage::Auto,
        }
    }

    /// What `--language` receives for a single pass. Code-switched mode also
    /// answers `auto` — its per-block calls are auto-detected individually,
    /// and this is what it falls back to when blocking cannot run.
    pub fn whisper_arg(&self) -> &str {
        match self {
            SttLanguage::Forced(code) => code,
            _ => "auto",
        }
    }
}

/// ISO 639 codes are two or three ASCII letters. Nothing else is accepted, so
/// a typo becomes "auto" instead of a failed whisper invocation.
fn is_language_code(value: &str) -> bool {
    (2..=3).contains(&value.len()) && value.chars().all(|c| c.is_ascii_alphabetic())
}

/// One transcription request. A struct rather than an argument list because
/// the code-switched path needs the VAD result and a place to log, neither of
/// which the single-pass path cares about.
pub struct SttRequest<'a> {
    pub paths: &'a ModelPaths,
    pub config: &'a PipelineConfig,
    /// 16 kHz mono PCM — see preprocess.rs.
    pub wav_path: &'a Path,
    /// Speech regions on `wav_path`'s timeline. Code-switched mode needs them
    /// to place its block boundaries in silence; without them it falls back.
    pub speech: Option<&'a [SpeechSegment]>,
    /// App data dir, for the whisper-server log.
    pub data_dir: Option<&'a Path>,
}

pub struct SttOutcome {
    pub segments: Vec<SttSegment>,
    pub metrics: StageMetrics,
    /// Anything the user should know about how this ran — in practice, that
    /// code-switched mode wanted to run and could not (never silent).
    pub warnings: Vec<String>,
    /// Blocks transcribed. `None` for a single pass.
    pub blocks: Option<usize>,
    /// Distinct detected languages, in first-appearance order.
    pub languages: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonOffsets {
    from: i64,
    to: i64,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonSegment {
    offsets: WhisperJsonOffsets,
    text: String,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonOutput {
    transcription: Vec<WhisperJsonSegment>,
}

/// Transcribes a recording in whichever language mode the settings ask for.
pub fn transcribe(request: &SttRequest) -> Result<SttOutcome, String> {
    let language = SttLanguage::parse(&request.config.stt_language);
    let mut warnings = Vec::new();

    if language == SttLanguage::CodeSwitched {
        match run_code_switched(request) {
            // Not worth blocking — a short recording detects on the same audio
            // either way, so the single pass below is the *correct* answer,
            // not a degraded one, and says nothing.
            Ok(None) => {}
            Ok(Some(outcome)) => return Ok(outcome),
            Err(reason) => warnings.push(format!(
                "Mixed-language transcription could not run, so this recording was transcribed \
in one language detected from its opening seconds. Tamil or English may be degraded wherever \
the speakers switched. Cause: {reason}"
            )),
        }
    }

    let (segments, metrics) = single_pass(request, language.whisper_arg())?;
    Ok(SttOutcome {
        segments,
        metrics,
        warnings,
        blocks: None,
        languages: Vec::new(),
    })
}

fn run_code_switched(request: &SttRequest) -> Result<Option<SttOutcome>, String> {
    let Some(speech) = request.speech.filter(|s| !s.is_empty()) else {
        return Err(
            "voice activity detection found no speech regions to place block boundaries in \
(it is either switched off in Settings or its model is not installed)"
                .to_string(),
        );
    };

    let outcome = codeswitch::transcribe(&codeswitch::BlockRun {
        paths: request.paths,
        wav_path: request.wav_path,
        speech,
        threads: request.config.stt_threads,
        pad_ms: request.config.vad_pad_ms,
        data_dir: request.data_dir,
    })?;

    Ok(outcome.map(|outcome| SttOutcome {
        languages: distinct_languages(&outcome.segments),
        segments: outcome.segments,
        metrics: outcome.metrics,
        warnings: Vec::new(),
        blocks: Some(outcome.blocks),
    }))
}

/// Detected languages in first-appearance order — what the metrics record and
/// what tells a reader at a glance whether the meeting actually code-switched.
fn distinct_languages(segments: &[SttSegment]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for language in segments.iter().filter_map(|s| s.language.as_ref()) {
        if !seen.contains(language) {
            seen.push(language.clone());
        }
    }
    seen
}

/// One `whisper-cli.exe` invocation over the whole file.
///
/// The `--max-len 80 --split-on-word --max-context 0` flags are load-bearing,
/// not tuning knobs — see docs/benchmarks.md: without them whisper.cpp has been
/// observed to silently collapse a real, multi-second span of speech into a
/// short, wrong sentence.
///
/// NOTE: `--translate` is never passed, and must never be added. The transcript must
/// preserve the language actually spoken; translating here
/// would destroy the original words for good. The minutes are the only thing
/// that gets rendered into another language, via `mom_language`.
fn single_pass(
    request: &SttRequest,
    language: &str,
) -> Result<(Vec<SttSegment>, StageMetrics), String> {
    let wav_path = request.wav_path;
    // whisper.cpp appends `.json` itself, so pass the base path without it.
    let output_base = wav_path.with_extension("");

    let mut command = Command::new(request.paths.whisper_cli());
    command
        .arg("-m")
        .arg(&request.paths.stt_model)
        .arg("-f")
        .arg(wav_path)
        .arg("--language")
        .arg(language)
        .arg("--max-len")
        .arg("80")
        .arg("--split-on-word")
        .arg("--max-context")
        .arg("0")
        .arg("-t")
        .arg(request.config.stt_threads.to_string())
        .arg("--output-json")
        .arg("--output-file")
        .arg(&output_base);

    let measured = proc::run_measured(command)?;
    if !measured.success {
        return Err(format!(
            "whisper-cli.exe exited with {}: {}",
            measured.status,
            last_lines(&measured.stderr, 4)
        ));
    }

    let json_path = wav_path.with_extension("json");
    let raw = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("could not read whisper output {json_path:?}: {e}"))?;
    let segments = parse_whisper_json(&raw)?;
    Ok((segments, measured.metrics))
}

fn parse_whisper_json(raw: &str) -> Result<Vec<SttSegment>, String> {
    let parsed: WhisperJsonOutput =
        serde_json::from_str(raw).map_err(|e| format!("could not parse whisper JSON: {e}"))?;

    Ok(parsed
        .transcription
        .into_iter()
        .filter_map(|seg| {
            let text = seg.text.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(SttSegment {
                start_ms: seg.offsets.from,
                end_ms: seg.offsets.to,
                text,
                words: Vec::new(),
                // A single pass detects one language for the whole file, which
                // is not a per-segment fact and must not be presented as one.
                language: None,
            })
        })
        .collect())
}

fn last_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(count)..].join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whisper_output() {
        let raw = r#"{
            "transcription": [
                {"offsets": {"from": 0, "to": 2400}, "text": " We should move this to the next sprint."},
                {"offsets": {"from": 2400, "to": 2600}, "text": "   "},
                {"offsets": {"from": 2600, "to": 5000}, "text": "இந்த feature ready ஆச்சா?"}
            ]
        }"#;
        let segments = parse_whisper_json(raw).unwrap();
        // The blank segment is dropped; the others keep their offsets and text.
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].text, "We should move this to the next sprint.");
        assert_eq!(segments[1].text, "இந்த feature ready ஆச்சா?");
        // A whole-file detection is not a per-segment fact.
        assert!(segments.iter().all(|s| s.language.is_none()));
    }

    #[test]
    fn reports_unparseable_output() {
        assert!(parse_whisper_json("not json").is_err());
    }

    #[test]
    fn summarises_the_tail_of_an_error() {
        let text = "line one\n\nline two\nline three\nline four\nline five";
        assert_eq!(last_lines(&text, 2), "line four | line five");
        assert_eq!(last_lines("", 3), "");
    }

    #[test]
    fn the_three_language_modes_parse() {
        assert_eq!(SttLanguage::parse("auto"), SttLanguage::Auto);
        assert_eq!(SttLanguage::parse("  AUTO "), SttLanguage::Auto);
        assert_eq!(SttLanguage::parse(""), SttLanguage::Auto);
        assert_eq!(
            SttLanguage::parse("code-switched"),
            SttLanguage::CodeSwitched
        );
        assert_eq!(
            SttLanguage::parse("Code_Switched"),
            SttLanguage::CodeSwitched
        );
        assert_eq!(
            SttLanguage::parse("ta"),
            SttLanguage::Forced("ta".to_string())
        );
        assert_eq!(
            SttLanguage::parse("EN"),
            SttLanguage::Forced("en".to_string())
        );
    }

    #[test]
    fn a_nonsense_setting_falls_back_to_auto_rather_than_failing_the_stage() {
        assert_eq!(SttLanguage::parse("klingon"), SttLanguage::Auto);
        assert_eq!(SttLanguage::parse("e"), SttLanguage::Auto);
        assert_eq!(SttLanguage::parse("t4"), SttLanguage::Auto);
    }

    #[test]
    fn only_a_forced_language_changes_the_whisper_argument() {
        assert_eq!(SttLanguage::Auto.whisper_arg(), "auto");
        // Code-switched falls back to the same single pass when blocking
        // cannot run, so it must not name a language either.
        assert_eq!(SttLanguage::CodeSwitched.whisper_arg(), "auto");
        assert_eq!(
            SttLanguage::Forced("ta".to_string()).whisper_arg(),
            "ta"
        );
    }

    #[test]
    fn the_default_setting_is_code_switched() {
        assert_eq!(
            SttLanguage::parse(crate::settings::DEFAULT_STT_LANGUAGE),
            SttLanguage::CodeSwitched
        );
    }

    #[test]
    fn distinct_languages_keep_first_appearance_order() {
        let segment = |language: Option<&str>| SttSegment {
            start_ms: 0,
            end_ms: 1,
            text: "x".to_string(),
            words: Vec::new(),
            language: language.map(|l| l.to_string()),
        };
        let segments = [
            segment(Some("ta")),
            segment(Some("en")),
            segment(None),
            segment(Some("ta")),
        ];
        assert_eq!(distinct_languages(&segments), vec!["ta", "en"]);
        assert!(distinct_languages(&[segment(None)]).is_empty());
    }
}
