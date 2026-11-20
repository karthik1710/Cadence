//! Speech-to-text, and the three ways a user can ask for a language.
//!
//! whisper.cpp's `--language auto` detects **once**, from the opening window,
//! and decodes the whole file in whatever it found. That is fine for a
//! monolingual meeting and wrong for the meetings this app exists for, which
//! switch between English and Tamil mid-sentence — see
//! [`crate::pipeline::codeswitch`], which is where the code-switched mode
//! actually lives.

use crate::pipeline::codeswitch;
use crate::pipeline::models::{SttSegment, SttWord};
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
struct WhisperJsonToken {
    #[serde(default)]
    text: String,
    /// whisper.cpp writes a token's `offsets` only when it actually has
    /// timestamps for it (`t0 > -1 && t1 > -1` in its own `output_json`), so
    /// this is genuinely optional. Declaring it non-optional would make one
    /// undated token fail the whole document — and the document is the
    /// transcript.
    #[serde(default)]
    offsets: Option<WhisperJsonOffsets>,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonSegment {
    offsets: WhisperJsonOffsets,
    text: String,
    /// Only written under [`FULL_JSON_FLAG`]. Defaults to empty so a
    /// plain `--output-json` file — every `stt.json` written before this
    /// existed, and the fallback run below — parses exactly as it used to.
    #[serde(default)]
    tokens: Vec<WhisperJsonToken>,
}

#[derive(Debug, Deserialize)]
struct WhisperJsonOutput {
    transcription: Vec<WhisperJsonSegment>,
}

// ------------------------------------------------------------ word timing

/// One whisper token placed on a timeline, before it is assembled into a word.
///
/// Both STT paths produce exactly this and nothing more: `whisper-cli.exe`'s
/// `--output-json-full` writes a `tokens` array with millisecond `offsets`,
/// and `whisper-server`'s `verbose_json` writes a `words` array with
/// fractional-second `start`/`end`. Despite the server's field name, its
/// entries are *tokens* too — a probe of the vendored binaries returned
/// `" di"`, `"ar"`, `"ization"` as three separate "words" from both — so the
/// two paths share [`assemble_words`] rather than each inventing a rule.
pub(crate) struct RawToken<'a> {
    pub text: &'a str,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Joins whisper's tokens into words with start/end times.
///
/// Three things in the raw stream are not words, all three observed in a real
/// run of the vendored `whisper-cli.exe`:
///
/// * **Special tokens.** `[_BEG_]` opens a segment and `[_TT_220]` closes one;
///   both carry a timestamp and no speech. They must not become words.
/// * **Sub-word pieces.** whisper's vocabulary is BPE, so `diarization` comes
///   back as `" di"` + `"ar"` + `"ization"` spanning 210-980 ms. whisper marks
///   the start of a word with a leading space, so a token *without* one
///   continues the token before it.
/// * **Punctuation.** `","` and `";"` arrive as their own tokens. They belong
///   on the word they follow — and only its text, not its timing: in that same
///   run the `;` was stamped at 4700 ms while the word it ends stopped at
///   4310 ms, and letting punctuation stretch a word by 390 ms of silence
///   would move the very boundary alignment is trying to find.
///
/// Never fails. A stream this cannot make sense of yields fewer words (or
/// none), and [`crate::pipeline::align`] falls back to whole-segment
/// attribution — worse speaker labels, never a lost transcript.
pub(crate) fn assemble_words(tokens: &[RawToken]) -> Vec<SttWord> {
    let mut words: Vec<SttWord> = Vec::new();

    for token in tokens {
        if is_special_token(token.text) {
            continue;
        }
        let trimmed = token.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        // A leading space is whisper's own word delimiter, so its absence is
        // what marks a continuation — of the word's spelling always, and of
        // its span only when the piece is actual speech rather than a mark.
        if !token.text.starts_with(char::is_whitespace) {
            if let Some(last) = words.last_mut() {
                last.text.push_str(trimmed);
                if !is_punctuation_only(trimmed) {
                    last.end_ms = last.end_ms.max(token.end_ms);
                }
                continue;
            }
        }

        words.push(SttWord {
            start_ms: token.start_ms,
            end_ms: token.end_ms.max(token.start_ms),
            text: trimmed.to_string(),
        });
    }

    words
}

/// whisper's own markers: `[_BEG_]`, `[_TT_220]`, `[_SOT_]`, `[_EOT_]`, and
/// the `<|0.00|>` timestamp spelling older builds use.
fn is_special_token(text: &str) -> bool {
    let text = text.trim();
    (text.starts_with("[_") && text.ends_with(']')) || (text.starts_with("<|") && text.ends_with("|>"))
}

fn is_punctuation_only(text: &str) -> bool {
    !text.is_empty() && !text.chars().any(char::is_alphanumeric)
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

/// whisper.cpp's verbose JSON output — the only way it will emit the
/// per-token offsets word timestamps are built from.
///
/// Spelling verified against the vendored binary's own `--help`, not assumed:
/// `-ojf, --output-json-full   [false  ] include more information in the JSON
/// file`. It is an *addition* to `--output-json`, not a replacement, so both
/// are passed and the fallback run simply drops this one.
const FULL_JSON_FLAG: &str = "--output-json-full";

/// whisper-cli's own defaults, pinned so a runtime update cannot switch
/// decoding strategy underneath the baseline (whisper-server already defaults
/// to greedy). Greedy is ruled out for this pipeline — see [`run_whisper`].
const BEAM_SIZE: &str = "5";
const BEST_OF: &str = "5";

/// Why a `whisper-cli.exe` run failed, to the only resolution that differs.
enum SttFailure {
    /// This build does not know [`FULL_JSON_FLAG`]. It bails on the argument
    /// before it loads a model, so re-running without the flag is cheap.
    NoFullJson,
    Other(String),
}

/// One `whisper-cli.exe` invocation over the whole file.
///
/// Runs twice only in the one case that is recoverable: a whisper build too
/// old for [`FULL_JSON_FLAG`] would otherwise cost the whole meeting for the
/// sake of a timing refinement. Every other failure is reported on the first
/// attempt, because re-decoding an hour of audio to produce the same error
/// helps nobody.
fn single_pass(
    request: &SttRequest,
    language: &str,
) -> Result<(Vec<SttSegment>, StageMetrics), String> {
    match run_whisper(request, language, true) {
        Ok(outcome) => Ok(outcome),
        Err(SttFailure::Other(reason)) => Err(reason),
        Err(SttFailure::NoFullJson) => {
            eprintln!(
                "[stt] this whisper-cli.exe does not support {FULL_JSON_FLAG}; transcribing \
without word timestamps. Speaker attribution falls back to whole-segment overlap."
            );
            run_whisper(request, language, false).map_err(|failure| match failure {
                SttFailure::Other(reason) => reason,
                SttFailure::NoFullJson => unreachable!("the flag was not passed"),
            })
        }
    }
}

/// The `--max-len 80 --split-on-word --max-context 0` flags are load-bearing,
/// not tuning knobs — see docs/benchmarks.md: without them whisper.cpp has been
/// observed to silently collapse a real, multi-second span of speech into a
/// short, wrong sentence. `full_json` only ever *adds* an output flag; it must
/// never be allowed to disturb those three.
///
/// NOTE: `--translate` is never passed, and must never be added. The transcript must
/// preserve the language actually spoken; translating here
/// would destroy the original words for good. The minutes are the only thing
/// that gets rendered into another language, via `mom_language`.
fn run_whisper(
    request: &SttRequest,
    language: &str,
    full_json: bool,
) -> Result<(Vec<SttSegment>, StageMetrics), SttFailure> {
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
        // Beam search, pinned rather than inherited from whisper-cli's
        // defaults (which happen to be the same today), because greedy
        // decoding is *unsafe* alongside `--max-context 0`. Measured
        // 2026-10-06 on the baseline council clip: `--beam-size 1` cut the
        // last ~20 s into segments truncated mid-word ("that reca", "council
        // membo") — WER 16.7% -> 21.9%, deletions 7 -> 29 — with zero
        // temperature fallbacks, so the fallback did not catch it; dropping
        // `--max-context 0` alone made the truncation disappear. It was ~28%
        // faster (2:00 excerpt, i7-1260P: 39-43 s -> 29 s), which is not worth
        // silently losing words. See docs/benchmarks.md before changing this.
        .arg("--beam-size")
        .arg(BEAM_SIZE)
        .arg("--best-of")
        .arg(BEST_OF)
        .arg("-t")
        .arg(request.config.stt_threads.to_string())
        .arg("--output-json");
    if full_json {
        command.arg(FULL_JSON_FLAG);
    }
    command.arg("--output-file").arg(&output_base);

    let measured = proc::run_measured(command).map_err(SttFailure::Other)?;
    if !measured.success {
        if full_json && rejected_full_json(&measured.stderr) {
            return Err(SttFailure::NoFullJson);
        }
        return Err(SttFailure::Other(format!(
            "whisper-cli.exe exited with {}: {}",
            measured.status,
            last_lines(&measured.stderr, 4)
        )));
    }

    let json_path = wav_path.with_extension("json");
    let raw = std::fs::read_to_string(&json_path)
        .map_err(|e| SttFailure::Other(format!("could not read whisper output {json_path:?}: {e}")))?;
    let segments = parse_whisper_json(&raw).map_err(SttFailure::Other)?;
    Ok((segments, measured.metrics))
}

/// whisper.cpp answers an unrecognised flag with `error: unknown argument:
/// <flag>` and then prints its whole usage block, so this has to search the
/// entire stderr — [`last_lines`] would only ever see the option list.
fn rejected_full_json(stderr: &str) -> bool {
    stderr.contains("unknown argument") && stderr.contains(FULL_JSON_FLAG)
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
            // An undated token cannot be placed, so it is dropped here — which
            // then leaves the assembled words unable to reconstruct `text`,
            // which is exactly the condition `align` refuses to split on.
            let tokens: Vec<RawToken> = seg
                .tokens
                .iter()
                .filter_map(|token| {
                    token.offsets.as_ref().map(|offsets| RawToken {
                        text: &token.text,
                        start_ms: offsets.from,
                        end_ms: offsets.to,
                    })
                })
                .collect();
            Some(SttSegment {
                start_ms: seg.offsets.from,
                end_ms: seg.offsets.to,
                text,
                words: assemble_words(&tokens),
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

    // --------------------------------------------------------- word timing

    /// Abridged from a real `whisper-cli.exe --output-json-full` run against
    /// the vendored `ggml-small.bin`, keeping every shape that matters: the
    /// `[_BEG_]` / `[_TT_220]` markers, two BPE-split words, and a `;` stamped
    /// 390 ms after the word it belongs to. Only the `id`/`p`/`t_dtw` noise is
    /// trimmed. whisper cannot be run in CI, so the fixture is the honest
    /// stand-in — but this one was recorded, not invented.
    const FULL_JSON: &str = r#"{
        "transcription": [
            {
                "offsets": {"from": 0, "to": 4780},
                "text": " The diarization threshold and the quantization settings need recalibrating;",
                "tokens": [
                    {"text": "[_BEG_]",    "offsets": {"from": 0,    "to": 0},    "id": 50364, "p": 0.98, "t_dtw": -1},
                    {"text": " The",       "offsets": {"from": 120,  "to": 200}},
                    {"text": " di",        "offsets": {"from": 210,  "to": 350}},
                    {"text": "ar",         "offsets": {"from": 350,  "to": 490}},
                    {"text": "ization",    "offsets": {"from": 490,  "to": 980}},
                    {"text": " threshold", "offsets": {"from": 980,  "to": 1600}},
                    {"text": " and",       "offsets": {"from": 1610, "to": 1810}},
                    {"text": " the",       "offsets": {"from": 1820, "to": 1990}},
                    {"text": " quant",     "offsets": {"from": 2100, "to": 2380}},
                    {"text": "ization",    "offsets": {"from": 2380, "to": 2870}},
                    {"text": " settings",  "offsets": {"from": 2870, "to": 3420}},
                    {"text": " need",      "offsets": {"from": 3430, "to": 3710}},
                    {"text": " rec",       "offsets": {"from": 3710, "to": 3910}},
                    {"text": "al",         "offsets": {"from": 3920, "to": 4060}},
                    {"text": "ibr",        "offsets": {"from": 4060, "to": 4270}},
                    {"text": "ating",      "offsets": {"from": 4270, "to": 4310}},
                    {"text": ";",          "offsets": {"from": 4700, "to": 4700}},
                    {"text": "[_TT_220]",  "offsets": {"from": 4780, "to": 4780}, "id": 50584, "p": 0.07, "t_dtw": -1}
                ]
            }
        ]
    }"#;

    #[test]
    fn assembles_whisper_tokens_into_timed_words() {
        let segments = parse_whisper_json(FULL_JSON).unwrap();
        let words: Vec<&str> = segments[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            words,
            vec![
                "The",
                "diarization",
                "threshold",
                "and",
                "the",
                "quantization",
                "settings",
                "need",
                "recalibrating;",
            ]
        );
        // The words must spell out the segment, or align refuses to split on
        // them — this is the property that promises a split loses no text.
        assert_eq!(words.join(" "), segments[0].text);
    }

    #[test]
    fn a_word_split_across_tokens_spans_all_of_them() {
        let words = parse_whisper_json(FULL_JSON).unwrap().remove(0).words;
        // " di" + "ar" + "ization": 210 through 980, not 210 through 350.
        let diarization = &words[1];
        assert_eq!(diarization.text, "diarization");
        assert_eq!((diarization.start_ms, diarization.end_ms), (210, 980));
    }

    #[test]
    fn punctuation_joins_the_previous_word_without_stretching_it() {
        let words = parse_whisper_json(FULL_JSON).unwrap().remove(0).words;
        let last = words.last().unwrap();
        assert_eq!(last.text, "recalibrating;");
        // The `;` was stamped at 4700, but the speech stopped at 4310. Taking
        // the mark's timestamp would push the word 390 ms into silence.
        assert_eq!((last.start_ms, last.end_ms), (3710, 4310));
    }

    #[test]
    fn special_tokens_never_become_words() {
        let words = parse_whisper_json(FULL_JSON).unwrap().remove(0).words;
        assert!(
            !words.iter().any(|w| w.text.contains("[_")),
            "markers leaked into the transcript: {words:?}"
        );
        assert_eq!(words.first().unwrap().text, "The");
    }

    #[test]
    fn an_undated_token_is_dropped_rather_than_failing_the_document() {
        // whisper omits `offsets` on a token it has no timestamps for. The
        // transcript must survive that; the word simply cannot be placed.
        let raw = r#"{
            "transcription": [
                {"offsets": {"from": 0, "to": 900}, "text": "yes really",
                 "tokens": [
                    {"text": " yes", "offsets": {"from": 0, "to": 400}},
                    {"text": " really", "id": 12}
                 ]}
            ]
        }"#;
        let segments = parse_whisper_json(raw).unwrap();
        assert_eq!(segments[0].text, "yes really");
        assert_eq!(segments[0].words.len(), 1);
    }

    #[test]
    fn a_plain_output_json_file_still_parses_and_carries_no_words() {
        // Everything written before `--output-json-full`, plus the fallback
        // run when a whisper build rejects the flag.
        let raw = r#"{"transcription": [{"offsets": {"from": 0, "to": 900}, "text": "hello"}]}"#;
        let segments = parse_whisper_json(raw).unwrap();
        assert_eq!(segments[0].text, "hello");
        assert!(segments[0].words.is_empty());
    }

    #[test]
    fn a_stray_leading_token_starts_a_word_rather_than_being_lost() {
        let tokens = [
            RawToken { text: "unspaced", start_ms: 0, end_ms: 100 },
            RawToken { text: " ", start_ms: 100, end_ms: 100 },
            RawToken { text: "!", start_ms: 100, end_ms: 120 },
        ];
        let words = assemble_words(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "unspaced!");
        // Blank tokens contribute nothing, and the `!` moved no boundary.
        assert_eq!(words[0].end_ms, 100);
    }

    #[test]
    fn only_an_unknown_argument_error_triggers_the_no_flag_retry() {
        // whisper prints the rejected flag and then its whole usage block, so
        // the needle is nowhere near the tail of stderr.
        let usage = format!(
            "error: unknown argument: {FULL_JSON_FLAG}\n\nusage: whisper-cli.exe [options]\n  -t N\n  -oj\n"
        );
        assert!(rejected_full_json(&usage));
        assert!(!rejected_full_json("error: failed to open the audio file"));
        assert!(!rejected_full_json("error: unknown argument: --nonsense"));
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
