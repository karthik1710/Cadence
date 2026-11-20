//! `bench_cli baseline` — the regression gate. A fixed set of real clips with
//! human captions (`tests/baseline/clips.json`) runs through the shipped
//! pipeline, and each transcript is scored against its captions and against
//! the numbers recorded the last time someone accepted a change.
//!
//! The clips are copyrighted YouTube content. `scripts/fetch-baseline-clips.ps1`
//! downloads them into `bench-runs/baseline-cache/` (gitignored); this module
//! only reads that cache and never touches the network. A clip whose cache
//! files are missing is an error that names the script to run.
//!
//! What is gated and what is not:
//! - WER is gated *relative* to the recorded `expected` value, within
//!   `tolerance.wer_pp` percentage points either way (worse is REGRESSED,
//!   better is IMPROVED, no recorded value is NEW).
//! - Distinct speakers, the longest repeat run and repeated STT segments are
//!   gated *absolutely* by the clip's tolerance — they encode failure modes
//!   (over-splitting one talker, whisper looping) rather than a score.
//! - Minutes generation (`--with-mom` only) fails the gate when it errors.
//! - RTF and stage timings depend on the machine and on whatever else is
//!   running: reported, never gated.
//!
//! - A clip's `known_failures` names gated metrics that are already known to
//!   fail. Such a metric still reports, as KNOWN, but does not fail the exit
//!   code; once it passes it reports FIXED so the entry gets removed.
//!
//! Every clip runs the real `pipeline::run` with shipped defaults (not the
//! user's `settings.json`), exactly as `bench_cli compare`'s `shipped` row
//! does.
//!
//! `--reuse <run-dir>` is the fast loop for diarization/alignment work: it
//! takes an earlier run's `processed.wav` and `stt.json` per clip (front end,
//! VAD and STT, which are most of the time) and re-runs only the real
//! `diarization::diarize` and `align::align`, then scores. Every clip directory
//! records the audio it ran on in `source.json`; reuse refuses when the cached
//! clip's hash (or, for a run older than `source.json`, its length) no longer
//! matches.

use super::compare;
use super::render;
use super::text::{self, EditCounts};
use app_lib::pipeline::models::{Mom, SttSegment, Transcript};
use app_lib::pipeline::paths::ModelPaths;
use app_lib::pipeline::{self, align, diarization, PipelineConfig, RunMetrics, RunRequest};
use app_lib::settings::Settings;
use app_lib::system::{self, HostInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: bench_cli baseline [--clips id,..] [--cache <dir>] [--update] [--out <dir>] [--with-mom] [--reuse <baseline-run-dir>]";

/// The gated metrics a clip's `known_failures` may name.
const GATED_METRICS: &[&str] = &["wer", "speakers", "longest_repeat_run", "repeated_segments", "minutes"];

const FETCH_SCRIPT: &str = "scripts/fetch-baseline-clips.ps1";

// ------------------------------------------------------------- manifest

/// `tests/baseline/clips.json`. Typed rather than a `serde_json::Value` so
/// `--update` writes the fields back in the order a person wrote them;
/// `extra` keeps any field this code does not know about.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    about: Vec<String>,
    clips: Vec<Clip>,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Clip {
    id: String,
    title: String,
    url: String,
    caption_track: String,
    /// Language of the caption text (`en`, `ta`, ...), shown in the report.
    /// Absent means English.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caption_language: Option<String>,
    duration_s: f64,
    what_it_tests: String,
    caption_quality: String,
    tolerance: Tolerance,
    /// Gated metrics (`GATED_METRICS`) already known to fail on this clip.
    /// They report KNOWN instead of failing the run, and FIXED once they pass.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    known_failures: Vec<String>,
    /// `null` until the first `--update`.
    expected: Option<Expected>,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tolerance {
    /// Allowed WER drift either way, in absolute percentage points.
    wer_pp: f64,
    speakers_min: usize,
    speakers_max: usize,
    /// Highest acceptable `longest repeat run` (times an n-gram repeats back
    /// to back). Natural speech reaches 2–3; a whisper loop goes far past it.
    max_repeat_run: usize,
    max_repeated_segments: usize,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Expected {
    wer_pct: f64,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    /// Words in the caption reference. If this changes, the caption track or
    /// the VTT parser changed, and WER is no longer comparable.
    reference_words: usize,
    words: usize,
    speakers: usize,
    longest_repeat_run: usize,
    repeated_segments: usize,
    /// Pipeline time (minutes excluded) over audio length. Machine-dependent.
    rtf: Option<f64>,
    total_s: Option<f64>,
    recorded: Recorded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Recorded {
    date: String,
    git_head: Option<String>,
    total_ram_mb: u64,
    logical_cores: usize,
    stt_model: String,
    stt_language: String,
    stt_threads: usize,
    diarization_threads: usize,
    how: String,
}

fn manifest_path() -> PathBuf {
    compare::repo_root().join("tests").join("baseline").join("clips.json")
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    parse_manifest(&raw).map_err(|e| format!("{} is not a valid manifest: {e}", path.display()))
}

fn parse_manifest(raw: &str) -> Result<Manifest, String> {
    let manifest: Manifest = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    for clip in &manifest.clips {
        for name in &clip.known_failures {
            if !GATED_METRICS.contains(&name.as_str()) {
                return Err(format!(
                    "clip `{}` lists unknown known_failures entry `{name}` (gated metrics: {})",
                    clip.id,
                    GATED_METRICS.join(", ")
                ));
            }
        }
    }
    Ok(manifest)
}

fn manifest_json(manifest: &Manifest) -> Result<String, String> {
    serde_json::to_string_pretty(manifest)
        .map(|s| s + "\n")
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- args

struct Args {
    clips: Option<Vec<String>>,
    cache: Option<PathBuf>,
    out: Option<PathBuf>,
    update: bool,
    with_mom: bool,
    /// An earlier baseline run whose processed audio and STT to reuse.
    reuse: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        clips: None,
        cache: None,
        out: None,
        update: false,
        with_mom: false,
        reuse: None,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |flag: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--clips" => {
                parsed.clips = Some(
                    value("--clips")?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                )
            }
            "--cache" => parsed.cache = Some(PathBuf::from(value("--cache")?)),
            "--out" => parsed.out = Some(PathBuf::from(value("--out")?)),
            "--update" => parsed.update = true,
            "--with-mom" => parsed.with_mom = true,
            "--reuse" => parsed.reuse = Some(PathBuf::from(value("--reuse")?)),
            other => return Err(format!("unknown argument: {other}\n{USAGE}")),
        }
    }
    if parsed.reuse.is_some() && parsed.update {
        // `expected` carries RTF and total time, which a reuse run does not
        // measure, and an expected block should come from one whole run.
        return Err("--reuse cannot be combined with --update: record `expected` from a full run".to_string());
    }
    if parsed.reuse.is_some() && parsed.with_mom {
        // Minutes read the redaction flags the full pipeline attaches; a
        // reuse run only diarizes and aligns.
        return Err("--reuse cannot be combined with --with-mom: minutes need a full run".to_string());
    }
    Ok(parsed)
}

// ------------------------------------------------------------------ VTT

/// WebVTT captions → plain reference text, one caption line per line.
///
/// Drops the header (`WEBVTT`, `Kind:`, `Language:`, `NOTE`), timing lines
/// (anything with `-->`, including `line:`/`align:` settings), inline tags
/// (`<00:00:01.033>`, `<c>`), broadcast `>>` speaker-change marks, and
/// bracketed sound annotations (`(Laughter)`, `[MUSIC]`). Rolling captions
/// repeat the previous line in the next cue, so a line equal to the one
/// before it is dropped. Pure, so it is unit-tested on a synthetic snippet.
pub fn vtt_to_reference(vtt: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for raw in vtt.lines() {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty()
            || line.starts_with("WEBVTT")
            || line.starts_with("Kind:")
            || line.starts_with("Language:")
            || line.starts_with("NOTE")
            || line.contains("-->")
        {
            continue;
        }
        let text = decode_entities(&strip_tags(line)).replace(">>", " ");
        let text = strip_annotations(&text);
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !collapsed.is_empty() && lines.last() != Some(&collapsed) {
            lines.push(collapsed);
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// Removes `<...>` tags. A `<` with no closing `>`, or `<>`, is kept as text.
fn strip_tags(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find('>') {
            Some(close) if close > 0 => rest = &after[close + 1..],
            _ => {
                out.push('<');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn decode_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

/// Replaces `( ... )` and `[ ... ]` spans (unnested, within the line) with a
/// space. An unclosed bracket is kept as text.
fn strip_annotations(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let close = match chars[i] {
            '(' => Some(')'),
            '[' => Some(']'),
            _ => None,
        };
        if let Some(close) = close {
            if let Some(offset) = chars[i + 1..].iter().position(|&c| c == close) {
                out.push(' ');
                i += offset + 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

// -------------------------------------------------------------- measuring

/// What one clip measured.
#[derive(Debug, Clone)]
struct Measured {
    wer: EditCounts,
    words: usize,
    speakers: usize,
    longest_repeat: usize,
    longest_repeat_ngram: String,
    repeated_segments: usize,
    audio_ms: i64,
    /// Pipeline time with minutes generation excluded.
    pipeline_ms: u64,
    metrics: RunMetrics,
    minutes: MinutesOutcome,
    /// Set when the front end and STT came from an earlier run (`--reuse`).
    /// `metrics` then mixes that run's front-end/STT stages with this run's
    /// diarization and alignment, and `pipeline_ms` covers only the latter.
    reused: Option<ReusePlan>,
}

#[derive(Debug, Clone)]
enum MinutesOutcome {
    NotRequested,
    Skipped(String),
    Parsed {
        chunks: usize,
        failed_chunks: usize,
        repaired: bool,
    },
    Failed(String),
}

fn measure(
    reference: &str,
    transcript: &Transcript,
    stt: &[SttSegment],
    metrics: RunMetrics,
    minutes: MinutesOutcome,
) -> Measured {
    let hypothesis = transcript
        .segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let words: Vec<String> = transcript
        .segments
        .iter()
        .flat_map(|s| text::normalize_words(&s.text))
        .collect();
    let repeat = text::longest_repeat_run(&words);
    let texts: Vec<String> = stt.iter().map(|s| s.text.clone()).collect();
    Measured {
        wer: text::wer(reference, &hypothesis),
        words: render::word_count(transcript),
        speakers: render::speaker_count(transcript),
        // No run at all counts as 1: every word appears once.
        longest_repeat: repeat.as_ref().map(|r| r.times).unwrap_or(1),
        longest_repeat_ngram: repeat.map(|r| r.ngram).unwrap_or_default(),
        repeated_segments: text::repeated_segments(&texts),
        audio_ms: metrics.audio_ms,
        pipeline_ms: metrics.total_ms.saturating_sub(metrics.mom_ms),
        metrics,
        minutes,
        reused: None,
    }
}

impl Measured {
    fn wer_pct(&self) -> f64 {
        self.wer.rate().unwrap_or(0.0) * 100.0
    }

    /// `None` for a reuse run: it timed only diarization and alignment, so a
    /// ratio against the recording would not be the pipeline's.
    fn rtf(&self) -> Option<f64> {
        (self.audio_ms > 0 && self.reused.is_none())
            .then(|| self.pipeline_ms as f64 / self.audio_ms as f64)
    }
}

// --------------------------------------------------------------- verdicts

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict {
    Pass,
    Regressed,
    Improved,
    New,
    /// Reported only — machine-dependent or derived from a gated metric.
    Info,
    /// Would be REGRESSED, but the clip lists it in `known_failures`: reported,
    /// not gating.
    Known,
    /// Listed in `known_failures` but now passing: the entry should go.
    Fixed,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Regressed => "**REGRESSED**",
            Verdict::Improved => "IMPROVED",
            Verdict::New => "NEW",
            Verdict::Info => "info",
            Verdict::Known => "KNOWN (regressed, in known_failures)",
            Verdict::Fixed => "**FIXED — remove from known_failures**",
        }
    }

    /// What a gated verdict becomes when its metric is a known failure. NEW
    /// and info rows have nothing to compare, so they are left alone.
    fn as_known_failure(self) -> Verdict {
        match self {
            Verdict::Regressed => Verdict::Known,
            Verdict::Pass | Verdict::Improved => Verdict::Fixed,
            other => other,
        }
    }
}

/// Relative gate: lower is better, `tolerance` either way.
fn relative(expected: Option<f64>, actual: f64, tolerance: f64) -> Verdict {
    match expected {
        None => Verdict::New,
        Some(e) if actual > e + tolerance + 1e-9 => Verdict::Regressed,
        Some(e) if actual < e - tolerance - 1e-9 => Verdict::Improved,
        Some(_) => Verdict::Pass,
    }
}

/// Absolute gate: `ok` decides; a baseline that already failed and now
/// passes is IMPROVED.
fn absolute(ok: impl Fn(usize) -> bool, expected: Option<usize>, actual: usize) -> Verdict {
    if !ok(actual) {
        Verdict::Regressed
    } else if expected.is_some_and(|e| !ok(e)) {
        Verdict::Improved
    } else {
        Verdict::Pass
    }
}

struct Row {
    /// The `GATED_METRICS` name a `known_failures` entry uses; `None` for
    /// rows that are never gated.
    key: Option<&'static str>,
    metric: String,
    expected: String,
    actual: String,
    delta: String,
    verdict: Verdict,
    /// Counts towards the exit code even under `--update` (absolute gates).
    hard: bool,
}

fn signed(delta: f64, decimals: usize, unit: &str) -> String {
    if delta.abs() < 0.5 * 10f64.powi(-(decimals as i32)) {
        return "0".to_string();
    }
    format!("{delta:+.decimals$}{unit}")
}

fn int_delta(expected: Option<usize>, actual: usize) -> String {
    match expected {
        Some(e) => signed(actual as f64 - e as f64, 0, ""),
        None => "—".to_string(),
    }
}

fn opt<T: ToString>(value: Option<T>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "—".to_string())
}

fn rows_for(clip: &Clip, m: &Measured) -> Vec<Row> {
    let e = clip.expected.as_ref();
    let t = &clip.tolerance;
    let mut rows = Vec::new();

    let wer = m.wer_pct();
    rows.push(Row {
        key: Some("wer"),
        metric: format!("WER (±{} pp)", t.wer_pp),
        expected: opt(e.map(|e| format!("{:.1}%", e.wer_pct))),
        actual: format!("{wer:.1}%"),
        delta: e
            .map(|e| signed(wer - e.wer_pct, 1, " pp"))
            .unwrap_or_else(|| "—".to_string()),
        verdict: relative(e.map(|e| e.wer_pct), wer, t.wer_pp),
        hard: false,
    });
    rows.push(Row {
        key: None,
        metric: "S / D / I".to_string(),
        expected: opt(e.map(|e| format!("{}/{}/{}", e.substitutions, e.deletions, e.insertions))),
        actual: format!(
            "{}/{}/{}",
            m.wer.substitutions, m.wer.deletions, m.wer.insertions
        ),
        delta: "—".to_string(),
        verdict: Verdict::Info,
        hard: false,
    });
    rows.push(Row {
        key: None,
        metric: "reference words".to_string(),
        expected: opt(e.map(|e| e.reference_words)),
        actual: m.wer.reference_words.to_string(),
        delta: int_delta(e.map(|e| e.reference_words), m.wer.reference_words),
        verdict: Verdict::Info,
        hard: false,
    });
    rows.push(Row {
        key: None,
        metric: "transcript words".to_string(),
        expected: opt(e.map(|e| e.words)),
        actual: m.words.to_string(),
        delta: int_delta(e.map(|e| e.words), m.words),
        verdict: Verdict::Info,
        hard: false,
    });
    let speakers_ok = |n: usize| n >= t.speakers_min && n <= t.speakers_max;
    rows.push(Row {
        key: Some("speakers"),
        metric: format!("speakers ({}–{})", t.speakers_min, t.speakers_max),
        expected: opt(e.map(|e| e.speakers)),
        actual: m.speakers.to_string(),
        delta: int_delta(e.map(|e| e.speakers), m.speakers),
        verdict: absolute(speakers_ok, e.map(|e| e.speakers), m.speakers),
        hard: true,
    });
    let repeat_ok = |n: usize| n <= t.max_repeat_run;
    rows.push(Row {
        key: Some("longest_repeat_run"),
        metric: format!("longest repeat run (≤{})", t.max_repeat_run),
        expected: opt(e.map(|e| e.longest_repeat_run)),
        actual: if m.longest_repeat_ngram.is_empty() {
            m.longest_repeat.to_string()
        } else {
            format!(
                "{}× \"{}\"",
                m.longest_repeat,
                render::cell(&m.longest_repeat_ngram)
            )
        },
        delta: int_delta(e.map(|e| e.longest_repeat_run), m.longest_repeat),
        verdict: absolute(repeat_ok, e.map(|e| e.longest_repeat_run), m.longest_repeat),
        hard: true,
    });
    let segs_ok = |n: usize| n <= t.max_repeated_segments;
    rows.push(Row {
        key: Some("repeated_segments"),
        metric: format!("repeated segments (≤{})", t.max_repeated_segments),
        expected: opt(e.map(|e| e.repeated_segments)),
        actual: m.repeated_segments.to_string(),
        delta: int_delta(e.map(|e| e.repeated_segments), m.repeated_segments),
        verdict: absolute(segs_ok, e.map(|e| e.repeated_segments), m.repeated_segments),
        hard: true,
    });
    let rtf = m.rtf();
    rows.push(Row {
        key: None,
        metric: "RTF (machine-dependent)".to_string(),
        expected: opt(e.and_then(|e| e.rtf).map(|r| format!("{r:.2}x"))),
        actual: if m.reused.is_some() {
            "— (reuse run: only diarization + align timed)".to_string()
        } else {
            opt(rtf.map(|r| format!("{r:.2}x")))
        },
        delta: match (e.and_then(|e| e.rtf), rtf) {
            (Some(a), Some(b)) => signed(b - a, 2, "x"),
            _ => "—".to_string(),
        },
        verdict: Verdict::Info,
        hard: false,
    });
    match &m.minutes {
        MinutesOutcome::NotRequested => {}
        MinutesOutcome::Skipped(why) => rows.push(Row {
            key: None,
            metric: "minutes".to_string(),
            expected: "—".to_string(),
            actual: format!("skipped: {}", render::cell(why)),
            delta: "—".to_string(),
            verdict: Verdict::Info,
            hard: false,
        }),
        MinutesOutcome::Parsed {
            chunks,
            failed_chunks,
            repaired,
        } => rows.push(Row {
            key: Some("minutes"),
            metric: "minutes".to_string(),
            expected: "parsed".to_string(),
            actual: format!(
                "parsed · {chunks} chunk(s), {failed_chunks} failed{}",
                if *repaired { ", repaired" } else { "" }
            ),
            delta: "—".to_string(),
            verdict: Verdict::Pass,
            hard: true,
        }),
        MinutesOutcome::Failed(err) => rows.push(Row {
            key: Some("minutes"),
            metric: "minutes".to_string(),
            expected: "parsed".to_string(),
            actual: format!("FAILED: {}", render::cell(err)),
            delta: "—".to_string(),
            verdict: Verdict::Regressed,
            hard: true,
        }),
    }
    for row in &mut rows {
        if row.key.is_some_and(|k| clip.known_failures.iter().any(|f| f == k)) {
            row.verdict = row.verdict.as_known_failure();
        }
    }
    rows
}

fn expected_from(m: &Measured, recorded: Recorded) -> Expected {
    let round = |x: f64, places: i32| {
        let scale = 10f64.powi(places);
        (x * scale).round() / scale
    };
    Expected {
        wer_pct: round(m.wer_pct(), 2),
        substitutions: m.wer.substitutions,
        deletions: m.wer.deletions,
        insertions: m.wer.insertions,
        reference_words: m.wer.reference_words,
        words: m.words,
        speakers: m.speakers,
        longest_repeat_run: m.longest_repeat,
        repeated_segments: m.repeated_segments,
        rtf: m.rtf().map(|r| round(r, 2)),
        total_s: Some(round(m.pipeline_ms as f64 / 1000.0, 1)),
        recorded,
    }
}

// ------------------------------------------------------------------- reuse

/// Identifies a clip's cached input audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct AudioFingerprint {
    sha256: String,
    bytes: u64,
    duration_ms: i64,
}

/// `<out>/<clip>/source.json`, written by every run: which audio the clip ran
/// on, and the clip directory holding the `processed.wav`, `stt.json` and
/// `metrics.json` its transcript was built from — the run's own directory, or
/// with `--reuse` the reused one's, so a reuse of a reuse still finds them.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SourceRecord {
    input_wav: String,
    input: AudioFingerprint,
    stages_from: String,
}

/// Where a reuse run takes the front end and STT for one clip from.
#[derive(Debug, Clone)]
struct ReusePlan {
    /// The clip directory holding `processed.wav`, `stt.json`, `metrics.json`.
    stages_from: PathBuf,
    /// How the cached audio was matched against that run, for the report.
    matched_by: String,
}

/// A WAV whose length differs from the cached clip's by more than this was
/// not made from it. Resampling and the front end preserve length exactly.
const REUSE_DURATION_SLACK_MS: i64 = 20;

fn wav_duration_ms(path: &Path) -> Result<i64, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| format!("could not read {} as WAV: {e}", path.display()))?;
    let rate = reader.spec().sample_rate.max(1) as i64;
    Ok(reader.duration() as i64 * 1000 / rate)
}

fn fingerprint(path: &Path) -> Result<AudioFingerprint, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(AudioFingerprint {
        sha256,
        bytes: bytes.len() as u64,
        duration_ms: wav_duration_ms(path)?,
    })
}

/// Checks that `run_dir`'s `label` clip can be reused for audio `input`, and
/// says where its stages are. Refuses — never guesses — when the cached audio
/// is not provably the audio that run processed.
fn plan_reuse(run_dir: &Path, label: &str, input: &AudioFingerprint) -> Result<ReusePlan, String> {
    let dir = run_dir.join(label);
    if !dir.is_dir() {
        return Err(format!(
            "{} has no `{label}` clip directory (was the clip in that run?)",
            run_dir.display()
        ));
    }
    let source_file = dir.join("source.json");
    let (stages_from, hash_checked) = if source_file.is_file() {
        let raw = std::fs::read_to_string(&source_file)
            .map_err(|e| format!("could not read {}: {e}", source_file.display()))?;
        let record: SourceRecord = serde_json::from_str(&raw)
            .map_err(|e| format!("{} is not a valid source record: {e}", source_file.display()))?;
        if record.input.sha256 != input.sha256 {
            return Err(format!(
                "refusing to reuse {}: the cached audio is not the audio that run processed (sha256 {} then, {} now). Re-fetched or replaced? Run without --reuse.",
                dir.display(),
                &record.input.sha256[..record.input.sha256.len().min(12)],
                &input.sha256[..12]
            ));
        }
        (PathBuf::from(record.stages_from), true)
    } else {
        // A run from before `source.json` existed: no hash was recorded, so
        // the processed audio's length is the only evidence left.
        (dir.clone(), false)
    };
    for file in ["processed.wav", "stt.json"] {
        if !stages_from.join(file).is_file() {
            return Err(format!(
                "cannot reuse {}: {} is missing",
                dir.display(),
                stages_from.join(file).display()
            ));
        }
    }
    let processed_ms = wav_duration_ms(&stages_from.join("processed.wav"))?;
    if (processed_ms - input.duration_ms).abs() > REUSE_DURATION_SLACK_MS {
        return Err(format!(
            "refusing to reuse {}: its processed.wav is {:.2} s but the cached clip is {:.2} s, so it was not made from this audio. Run without --reuse.",
            dir.display(),
            processed_ms as f64 / 1000.0,
            input.duration_ms as f64 / 1000.0
        ));
    }
    let matched_by = if hash_checked {
        format!("input sha256 matched ({}…)", &input.sha256[..12])
    } else {
        format!(
            "that run predates source.json, so no input hash was recorded; its processed.wav length matched the cached clip ({:.2} s)",
            input.duration_ms as f64 / 1000.0
        )
    };
    Ok(ReusePlan {
        stages_from,
        matched_by,
    })
}

fn write_source(dir: &Path, wav: &Path, input: &AudioFingerprint, stages_from: &Path) {
    let absolute = |p: &Path| {
        std::path::absolute(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .display()
            .to_string()
    };
    let record = SourceRecord {
        input_wav: absolute(wav),
        input: input.clone(),
        stages_from: absolute(stages_from),
    };
    if let Ok(json) = serde_json::to_string_pretty(&record) {
        let _ = std::fs::write(dir.join("source.json"), json);
    }
}

// ----------------------------------------------------------------- running

enum ClipResult {
    Ran(Box<Measured>),
    Failed(String),
}

struct Ctx {
    data_dir: PathBuf,
    host: HostInfo,
    settings: Settings,
    config: PipelineConfig,
    out: PathBuf,
}

pub fn run(raw_args: &[String]) -> Result<(), String> {
    let args = parse_args(raw_args)?;
    let manifest_file = manifest_path();
    let mut manifest = load_manifest(&manifest_file)?;
    let stamp = compare::utc_stamp();
    let data_dir = pipeline::paths::default_data_dir()?;
    let cache = args
        .cache
        .clone()
        .unwrap_or_else(|| compare::repo_root().join("bench-runs").join("baseline-cache"));
    let out = args.out.clone().unwrap_or_else(|| {
        compare::repo_root()
            .join("bench-runs")
            .join(format!("{stamp}-baseline"))
    });
    compare::guard_out_dir(&out, &data_dir)?;

    // Which clips, and are they all cached? Checked before any work.
    let selected: Vec<usize> = match &args.clips {
        None => (0..manifest.clips.len()).collect(),
        Some(wanted) => {
            let mut indices = Vec::new();
            for id in wanted {
                let index = manifest.clips.iter().position(|c| &c.id == id).ok_or_else(|| {
                    format!(
                        "unknown clip `{id}`. Known: {}",
                        manifest
                            .clips
                            .iter()
                            .map(|c| c.id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
                indices.push(index);
            }
            indices
        }
    };
    if selected.is_empty() {
        return Err(format!("no clips in {}", manifest_file.display()));
    }
    let mut missing = Vec::new();
    for &i in &selected {
        let id = &manifest.clips[i].id;
        for file in [cache.join(format!("{id}.16k.wav")), cache.join(format!("{id}.vtt"))] {
            if !file.is_file() {
                missing.push(file.display().to_string());
            }
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "baseline clips are not cached:\n  {}\nRun `powershell -ExecutionPolicy Bypass -File {FETCH_SCRIPT}` from the repo root first \
             (audio is copyrighted YouTube content, cached locally only).",
            missing.join("\n  ")
        ));
    }

    // Every clip's audio is fingerprinted for its source.json; with --reuse,
    // every clip's reuse is checked before any work starts.
    if let Some(reuse) = &args.reuse {
        if !reuse.is_dir() {
            return Err(format!("--reuse: {} is not a directory", reuse.display()));
        }
        if std::path::absolute(reuse).ok() == std::path::absolute(&out).ok() {
            return Err("--reuse and --out must be different directories".to_string());
        }
    }
    let mut inputs: Vec<AudioFingerprint> = Vec::new();
    let mut plans: Vec<Option<ReusePlan>> = Vec::new();
    let mut refusals: Vec<String> = Vec::new();
    for &i in &selected {
        let id = &manifest.clips[i].id;
        let input = fingerprint(&cache.join(format!("{id}.16k.wav")))?;
        match &args.reuse {
            None => plans.push(None),
            Some(run_dir) => match plan_reuse(run_dir, &compare::file_safe(id), &input) {
                Ok(plan) => plans.push(Some(plan)),
                Err(err) => refusals.push(format!("{id}: {err}")),
            },
        }
        inputs.push(input);
    }
    if !refusals.is_empty() {
        return Err(format!("--reuse refused:\n  {}", refusals.join("\n  ")));
    }

    // Shipped defaults, not the user's settings.json.
    let host = HostInfo::detect();
    let mut settings = Settings {
        generate_mom: false,
        ..Settings::default()
    };
    let required = ModelPaths::unchecked(&data_dir, &settings).missing(&data_dir, &settings);
    if !required.is_empty() {
        return Err(format!(
            "required runtime/models not installed: {} — run the app's first-run setup (or scripts/setup-models.ps1)",
            required.join(", ")
        ));
    }
    let mut mom_skip: Option<String> = None;
    if args.with_mom {
        let with_llm = Settings {
            generate_mom: true,
            ..settings.clone()
        };
        let llm_missing = ModelPaths::unchecked(&data_dir, &with_llm).missing(&data_dir, &with_llm);
        if llm_missing.is_empty() {
            settings = with_llm;
        } else {
            mom_skip = Some(format!("not installed: {}", llm_missing.join(", ")));
        }
    }
    let config = PipelineConfig::from_settings(&settings, &host);
    std::fs::create_dir_all(&out).map_err(|e| format!("could not create {}: {e}", out.display()))?;
    let ctx = Ctx {
        data_dir,
        host,
        settings,
        config,
        out,
    };

    eprintln!(
        "bench baseline: {} clip(s) from {} → {}",
        selected.len(),
        cache.display(),
        ctx.out.display()
    );

    let mut results: Vec<(usize, ClipResult)> = Vec::new();
    for (n, &i) in selected.iter().enumerate() {
        let clip = &manifest.clips[i];
        eprintln!("\n== {} — {}", clip.id, clip.title);
        let result = run_clip(
            &ctx,
            clip,
            &cache,
            &inputs[n],
            plans[n].as_ref(),
            args.with_mom,
            mom_skip.as_deref(),
        );
        match &result {
            ClipResult::Ran(m) => eprintln!(
                "   done in {:.1}s, {} words, WER {:.1}%",
                m.pipeline_ms as f64 / 1000.0,
                m.words,
                m.wer_pct()
            ),
            ClipResult::Failed(err) => eprintln!("   FAILED: {err}"),
        }
        results.push((i, result));
    }

    // ------------------------------------------------------------ verdicts
    let mut report = report_header(&ctx, &stamp, &manifest_file, &cache, &args);
    let mut regressions: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut known: Vec<String> = Vec::new();
    let mut fixed: Vec<String> = Vec::new();
    for (i, result) in &results {
        let clip = &manifest.clips[*i];
        report.push_str(&format!("## {} — {}\n\n", clip.id, clip.title));
        report.push_str(&format!(
            "_{}_ Captions{}: {}\n\n",
            clip.what_it_tests,
            clip.caption_language
                .as_ref()
                .map(|l| format!(" (`{l}`)"))
                .unwrap_or_default(),
            clip.caption_quality
        ));
        match result {
            ClipResult::Failed(err) => {
                report.push_str(&format!("**Pipeline FAILED:** {}\n\n", render::cell(err)));
                failures.push(format!("{}: pipeline failed", clip.id));
            }
            ClipResult::Ran(m) => {
                let rows = rows_for(clip, m);
                if let Some(plan) = &m.reused {
                    report.push_str(&format!(
                        "**Reused** front end, VAD and STT (`processed.wav`, `stt.json`) from `{}` ({}); diarization and alignment re-run now.\n\n",
                        plan.stages_from.display(),
                        plan.matched_by
                    ));
                }
                report.push_str(&clip_table(&rows));
                if let Some(e) = &clip.expected {
                    if e.reference_words != m.wer.reference_words {
                        report.push_str(&format!(
                            "> **Reference changed** ({} → {} words): the caption track or the VTT parser changed, so this WER is not comparable with the recorded one.\n\n",
                            e.reference_words, m.wer.reference_words
                        ));
                    }
                    report.push_str(&format!(
                        "Expected recorded {} on {} MB RAM / {} logical cores{} ({}).\n\n",
                        e.recorded.date,
                        e.recorded.total_ram_mb,
                        e.recorded.logical_cores,
                        e.recorded
                            .git_head
                            .as_ref()
                            .map(|h| format!(", git {h}"))
                            .unwrap_or_default(),
                        e.recorded.how
                    ));
                }
                report.push_str(&stage_table(m));
                for row in &rows {
                    let line = format!("{}: {} {} (expected {})", clip.id, row.metric, row.actual, row.expected);
                    match row.verdict {
                        Verdict::Regressed if row.hard => failures.push(line),
                        Verdict::Regressed => regressions.push(line),
                        Verdict::Known => known.push(line),
                        Verdict::Fixed => fixed.push(format!(
                            "{}: {} now passes ({}): remove `{}` from its known_failures",
                            clip.id,
                            row.metric,
                            row.actual,
                            row.key.unwrap_or_default()
                        )),
                        _ => {}
                    }
                }
                if !m.metrics.warnings.is_empty() {
                    report.push_str(if m.reused.is_some() {
                        "Pipeline warnings (from the reused run's front end and STT):\n\n"
                    } else {
                        "Pipeline warnings:\n\n"
                    });
                    for warning in &m.metrics.warnings {
                        report.push_str(&format!("- {}\n", render::cell(warning)));
                    }
                    report.push('\n');
                }
            }
        }
    }

    // ------------------------------------------------------------- update
    if args.update {
        let recorded = Recorded {
            date: format!("{}-{}-{}", &stamp[0..4], &stamp[4..6], &stamp[6..8]),
            git_head: git_head(&compare::repo_root()),
            total_ram_mb: ctx.host.total_ram_mb,
            logical_cores: ctx.host.logical_cores,
            stt_model: system::STT_MODEL_ID.to_string(),
            stt_language: ctx.config.stt_language.clone(),
            stt_threads: ctx.config.stt_threads,
            diarization_threads: ctx.config.diarization_threads,
            how: "bench_cli baseline --update".to_string(),
        };
        let mut updated = Vec::new();
        for (i, result) in &results {
            if let ClipResult::Ran(m) = result {
                manifest.clips[*i].expected = Some(expected_from(m, recorded.clone()));
                updated.push(manifest.clips[*i].id.clone());
            }
        }
        if !updated.is_empty() {
            std::fs::write(&manifest_file, manifest_json(&manifest)?)
                .map_err(|e| format!("could not write {}: {e}", manifest_file.display()))?;
        }
        report.push_str(&format!(
            "## Updated\n\n`expected` rewritten in `{}` for: {}. Relative (WER) changes are accepted by this run; absolute gates still apply.\n\n",
            manifest_file.display(),
            if updated.is_empty() { "nothing".to_string() } else { updated.join(", ") }
        ));
    }

    let gate_failures: Vec<&String> = if args.update {
        failures.iter().collect()
    } else {
        failures.iter().chain(regressions.iter()).collect()
    };
    report.push_str("## Verdict\n\n");
    if gate_failures.is_empty() {
        report.push_str(if known.is_empty() {
            "**PASS** — no gated metric regressed beyond tolerance.\n"
        } else {
            "**PASS** — no gated metric regressed beyond tolerance, apart from the known failures below.\n"
        });
    } else {
        report.push_str("**FAIL**\n\n");
        for line in &gate_failures {
            report.push_str(&format!("- {}\n", render::cell(line)));
        }
    }
    if !known.is_empty() {
        report.push_str("\nKnown failures (listed in the clip's `known_failures`; still failing, not gating):\n\n");
        for line in &known {
            report.push_str(&format!("- {}\n", render::cell(line)));
        }
    }
    if !fixed.is_empty() {
        report.push_str("\n**FIXED — remove from known_failures** (a listed known failure now passes; take it out of `clips.json` so a relapse fails the gate again):\n\n");
        for line in &fixed {
            report.push_str(&format!("- {}\n", render::cell(line)));
        }
    }

    let report_path = ctx.out.join("report.md");
    std::fs::write(&report_path, &report)
        .map_err(|e| format!("could not write {}: {e}", report_path.display()))?;
    println!("{report}");
    eprintln!("\nReport written to {}", report_path.display());

    if gate_failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "baseline regressed ({} gated failure(s)); see {}",
            gate_failures.len(),
            report_path.display()
        ))
    }
}

fn run_clip(
    ctx: &Ctx,
    clip: &Clip,
    cache: &Path,
    input: &AudioFingerprint,
    reuse: Option<&ReusePlan>,
    with_mom: bool,
    mom_skip: Option<&str>,
) -> ClipResult {
    let wav = cache.join(format!("{}.16k.wav", clip.id));
    let vtt = cache.join(format!("{}.vtt", clip.id));
    let reference = match std::fs::read_to_string(&vtt) {
        Ok(raw) => vtt_to_reference(&raw),
        Err(e) => return ClipResult::Failed(format!("could not read {}: {e}", vtt.display())),
    };
    let label = compare::file_safe(&clip.id);
    let _ = std::fs::write(ctx.out.join(format!("{label}.ref.txt")), &reference);

    let dir = ctx.out.join(&label);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return ClipResult::Failed(e.to_string());
    }

    if let Some(plan) = reuse {
        return match rerun_after_stt(ctx, clip, &dir, plan) {
            Ok((transcript, stt, metrics)) => {
                write_source(&dir, &wav, input, &plan.stages_from);
                let mut measured = measure(
                    &reference,
                    &transcript,
                    &stt,
                    metrics,
                    MinutesOutcome::NotRequested,
                );
                measured.reused = Some(plan.clone());
                ClipResult::Ran(Box::new(measured))
            }
            Err(err) => ClipResult::Failed(err),
        };
    }

    let paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    let outcome = match pipeline::run(
        &RunRequest {
            paths: &paths,
            config: &ctx.config,
            settings: &ctx.settings,
            host: &ctx.host,
            input_wav: &wav,
            out_dir: &dir,
            data_dir: &ctx.data_dir,
            meeting_id: &clip.id,
            title: &clip.title,
        },
        &|stage, pct| eprintln!("   [{}] {stage} {pct}%", clip.id),
    ) {
        Ok(outcome) => outcome,
        Err(err) => return ClipResult::Failed(err),
    };
    for scratch in compare::SCRATCH {
        let _ = std::fs::remove_file(dir.join(scratch));
    }
    write_source(&dir, &wav, input, &dir);
    let stt: Vec<SttSegment> = std::fs::read_to_string(dir.join("stt.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    let _ = std::fs::write(
        dir.join("transcript.txt"),
        render::transcript_text(&outcome.transcript),
    );

    let minutes = match (with_mom, mom_skip) {
        (false, _) => MinutesOutcome::NotRequested,
        (true, Some(why)) => MinutesOutcome::Skipped(why.to_string()),
        (true, None) => minutes_outcome(&dir, outcome.mom.as_ref(), outcome.mom_error.as_deref()),
    };
    ClipResult::Ran(Box::new(measure(
        &reference,
        &outcome.transcript,
        &stt,
        outcome.metrics,
        minutes,
    )))
}

/// `--reuse`: an earlier run's front end and STT output, then this build's
/// `diarization::diarize` and `align::align` — the same two calls, on the same
/// inputs (`processed.wav`; `stt.json` already on the original timeline), that
/// `pipeline::run` makes. Redaction and classification are skipped: they only
/// annotate, and nothing scored here reads them.
fn rerun_after_stt(
    ctx: &Ctx,
    clip: &Clip,
    dir: &Path,
    plan: &ReusePlan,
) -> Result<(Transcript, Vec<SttSegment>, RunMetrics), String> {
    let read = |name: &str| {
        let path = plan.stages_from.join(name);
        std::fs::read_to_string(&path).map_err(|e| format!("could not read {}: {e}", path.display()))
    };
    let stt: Vec<SttSegment> = serde_json::from_str(&read("stt.json")?)
        .map_err(|e| format!("reused stt.json does not parse: {e}"))?;
    // Front-end and STT timings and warnings, for the report. A run without
    // a readable metrics.json just reports them as unknown.
    let mut metrics: RunMetrics = read("metrics.json")
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();

    let paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    eprintln!("   [{}] DIARIZING (front end + STT reused)", clip.id);
    let (diarization, diarization_metrics) =
        diarization::diarize(&paths, &ctx.config, &plan.stages_from.join("processed.wav"))?;
    let write = |name: &str, json: Result<String, serde_json::Error>| {
        let path = dir.join(name);
        json.map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display())))
    };
    write("diarization.json", serde_json::to_string_pretty(&diarization))?;

    eprintln!("   [{}] ALIGNING", clip.id);
    let started = std::time::Instant::now();
    let transcript = Transcript {
        meeting_id: clip.id.clone(),
        segments: align::align(&stt, &diarization),
    };
    let align_ms = started.elapsed().as_millis() as u64;
    write("transcript.json", serde_json::to_string_pretty(&transcript))?;
    let _ = std::fs::write(dir.join("transcript.txt"), render::transcript_text(&transcript));

    metrics.diarization = diarization_metrics;
    metrics.diarization_threads = ctx.config.diarization_threads;
    metrics.align_ms = align_ms;
    metrics.audio_ms = transcript.duration_ms();
    metrics.mom_ms = 0;
    // This run's own work only; `Measured::rtf` is withheld for reuse runs.
    metrics.total_ms = metrics.diarization.wall_ms + align_ms;
    Ok((transcript, stt, metrics))
}

fn minutes_outcome(dir: &Path, mom: Option<&Mom>, error: Option<&str>) -> MinutesOutcome {
    match (mom, error) {
        (Some(minutes), _) => {
            let _ = std::fs::write(
                dir.join("mom.md"),
                render::mom_markdown(minutes, "Minutes"),
            );
            let meta = minutes.meta.clone().unwrap_or_default();
            MinutesOutcome::Parsed {
                chunks: meta.chunk_count,
                failed_chunks: meta.failed_chunks,
                repaired: meta.repaired,
            }
        }
        (None, Some(err)) => MinutesOutcome::Failed(err.to_string()),
        (None, None) => MinutesOutcome::Failed("no minutes and no error reported".to_string()),
    }
}

// ----------------------------------------------------------------- report

fn report_header(ctx: &Ctx, stamp: &str, manifest: &Path, cache: &Path, args: &Args) -> String {
    let mut out = String::from("# Baseline regression test\n\n");
    out.push_str(&format!(
        "- **Run:** {stamp} UTC · git {} · shipped models {} / {} · STT language `{}` · threads STT {} / diarization {}\n",
        git_head(&compare::repo_root()).unwrap_or_else(|| "unknown".to_string()),
        system::STT_MODEL_ID,
        system::LLM_MODEL_ID,
        ctx.config.stt_language,
        ctx.config.stt_threads,
        ctx.config.diarization_threads
    ));
    out.push_str(&format!(
        "- **Machine:** {} MB RAM ({} MB free at start), {} logical cores\n",
        ctx.host.total_ram_mb, ctx.host.available_ram_mb, ctx.host.logical_cores
    ));
    out.push_str(&format!("- **Manifest:** `{}`\n", manifest.display()));
    out.push_str(&format!("- **Cache:** `{}`\n", cache.display()));
    out.push_str(&format!("- **Output:** `{}`\n", ctx.out.display()));
    out.push_str(&format!(
        "- **Minutes:** {}\n",
        if args.with_mom { "generated (`--with-mom`)" } else { "not generated (pass `--with-mom`)" }
    ));
    match &args.reuse {
        None => out.push_str("- **Stages:** full pipeline run\n\n"),
        Some(dir) => out.push_str(&format!(
            "- **Stages: REUSE RUN** (`--reuse {}`). Front end (preprocess, denoise), VAD and STT were **not** run: each clip's `processed.wav` and `stt.json` come from that run. Only **diarization and alignment** ran now, with this build. A change to the front end, VAD or STT is not reflected here; run without `--reuse` for that. RTF is withheld.\n\n",
            dir.display()
        )),
    }
    out.push_str("WER is gated relative to the recorded value; speakers, repeat run and repeated segments are gated by absolute bounds; RTF and timings are reported only. WER is against human captions, normalised (lowercase, no punctuation).\n\n");
    out
}

fn clip_table(rows: &[Row]) -> String {
    let mut out = String::from("| metric | expected | actual | Δ | status |\n|---|---|---|---|---|\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            row.metric,
            row.expected,
            row.actual,
            row.delta,
            row.verdict.label()
        ));
    }
    out.push('\n');
    out
}

fn stage_table(m: &Measured) -> String {
    let secs = |ms: u64| format!("{:.1}s", ms as f64 / 1000.0);
    let x = &m.metrics;
    // A reused stage shows the time it took in the run it came from.
    let carried = |s: String| {
        if m.reused.is_some() {
            format!("reused ({s} then)")
        } else {
            s
        }
    };
    let mut out = String::from(
        "| preprocess | denoise | VAD | diarization | STT | align | minutes | total (excl. minutes) |\n|---|---|---|---|---|---|---|---|\n",
    );
    out.push_str(&format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} |\n\n",
        carried(secs(x.preprocess_ms)),
        carried(x.denoise.map(|d| secs(d.wall_ms)).unwrap_or_else(|| "—".to_string())),
        carried(x.vad.map(|v| secs(v.wall_ms)).unwrap_or_else(|| "—".to_string())),
        secs(x.diarization.wall_ms),
        carried(secs(x.stt.wall_ms)),
        secs(x.align_ms),
        if x.mom_ms > 0 { secs(x.mom_ms) } else { "—".to_string() },
        if m.reused.is_some() {
            format!("{} (diarization + align only)", secs(m.pipeline_ms))
        } else {
            secs(m.pipeline_ms)
        }
    ));
    out
}

/// The commit `HEAD` points at, read from `.git` directly (no `git`
/// subprocess). The working tree may differ from it.
fn git_head(repo: &Path) -> Option<String> {
    let git = repo.join(".git");
    let head = std::fs::read_to_string(git.join("HEAD")).ok()?;
    let head = head.trim();
    let hash = match head.strip_prefix("ref: ") {
        None => head.to_string(),
        Some(name) => match std::fs::read_to_string(git.join(name)) {
            Ok(hash) => hash.trim().to_string(),
            Err(_) => std::fs::read_to_string(git.join("packed-refs"))
                .ok()?
                .lines()
                .find_map(|line| {
                    let (hash, r) = line.split_once(' ')?;
                    (r.trim() == name).then(|| hash.to_string())
                })?,
        },
    };
    (hash.len() >= 12 && hash.chars().all(|c| c.is_ascii_hexdigit()))
        .then(|| hash[..12].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// Synthetic captions exercising every rule — no real caption text.
    const SAMPLE_VTT: &str = "WEBVTT\r\nKind: captions\r\nLanguage: en\r\n\r\n\
00:00:00.467 --> 00:00:00.934\r\n \r\n\r\n\
00:00:00.967 --> 00:00:06.206 line:68%\r\n\
>> THE BLUE KITE... \r\n\r\n\
00:00:01.000 --> 00:00:02.000 align:start position:0%\r\n\
the<00:00:01.033><c> green</c><00:00:01.400><c> lamp</c>\r\n\r\n\
00:00:02.000 --> 00:00:03.000\r\n\
the green lamp\r\n\
sat (Laughter) on [MUSIC] a shelf\r\n\r\n\
NOTE a comment block\r\n\r\n\
00:00:03.000 --> 00:00:04.000\r\n\
(Applause)\r\n\
Tom &amp; Ann &gt;&gt; said (unclosed\r\n";

    #[test]
    fn vtt_becomes_plain_reference_text() {
        let reference = vtt_to_reference(SAMPLE_VTT);
        assert_eq!(
            reference,
            "THE BLUE KITE...\nthe green lamp\nsat on a shelf\nTom & Ann said (unclosed\n"
        );
        // WER normalisation of it: lowercase, punctuation gone.
        assert_eq!(
            text::normalize_words(&reference),
            ["the", "blue", "kite", "the", "green", "lamp", "sat", "on", "a", "shelf", "tom", "ann", "said", "unclosed"]
        );
    }

    #[test]
    fn rolling_duplicates_drop_but_a_later_repeat_survives() {
        let vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nalpha beta\n\n\
00:00:01.000 --> 00:00:02.000\nalpha beta\ngamma\n\n\
00:00:02.000 --> 00:00:03.000\nalpha beta\n";
        assert_eq!(vtt_to_reference(vtt), "alpha beta\ngamma\nalpha beta\n");
    }

    #[test]
    fn tags_and_brackets_strip_only_when_closed() {
        assert_eq!(strip_tags("a <c>b</c> <> c < d"), "a b <> c < d");
        assert_eq!(strip_annotations("x (y) [z] (w"), "x     (w");
    }

    #[test]
    fn parses_the_documented_flags() {
        let parsed = parse_args(&args(&[
            "--clips", "a, b", "--cache", "c", "--out", "o", "--update", "--with-mom",
        ]))
        .unwrap();
        assert_eq!(parsed.clips, Some(vec!["a".to_string(), "b".to_string()]));
        assert!(parsed.update && parsed.with_mom);
        assert_eq!(parsed.cache, Some(PathBuf::from("c")));
        assert_eq!(parsed.out, Some(PathBuf::from("o")));
        assert!(parse_args(&args(&["--bogus"])).is_err());
        assert!(parse_args(&args(&["--clips"])).is_err());
    }

    #[test]
    fn relative_and_absolute_gates() {
        assert_eq!(relative(None, 20.0, 2.0), Verdict::New);
        assert_eq!(relative(Some(16.7), 18.7, 2.0), Verdict::Pass);
        assert_eq!(relative(Some(16.7), 18.8, 2.0), Verdict::Regressed);
        assert_eq!(relative(Some(16.7), 14.6, 2.0), Verdict::Improved);
        let in_range = |n: usize| (2..=12).contains(&n);
        assert_eq!(absolute(in_range, Some(10), 12), Verdict::Pass);
        assert_eq!(absolute(in_range, None, 13), Verdict::Regressed);
        assert_eq!(absolute(in_range, Some(14), 5), Verdict::Improved);
    }

    /// The checked-in manifest must parse, name each clip once, and keep its
    /// tolerances coherent — and a parse → write round trip must not lose a
    /// field (that is what `--update` does to it).
    #[test]
    fn the_checked_in_manifest_parses_and_round_trips() {
        let manifest = load_manifest(&manifest_path()).unwrap();
        assert!(!manifest.clips.is_empty());
        let mut ids: Vec<&str> = manifest.clips.iter().map(|c| c.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), manifest.clips.len(), "duplicate clip id");
        for clip in &manifest.clips {
            assert!(clip.tolerance.speakers_min <= clip.tolerance.speakers_max, "{}", clip.id);
            assert!(clip.tolerance.wer_pp > 0.0, "{}", clip.id);
        }
        let written = manifest_json(&manifest).unwrap();
        let reread: serde_json::Value = serde_json::from_str(&written).unwrap();
        let original: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(manifest_path()).unwrap()).unwrap();
        assert_eq!(reread, original);
    }

    #[test]
    fn unknown_manifest_fields_survive_an_update() {
        let raw = r#"{"clips":[{"id":"x","title":"t","url":"u","caption_track":"en",
            "duration_s":1.0,"what_it_tests":"w","caption_quality":"q","owner":"me",
            "tolerance":{"wer_pp":2.0,"speakers_min":1,"speakers_max":2,"max_repeat_run":4,
            "max_repeated_segments":1},"expected":null}],"version":3}"#;
        let manifest: Manifest = serde_json::from_str(raw).unwrap();
        let written = manifest_json(&manifest).unwrap();
        assert!(written.contains("\"owner\": \"me\""));
        assert!(written.contains("\"version\": 3"));
        assert!(written.contains("\"expected\": null"));
        // Optional fields stay absent when they were absent...
        assert!(!written.contains("known_failures") && !written.contains("caption_language"));
        // ...and survive when present.
        let with = raw.replace(
            "\"owner\":\"me\",",
            "\"owner\":\"me\",\"caption_language\":\"ta\",\"known_failures\":[\"speakers\"],",
        );
        let written = manifest_json(&parse_manifest(&with).unwrap()).unwrap();
        assert!(written.contains("\"caption_language\": \"ta\""));
        assert!(written.contains("\"known_failures\": [\n        \"speakers\"\n      ]"));
        // Known fields keep their declared order.
        assert!(written.find("\"id\"").unwrap() < written.find("\"tolerance\"").unwrap());
    }

    #[test]
    fn reuse_parses_and_refuses_what_it_cannot_honour() {
        let parsed = parse_args(&args(&["--reuse", "runs/x", "--clips", "a"])).unwrap();
        assert_eq!(parsed.reuse, Some(PathBuf::from("runs/x")));
        assert!(parse_args(&args(&["--reuse"])).is_err());
        assert!(parse_args(&args(&["--reuse", "r", "--update"])).is_err());
        assert!(parse_args(&args(&["--reuse", "r", "--with-mom"])).is_err());
    }

    fn clip_with(known: &[&str]) -> Clip {
        let raw = format!(
            r#"{{"id":"x","title":"t","url":"u","caption_track":"en","duration_s":1.0,
            "what_it_tests":"w","caption_quality":"q",
            "tolerance":{{"wer_pp":2.0,"speakers_min":1,"speakers_max":2,"max_repeat_run":4,
            "max_repeated_segments":1}},"known_failures":{},"expected":null}}"#,
            serde_json::to_string(known).unwrap()
        );
        serde_json::from_str(&raw).unwrap()
    }

    fn measured(speakers: usize) -> Measured {
        Measured {
            wer: EditCounts {
                reference_words: 100,
                substitutions: 2,
                deletions: 0,
                insertions: 0,
            },
            words: 100,
            speakers,
            longest_repeat: 1,
            longest_repeat_ngram: String::new(),
            repeated_segments: 0,
            audio_ms: 10_000,
            pipeline_ms: 5_000,
            metrics: RunMetrics::default(),
            minutes: MinutesOutcome::NotRequested,
            reused: None,
        }
    }

    fn verdict_of(rows: &[Row], key: &str) -> Verdict {
        rows.iter().find(|r| r.key == Some(key)).unwrap().verdict
    }

    #[test]
    fn known_failures_report_known_and_then_fixed() {
        assert_eq!(Verdict::Regressed.as_known_failure(), Verdict::Known);
        assert_eq!(Verdict::Pass.as_known_failure(), Verdict::Fixed);
        assert_eq!(Verdict::Improved.as_known_failure(), Verdict::Fixed);
        assert_eq!(Verdict::New.as_known_failure(), Verdict::New);
        assert_eq!(Verdict::Info.as_known_failure(), Verdict::Info);

        // 8 speakers against a 1..2 bound: REGRESSED, unless it is known.
        assert_eq!(verdict_of(&rows_for(&clip_with(&[]), &measured(8)), "speakers"), Verdict::Regressed);
        let known = clip_with(&["speakers"]);
        let rows = rows_for(&known, &measured(8));
        assert_eq!(verdict_of(&rows, "speakers"), Verdict::Known);
        // Only the listed metric is affected.
        assert_eq!(verdict_of(&rows, "repeated_segments"), Verdict::Pass);
        assert!(rows.iter().all(|r| r.verdict != Verdict::Regressed));
        // Fixed once it passes, so the entry gets removed.
        assert_eq!(verdict_of(&rows_for(&known, &measured(1)), "speakers"), Verdict::Fixed);
        assert!(Verdict::Fixed.label().contains("remove from known_failures"));
    }

    #[test]
    fn an_unknown_known_failure_name_is_rejected() {
        let raw = r#"{"clips":[{"id":"x","title":"t","url":"u","caption_track":"en",
            "duration_s":1.0,"what_it_tests":"w","caption_quality":"q",
            "tolerance":{"wer_pp":2.0,"speakers_min":1,"speakers_max":2,"max_repeat_run":4,
            "max_repeated_segments":1},"known_failures":["speaker"],"expected":null}]}"#;
        let err = parse_manifest(raw).unwrap_err();
        assert!(err.contains("speaker") && err.contains("gated metrics"), "{err}");
        assert!(parse_manifest(&raw.replace("[\"speaker\"]", "[\"speakers\", \"wer\"]")).is_ok());
    }

    #[test]
    fn the_ted_clip_carries_its_known_speaker_failure() {
        let manifest = load_manifest(&manifest_path()).unwrap();
        let ted = manifest.clips.iter().find(|c| c.id == "R1vskiVDwl4").unwrap();
        assert_eq!(ted.known_failures, ["speakers"]);
        // The bound stays the target; known_failures is what lets it fail.
        assert_eq!((ted.tolerance.speakers_min, ted.tolerance.speakers_max), (1, 2));
    }

    /// Synthetic Tamil / code-switched captions (invented sentences, not from
    /// any clip): the parser and WER normalisation must keep Tamil words —
    /// vowel signs and virama included — whole.
    #[test]
    fn tamil_and_tanglish_captions_survive_parsing_and_normalisation() {
        let vtt = "WEBVTT\nKind: captions\nLanguage: ta\n\n\
00:00:00.000 --> 00:00:02.000\n<c>வணக்கம்,</c> எல்லாரும் வந்தாச்சா?\n\n\
00:00:02.000 --> 00:00:04.000 align:start\nசரி, meeting-ஐ start பண்ணலாம்... (சிரிப்பு)\n";
        let reference = vtt_to_reference(vtt);
        assert_eq!(
            reference,
            "வணக்கம், எல்லாரும் வந்தாச்சா?\nசரி, meeting-ஐ start பண்ணலாம்...\n"
        );
        assert_eq!(
            text::normalize_words(&reference),
            ["வணக்கம்", "எல்லாரும்", "வந்தாச்சா", "சரி", "meeting", "ஐ", "start", "பண்ணலாம்"]
        );
        assert_eq!(text::wer(&reference, &reference).errors(), 0);
        // A dropped virama is a different word, not a near miss.
        let hypothesis = reference.replace("பண்ணலாம்", "பண்ணலாம");
        let counts = text::wer(&reference, &hypothesis);
        assert_eq!((counts.substitutions, counts.reference_words), (1, 8));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cadence-baseline-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_wav(path: &Path, samples: usize, seed: i16) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..samples {
            writer.write_sample((i as i16).wrapping_mul(seed)).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn reuse_checks_the_audio_it_would_be_reusing() {
        let root = scratch("reuse");
        let cached = root.join("clip.16k.wav");
        write_wav(&cached, 16_000, 3);
        let input = fingerprint(&cached).unwrap();
        assert_eq!(input.duration_ms, 1_000);
        assert_eq!(input.sha256.len(), 64);

        // A run from before source.json: the length is all there is to check.
        let legacy = root.join("legacy");
        std::fs::create_dir_all(legacy.join("clip")).unwrap();
        write_wav(&legacy.join("clip").join("processed.wav"), 16_000, 5);
        std::fs::write(legacy.join("clip").join("stt.json"), "[]").unwrap();
        let plan = plan_reuse(&legacy, "clip", &input).unwrap();
        assert_eq!(plan.stages_from, legacy.join("clip"));
        assert!(plan.matched_by.contains("no input hash"), "{}", plan.matched_by);
        assert!(plan_reuse(&legacy, "other", &input).is_err());

        let short = root.join("short");
        std::fs::create_dir_all(short.join("clip")).unwrap();
        write_wav(&short.join("clip").join("processed.wav"), 12_000, 5);
        std::fs::write(short.join("clip").join("stt.json"), "[]").unwrap();
        let err = plan_reuse(&short, "clip", &input).unwrap_err();
        assert!(err.contains("refusing"), "{err}");

        // A newer run (here: a reuse of the legacy one) records the hash and
        // points at the directory that actually holds the stages.
        let newer = root.join("newer");
        std::fs::create_dir_all(newer.join("clip")).unwrap();
        write_source(&newer.join("clip"), &cached, &input, &legacy.join("clip"));
        let plan = plan_reuse(&newer, "clip", &input).unwrap();
        assert_eq!(
            std::path::absolute(&plan.stages_from).unwrap(),
            std::path::absolute(legacy.join("clip")).unwrap()
        );
        assert!(plan.matched_by.contains("sha256 matched"), "{}", plan.matched_by);

        // Same length, different audio: only the hash catches it.
        let replaced = root.join("replaced.16k.wav");
        write_wav(&replaced, 16_000, 7);
        let other = fingerprint(&replaced).unwrap();
        assert_eq!(other.duration_ms, input.duration_ms);
        let err = plan_reuse(&newer, "clip", &other).unwrap_err();
        assert!(err.contains("not the audio that run processed"), "{err}");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn signed_deltas_read_cleanly() {
        assert_eq!(signed(0.04, 1, " pp"), "0");
        assert_eq!(signed(1.26, 1, " pp"), "+1.3 pp");
        assert_eq!(signed(-2.0, 0, ""), "-2");
        assert_eq!(int_delta(None, 3), "—");
    }
}
