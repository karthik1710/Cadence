//! `bench_cli compare` — one recording through a matrix of front-end, STT and
//! LLM variants, written out as readable transcripts, minutes and one
//! `report.md` whose tables paste straight into `docs/benchmarks.md`.
//!
//! Every variant runs the **real** pipeline code: front-end and STT variants
//! are `pipeline::run` with an adjusted `PipelineConfig` or an overridden
//! `ModelPaths::stt_model`, so the required whisper flags, the code-switched
//! language handling, VAD, diarization and alignment are exactly the app's.
//! LLM variants are `mom::generate` on a `LlamaCppProvider` started on the
//! candidate's weights. The one bench-side composition is the DPDFNet
//! variant, whose denoiser the pipeline has no switch for: it runs the same
//! front half / denoise / `apply_gain` split `pipeline::run_front_end` does,
//! then hands the result to `pipeline::run` with every front-end stage off.
//!
//! Nothing is written into a meeting directory: a meeting id is resolved to
//! its `original.wav`, which is only ever read.

use super::catalog;
use super::render::{self, cell};
use super::text::{self, EditCounts};
use app_lib::models_manager::{self, CatalogEntry, ModelKind};
use app_lib::pipeline::llm::{self, LlmProvider};
use app_lib::pipeline::models::{Mom, SttSegment, Transcript};
use app_lib::pipeline::paths::ModelPaths;
use app_lib::pipeline::preprocess::{self, PreprocessOptions};
use app_lib::pipeline::proc::{self, StageMetrics};
use app_lib::pipeline::{
    self, audio_profile, chunk, denoise, enhance, mom, prompts, PipelineConfig, RunMetrics,
    RunRequest,
};
use app_lib::settings::Settings;
use app_lib::system::{self, HostInfo};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: bench_cli compare <audio.wav | meeting-id> [--out <dir>] [--quick] \
[--variants a,b,...] [--reference <ref.txt>] [--reference-range mm:ss-mm:ss] [--llm-cleanup] \
[--llm-file <model.gguf>]";

/// Front-end variants, in report order. STT is fixed to the shipped model.
const FRONT_ENDS: &[(&str, &str)] = &[
    ("raw", "resample to 16 kHz mono only — no enhancement at all"),
    ("shipped", "the exact default chain"),
    ("no-leveler", "the default chain without the leveler"),
    ("nr-instead-of-gtcrn", "model-free noise reduction in place of GTCRN"),
    ("gtcrn-only", "DC + high-pass + GTCRN, no level stages"),
    ("dpdfnet4", "the default chain with DPDFNet-4 in place of GTCRN"),
];

const DPDFNET_ID: &str = "denoise-dpdfnet4";

/// sherpa-onnx's default for DPDFNet is 0 = no limit, and unlimited it mutes
/// whole seconds of real speech: on a clean 30 s meeting recording it cut
/// individual speech seconds to -80..-105 dBFS, and whisper then lost the
/// first 10 s and looped on the rest (WER 100% vs GTCRN's 14%). 12 dB is the
/// value sherpa-onnx's own `--help` example uses, and keeps the worst second
/// within ~15 dB of its neighbours.
const DPDFNET_ATTENUATION_LIMIT_DB: u32 = 12;

/// Front-end variant names, for `--help`'s consistency test.
#[cfg(test)]
pub fn front_end_names() -> impl Iterator<Item = &'static str> {
    FRONT_ENDS.iter().map(|(name, _)| *name)
}

/// Scratch the pipeline leaves in its out dir. `processed.wav` is kept — it is
/// what a person listens to when two variants' transcripts disagree.
pub(super) const SCRATCH: &[&str] = &[
    "converted.wav",
    "denoised.wav",
    "converted_nr.wav",
    "vad_scratch.wav",
    "speech.wav",
    "dpdf_converted.wav",
    "dpdf_denoised.wav",
];

const CLEANUP_SYSTEM: &str = "You are a careful copy editor for automatic speech-recognition \
transcripts. You fix obvious recognition errors, punctuation and casing only. You never add, \
remove, summarise or translate content, and you never change who said what. You always answer \
with a single JSON object.";

struct Args {
    input: String,
    out: Option<PathBuf>,
    quick: bool,
    variants: Option<Vec<String>>,
    reference: Option<PathBuf>,
    reference_range: Option<(i64, i64)>,
    llm_cleanup: bool,
    llm_files: Vec<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut iter = args.iter();
    let mut parsed = Args {
        input: String::new(),
        out: None,
        quick: false,
        variants: None,
        reference: None,
        reference_range: None,
        llm_cleanup: false,
        llm_files: Vec::new(),
    };
    while let Some(arg) = iter.next() {
        let mut value = |flag: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--out" => parsed.out = Some(PathBuf::from(value("--out")?)),
            "--quick" => parsed.quick = true,
            "--llm-cleanup" => parsed.llm_cleanup = true,
            "--variants" => {
                parsed.variants = Some(
                    value("--variants")?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                )
            }
            "--reference" => parsed.reference = Some(PathBuf::from(value("--reference")?)),
            "--reference-range" => {
                parsed.reference_range = Some(text::parse_range(&value("--reference-range")?)?)
            }
            "--llm-file" => {
                for path in value("--llm-file")?.split(',') {
                    parsed.llm_files.push(PathBuf::from(path.trim()));
                }
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option: {flag}\n{USAGE}")),
            positional if parsed.input.is_empty() => parsed.input = positional.to_string(),
            extra => return Err(format!("unexpected argument: {extra}\n{USAGE}")),
        }
    }
    if parsed.input.is_empty() {
        return Err(USAGE.to_string());
    }
    if parsed.reference_range.is_some() && parsed.reference.is_none() {
        return Err("--reference-range needs --reference".to_string());
    }
    if parsed.quick && parsed.variants.is_some() {
        return Err("--quick and --variants are mutually exclusive".to_string());
    }
    Ok(parsed)
}

/// An LLM to generate minutes with.
struct LlmCandidate {
    name: String,
    path: PathBuf,
    /// For the memory pre-flight. `None` for an ad-hoc `--llm-file`.
    entry: Option<CatalogEntry>,
    missing_hint: &'static str,
}

/// Which variants this run executes.
struct Plan {
    front_ends: Vec<&'static str>,
    stt: Vec<String>,
    llm: Vec<LlmCandidate>,
}

fn build_plan(args: &Args, data_dir: &Path) -> Result<Plan, String> {
    let models = pipeline::paths::models_dir(data_dir);
    let mut stt_all = vec![system::STT_MODEL_ID.to_string()];
    stt_all.extend(catalog::ids_of_kind(ModelKind::Stt).into_iter().map(str::to_string));

    let baseline = models_manager::entry(system::LLM_MODEL_ID)
        .ok_or("the shipped LLM is missing from the catalog")?;
    let mut llm_all = vec![LlmCandidate {
        name: baseline.id.to_string(),
        path: baseline.path_in(&models),
        entry: Some(baseline.clone()),
        missing_hint: "not installed (it is the shipped model — install it from the app's setup screen)",
    }];
    for id in catalog::ids_of_kind(ModelKind::Llm) {
        if let Some(candidate) = catalog::find(id) {
            llm_all.push(LlmCandidate {
                name: id.to_string(),
                path: catalog::path_of(data_dir, id).unwrap_or_default(),
                entry: Some(candidate.entry),
                missing_hint: if candidate.large {
                    "not installed, run `bench_cli setup --with-large`"
                } else {
                    "not installed, run `bench_cli setup`"
                },
            });
        }
    }
    let ad_hoc: Vec<LlmCandidate> = args
        .llm_files
        .iter()
        .map(|path| LlmCandidate {
            name: format!(
                "file-{}",
                path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string())
            ),
            path: path.clone(),
            entry: None,
            missing_hint: "file not found",
        })
        .collect();

    let mut plan = if args.quick {
        Plan {
            front_ends: vec!["shipped"],
            stt: vec![system::STT_MODEL_ID.to_string()],
            llm: llm_all.into_iter().take(1).collect(),
        }
    } else if let Some(wanted) = &args.variants {
        let known: Vec<String> = FRONT_ENDS
            .iter()
            .map(|(n, _)| n.to_string())
            .chain(stt_all.iter().cloned())
            .chain(llm_all.iter().map(|c| c.name.clone()))
            .collect();
        for name in wanted {
            if !known.contains(name) {
                return Err(format!(
                    "unknown variant `{name}`. Known: {}",
                    known.join(", ")
                ));
            }
        }
        Plan {
            front_ends: FRONT_ENDS
                .iter()
                .map(|(n, _)| *n)
                .filter(|n| wanted.iter().any(|w| w == n))
                .collect(),
            stt: stt_all.into_iter().filter(|n| wanted.contains(n)).collect(),
            llm: llm_all.into_iter().filter(|c| wanted.contains(&c.name)).collect(),
        }
    } else {
        Plan {
            front_ends: FRONT_ENDS.iter().map(|(n, _)| *n).collect(),
            stt: stt_all,
            llm: llm_all,
        }
    };
    plan.llm.extend(ad_hoc);

    // The baseline STT row *is* the shipped front-end run, and minutes are
    // generated from its transcript — so either one pulls `shipped` in.
    let needs_shipped = plan.stt.iter().any(|s| s == system::STT_MODEL_ID) || !plan.llm.is_empty();
    if needs_shipped && !plan.front_ends.contains(&"shipped") {
        plan.front_ends.insert(0, "shipped");
    }
    Ok(plan)
}

/// Level/noise measurements of one WAV.
#[derive(Clone, Copy)]
struct AudioStats {
    noise_floor_dbfs: f32,
    speech_dbfs: f32,
    snr_db: f32,
    /// Speech-level spread across the recording, as the leveler measures it —
    /// how far apart the near and far ends of the table are.
    spread_db: Option<f32>,
}

fn audio_stats(path: &Path) -> Option<AudioStats> {
    let profile = audio_profile::profile_wav(path).ok()?;
    let spread = preprocess::read_mono_16k(path).ok().map(|(mut samples, _, _)| {
        // Measure-only: the buffer is a throwaway copy.
        enhance::level(
            &mut samples,
            preprocess::TARGET_SAMPLE_RATE,
            &enhance::LevelerOptions::default(),
        )
        .speech_spread_before_db
    });
    Some(AudioStats {
        noise_floor_dbfs: profile.noise_floor_dbfs,
        speech_dbfs: profile.speech_dbfs,
        snr_db: profile.estimated_snr_db,
        spread_db: spread,
    })
}

/// One completed pipeline run.
struct RunData {
    transcript: Transcript,
    stt: Vec<SttSegment>,
    metrics: RunMetrics,
    /// Front-end work done bench-side, before `pipeline::run` (DPDFNet only).
    extra_front_ms: u64,
    extra_denoise: Option<StageMetrics>,
    denoiser: String,
    level_stages: String,
    after: Option<AudioStats>,
    wer: Option<EditCounts>,
}

impl RunData {
    fn total_ms(&self) -> u64 {
        self.metrics.total_ms + self.extra_front_ms
    }
}

enum Status {
    Ran(Box<RunData>),
    Skipped(String),
    Failed(String),
}

struct LlmRow {
    name: String,
    quantization: String,
    load_ms: u64,
    generate_ms: u64,
    peak_ram_mb: Option<u64>,
    outcome: Result<Mom, String>,
    cleanup: Option<Result<CleanupResult, String>>,
}

struct CleanupResult {
    wall_ms: u64,
    chunks: usize,
    failed_chunks: usize,
    lines: usize,
    lines_kept_raw: usize,
    raw_tokens: usize,
    changed_tokens: usize,
    content: EditCounts,
}

struct Ctx {
    data_dir: PathBuf,
    host: HostInfo,
    settings: Settings,
    base: PipelineConfig,
    out: PathBuf,
    input: PathBuf,
    /// UTC start time, `YYYYMMDD-HHMMSS` — names the default output dir and
    /// heads the report, so the two always agree.
    stamp: String,
}

pub fn run(raw_args: &[String]) -> Result<(), String> {
    let args = parse_args(raw_args)?;
    let stamp = utc_stamp();
    let data_dir = pipeline::paths::default_data_dir()?;
    let (input, stem) = resolve_input(&args.input, &data_dir)?;
    let out = match &args.out {
        Some(dir) => dir.clone(),
        None => repo_root()
            .join("bench-runs")
            .join(format!("{}-{}", stamp, file_safe(&stem))),
    };
    guard_out_dir(&out, &data_dir)?;
    std::fs::create_dir_all(out.join("runs"))
        .map_err(|e| format!("could not create {}: {e}", out.display()))?;

    let reference = match &args.reference {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| format!("could not read reference {}: {e}", path.display()))?,
        ),
        None => None,
    };

    let host = HostInfo::detect();
    // Shipped defaults, not the user's settings.json: this compares the
    // product as it ships. MoM is driven separately below.
    let settings = Settings {
        generate_mom: false,
        ..Settings::default()
    };
    let base = PipelineConfig::from_settings(&settings, &host);
    let plan = build_plan(&args, &data_dir)?;
    let ctx = Ctx {
        data_dir,
        host,
        settings,
        base,
        out,
        input,
        stamp,
    };

    // The runtimes and diarization models every pipeline variant needs. A
    // missing one would fail every row identically, so it is fatal up front.
    let base_paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    let missing: Vec<String> = base_paths
        .missing(&ctx.data_dir, &ctx.settings)
        .into_iter()
        .filter(|id| id != system::STT_MODEL_ID)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "required runtime/models not installed: {} — run the app's first-run setup (or scripts/setup-models.ps1)",
            missing.join(", ")
        ));
    }
    let baseline_stt_installed = base_paths.stt_model.exists();

    let audio_ms = app_lib::storage::wav_duration_ms(&ctx.input).unwrap_or(0);
    let before = audio_stats(&ctx.input);
    let input_profile = audio_profile::profile_wav(&ctx.input).ok();

    eprintln!(
        "bench compare: {} ({}) → {}",
        ctx.input.display(),
        text::clock(audio_ms),
        ctx.out.display()
    );

    // ------------------------------------------------------------ front end
    let mut front_rows: Vec<(String, Status)> = Vec::new();
    for &name in &plan.front_ends {
        eprintln!("\n== front end: {name}");
        let status = if !baseline_stt_installed {
            Status::Skipped(format!(
                "{} (the shipped STT model) is not installed",
                system::STT_MODEL_ID
            ))
        } else {
            run_front_end_variant(&ctx, name)
        };
        report_status(name, &status);
        front_rows.push((name.to_string(), status));
    }

    // ------------------------------------------------------------------ STT
    let mut stt_rows: Vec<(String, Status)> = Vec::new();
    for model in &plan.stt {
        if model == system::STT_MODEL_ID {
            // Same run as the `shipped` front end; not repeated.
            continue;
        }
        eprintln!("\n== STT: {model}");
        let status = match catalog::path_of(&ctx.data_dir, model) {
            Some(path) if path.exists() => run_stt_variant(&ctx, model, &path),
            _ => Status::Skipped("not installed, run `bench_cli setup`".to_string()),
        };
        report_status(model, &status);
        stt_rows.push((model.clone(), status));
    }

    // ---------------------------------------------------------------- WER
    if let Some(reference) = &reference {
        for (_, status) in front_rows.iter_mut().chain(stt_rows.iter_mut()) {
            if let Status::Ran(run) = status {
                let hypothesis = hypothesis_text(run, args.reference_range);
                run.wer = Some(text::wer(reference, &hypothesis));
            }
        }
    }

    // ------------------------------------------------------------------ LLM
    let shipped_transcript = front_rows.iter().find_map(|(name, status)| match status {
        Status::Ran(run) if name == "shipped" => Some(run.transcript.clone()),
        _ => None,
    });
    let mut llm_rows: Vec<LlmRow> = Vec::new();
    let mut llm_skipped: Vec<(String, String)> = Vec::new();
    for candidate in &plan.llm {
        eprintln!("\n== LLM: {}", candidate.name);
        let Some(transcript) = shipped_transcript.as_ref() else {
            llm_skipped.push((
                candidate.name.clone(),
                "no `shipped` transcript to summarise (that run was skipped or failed)".to_string(),
            ));
            continue;
        };
        if !candidate.path.exists() {
            llm_skipped.push((candidate.name.clone(), candidate.missing_hint.to_string()));
            continue;
        }
        if transcript.segments.is_empty() {
            llm_skipped.push((candidate.name.clone(), "the shipped transcript is empty".to_string()));
            continue;
        }
        match run_llm(&ctx, candidate, transcript, args.llm_cleanup) {
            Ok(row) => llm_rows.push(row),
            Err(err) => llm_skipped.push((candidate.name.clone(), format!("failed: {err}"))),
        }
    }

    // --------------------------------------------------------------- report
    let report = build_report(&ReportInput {
        ctx: &ctx,
        args: &args,
        audio_ms,
        input_profile: input_profile.as_ref(),
        before,
        front_rows: &front_rows,
        stt_rows: &stt_rows,
        llm_rows: &llm_rows,
        llm_skipped: &llm_skipped,
        reference: reference.as_deref(),
    });
    let report_path = ctx.out.join("report.md");
    std::fs::write(&report_path, &report)
        .map_err(|e| format!("could not write {}: {e}", report_path.display()))?;
    println!("{report}");
    eprintln!("\nReport written to {}", report_path.display());
    Ok(())
}

fn report_status(name: &str, status: &Status) {
    match status {
        Status::Ran(run) => eprintln!(
            "   {name}: done in {:.1}s, {} words",
            run.total_ms() as f64 / 1000.0,
            render::word_count(&run.transcript)
        ),
        Status::Skipped(why) => eprintln!("   {name}: skipped: {why}"),
        Status::Failed(why) => eprintln!("   {name}: FAILED: {why}"),
    }
}

// ------------------------------------------------------------- input / out

fn resolve_input(raw: &str, data_dir: &Path) -> Result<(PathBuf, String), String> {
    let as_path = PathBuf::from(raw);
    if as_path.is_file() {
        let stem = as_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "input".to_string());
        return Ok((as_path, stem));
    }
    let meeting_wav = app_lib::storage::meeting_dir(data_dir, raw)?.join("original.wav");
    if meeting_wav.is_file() {
        return Ok((meeting_wav, raw.to_string()));
    }
    Err(format!(
        "`{raw}` is neither a WAV file nor a meeting id with an original.wav under {}",
        app_lib::storage::meetings_dir(data_dir).display()
    ))
}

/// Refuses to write anywhere under the meetings directory — a meeting's
/// artifacts are the app's, and a bench run must never add to or overwrite them.
pub(super) fn guard_out_dir(out: &Path, data_dir: &Path) -> Result<(), String> {
    let meetings = app_lib::storage::meetings_dir(data_dir);
    let canonical_meetings = meetings.canonicalize().unwrap_or(meetings);
    let mut probe = out.to_path_buf();
    // `out` may not exist yet: canonicalise its nearest existing ancestor.
    let resolved = loop {
        if let Ok(real) = probe.canonicalize() {
            let rest = out.strip_prefix(&probe).unwrap_or(Path::new(""));
            break real.join(rest);
        }
        if !probe.pop() {
            break out.to_path_buf();
        }
    };
    if resolved.starts_with(&canonical_meetings) {
        return Err(format!(
            "refusing to write bench output under the meetings directory ({})",
            canonical_meetings.display()
        ));
    }
    Ok(())
}

pub(super) fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

pub(super) fn file_safe(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
        .collect()
}

/// `YYYYMMDD-HHMMSS` in UTC, from the system clock — hand-rolled rather than
/// pulling in a date crate for one directory name.
pub(super) fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's days-to-civil.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

// --------------------------------------------------------- pipeline runs

fn resample_only(options: &PreprocessOptions) -> PreprocessOptions {
    PreprocessOptions {
        remove_dc: false,
        high_pass_hz: 0.0,
        normalize: false,
        compressor: None,
        leveler: None,
        noise_reduction: None,
        ..*options
    }
}

fn level_stages(options: &PreprocessOptions) -> String {
    let mut stages = Vec::new();
    if options.leveler.is_some() {
        stages.push("leveler");
    }
    if options.compressor.is_some() {
        stages.push("compressor");
    }
    if options.normalize {
        stages.push("normalize");
    }
    if stages.is_empty() {
        "none".to_string()
    } else {
        stages.join(" + ")
    }
}

fn denoiser_label(metrics: &RunMetrics) -> String {
    if metrics.denoise.is_some() {
        "GTCRN".to_string()
    } else if metrics
        .preprocess_report
        .as_ref()
        .and_then(|r| r.noise_reduction.as_ref())
        .is_some()
    {
        "model-free NR".to_string()
    } else {
        "none".to_string()
    }
}

fn run_front_end_variant(ctx: &Ctx, name: &str) -> Status {
    let dir = ctx.out.join("runs").join(name);
    let paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    let mut config = ctx.base.clone();
    match name {
        "raw" => {
            config.preprocess = resample_only(&ctx.base.preprocess);
            config.denoise_enabled = false;
        }
        "shipped" => {}
        "no-leveler" => config.preprocess.leveler = None,
        "nr-instead-of-gtcrn" => {
            config.denoise_enabled = false;
            if config.preprocess.noise_reduction.is_none() {
                config.preprocess.noise_reduction = Some(enhance::NoiseReductionOptions::default());
            }
        }
        "gtcrn-only" => {
            if let Some(reason) = denoise::unavailable_reason(&ctx.data_dir) {
                return Status::Skipped(reason);
            }
            config.denoise_enabled = true;
            config.preprocess.normalize = false;
            config.preprocess.compressor = None;
            config.preprocess.leveler = None;
            config.preprocess.noise_reduction = None;
        }
        "dpdfnet4" => return run_dpdfnet_variant(ctx, &dir),
        other => return Status::Failed(format!("unknown front end {other}")),
    }
    let level = level_stages(&config.preprocess);
    match run_pipeline(ctx, name, &config, &paths, &ctx.input, &dir) {
        Ok((transcript, stt, metrics)) => {
            let denoiser = denoiser_label(&metrics);
            finish_run(ctx, name, &dir, transcript, stt, metrics, 0, None, denoiser, level)
        }
        Err(err) => Status::Failed(err),
    }
}

/// The DPDFNet variant: the shipped chain with DPDFNet-4 where GTCRN would
/// run, composed from the same public stages `pipeline::run_front_end` uses.
fn run_dpdfnet_variant(ctx: &Ctx, dir: &Path) -> Status {
    let Some(model) = catalog::path_of(&ctx.data_dir, DPDFNET_ID).filter(|p| p.exists()) else {
        return Status::Skipped("not installed, run `bench_cli setup`".to_string());
    };
    let binary = denoise::binary_path(&ctx.data_dir);
    if !binary.exists() {
        return Status::Skipped(format!("sherpa-onnx denoiser not installed at {}", binary.display()));
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        return Status::Failed(e.to_string());
    }
    let options = ctx.base.preprocess;
    let started = Instant::now();
    let converted = dir.join("dpdf_converted.wav");
    let denoised = dir.join("dpdf_denoised.wav");
    let front_end = dir.join("dpdf_front_end.wav");

    let front = (|| -> Result<StageMetrics, String> {
        preprocess::preprocess(
            &ctx.input,
            &converted,
            &options.without_gain().without_noise_reduction(),
        )?;
        let mut command = std::process::Command::new(&binary);
        command
            .arg(format!("--speech-denoiser-dpdfnet-model={}", model.display()))
            .arg(format!(
                "--speech-denoiser-dpdfnet-attenuation-limit-db={DPDFNET_ATTENUATION_LIMIT_DB}"
            ))
            .arg(format!("--num-threads={}", ctx.base.diarization_threads.max(1)))
            .arg(format!("--input-wav={}", converted.display()))
            .arg(format!("--output-wav={}", denoised.display()));
        let measured = proc::run_measured(command)?;
        if !measured.success
            || std::fs::metadata(&denoised).map(|m| m.len()).unwrap_or(0) <= 44
        {
            return Err(format!(
                "DPDFNet denoiser failed ({}): {}",
                measured.status,
                measured.stderr.trim()
            ));
        }
        preprocess::apply_gain(&denoised, &front_end, &options)?;
        Ok(measured.metrics)
    })();
    let denoise_metrics = match front {
        Ok(metrics) => metrics,
        Err(err) => return Status::Failed(err),
    };
    let extra_ms = started.elapsed().as_millis() as u64;

    // Everything front-end already happened; the pipeline only resamples
    // (a no-op on a 16 kHz mono file) and runs the remaining stages.
    let mut config = ctx.base.clone();
    config.preprocess = resample_only(&ctx.base.preprocess);
    config.denoise_enabled = false;
    let paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    let result = run_pipeline(ctx, "dpdfnet4", &config, &paths, &front_end, dir);
    let _ = std::fs::remove_file(&front_end);
    match result {
        Ok((transcript, stt, metrics)) => finish_run(
            ctx,
            "dpdfnet4",
            dir,
            transcript,
            stt,
            metrics,
            extra_ms,
            Some(denoise_metrics),
            "DPDFNet-4".to_string(),
            level_stages(&options),
        ),
        Err(err) => Status::Failed(err),
    }
}

fn run_stt_variant(ctx: &Ctx, model: &str, model_path: &Path) -> Status {
    let dir = ctx.out.join("runs").join(format!("stt-{model}"));
    let mut paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    paths.stt_model = model_path.to_path_buf();
    let config = ctx.base.clone();
    let level = level_stages(&config.preprocess);
    match run_pipeline(ctx, model, &config, &paths, &ctx.input, &dir) {
        Ok((transcript, stt, mut metrics)) => {
            // `RunMetrics::stt_model` is read from Settings, which knows only
            // the shipped model; record what actually ran.
            metrics.stt_model = model.to_string();
            let denoiser = denoiser_label(&metrics);
            finish_run(ctx, model, &dir, transcript, stt, metrics, 0, None, denoiser, level)
        }
        Err(err) => Status::Failed(err),
    }
}

type PipelineResult = Result<(Transcript, Vec<SttSegment>, RunMetrics), String>;

fn run_pipeline(
    ctx: &Ctx,
    name: &str,
    config: &PipelineConfig,
    paths: &ModelPaths,
    input: &Path,
    dir: &Path,
) -> PipelineResult {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let outcome = pipeline::run(
        &RunRequest {
            paths,
            config,
            settings: &ctx.settings,
            host: &ctx.host,
            input_wav: input,
            out_dir: dir,
            data_dir: &ctx.data_dir,
            meeting_id: "bench",
            title: "Benchmark meeting",
        },
        &|stage, pct| eprintln!("   [{name}] {stage} {pct}%"),
    )?;
    let stt: Vec<SttSegment> = std::fs::read_to_string(dir.join("stt.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    for scratch in SCRATCH {
        let _ = std::fs::remove_file(dir.join(scratch));
    }
    Ok((outcome.transcript, stt, outcome.metrics))
}

#[allow(clippy::too_many_arguments)]
fn finish_run(
    ctx: &Ctx,
    name: &str,
    dir: &Path,
    transcript: Transcript,
    stt: Vec<SttSegment>,
    metrics: RunMetrics,
    extra_front_ms: u64,
    extra_denoise: Option<StageMetrics>,
    denoiser: String,
    level_stages: String,
) -> Status {
    let label = file_safe(name);
    let _ = std::fs::write(
        ctx.out.join(format!("transcript-{label}.txt")),
        render::transcript_text(&transcript),
    );
    let _ = std::fs::write(
        ctx.out.join(format!("transcript-{label}.json")),
        serde_json::to_string_pretty(&transcript).unwrap_or_default(),
    );
    let after = audio_stats(&dir.join("processed.wav"));
    Status::Ran(Box::new(RunData {
        transcript,
        stt,
        metrics,
        extra_front_ms,
        extra_denoise,
        denoiser,
        level_stages,
        after,
        wer: None,
    }))
}

/// What a run said inside the reference range — word-timed where whisper
/// gave word times, whole segments (by midpoint) where it did not.
fn hypothesis_text(run: &RunData, range: Option<(i64, i64)>) -> String {
    let Some((start, end)) = range else {
        return run
            .transcript
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    };
    let inside = |a: i64, b: i64| {
        let mid = (a + b) / 2;
        mid >= start && mid < end
    };
    let mut words: Vec<&str> = Vec::new();
    for segment in &run.stt {
        if segment.words.is_empty() {
            if inside(segment.start_ms, segment.end_ms) {
                words.push(segment.text.as_str());
            }
        } else {
            for word in &segment.words {
                if inside(word.start_ms, word.end_ms) {
                    words.push(word.text.as_str());
                }
            }
        }
    }
    words.join(" ")
}

// ------------------------------------------------------------------- LLM

fn run_llm(
    ctx: &Ctx,
    candidate: &LlmCandidate,
    transcript: &Transcript,
    cleanup: bool,
) -> Result<LlmRow, String> {
    let mut paths = ModelPaths::unchecked(&ctx.data_dir, &ctx.settings);
    paths.llm_model = candidate.path.clone();
    let context_tokens = ctx.settings.effective_llm_context();
    let started = Instant::now();
    let provider = llm::LlamaCppProvider::start_with_model(
        &paths,
        &candidate.name,
        candidate.entry.as_ref(),
        context_tokens,
        ctx.settings.effective_llm_threads(&ctx.host),
        &ctx.host,
        &ctx.data_dir,
    )?;
    let load_ms = started.elapsed().as_millis() as u64;
    let quantization = provider.describe().quantization;

    let started = Instant::now();
    let outcome = mom::generate(
        &provider,
        &mom::MomRequest {
            transcript,
            title: "Benchmark meeting",
            language: &ctx.base.mom_language,
            data_dir: Some(&ctx.data_dir),
            redaction: ctx.settings.effective_redaction_mode(),
        },
        &|stage, pct| eprintln!("   [{}] {stage} {pct}%", candidate.name),
    );
    let generate_ms = started.elapsed().as_millis() as u64;
    // Peak working set over the server's life so far — i.e. the minutes run,
    // read before the cleanup pass can add to it.
    let peak_ram_mb = provider.peak_ram_mb();

    let label = file_safe(&candidate.name);
    match &outcome {
        Ok(minutes) => {
            let _ = std::fs::write(
                ctx.out.join(format!("mom-{label}.md")),
                render::mom_markdown(minutes, &format!("Minutes — {}", candidate.name)),
            );
            let _ = std::fs::write(
                ctx.out.join(format!("mom-{label}.json")),
                serde_json::to_string_pretty(minutes).unwrap_or_default(),
            );
        }
        Err(err) => eprintln!("   [{}] minutes failed: {err}", candidate.name),
    }

    let cleanup = cleanup.then(|| {
        eprintln!("   [{}] transcript cleanup", candidate.name);
        let result = run_cleanup(&provider, transcript, &ctx.data_dir, context_tokens);
        if let Ok((cleaned, stats, diff)) = &result {
            let _ = std::fs::write(
                ctx.out.join(format!("cleanup-{label}.txt")),
                render::transcript_text(cleaned),
            );
            let _ = std::fs::write(ctx.out.join(format!("cleanup-{label}.diff.txt")), diff);
            eprintln!(
                "   [{}] cleanup changed {} of {} tokens",
                candidate.name, stats.changed_tokens, stats.raw_tokens
            );
        }
        result.map(|(_, stats, _)| stats)
    });
    drop(provider);

    Ok(LlmRow {
        name: candidate.name.clone(),
        quantization,
        load_ms,
        generate_ms,
        peak_ram_mb,
        outcome,
        cleanup,
    })
}

/// The experimental correction pass: per chunk, ask the model to fix obvious
/// ASR errors, punctuation and casing only, and measure how much it changed.
/// Bench-only — nothing in the app calls this.
///
/// Speaker labels and turn boundaries are preserved by construction: the
/// model sees numbered lines and returns text per id, and only the text is
/// taken back. A line the model drops or blanks keeps its original text.
fn run_cleanup(
    llm: &dyn LlmProvider,
    transcript: &Transcript,
    data_dir: &Path,
    context_tokens: usize,
) -> Result<(Transcript, CleanupResult, String), String> {
    let started = Instant::now();
    let template = prompts::load(Some(data_dir), prompts::CLEANUP_VERSION)?;
    // Half the extraction budget: the answer is as long as the input here.
    let budget = (chunk::max_chunk_chars(context_tokens) / 2).max(600);
    let max_tokens = (context_tokens / 2).clamp(512, 4096);
    let chunks = chunk::chunk_transcript(&transcript.segments, budget, 0);
    let covered: usize = chunks.iter().map(|c| c.segments.len()).sum();
    if covered != transcript.segments.len() {
        return Err(format!(
            "chunking covered {covered} of {} turns — refusing to guess the mapping",
            transcript.segments.len()
        ));
    }

    let mut cleaned = transcript.clone();
    let mut failed_chunks = 0;
    let mut kept_raw = 0;
    let mut first = 0usize;
    for chunk in &chunks {
        let lines: Vec<serde_json::Value> = chunk
            .segments
            .iter()
            .enumerate()
            .map(|(k, s)| serde_json::json!({"id": first + k, "speaker": s.speaker_label, "text": s.text}))
            .collect();
        let input = serde_json::to_string_pretty(&serde_json::json!({ "lines": lines }))
            .map_err(|e| e.to_string())?;
        let prompt = prompts::fill(
            &template,
            &[
                ("CHUNK_INDEX", &(chunk.index + 1).to_string()),
                ("CHUNK_COUNT", &chunks.len().to_string()),
                ("TRANSCRIPT", &input),
            ],
        );
        match llm
            .complete_json(CLEANUP_SYSTEM, &prompt, max_tokens)
            .and_then(|raw| parse_cleanup(&raw))
        {
            Ok(fixed) => {
                for index in first..first + chunk.segments.len() {
                    match fixed.get(&index) {
                        Some(text) if !text.trim().is_empty() => {
                            cleaned.segments[index].text = text.trim().to_string();
                        }
                        _ => kept_raw += 1,
                    }
                }
            }
            Err(err) => {
                eprintln!("   cleanup chunk {} failed: {err}", chunk.index + 1);
                failed_chunks += 1;
                kept_raw += chunk.segments.len();
            }
        }
        first += chunk.segments.len();
    }
    // Flag offsets point into the original text, which no longer matches.
    for segment in &mut cleaned.segments {
        segment.flags.clear();
    }

    let mut raw_tokens = 0;
    let mut changed_tokens = 0;
    let mut content = EditCounts::default();
    let mut diff = String::from(
        "Word diff of the LLM cleanup pass against the raw transcript.\n\
         [-removed-] {+added+}. Only changed turns are listed.\n\n",
    );
    for (before, after) in transcript.segments.iter().zip(&cleaned.segments) {
        let (b, a) = (text::raw_words(&before.text), text::raw_words(&after.text));
        let exact = text::edit_counts(&b, &a);
        raw_tokens += b.len();
        changed_tokens += exact.errors();
        content.add(&text::edit_counts(
            &text::normalize_words(&before.text),
            &text::normalize_words(&after.text),
        ));
        if exact.errors() > 0 {
            diff.push_str(&format!(
                "[{}] {} ({} token edit(s))\n  {}\n\n",
                text::clock(before.start_ms),
                before.speaker_label,
                exact.errors(),
                text::render_diff(&text::diff_words(&b, &a))
            ));
        }
    }

    let stats = CleanupResult {
        wall_ms: started.elapsed().as_millis() as u64,
        chunks: chunks.len(),
        failed_chunks,
        lines: transcript.segments.len(),
        lines_kept_raw: kept_raw,
        raw_tokens,
        changed_tokens,
        content,
    };
    Ok((cleaned, stats, diff))
}

fn parse_cleanup(raw: &str) -> Result<HashMap<usize, String>, String> {
    let json = llm::extract_json(raw).ok_or("no JSON object in the response")?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    let lines = value
        .get("lines")
        .and_then(|l| l.as_array())
        .ok_or("response has no \"lines\" array")?;
    let mut fixed = HashMap::new();
    for line in lines {
        let id = line.get("id").and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
        });
        let text = line.get("text").and_then(|t| t.as_str());
        if let (Some(id), Some(text)) = (id, text) {
            fixed.insert(id as usize, text.to_string());
        }
    }
    Ok(fixed)
}

// ---------------------------------------------------------------- report

struct ReportInput<'a> {
    ctx: &'a Ctx,
    args: &'a Args,
    audio_ms: i64,
    input_profile: Option<&'a audio_profile::AudioProfile>,
    before: Option<AudioStats>,
    front_rows: &'a [(String, Status)],
    stt_rows: &'a [(String, Status)],
    llm_rows: &'a [LlmRow],
    llm_skipped: &'a [(String, String)],
    reference: Option<&'a str>,
}

fn secs(ms: u64) -> String {
    format!("{:.1}s", ms as f64 / 1000.0)
}

fn ram(mb: Option<u64>) -> String {
    mb.map(|mb| format!("{mb} MB")).unwrap_or_else(|| "n/a".to_string())
}

fn db(value: f32) -> String {
    format!("{value:.1}")
}

fn arrow(before: Option<f32>, after: Option<f32>) -> String {
    match (before, after) {
        (Some(b), Some(a)) => format!("{} → {}", db(b), db(a)),
        (None, Some(a)) => format!("? → {}", db(a)),
        (Some(b), None) => format!("{} → ?", db(b)),
        (None, None) => "—".to_string(),
    }
}

fn wer_cell(counts: Option<&EditCounts>) -> String {
    match counts.and_then(|c| c.rate().map(|r| (c, r))) {
        Some((c, rate)) => format!(
            "{:.1}% ({}/{}/{})",
            rate * 100.0,
            c.substitutions,
            c.deletions,
            c.insertions
        ),
        None => "—".to_string(),
    }
}

fn realtime(ms: u64, audio_ms: i64) -> String {
    if audio_ms <= 0 {
        return "—".to_string();
    }
    format!("{:.2}x", ms as f64 / audio_ms as f64)
}

fn hallucination_cells(run: &RunData) -> (String, usize) {
    let words: Vec<String> = run
        .transcript
        .segments
        .iter()
        .flat_map(|s| text::normalize_words(&s.text))
        .collect();
    let longest = match text::longest_repeat_run(&words) {
        Some(r) => format!("{}× \"{}\"", r.times, cell(&r.ngram)),
        None => "—".to_string(),
    };
    let texts: Vec<String> = run.stt.iter().map(|s| s.text.clone()).collect();
    (longest, text::repeated_segments(&texts))
}

fn build_report(input: &ReportInput) -> String {
    let ctx = input.ctx;
    let mut out = String::new();
    let stem = ctx
        .input
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    out.push_str(&format!("# Bench comparison — {}\n\n", input.args.input));
    out.push_str(&format!("- **Input:** `{}` ({stem})\n", ctx.input.display()));
    out.push_str(&format!(
        "- **Audio:** {} ({} ms){}\n",
        text::clock(input.audio_ms),
        input.audio_ms,
        input
            .input_profile
            .map(|p| format!(", {} Hz, {} ch", p.sample_rate, p.channels))
            .unwrap_or_default()
    ));
    out.push_str(&format!(
        "- **Machine:** {} MB RAM ({} MB free at start), {} logical cores\n",
        ctx.host.total_ram_mb, ctx.host.available_ram_mb, ctx.host.logical_cores
    ));
    out.push_str(&format!(
        "- **Run:** {} UTC · shipped models {} / {} · STT language `{}` · threads STT {} / diarization {} / LLM {} · LLM context {} tokens\n",
        ctx.stamp,
        system::STT_MODEL_ID,
        system::LLM_MODEL_ID,
        ctx.base.stt_language,
        ctx.base.stt_threads,
        ctx.base.diarization_threads,
        ctx.settings.effective_llm_threads(&ctx.host),
        ctx.settings.effective_llm_context()
    ));
    match (input.reference, &input.args.reference) {
        (Some(reference), Some(path)) => out.push_str(&format!(
            "- **Reference:** `{}` ({} words){}\n",
            path.display(),
            text::normalize_words(reference).len(),
            input
                .args
                .reference_range
                .map(|(a, b)| format!(", compared over {}–{} only", text::clock(a), text::clock(b)))
                .unwrap_or_default()
        )),
        _ => out.push_str("- **Reference:** none (no WER — pass `--reference`)\n"),
    }
    out.push_str(&format!("- **Output:** `{}`\n\n", ctx.out.display()));

    if let Some(p) = input.input_profile {
        out.push_str("## Input audio\n\n");
        out.push_str("| grade | score | speech | floor | SNR | speech-band SNR | speech spread | clipped | noise found |\n");
        out.push_str("|---|---|---|---|---|---|---|---|---|\n");
        let noise: Vec<String> = p
            .noise
            .findings
            .iter()
            .map(|f| format!("{} ({:.0}%)", f.label, f.confidence * 100.0))
            .collect();
        out.push_str(&format!(
            "| {} | {} | {} dBFS | {} dBFS | {} dB | {} dB | {} | {:.2}% | {} |\n\n",
            p.quality_grade,
            p.quality_score,
            db(p.speech_dbfs),
            db(p.noise_floor_dbfs),
            db(p.estimated_snr_db),
            db(p.speech_band_snr_db),
            input
                .before
                .and_then(|b| b.spread_db)
                .map(|s| format!("{} dB", db(s)))
                .unwrap_or_else(|| "—".to_string()),
            p.clipped_percent,
            if noise.is_empty() { "none".to_string() } else { noise.join(", ") }
        ));
    }

    let mut skipped: Vec<String> = Vec::new();

    // Front end.
    out.push_str(&format!(
        "## Front end (STT {}, diarization as shipped)\n\n",
        system::STT_MODEL_ID
    ));
    out.push_str("Audio columns are before → after the front end (dBFS / dB). Spread is the leveler's speech-level spread — the near/far gap it exists to close. Repeat and repeated-segment counts are cheap hallucination signals; WER is `S/D/I` in brackets.\n\n");
    out.push_str("| variant | denoiser | level stages | time | RTF | noise floor | speech | SNR | spread | words | speakers | longest repeat | repeated segs | WER |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    let before = input.before;
    for (name, status) in input.front_rows {
        match status {
            Status::Ran(run) => {
                let after = run.after;
                let (longest, repeated) = hallucination_cells(run);
                out.push_str(&format!(
                    "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {longest} | {repeated} | {} |\n",
                    run.denoiser,
                    run.level_stages,
                    secs(run.total_ms()),
                    realtime(run.total_ms(), input.audio_ms),
                    arrow(before.map(|b| b.noise_floor_dbfs), after.map(|a| a.noise_floor_dbfs)),
                    arrow(before.map(|b| b.speech_dbfs), after.map(|a| a.speech_dbfs)),
                    arrow(before.map(|b| b.snr_db), after.map(|a| a.snr_db)),
                    arrow(before.and_then(|b| b.spread_db), after.and_then(|a| a.spread_db)),
                    render::word_count(&run.transcript),
                    render::speaker_count(&run.transcript),
                    wer_cell(run.wer.as_ref())
                ));
            }
            Status::Skipped(why) => {
                out.push_str(&format!("| {name} | _skipped_ | | | | | | | | | | | | |\n"));
                skipped.push(format!("front end `{name}`: skipped: {why}"));
            }
            Status::Failed(why) => {
                out.push_str(&format!("| {name} | _failed_ | | | | | | | | | | | | |\n"));
                skipped.push(format!("front end `{name}`: FAILED: {}", cell(why)));
            }
        }
    }
    out.push('\n');
    for (name, description) in FRONT_ENDS {
        if input.front_rows.iter().any(|(n, _)| n == name) {
            out.push_str(&format!("- `{name}` — {description}\n"));
        }
    }
    out.push('\n');

    // Stage timings, for every pipeline run.
    out.push_str("### Stage timings\n\n");
    out.push_str("| run | preprocess | denoise | VAD | diarization | STT | align | total |\n");
    out.push_str("|---|---|---|---|---|---|---|---|\n");
    let all_runs = input
        .front_rows
        .iter()
        .map(|(n, s)| (n.clone(), s))
        .chain(input.stt_rows.iter().map(|(n, s)| (format!("stt: {n}"), s)));
    for (name, status) in all_runs {
        if let Status::Ran(run) = status {
            let m = &run.metrics;
            let denoise = run.extra_denoise.or(m.denoise);
            out.push_str(&format!(
                "| {name} | {} | {} | {} | {} | {} | {} | {} |\n",
                secs(m.preprocess_ms + run.extra_front_ms),
                denoise.map(|d| secs(d.wall_ms)).unwrap_or_else(|| "—".to_string()),
                m.vad.map(|v| secs(v.wall_ms)).unwrap_or_else(|| "—".to_string()),
                secs(m.diarization.wall_ms),
                secs(m.stt.wall_ms),
                secs(m.align_ms),
                secs(run.total_ms())
            ));
        }
    }
    out.push_str("\n_Preprocess includes denoising (it runs inside the front end); the denoise column breaks it out. For `dpdfnet4`, preprocess includes the bench-side front half._\n\n");

    // STT.
    out.push_str("## STT (shipped front end)\n\n");
    out.push_str("| model | STT wall | RTF | peak RAM | words | speakers | languages | blocks | WER |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|\n");
    let baseline = input
        .front_rows
        .iter()
        .find(|(n, _)| n == "shipped")
        .map(|(_, s)| (system::STT_MODEL_ID.to_string(), s));
    let stt_all = baseline
        .into_iter()
        .chain(input.stt_rows.iter().map(|(n, s)| (n.clone(), s)));
    for (model, status) in stt_all {
        match status {
            Status::Ran(run) => {
                let m = &run.metrics;
                out.push_str(&format!(
                    "| {model} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    secs(m.stt.wall_ms),
                    realtime(m.stt.wall_ms, input.audio_ms),
                    ram(m.stt.peak_ram_mb),
                    render::word_count(&run.transcript),
                    render::speaker_count(&run.transcript),
                    if m.stt_languages.is_empty() { "—".to_string() } else { m.stt_languages.join(" ") },
                    m.stt_blocks.map(|b| b.to_string()).unwrap_or_else(|| "single pass".to_string()),
                    wer_cell(run.wer.as_ref())
                ));
            }
            Status::Skipped(why) => {
                out.push_str(&format!("| {model} | _skipped_ | | | | | | | |\n"));
                skipped.push(format!("STT `{model}`: skipped: {why}"));
            }
            Status::Failed(why) => {
                out.push_str(&format!("| {model} | _failed_ | | | | | | | |\n"));
                skipped.push(format!("STT `{model}`: FAILED: {}", cell(why)));
            }
        }
    }
    out.push('\n');

    // LLM.
    out.push_str(&format!(
        "## LLM (minutes from the `shipped` transcript, {} MB budget)\n\n",
        system::LLM_RAM_BUDGET_MB
    ));
    out.push_str("| model | quant | load | generate | peak RAM | budget | chunks | failed | repaired | decisions | actions | questions | risks |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for row in input.llm_rows {
        let budget = match row.peak_ram_mb {
            Some(mb) if mb > system::LLM_RAM_BUDGET_MB => "**OVER**".to_string(),
            Some(_) => "ok".to_string(),
            None => "n/a".to_string(),
        };
        match &row.outcome {
            Ok(minutes) => {
                let meta = minutes.meta.clone().unwrap_or_default();
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {budget} | {} | {} | {} | {} | {} | {} | {} |\n",
                    row.name,
                    row.quantization,
                    secs(row.load_ms),
                    secs(row.generate_ms),
                    ram(row.peak_ram_mb),
                    meta.chunk_count,
                    meta.failed_chunks,
                    if meta.repaired { "yes" } else { "no" },
                    minutes.decisions.len(),
                    minutes.action_items.len(),
                    minutes.open_questions.len(),
                    minutes.risks.len()
                ));
            }
            Err(err) => {
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {budget} | _failed_ | | | | | | |\n",
                    row.name,
                    row.quantization,
                    secs(row.load_ms),
                    secs(row.generate_ms),
                    ram(row.peak_ram_mb)
                ));
                skipped.push(format!("LLM `{}`: minutes FAILED: {}", row.name, cell(err)));
            }
        }
    }
    for (name, why) in input.llm_skipped {
        out.push_str(&format!("| {name} | _skipped_ | | | | | | | | | | | |\n"));
        skipped.push(format!("LLM `{name}`: {why}"));
    }
    out.push_str("\n_Peak RAM is llama-server's peak working set over the load and the minutes run. Score each `mom-<model>.md` by hand against what was actually decided and assigned._\n\n");

    // Cleanup.
    if input.args.llm_cleanup {
        out.push_str("## Transcript cleanup (experimental, bench-only)\n\n");
        out.push_str("Tokens changed counts every edit including casing and punctuation; content words changed is the same comparison after WER normalisation, i.e. words actually added, removed or replaced. High content change is the failure this pass must not have.\n\n");
        out.push_str("| model | wall | chunks | failed | turns kept raw | tokens changed | content words changed |\n");
        out.push_str("|---|---|---|---|---|---|---|\n");
        for row in input.llm_rows {
            match &row.cleanup {
                Some(Ok(c)) => {
                    let pct = |n: usize, d: usize| {
                        if d == 0 { 0.0 } else { n as f64 * 100.0 / d as f64 }
                    };
                    out.push_str(&format!(
                        "| {} | {} | {} | {} | {}/{} | {:.1}% ({}/{}) | {:.1}% ({}/{}/{}) |\n",
                        row.name,
                        secs(c.wall_ms),
                        c.chunks,
                        c.failed_chunks,
                        c.lines_kept_raw,
                        c.lines,
                        pct(c.changed_tokens, c.raw_tokens),
                        c.changed_tokens,
                        c.raw_tokens,
                        c.content.rate().unwrap_or(0.0) * 100.0,
                        c.content.substitutions,
                        c.content.deletions,
                        c.content.insertions
                    ));
                }
                Some(Err(err)) => {
                    out.push_str(&format!("| {} | _failed_ | | | | | |\n", row.name));
                    skipped.push(format!("cleanup `{}`: FAILED: {}", row.name, cell(err)));
                }
                None => {}
            }
        }
        out.push('\n');
    }

    if !skipped.is_empty() {
        out.push_str("## Skipped and failed\n\n");
        for line in &skipped {
            out.push_str(&format!("- {line}\n"));
        }
        out.push('\n');
    }

    // Per-run warnings, deduplicated.
    let mut warnings: Vec<String> = Vec::new();
    for (name, status) in input.front_rows.iter().chain(input.stt_rows.iter()) {
        if let Status::Ran(run) = status {
            for warning in &run.metrics.warnings {
                let line = format!("{name}: {warning}");
                if !warnings.contains(&line) {
                    warnings.push(line);
                }
            }
        }
    }
    if !warnings.is_empty() {
        out.push_str("## Pipeline warnings\n\n");
        for line in &warnings {
            out.push_str(&format!("- {}\n", cell(line)));
        }
        out.push('\n');
    }

    out.push_str("## Files\n\n");
    out.push_str("- `transcript-<variant>.txt` / `.json` — readable and raw transcript per run\n");
    out.push_str("- `mom-<model>.md` / `.json` — minutes per LLM\n");
    if input.args.llm_cleanup {
        out.push_str("- `cleanup-<model>.txt` / `.diff.txt` — cleaned transcript and word diff against the raw one\n");
    }
    out.push_str("- `runs/<variant>/` — the pipeline's own artifacts (`processed.wav` to listen to, `stt.json`, `diarization.json`, `metrics.json`, `audio_profile.json`)\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_the_documented_flags() {
        let parsed = parse_args(&args(&[
            "abc123",
            "--quick",
            "--reference",
            "ref.txt",
            "--reference-range",
            "01:00-02:30",
            "--llm-cleanup",
        ]))
        .unwrap();
        assert_eq!(parsed.input, "abc123");
        assert!(parsed.quick && parsed.llm_cleanup);
        assert_eq!(parsed.reference_range, Some((60_000, 150_000)));
        assert!(parse_args(&args(&["x", "--reference-range", "01:00-02:00"])).is_err());
        assert!(parse_args(&args(&["x", "--bogus"])).is_err());
        assert!(parse_args(&args(&["x", "--quick", "--variants", "raw"])).is_err());
    }

    #[test]
    fn quick_runs_the_shipped_set_only() {
        let parsed = parse_args(&args(&["x", "--quick"])).unwrap();
        let plan = build_plan(&parsed, Path::new("C:\\data")).unwrap();
        assert_eq!(plan.front_ends, vec!["shipped"]);
        assert_eq!(plan.stt, vec![system::STT_MODEL_ID.to_string()]);
        let llm: Vec<&str> = plan.llm.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(llm, vec![system::LLM_MODEL_ID]);
    }

    #[test]
    fn an_llm_variant_pulls_in_the_shipped_transcript_it_summarises() {
        let parsed = parse_args(&args(&["x", "--variants", "raw,gemma3-4b-q4"])).unwrap();
        let plan = build_plan(&parsed, Path::new("C:\\data")).unwrap();
        assert_eq!(plan.front_ends, vec!["shipped", "raw"]);
        assert!(plan.stt.is_empty());
        assert_eq!(plan.llm.len(), 1);
        assert!(build_plan(
            &parse_args(&args(&["x", "--variants", "nope"])).unwrap(),
            Path::new("C:\\data")
        )
        .is_err());
    }

    #[test]
    fn the_default_plan_covers_every_candidate() {
        let plan = build_plan(&parse_args(&args(&["x"])).unwrap(), Path::new("C:\\data")).unwrap();
        assert_eq!(plan.front_ends.len(), FRONT_ENDS.len());
        assert_eq!(plan.stt.len(), 1 + catalog::ids_of_kind(ModelKind::Stt).len());
        assert_eq!(plan.llm.len(), 1 + catalog::ids_of_kind(ModelKind::Llm).len());
    }

    #[test]
    fn the_output_directory_may_not_be_a_meeting_directory() {
        let data = std::env::temp_dir().join("cadence-bench-guard");
        let meetings = app_lib::storage::meetings_dir(&data);
        std::fs::create_dir_all(meetings.join("m1")).unwrap();
        assert!(guard_out_dir(&meetings.join("m1").join("bench"), &data).is_err());
        assert!(guard_out_dir(&data.join("elsewhere"), &data).is_ok());
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn the_utc_stamp_is_sortable() {
        let stamp = utc_stamp();
        assert_eq!(stamp.len(), 15);
        assert_eq!(&stamp[8..9], "-");
        assert!(stamp.starts_with("20"));
    }

    #[test]
    fn cleanup_responses_map_back_by_id() {
        let fixed = parse_cleanup(r#"{"lines":[{"id":0,"text":"Hello."},{"id":"2","text":"Bye."}]}"#)
            .unwrap();
        assert_eq!(fixed.get(&0).map(String::as_str), Some("Hello."));
        assert_eq!(fixed.get(&2).map(String::as_str), Some("Bye."));
        assert!(parse_cleanup(r#"{"nope":[]}"#).is_err());
    }

    #[test]
    fn cleanup_keeps_speakers_and_counts_changes_against_a_mock_model() {
        use app_lib::pipeline::models::TranscriptSegment;
        let segment = |speaker: &str, text: &str, start: i64| TranscriptSegment {
            speaker: speaker.to_string(),
            speaker_label: speaker.replace("SPEAKER_0", "Person "),
            start_ms: start,
            end_ms: start + 1000,
            text: text.to_string(),
            language: None,
            flags: Vec::new(),
        };
        let transcript = Transcript {
            meeting_id: "t".to_string(),
            segments: vec![
                segment("SPEAKER_01", "we will ship it friday", 0),
                segment("SPEAKER_02", "நாளைக்கு meeting இருக்கு", 1000),
            ],
        };
        // Line 0 recased and punctuated; line 1 omitted by the model.
        let mock = llm::MockLlm::new(
            vec![r#"{"lines":[{"id":0,"text":"We will ship it Friday."}]}"#.to_string()],
            "{}",
        );
        let dir = std::env::temp_dir();
        let (cleaned, stats, diff) = run_cleanup(&mock, &transcript, &dir, 8192).unwrap();
        assert_eq!(cleaned.segments[0].text, "We will ship it Friday.");
        assert_eq!(cleaned.segments[1].text, transcript.segments[1].text);
        assert_eq!(cleaned.segments[0].speaker_label, "Person 1");
        assert_eq!(stats.lines_kept_raw, 1);
        assert_eq!(stats.changed_tokens, 2);
        // Casing and punctuation only: no content words changed.
        assert_eq!(stats.content.errors(), 0);
        assert!(diff.contains("[-we-] {+We+}"), "{diff}");
    }
}
