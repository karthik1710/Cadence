//! Headless runner for the whole local pipeline: audio → preprocess →
//! diarize → STT → align → minutes (master prompt Phase 5).
//!
//! It exists so the AI path can be exercised and re-run on test recordings
//! without the UI — which is also how the §29 benchmarks are collected.
//!
//! ```text
//! cargo run --bin pipeline_cli -- <audio.wav> [out-dir] [options]
//!
//!   --no-mom            transcript only; skip minutes generation
//!   --profile <name>    force low | balanced | high instead of detecting
//!   --stt <model-id>    override the STT model (e.g. whisper-base)
//!   --llm <model-id>    override the language model (e.g. qwen3-4b-q4)
//!   --language <name>   output language for the minutes (default English)
//!   --stt-language <m>  auto | code-switched | an ISO code (§8, default
//!                       code-switched)
//! ```

use app_lib::pipeline::{self, paths::ModelPaths, PipelineConfig, RunRequest};
use app_lib::settings::Settings;
use app_lib::system::{HostInfo, MemoryProfile};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(err) = run() {
        eprintln!("\nPipeline failed: {err}");
        std::process::exit(1);
    }
}

struct Args {
    input: PathBuf,
    out_dir: PathBuf,
    settings: Settings,
}

fn parse_args() -> Result<Args, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut settings = Settings::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-mom" => settings.generate_mom = false,
            "--profile" => {
                let value = args.next().ok_or("--profile needs a value")?;
                MemoryProfile::parse(&value)
                    .ok_or_else(|| format!("unknown profile: {value}"))?;
                settings.profile = value;
            }
            "--stt" => settings.stt_model = Some(args.next().ok_or("--stt needs a value")?),
            "--llm" => settings.llm_model = Some(args.next().ok_or("--llm needs a value")?),
            "--language" => {
                settings.mom_language = args.next().ok_or("--language needs a value")?
            }
            "--stt-language" => {
                settings.stt_language =
                    args.next().ok_or("--stt-language needs a value")?
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option: {other}"))
            }
            other => positional.push(other.to_string()),
        }
    }

    let input = PathBuf::from(
        positional
            .first()
            .ok_or("usage: pipeline_cli <audio.wav> [out-dir] [--no-mom] [--profile <name>]")?,
    );
    if !input.exists() {
        return Err(format!("input audio not found: {}", input.display()));
    }
    let out_dir = positional
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("").with_extension("pipeline"));

    Ok(Args {
        input,
        out_dir,
        settings,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let meeting_id = args
        .out_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "cli-run".to_string());

    let data_dir = pipeline::paths::default_data_dir()?;
    let paths = ModelPaths::from_data_dir(&data_dir, &args.settings)?;
    let host = HostInfo::detect();
    let config = PipelineConfig::from_settings(&args.settings, &host);
    let profile = args.settings.effective_profile();

    println!("input:    {}", args.input.display());
    println!("output:   {}", args.out_dir.display());
    println!(
        "machine:  {} MB RAM ({} free), {} logical cores → {} profile",
        host.total_ram_mb,
        host.available_ram_mb,
        host.logical_cores,
        profile.as_str()
    );
    println!(
        "models:   stt {}, llm {}{}",
        args.settings.effective_stt_model(),
        args.settings.effective_llm_model(),
        if config.generate_mom {
            String::new()
        } else {
            " (minutes disabled)".to_string()
        }
    );
    println!(
        "config:   stt {} threads, diarization {} threads, cluster threshold {}, context {}",
        config.stt_threads,
        config.diarization_threads,
        config.cluster_threshold,
        args.settings.effective_llm_context()
    );

    let started = Instant::now();
    let outcome = pipeline::run(
        &RunRequest {
            paths: &paths,
            config: &config,
            settings: &args.settings,
            host: &host,
            input_wav: &args.input,
            out_dir: &args.out_dir,
            data_dir: &data_dir,
            meeting_id: &meeting_id,
            title: &meeting_id,
        },
        &|stage, pct| {
            println!(
                "[{:>7.1}s] {pct:>3}%  {stage}",
                started.elapsed().as_secs_f64()
            );
        },
    )?;

    let transcript = &outcome.transcript;
    let audio_secs = transcript.duration_ms() as f64 / 1000.0;
    let elapsed = started.elapsed().as_secs_f64();
    let speakers = transcript
        .segments
        .iter()
        .map(|s| s.speaker.as_str())
        .collect::<std::collections::BTreeSet<_>>();

    println!();
    println!("turns:    {}", transcript.segments.len());
    println!("speakers: {}", speakers.len());
    println!(
        "elapsed:  {elapsed:.1}s for {audio_secs:.1}s of audio ({:.2}x realtime)",
        if audio_secs > 0.0 {
            elapsed / audio_secs
        } else {
            0.0
        }
    );

    let m = &outcome.metrics;
    println!("\nper stage (§29):");
    println!("  preprocess:  {:>7.1}s", m.preprocess_ms as f64 / 1000.0);
    println!(
        "  diarization: {:>7.1}s  peak {}",
        m.diarization.wall_ms as f64 / 1000.0,
        ram(m.diarization.peak_ram_mb)
    );
    println!(
        "  stt:         {:>7.1}s  peak {}  {}",
        m.stt.wall_ms as f64 / 1000.0,
        ram(m.stt.peak_ram_mb),
        match m.stt_blocks {
            // §8: how many blocks whisper saw, and what it detected in them —
            // one language on a meeting known to be mixed is the thing to look
            // at, and it is invisible without this line.
            Some(blocks) => format!("{blocks} blocks, detected: {}", m.stt_languages.join(" ")),
            None => "single pass".to_string(),
        }
    );
    println!("  align:       {:>7.1}s", m.align_ms as f64 / 1000.0);
    println!("  minutes:     {:>7.1}s", m.mom_ms as f64 / 1000.0);

    match (&outcome.mom, &outcome.mom_error) {
        (Some(mom), _) => {
            println!("\nminutes:");
            println!("  summary:      {}", first_line(&mom.summary));
            println!("  decisions:    {}", mom.decisions.len());
            println!("  action items: {}", mom.action_items.len());
            println!("  questions:    {}", mom.open_questions.len());
            println!("  risks:        {}", mom.risks.len());
        }
        (None, Some(err)) => {
            println!("\nminutes: FAILED — {err}");
            println!("(the transcript above is unaffected and saved)");
        }
        (None, None) => println!("\nminutes: skipped"),
    }

    // §33: a stage that could not run says so. Silently degraded output is the
    // failure mode this project cares most about.
    if !m.warnings.is_empty() {
        println!("\nwarnings:");
        for warning in &m.warnings {
            println!("  - {warning}");
        }
    }

    print_preview(&args.out_dir, transcript);
    Ok(())
}

fn ram(mb: Option<u64>) -> String {
    mb.map(|mb| format!("{mb} MB"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_string()
}

fn print_preview(out_dir: &Path, transcript: &app_lib::pipeline::models::Transcript) {
    println!("\nfirst turns:");
    for seg in transcript.segments.iter().take(5) {
        let secs = seg.start_ms / 1000;
        println!(
            "  [{:02}:{:02}:{:02}] {}: {}",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60,
            seg.speaker_label,
            seg.text
        );
    }
    println!(
        "\nartifacts: {}",
        out_dir.join("transcript.json").display()
    );
}
