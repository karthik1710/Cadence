//! Benchmark harness for the model-selection phases (master prompt Phases 2,
//! 3, 6 and 14).
//!
//! The master prompt forbids committing to an STT model, a diarization
//! threshold, or an LLM before benchmarks exist (§39.3, §39.4). This is the
//! tool that produces them, on the machine that will run them, and prints
//! Markdown tables ready to paste into `docs/model-benchmark.md`.
//!
//! ```text
//! cargo run --release --bin bench_cli -- stt <audio.wav> [--models a,b] [--threads 4,8,24]
//! cargo run --release --bin bench_cli -- diarize <audio.wav> [--thresholds 0.5,0.7,0.8]
//! cargo run --release --bin bench_cli -- llm <transcript.json> [--models qwen3-4b-q4,qwen3-8b-q4]
//! cargo run --release --bin bench_cli -- full <audio.wav>
//! ```
//!
//! Every mode needs the relevant models installed (Settings → Models, or
//! `scripts/setup-models.ps1`). Nothing here touches the network.

use app_lib::pipeline::models::Transcript;
use app_lib::pipeline::{
    self, diarization, llm, mom, paths::ModelPaths, preprocess, stt, PipelineConfig, RunRequest,
};
use app_lib::settings::Settings;
use app_lib::system::HostInfo;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(err) = run() {
        eprintln!("\nBenchmark failed: {err}");
        std::process::exit(1);
    }
}

const USAGE: &str = "usage: bench_cli <stt|diarize|llm|full> <input> [--models a,b] [--threads 4,8] [--thresholds 0.5,0.8]";

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or(USAGE)?;
    let input = PathBuf::from(args.next().ok_or(USAGE)?);
    if !input.exists() {
        return Err(format!("input not found: {}", input.display()));
    }

    let mut models: Vec<String> = Vec::new();
    let mut threads: Vec<usize> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--models" => models = value.split(',').map(|s| s.trim().to_string()).collect(),
            "--threads" => {
                threads = value
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            }
            "--thresholds" => {
                thresholds = value
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            }
            other => return Err(format!("unknown option: {other}")),
        }
    }

    let host = HostInfo::detect();
    println!(
        "machine: {} MB RAM ({} free), {} logical cores, profile {}\n",
        host.total_ram_mb,
        host.available_ram_mb,
        host.logical_cores,
        Settings::default().effective_profile().as_str()
    );

    match mode.as_str() {
        "stt" => bench_stt(&input, &models, &threads, &host),
        "diarize" => bench_diarize(&input, &thresholds, &threads, &host),
        "llm" => bench_llm(&input, &models, &host),
        "full" => bench_full(&input, &host),
        other => Err(format!("unknown mode: {other}\n{USAGE}")),
    }
}

fn work_dir(name: &str) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("echomind-bench-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// 16 kHz mono is what every stage consumes, and resampling once keeps the
/// comparison about the model rather than about the audio prep.
fn prepared_audio(input: &Path, dir: &Path) -> Result<PathBuf, String> {
    let processed = dir.join("processed.wav");
    preprocess::resample_to_16k_mono(input, &processed)?;
    Ok(processed)
}

// ------------------------------------------------------------- Phase 2: STT

fn bench_stt(
    input: &Path,
    models: &[String],
    threads: &[usize],
    host: &HostInfo,
) -> Result<(), String> {
    let models = if models.is_empty() {
        vec![
            "whisper-base".to_string(),
            "whisper-small".to_string(),
            "whisper-medium".to_string(),
        ]
    } else {
        models.to_vec()
    };
    let threads = if threads.is_empty() {
        vec![host.logical_cores.min(24)]
    } else {
        threads.to_vec()
    };

    let dir = work_dir("stt")?;
    let audio = prepared_audio(input, &dir)?;
    let audio_secs = wav_seconds(&audio);

    println!("## STT (Phase 2) — {}\n", input.display());
    println!("| model | threads | wall | realtime | peak RAM | segments | characters |");
    println!("|---|---|---|---|---|---|---|");

    for model in &models {
        for &thread_count in &threads {
            let settings = Settings {
                stt_model: Some(model.clone()),
                stt_threads: Some(thread_count),
                generate_mom: false,
                ..Settings::default()
            };
            let data_dir = pipeline::paths::default_data_dir()?;
            let paths = match ModelPaths::from_data_dir(&data_dir, &settings) {
                Ok(paths) => paths,
                Err(err) => {
                    println!("| {model} | {thread_count} | _not installed_ | | | | |");
                    eprintln!("  skipping {model}: {err}");
                    continue;
                }
            };
            let config = PipelineConfig::from_settings(&settings, host);

            match stt::transcribe(&paths, &config, &audio) {
                Ok((segments, metrics)) => {
                    let characters: usize = segments.iter().map(|s| s.text.chars().count()).sum();
                    println!(
                        "| {model} | {thread_count} | {:.1}s | {:.2}x | {} | {} | {} |",
                        metrics.wall_ms as f64 / 1000.0,
                        realtime(metrics.wall_ms, audio_secs),
                        ram(metrics.peak_ram_mb),
                        segments.len(),
                        characters
                    );
                    // The transcript itself is what accuracy is judged on;
                    // WER needs a reference, which only a human can supply.
                    let out = dir.join(format!("{model}-{thread_count}.json"));
                    let _ = std::fs::write(
                        &out,
                        serde_json::to_string_pretty(&segments).unwrap_or_default(),
                    );
                }
                Err(err) => println!("| {model} | {thread_count} | _failed_ | | | | {err} |"),
            }
        }
    }

    println!(
        "\nTranscripts written to {} — compare them by hand for accuracy, especially on Tamil and code-switched audio (§6, §8).",
        dir.display()
    );
    Ok(())
}

// ----------------------------------------------------- Phase 3: diarization

fn bench_diarize(
    input: &Path,
    thresholds: &[f32],
    threads: &[usize],
    host: &HostInfo,
) -> Result<(), String> {
    let thresholds = if thresholds.is_empty() {
        vec![0.5, 0.6, 0.7, 0.8, 0.9]
    } else {
        thresholds.to_vec()
    };
    let threads = if threads.is_empty() {
        vec![host.logical_cores.min(8)]
    } else {
        threads.to_vec()
    };

    let dir = work_dir("diarize")?;
    let audio = prepared_audio(input, &dir)?;
    let audio_secs = wav_seconds(&audio);
    let data_dir = pipeline::paths::default_data_dir()?;
    let settings = Settings {
        generate_mom: false,
        ..Settings::default()
    };
    let paths = ModelPaths::from_data_dir(&data_dir, &settings)?;

    println!("## Diarization (Phase 3) — {}\n", input.display());
    println!("| threshold | threads | wall | realtime | peak RAM | speakers | segments | longest speaker share |");
    println!("|---|---|---|---|---|---|---|---|");

    for &threshold in &thresholds {
        for &thread_count in &threads {
            let config = PipelineConfig {
                cluster_threshold: threshold,
                diarization_threads: thread_count,
                ..PipelineConfig::from_settings(&settings, host)
            };
            match diarization::diarize(&paths, &config, &audio) {
                Ok((segments, metrics)) => {
                    let speakers: BTreeSet<&str> =
                        segments.iter().map(|s| s.speaker_id.as_str()).collect();
                    let total: i64 = segments.iter().map(|s| s.end_ms - s.start_ms).sum();
                    let longest = speakers
                        .iter()
                        .map(|id| {
                            segments
                                .iter()
                                .filter(|s| &s.speaker_id == id)
                                .map(|s| s.end_ms - s.start_ms)
                                .sum::<i64>()
                        })
                        .max()
                        .unwrap_or(0);
                    println!(
                        "| {threshold} | {thread_count} | {:.1}s | {:.2}x | {} | {} | {} | {:.0}% |",
                        metrics.wall_ms as f64 / 1000.0,
                        realtime(metrics.wall_ms, audio_secs),
                        ram(metrics.peak_ram_mb),
                        speakers.len(),
                        segments.len(),
                        if total > 0 {
                            longest as f64 * 100.0 / total as f64
                        } else {
                            0.0
                        }
                    );
                }
                Err(err) => println!("| {threshold} | {thread_count} | _failed_ | | | | | {err} |"),
            }
        }
    }

    println!("\nCompare the speaker count against the real headcount for this recording (§7). A count far above it means the threshold is too low.");
    Ok(())
}

// ------------------------------------------------------------- Phase 6: LLM

fn bench_llm(transcript_path: &Path, models: &[String], host: &HostInfo) -> Result<(), String> {
    let raw = std::fs::read_to_string(transcript_path)
        .map_err(|e| format!("could not read {}: {e}", transcript_path.display()))?;
    let transcript: Transcript = serde_json::from_str(&raw)
        .map_err(|e| format!("{} is not a transcript.json: {e}", transcript_path.display()))?;

    let models = if models.is_empty() {
        vec![
            "qwen3-4b-q4".to_string(),
            "qwen3-8b-q4".to_string(),
            "gemma3-4b-q4".to_string(),
        ]
    } else {
        models.to_vec()
    };

    let data_dir = pipeline::paths::default_data_dir()?;
    let dir = work_dir("llm")?;

    println!(
        "## LLM (Phase 6) — {} turns, {}\n",
        transcript.segments.len(),
        mom::format_duration(transcript.duration_ms())
    );
    println!("| model | context | wall | peak RAM | valid JSON | repaired | decisions | actions | questions | risks |");
    println!("|---|---|---|---|---|---|---|---|---|---|");

    for model in &models {
        let settings = Settings {
            llm_model: Some(model.clone()),
            ..Settings::default()
        };
        let paths = match ModelPaths::from_data_dir(&data_dir, &settings) {
            Ok(paths) => paths,
            Err(err) => {
                println!("| {model} | | _not installed_ | | | | | | | |");
                eprintln!("  skipping {model}: {err}");
                continue;
            }
        };

        let started = Instant::now();
        let provider =
            match llm::LlamaCppProvider::start(&paths, &settings, host, &data_dir) {
                Ok(provider) => provider,
                Err(err) => {
                    println!("| {model} | | _failed to start_ | | | | | | | |");
                    eprintln!("  {model}: {err}");
                    continue;
                }
            };

        let result = mom::generate(
            &provider,
            &mom::MomRequest {
                transcript: &transcript,
                title: "Benchmark meeting",
                language: "English",
                data_dir: Some(&data_dir),
            },
            &|_, _| {},
        );
        let wall = started.elapsed().as_secs_f64();
        let peak = provider.peak_ram_mb();

        match result {
            Ok(minutes) => {
                let meta = minutes.meta.clone().unwrap_or_default();
                println!(
                    "| {model} | {} | {wall:.1}s | {} | yes | {} | {} | {} | {} | {} |",
                    settings.effective_llm_context(),
                    ram(peak),
                    meta.repaired,
                    minutes.decisions.len(),
                    minutes.action_items.len(),
                    minutes.open_questions.len(),
                    minutes.risks.len()
                );
                let out = dir.join(format!("{model}-mom.json"));
                let _ = std::fs::write(
                    &out,
                    serde_json::to_string_pretty(&minutes).unwrap_or_default(),
                );
            }
            Err(err) => {
                println!(
                    "| {model} | {} | {wall:.1}s | {} | no | | | | | |",
                    settings.effective_llm_context(),
                    ram(peak)
                );
                eprintln!("  {model}: {err}");
            }
        }
        // Dropping the provider stops the server, so the next model starts on
        // a machine with its RAM back.
        drop(provider);
    }

    println!(
        "\nMinutes written to {} — score them against §30 (factual accuracy, action items, decisions, Tamil handling, hallucination resistance) before choosing.",
        dir.display()
    );
    println!("Any model whose peak RAM exceeds 15 GB is rejected outright (§4/§30).");
    Ok(())
}

// ------------------------------------------------- Phase 14: whole pipeline

fn bench_full(input: &Path, host: &HostInfo) -> Result<(), String> {
    let dir = work_dir("full")?;
    let data_dir = pipeline::paths::default_data_dir()?;
    let settings = Settings::default();
    let paths = ModelPaths::from_data_dir(&data_dir, &settings)?;
    let config = PipelineConfig::from_settings(&settings, host);

    let outcome = pipeline::run(
        &RunRequest {
            paths: &paths,
            config: &config,
            settings: &settings,
            host,
            input_wav: input,
            out_dir: &dir,
            data_dir: &data_dir,
            meeting_id: "bench",
            title: "Benchmark meeting",
        },
        &|stage, pct| eprintln!("  {stage} {pct}%"),
    )?;

    let m = &outcome.metrics;
    println!("## Full pipeline (Phase 14) — {}\n", input.display());
    println!("| stage | wall | peak RAM |");
    println!("|---|---|---|");
    println!("| preprocess | {:.1}s | |", m.preprocess_ms as f64 / 1000.0);
    println!(
        "| diarization | {:.1}s | {} |",
        m.diarization.wall_ms as f64 / 1000.0,
        ram(m.diarization.peak_ram_mb)
    );
    println!(
        "| STT | {:.1}s | {} |",
        m.stt.wall_ms as f64 / 1000.0,
        ram(m.stt.peak_ram_mb)
    );
    println!("| alignment | {:.1}s | |", m.align_ms as f64 / 1000.0);
    println!("| minutes | {:.1}s | |", m.mom_ms as f64 / 1000.0);
    println!("| **total** | **{:.1}s** | |", m.total_ms as f64 / 1000.0);
    println!(
        "\nAudio {} · {:.2}x realtime · models {} / {}",
        mom::format_duration(m.audio_ms),
        m.realtime_factor(),
        m.stt_model,
        m.llm_model
    );
    let budget = m.audio_ms as f64 * 0.167;
    println!(
        "§29 target (10 min for a 60 min meeting) — {}",
        if (m.total_ms as f64) <= budget {
            "MET".to_string()
        } else {
            format!(
                "MISSED by {:.1}s",
                (m.total_ms as f64 - budget) / 1000.0
            )
        }
    );
    if let Some(err) = &outcome.mom_error {
        println!("\nMinutes failed: {err}");
    }
    Ok(())
}

// --------------------------------------------------------------- utilities

fn wav_seconds(path: &Path) -> f64 {
    app_lib::storage::wav_duration_ms(path).unwrap_or(0) as f64 / 1000.0
}

fn realtime(wall_ms: u64, audio_secs: f64) -> f64 {
    if audio_secs <= 0.0 {
        return 0.0;
    }
    (wall_ms as f64 / 1000.0) / audio_secs
}

fn ram(mb: Option<u64>) -> String {
    mb.map(|mb| format!("{mb} MB"))
        .unwrap_or_else(|| "n/a".to_string())
}
