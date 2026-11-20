//! Benchmark harness for the model-selection phases (Phases 2,
//! 3, 6 and 14).
//!
//! Project rule: never commit to an STT model, a diarization
//! threshold, or an LLM before benchmarks exist. This is the
//! tool that produces them, on the machine that will run them, and prints
//! Markdown tables ready to paste into `docs/benchmarks.md`.
//!
//! ```text
//! cargo run --release --bin bench_cli -- stt <audio.wav> [--threads 4,8,24] [--languages auto,code-switched]
//! cargo run --release --bin bench_cli -- diarize <audio.wav> [--thresholds 0.5,0.7,0.8]
//! cargo run --release --bin bench_cli -- llm <transcript.json>
//! cargo run --release --bin bench_cli -- full <audio.wav>
//! cargo run --release --bin bench_cli -- setup [--with-large] [--dry-run]
//! cargo run --release --bin bench_cli -- compare <audio.wav | meeting-id> [--out <dir>] [--quick] [--variants a,b,...]
//!     [--reference ref.txt] [--reference-range mm:ss-mm:ss] [--llm-cleanup] [--llm-file model.gguf]
//! cargo run --release --bin bench_cli -- baseline [--clips id,..] [--cache <dir>] [--update] [--out <dir>] [--with-mom] [--reuse <run-dir>]
//! cargo run --release --bin bench_cli -- --help
//! ```
//!
//! Every machine runs the same STT and LLM (`app_lib::system`), so the stt
//! and llm modes measure that one model. Comparing candidate replacements is
//! what `setup` + `compare` are for (see `bench/`): the candidates live in a
//! bench-only catalog, install under `<app_data>/models/bench`, and never
//! appear in the app.
//!
//! Every mode needs the relevant models installed (Settings → Models, or
//! `scripts/setup-models.ps1`). Only `setup` touches the network, through the
//! app's own model installer.

mod bench;

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

const USAGE: &str = "usage: bench_cli <stt|diarize|llm|full> <input> [--threads 4,8] [--thresholds 0.5,0.8] [--languages auto,code-switched,ta]
       bench_cli setup [--with-large] [--dry-run]
       bench_cli compare <audio.wav | meeting-id> [--out <dir>] [--quick] [--variants a,b,...] [--reference <ref.txt>] [--reference-range mm:ss-mm:ss] [--llm-cleanup] [--llm-file <model.gguf>]
       bench_cli baseline [--clips id,..] [--cache <dir>] [--update] [--out <dir>] [--with-mom] [--reuse <run-dir>]";

const HELP: &str = "bench_cli — Cadence benchmark harness (dev-only; never shipped)

MODES
  stt <audio.wav> [--threads 4,8,24] [--languages auto,code-switched]
      Time the shipped STT model per thread count / language mode.
  diarize <audio.wav> [--thresholds 0.5,0.7,0.8] [--threads 8]
      Sweep the diarization cluster threshold.
  llm <transcript.json>
      Generate minutes from a transcript with the shipped LLM.
  full <audio.wav>
      Run the whole pipeline once and print per-stage timings.

  setup [--with-large] [--dry-run]
      Install the bench-only comparison candidates into
      %APPDATA%\\com.cadence.app\\models\\bench\\, one at a time, skipping any
      already installed. The only mode that touches the network.
      Candidates: whisper-medium, whisper-large-v3-turbo (q5_0),
      gemma3-4b-q4, qwen35-4b-q4, gemma4-e2b-q4, llama3.1-8b-q4,
      gemma4-e4b-qat-q4, qwen35-9b-q4, qwen3-8b-q5, denoise-dpdfnet4 (the
      DPDFNet-4 denoiser behind the dpdfnet4 front-end variant).
        --with-large   also install gemma4-12b-qat-q4 and gemma3-12b-q4
                       (over the RAM budget; quality-ceiling references only)
        --dry-run      list each item, its size and the total; download nothing

  compare <audio.wav | meeting-id> [options]
      Run one recording through a matrix of front-end, STT and LLM variants
      on the real pipeline code and write transcripts, minutes and report.md.
      A meeting id is read from <app_data>/meetings/<id>/original.wav,
      read-only. Uninstalled candidates are skipped, never downloaded.
        --out <dir>                 output directory (default:
                                    <repo>/bench-runs/<timestamp>-<stem>/)
        --quick                     shipped front end + whisper-small +
                                    qwen3-8b-q4 only
        --variants a,b,...          explicit subset of variant names:
                                    front end: raw, shipped, no-leveler,
                                      nr-instead-of-gtcrn, gtcrn-only, dpdfnet4
                                    STT: whisper-small, whisper-medium,
                                      whisper-large-v3-turbo
                                    LLM: qwen3-8b-q4, gemma3-4b-q4,
                                      qwen35-4b-q4, gemma4-e2b-q4,
                                      llama3.1-8b-q4, gemma4-e4b-qat-q4,
                                      qwen35-9b-q4, qwen3-8b-q5,
                                      gemma4-12b-qat-q4, gemma3-12b-q4
        --reference <ref.txt>       hand transcript; adds WER columns
        --reference-range mm:ss-mm:ss
                                    compute WER only over that time range
        --llm-cleanup               experimental: per-chunk LLM transcript
                                    correction, with word diff and changed-%
        --llm-file <model.gguf>[,..]
                                    also generate minutes with an ad-hoc GGUF

  baseline [options]
      Regression test: run the clips in tests/baseline/clips.json (real
      YouTube audio + human captions) through the shipped pipeline, score each
      transcript against its captions and against the recorded `expected`
      values, and print metric | expected | actual | delta | status. Exits 1
      if a gated metric regressed: WER beyond the clip's tolerance (relative),
      or speakers / repeat run / repeated segments outside their bounds
      (absolute). RTF and timings are reported, never gated. A metric named
      in a clip's `known_failures` reports KNOWN instead of failing, and
      FIXED (remove the entry) once it passes. Fetch the clips first with
      scripts/fetch-baseline-clips.ps1; nothing here downloads.
        --clips id,..               run only these clip ids (default: all)
        --cache <dir>               cached clips (default:
                                    <repo>/bench-runs/baseline-cache)
        --out <dir>                 output directory (default:
                                    <repo>/bench-runs/<timestamp>-baseline/)
        --update                    rewrite the `expected` blocks in clips.json
                                    with this run's numbers; only for an
                                    intentional improvement. WER changes are
                                    then accepted; absolute gates still apply
        --with-mom                  also generate minutes with the shipped LLM
                                    and report whether they parsed (skipped if
                                    the LLM is not installed)
        --reuse <run-dir>           fast loop for diarization/alignment work:
                                    take each clip's processed.wav + stt.json
                                    from an earlier baseline run and re-run
                                    only diarization and alignment. Refuses if
                                    the cached audio no longer matches that
                                    run. Not with --update or --with-mom

  --help, -h                        show this help
";

fn run() -> Result<(), String> {
    let all: Vec<String> = std::env::args().skip(1).collect();
    let wants_help = |rest: &[String]| rest.iter().any(|a| a == "--help" || a == "-h");
    match all.first().map(String::as_str) {
        None | Some("--help") | Some("-h") | Some("help") => {
            print!("{HELP}");
            return Ok(());
        }
        Some("setup" | "compare" | "baseline") if wants_help(&all[1..]) => {
            print!("{HELP}");
            return Ok(());
        }
        Some("setup") => return bench::setup::run(&all[1..]),
        Some("compare") => return bench::compare::run(&all[1..]),
        Some("baseline") => return bench::baseline::run(&all[1..]),
        _ => {}
    }

    let mut args = all.into_iter();
    let mode = args.next().ok_or(USAGE)?;
    let input = PathBuf::from(args.next().ok_or(USAGE)?);
    if !input.exists() {
        return Err(format!("input not found: {}", input.display()));
    }

    let mut threads: Vec<usize> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    let mut languages: Vec<String> = Vec::new();
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--threads" => {
                threads = value
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            }
            "--languages" => {
                languages = value.split(',').map(|s| s.trim().to_string()).collect()
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
        "machine: {} MB RAM ({} free), {} logical cores{}\n",
        host.total_ram_mb,
        host.available_ram_mb,
        host.logical_cores,
        if host.meets_minimum() {
            ""
        } else {
            " (below the supported minimum)"
        }
    );

    match mode.as_str() {
        "stt" => bench_stt(&input, &threads, &languages, &host),
        "diarize" => bench_diarize(&input, &thresholds, &threads, &host),
        "llm" => bench_llm(&input, &host),
        "full" => bench_full(&input, &host),
        other => Err(format!("unknown mode: {other}\n{USAGE}")),
    }
}

fn work_dir(name: &str) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("cadence-bench-{name}"));
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
    threads: &[usize],
    languages: &[String],
    host: &HostInfo,
) -> Result<(), String> {
    let model = Settings::default().effective_stt_model();
    let threads = if threads.is_empty() {
        vec![host.logical_cores.min(24)]
    } else {
        threads.to_vec()
    };
    let languages = if languages.is_empty() {
        vec!["auto".to_string()]
    } else {
        languages.to_vec()
    };

    let dir = work_dir("stt")?;
    let audio = prepared_audio(input, &dir)?;
    let audio_secs = wav_seconds(&audio);
    let data_dir = pipeline::paths::default_data_dir()?;
    // Code-switched mode places its block boundaries in silence, so it needs
    // the same VAD pass the real pipeline runs.
    let speech = detected_speech(&data_dir, &dir, &audio, host);

    println!("## STT (Phase 2) — {}\n", input.display());
    println!("| model | language | threads | wall | realtime | peak RAM | blocks | detected | segments | characters |");
    println!("|---|---|---|---|---|---|---|---|---|---|");

    for language in &languages {
        for &thread_count in &threads {
            let settings = Settings {
                stt_threads: Some(thread_count),
                stt_language: language.clone(),
                generate_mom: false,
                ..Settings::default()
            };
            let paths = match ModelPaths::from_data_dir(&data_dir, &settings) {
                Ok(paths) => paths,
                Err(err) => {
                    println!("| {model} | {language} | {thread_count} | _not installed_ | | | | | | |");
                    eprintln!("  skipping {model}: {err}");
                    continue;
                }
            };
            let config = PipelineConfig::from_settings(&settings, host);

            match stt::transcribe(&stt::SttRequest {
                paths: &paths,
                config: &config,
                wav_path: &audio,
                speech: speech.as_deref(),
                data_dir: Some(&data_dir),
            }) {
                Ok(outcome) => {
                    let characters: usize =
                        outcome.segments.iter().map(|s| s.text.chars().count()).sum();
                    println!(
                        "| {model} | {language} | {thread_count} | {:.1}s | {:.2}x | {} | {} | {} | {} | {} |",
                        outcome.metrics.wall_ms as f64 / 1000.0,
                        realtime(outcome.metrics.wall_ms, audio_secs),
                        ram(outcome.metrics.peak_ram_mb),
                        outcome
                            .blocks
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "single pass".to_string()),
                        if outcome.languages.is_empty() {
                            "—".to_string()
                        } else {
                            outcome.languages.join(" ")
                        },
                        outcome.segments.len(),
                        characters
                    );
                    for warning in &outcome.warnings {
                        eprintln!("  warning: {warning}");
                    }
                    // The transcript itself is what accuracy is judged on;
                    // WER needs a reference, which only a human can supply.
                    let out = dir.join(format!("{model}-{language}-{thread_count}.json"));
                    let _ = std::fs::write(
                        &out,
                        serde_json::to_string_pretty(&outcome.segments).unwrap_or_default(),
                    );
                }
                Err(err) => {
                    println!("| {model} | {language} | {thread_count} | _failed_ | | | | | | {err} |")
                }
            }
        }
    }

    println!(
        "\nTranscripts written to {} — compare them by hand for accuracy, especially on Tamil and code-switched audio.",
        dir.display()
    );
    Ok(())
}

/// Runs the VAD stage the way the pipeline does, so `--languages
/// code-switched` benchmarks the same thing the app would do. `None` when the
/// VAD model is not installed (it is part of first-run setup, but the
/// pipeline still runs without it) — the block path then reports why it
/// fell back rather than pretending it ran.
fn detected_speech(
    data_dir: &Path,
    work: &Path,
    audio: &Path,
    host: &HostInfo,
) -> Option<Vec<pipeline::vad::SpeechSegment>> {
    if !pipeline::vad::is_available(data_dir) {
        return None;
    }
    let threads = Settings::default().effective_diarization_threads(host);
    match pipeline::vad::detect(data_dir, threads, audio, &work.join("vad_scratch.wav")) {
        Ok((report, _)) => Some(report.segments),
        Err(err) => {
            eprintln!("  voice activity detection failed: {err}");
            None
        }
    }
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

    println!("\nCompare the speaker count against the real headcount for this recording. A count far above it means the threshold is too low.");
    Ok(())
}

// ------------------------------------------------------------- Phase 6: LLM

fn bench_llm(transcript_path: &Path, host: &HostInfo) -> Result<(), String> {
    let raw = std::fs::read_to_string(transcript_path)
        .map_err(|e| format!("could not read {}: {e}", transcript_path.display()))?;
    let transcript: Transcript = serde_json::from_str(&raw)
        .map_err(|e| format!("{} is not a transcript.json: {e}", transcript_path.display()))?;

    let data_dir = pipeline::paths::default_data_dir()?;
    let dir = work_dir("llm")?;

    println!(
        "## LLM (Phase 6) — {} turns, {}\n",
        transcript.segments.len(),
        mom::format_duration(transcript.duration_ms())
    );
    println!("| model | context | wall | peak RAM | valid JSON | repaired | decisions | actions | questions | risks |");
    println!("|---|---|---|---|---|---|---|---|---|---|");

    let settings = Settings::default();
    let model = settings.effective_llm_model();
    let paths = ModelPaths::from_data_dir(&data_dir, &settings)?;

    let started = Instant::now();
    let provider = llm::LlamaCppProvider::start(&paths, &settings, host, &data_dir)
        .map_err(|err| format!("{model} failed to start: {err}"))?;

    let result = mom::generate(
        &provider,
        &mom::MomRequest {
            transcript: &transcript,
            title: "Benchmark meeting",
            language: "English",
            redaction: settings.effective_redaction_mode(),
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
    // Stop the server before printing, so nothing holds its RAM meanwhile.
    drop(provider);

    println!(
        "\nMinutes written to {} — score them against the rubric (factual accuracy, action items, decisions, Tamil handling, hallucination resistance).",
        dir.display()
    );
    println!(
        "Peak RAM must stay within the {} MB LLM budget on a 16 GB machine.",
        app_lib::system::LLM_RAM_BUDGET_MB
    );
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
    // Both new stages are sub-second in every measurement so far, which is
    // precisely why they are in the table: the claim that they are free is
    // only worth anything while it keeps being re-measured.
    println!("| redaction | {:.1}s | |", m.redact_ms as f64 / 1000.0);
    println!("| classification | {:.1}s | |", m.classify_ms as f64 / 1000.0);
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
        "Processing-time target (10 min for a 60 min meeting) — {}",
        if (m.total_ms as f64) <= budget {
            "MET".to_string()
        } else {
            format!(
                "MISSED by {:.1}s",
                (m.total_ms as f64 - budget) / 1000.0
            )
        }
    );
    print_audio_quality(&dir);
    print_meeting_shape(&dir, m.flags_found);

    if !m.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &m.warnings {
            println!("- {warning}");
        }
    }
    if let Some(err) = &outcome.mom_error {
        println!("\nMinutes failed: {err}");
    }
    Ok(())
}

/// The audio-quality row of a full-pipeline benchmark.
///
/// A benchmark table without it is unreadable a month later: a WER or a
/// realtime factor only means something next to what the recording was
/// actually like, and "the same audio" is a claim that needs a number.
fn print_audio_quality(dir: &Path) {
    use app_lib::pipeline::audio_profile::AudioProfile;
    let Some(profile) = artifact::<AudioProfile>(dir, "audio_profile.json") else {
        return;
    };
    println!("\n### Input audio\n");
    println!("| grade | score | speech | floor | SNR | speech-band SNR | crest | clipped |");
    println!("|---|---|---|---|---|---|---|---|");
    println!(
        "| {} | {} | {:.1} dBFS | {:.1} dBFS | {:.1} dB | {:.1} dB | {:.1} dB | {:.2}% |",
        profile.quality_grade,
        profile.quality_score,
        profile.speech_dbfs,
        profile.noise_floor_dbfs,
        profile.estimated_snr_db,
        profile.speech_band_snr_db,
        profile.crest_factor_db,
        profile.clipped_percent
    );
    if !profile.noise.findings.is_empty() {
        let findings: Vec<String> = profile
            .noise
            .findings
            .iter()
            .map(|f| format!("{} ({:.0}%)", f.label, f.confidence * 100.0))
            .collect();
        println!("\nNoise found: {}", findings.join(", "));
    }
}

fn print_meeting_shape(dir: &Path, flags_found: usize) {
    use app_lib::pipeline::topic::MeetingClassification;
    if let Some(classification) = artifact::<MeetingClassification>(dir, "classification.json") {
        let topics: Vec<&str> = classification
            .topics
            .iter()
            .take(6)
            .map(|t| t.term.as_str())
            .collect();
        println!(
            "\nClassified as **{}** ({:.0}% confidence) · topics: {}",
            classification.label,
            classification.confidence * 100.0,
            if topics.is_empty() {
                "none".to_string()
            } else {
                topics.join(", ")
            }
        );
    }
    println!("Redaction flags: {flags_found}");
}

fn artifact<T: serde::de::DeserializeOwned>(dir: &Path, name: &str) -> Option<T> {
    let raw = std::fs::read_to_string(dir.join(name)).ok()?;
    serde_json::from_str(&raw).ok()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The help text names every variant by hand; this keeps it honest when a
    /// candidate or front end is added, renamed or dropped.
    #[test]
    fn help_names_every_variant_and_flag() {
        // `setup`'s --with-large paragraph, and the LLM list under --variants.
        let with_large = &HELP[HELP.find("--with-large   ").unwrap()..HELP.find("--dry-run      ").unwrap()];
        let llm_variants = &HELP[HELP.find("LLM: ").unwrap()..HELP.find("--reference <").unwrap()];
        for candidate in bench::catalog::candidates() {
            let id = candidate.entry.id;
            assert!(HELP.contains(id), "help misses {id}");
            assert_eq!(
                with_large.contains(id),
                candidate.large,
                "--with-large help and the catalog disagree about {id}"
            );
            if candidate.entry.kind == app_lib::models_manager::ModelKind::Llm {
                assert!(llm_variants.contains(id), "--variants help misses LLM {id}");
            }
        }
        for name in bench::compare::front_end_names() {
            assert!(HELP.contains(name), "help misses front end {name}");
        }
        for flag in [
            "--with-large", "--dry-run", "--out", "--quick", "--variants", "--reference",
            "--reference-range", "--llm-cleanup", "--llm-file", "--clips", "--cache",
            "--update", "--with-mom", "--reuse", "known_failures", "baseline",
        ] {
            assert!(HELP.contains(flag), "help misses {flag}");
        }
        assert!(HELP.contains(app_lib::system::STT_MODEL_ID));
        assert!(HELP.contains(app_lib::system::LLM_MODEL_ID));
    }
}
