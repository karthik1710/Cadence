//! Headless runner for the whole local pipeline: audio → preprocess →
//! diarize → STT → align → minutes (Phase 5).
//!
//! It exists so the AI path can be exercised and re-run on test recordings
//! without the UI — which is also how the benchmarks are collected.
//!
//! ```text
//! cargo run --bin pipeline_cli -- <audio.wav> [out-dir] [options]
//!
//!   --no-mom            transcript only; skip minutes generation
//!   --language <name>   output language for the minutes (default English)
//!   --stt-language <m>  auto | code-switched | an ISO code (default
//!                       code-switched)
//!   --speakers <n>      how many people spoke (1-4): diarize into exactly n
//!                       speakers instead of clustering by threshold; omit
//!                       for five or more (automatic, as in the app)
//! ```

use app_lib::pipeline::{self, paths::ModelPaths, PipelineConfig, RunRequest};
use app_lib::settings::Settings;
use app_lib::system::{self, HostInfo};
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
    /// Per-run, like the app's per-meeting hint — not a `Settings` field.
    expected_speakers: Option<u32>,
}

fn parse_args() -> Result<Args, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut settings = Settings::default();
    let mut expected_speakers = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-mom" => settings.generate_mom = false,
            "--language" => {
                settings.mom_language = args.next().ok_or("--language needs a value")?
            }
            "--stt-language" => {
                settings.stt_language =
                    args.next().ok_or("--stt-language needs a value")?
            }
            "--speakers" => {
                let value = args.next().ok_or("--speakers needs a value")?;
                let count: u32 = value
                    .parse()
                    .map_err(|_| format!("--speakers needs a whole number, got {value:?}"))?;
                expected_speakers = pipeline::validate_expected_speakers(Some(count))?;
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
            .ok_or("usage: pipeline_cli <audio.wav> [out-dir] [--no-mom] [--language <name>] [--stt-language <mode>] [--speakers <n>]")?,
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
        expected_speakers,
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
    let config = PipelineConfig {
        expected_speakers: args.expected_speakers,
        ..PipelineConfig::from_settings(&args.settings, &host)
    };

    println!("input:    {}", args.input.display());
    println!("output:   {}", args.out_dir.display());
    println!(
        "machine:  {} MB RAM ({} free), {} logical cores{}",
        host.total_ram_mb,
        host.available_ram_mb,
        host.logical_cores,
        if host.meets_minimum() {
            String::new()
        } else {
            format!(
                " — below the {} MB minimum; minutes may not fit",
                system::MIN_TOTAL_RAM_MB
            )
        }
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
        "config:   stt {} threads, diarization {} threads, {}, context {}",
        config.stt_threads,
        config.diarization_threads,
        // Only one of the two decides the speaker count; print that one.
        match config.expected_speakers {
            Some(n) => format!("{n} expected speaker(s)"),
            None => format!("cluster threshold {}", config.cluster_threshold),
        },
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
    println!("\nper stage:");
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
            // how many blocks whisper saw, and what it detected in them —
            // one language on a meeting known to be mixed is the thing to look
            // at, and it is invisible without this line.
            Some(blocks) => format!("{blocks} blocks, detected: {}", m.stt_languages.join(" ")),
            None => "single pass".to_string(),
        }
    );
    println!("  align:       {:>7.1}s", m.align_ms as f64 / 1000.0);
    println!(
        "  redaction:   {:>7.1}s  {}",
        m.redact_ms as f64 / 1000.0,
        if m.redact_ms == 0 && m.flags_found == 0 {
            "(off, or nothing flagged)".to_string()
        } else {
            format!("{} span(s) flagged", m.flags_found)
        }
    );
    println!("  classify:    {:>7.1}s", m.classify_ms as f64 / 1000.0);
    println!("  minutes:     {:>7.1}s", m.mom_ms as f64 / 1000.0);

    print_front_end(m);
    print_audio_profile(&args.out_dir);
    print_classification(&args.out_dir);
    print_redaction(&args.out_dir);

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

    // a stage that could not run says so. Silently degraded output is the
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

/// Reads one of the derived side-cars back off disk.
///
/// The CLI deliberately reads the *files* rather than holding the values from
/// the run: it is the artifacts the UI will read, so printing anything else
/// would let a broken write pass unnoticed here.
fn artifact<T: serde::de::DeserializeOwned>(out_dir: &Path, name: &str) -> Option<T> {
    let raw = std::fs::read_to_string(out_dir.join(name)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// What the noise reducer, the leveler and the compressor actually did — the
/// stages whose effect is otherwise invisible in a transcript.
fn print_front_end(metrics: &app_lib::pipeline::RunMetrics) {
    let Some(report) = metrics.preprocess_report.as_ref() else {
        return;
    };
    println!("\nfront end:");
    println!(
        "  input:       {} Hz, {} ch, {:.1} dBFS RMS",
        report.input_sample_rate, report.input_channels, report.input_rms_dbfs
    );
    if let Some(nr) = report.noise_reduction.as_ref() {
        println!(
            "  noise red.:  floor {:+.1} dB, speech {:+.1} dB over {} frames",
            nr.noise_floor_delta_db, nr.speech_level_delta_db, nr.frames
        );
    }
    if let Some(lev) = report.leveler.as_ref() {
        println!(
            "  leveler:     spread {:.1} → {:.1} dB, speech {:.0}%, gain {:+.1} dB mean (+{:.1} / -{:.1} max)",
            lev.speech_spread_before_db,
            lev.speech_spread_after_db,
            lev.speech_share * 100.0,
            lev.mean_speech_gain_db,
            lev.max_boost_db,
            lev.max_cut_db
        );
    }
    if let Some(comp) = report.compressor.as_ref() {
        println!(
            "  compressor:  crest {:.1} → {:.1} dB, up to {:.1} dB reduction, makeup {:+.1} dB",
            comp.crest_before_db, comp.crest_after_db, comp.max_reduction_db, comp.makeup_db
        );
    }
    println!(
        "  output:      {:.1} dBFS RMS, {:.1} dBFS peak (gain {:+.1} dB)",
        report.output_rms_dbfs, report.output_peak_dbfs, report.applied_gain_db
    );
}

fn print_audio_profile(out_dir: &Path) {
    use app_lib::pipeline::audio_profile::AudioProfile;
    let Some(profile) = artifact::<AudioProfile>(out_dir, "audio_profile.json") else {
        return;
    };
    println!("\naudio quality: {} ({}/100)", profile.quality_grade, profile.quality_score);
    println!(
        "  speech {:.1} dBFS · peak {:.1} dBFS · floor {:.1} dBFS · crest {:.1} dB",
        profile.speech_dbfs, profile.peak_dbfs, profile.noise_floor_dbfs, profile.crest_factor_db
    );
    println!(
        "  SNR {:.1} dB broadband, {:.1} dB in the speech band · clipped {:.2}% · silence {:.1}%",
        profile.estimated_snr_db,
        profile.speech_band_snr_db,
        profile.clipped_percent,
        profile.silence_percent
    );
    if profile.noise.findings.is_empty() {
        println!("  room: nothing worth reporting");
    } else {
        println!("  room:");
        for finding in &profile.noise.findings {
            println!(
                "    - {} ({:.0}%) — {} {}",
                finding.label,
                finding.confidence * 100.0,
                finding.detail,
                finding.advice
            );
        }
    }
}

fn print_classification(out_dir: &Path) {
    use app_lib::pipeline::topic::MeetingClassification;
    let Some(classification) = artifact::<MeetingClassification>(out_dir, "classification.json")
    else {
        return;
    };
    println!(
        "\nmeeting type: {} ({:.0}% confidence)",
        classification.label,
        classification.confidence * 100.0
    );
    let alternatives: Vec<String> = classification
        .alternatives
        .iter()
        .skip(1)
        .take(3)
        .map(|alt| format!("{} {:.2}", alt.label, alt.score))
        .collect();
    if !alternatives.is_empty() {
        println!("  also scored: {}", alternatives.join(", "));
    }
    let topics: Vec<String> = classification
        .topics
        .iter()
        .take(8)
        .map(|topic| format!("{} ({})", topic.term, topic.count))
        .collect();
    if !topics.is_empty() {
        println!("  topics:      {}", topics.join(", "));
    }
    let s = &classification.signals;
    println!(
        "  signals:     {} turns, {} speakers, {} words, {:.0}% questions, top speaker {:.0}%",
        s.turn_count,
        s.speaker_count,
        s.word_count,
        s.question_ratio * 100.0,
        s.dominant_speaker_share * 100.0
    );
}

fn print_redaction(out_dir: &Path) {
    use app_lib::pipeline::redact::RedactionSummary;
    let Some(summary) = artifact::<RedactionSummary>(out_dir, "redaction.json") else {
        return;
    };
    println!(
        "\nredaction: {} span(s) across {} turn(s), lexicon {} terms{}",
        summary.total,
        summary.turns_affected,
        summary.lexicon_terms,
        if summary.fuzzy { ", fuzzy on" } else { "" }
    );
    if summary.total == 0 {
        return;
    }
    let by_severity: Vec<String> = summary
        .by_severity
        .iter()
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    println!("  by severity: {}", by_severity.join(", "));
    let by_category: Vec<String> = summary
        .by_category
        .iter()
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    println!("  by category: {}", by_category.join(", "));
    // Counts only, no term list: this prints to a terminal someone may be
    // screen-sharing, and the file beside it has the detail if it is wanted.
    println!("  distinct terms: {}", summary.terms.len());
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
