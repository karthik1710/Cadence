//! Phase 5 CLI runner: drives the local audio → transcript pipeline without
//! the Tauri UI, so STT/diarization/alignment can be benchmarked and re-run on
//! test recordings (master prompt §29) before UI integration lands in Phase 8.
//!
//! ```text
//! cargo run --bin pipeline_cli -- <audio.wav> [out-dir]
//! ```

use app_lib::pipeline::{self, paths::ModelPaths, PipelineConfig};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(err) = run() {
        eprintln!("\nPipeline failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: pipeline_cli <audio.wav> [out-dir]")?,
    );
    if !input.exists() {
        return Err(format!("input audio not found: {}", input.display()));
    }

    let out_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("").with_extension("pipeline"));
    let meeting_id = out_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "cli-run".to_string());

    let paths = ModelPaths::from_data_dir(&pipeline::paths::default_data_dir()?)?;

    let config = PipelineConfig::default();
    println!("input:   {}", input.display());
    println!("output:  {}", out_dir.display());
    println!(
        "config:  stt {} threads, diarization {} threads, cluster threshold {}",
        config.stt_threads, config.diarization_threads, config.cluster_threshold
    );

    let started = Instant::now();
    let transcript = pipeline::run(
        &paths,
        &config,
        &input,
        &out_dir,
        &meeting_id,
        &|stage, pct| {
            println!("[{:>7.1}s] {pct:>3}%  {stage}", started.elapsed().as_secs_f64());
        },
    )?;

    let audio_secs = transcript
        .segments
        .last()
        .map(|s| s.end_ms as f64 / 1000.0)
        .unwrap_or(0.0);
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
        if audio_secs > 0.0 { elapsed / audio_secs } else { 0.0 }
    );
    print_preview(&out_dir, &transcript);
    Ok(())
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
    println!("\nfull transcript: {}", out_dir.join("transcript.json").display());
}
