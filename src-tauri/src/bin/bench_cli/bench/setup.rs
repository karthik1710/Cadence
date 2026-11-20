//! `bench_cli setup [--with-large] [--dry-run]` — installs the comparison
//! candidates, one at a time, skipping any already on disk.
//!
//! This is the only part of `bench_cli` that touches the network, and only
//! through `models_manager::install_entry` — the app's own installer, which
//! shells out to Windows' bundled downloader. `compare` never downloads.

use super::catalog;
use app_lib::models_manager::{self, InstallProgress};
use std::cell::Cell;
use std::io::Write;

pub fn run(args: &[String]) -> Result<(), String> {
    let mut with_large = false;
    let mut dry_run = false;
    for arg in args {
        match arg.as_str() {
            "--with-large" => with_large = true,
            "--dry-run" => dry_run = true,
            other => return Err(format!("unknown option for setup: {other}")),
        }
    }

    let data_dir = app_lib::pipeline::paths::default_data_dir()?;
    let bench_dir = catalog::models_dir(&data_dir);
    let chosen: Vec<catalog::Candidate> = catalog::candidates()
        .into_iter()
        .filter(|c| with_large || !c.large)
        .collect();

    println!("Bench candidates → {}\n", bench_dir.display());
    println!("| id | kind | size | status |");
    println!("|---|---|---|---|");
    let mut to_install = Vec::new();
    let mut total_mb = 0u64;
    for candidate in &chosen {
        let installed = candidate.entry.is_installed(&bench_dir);
        println!(
            "| {} | {:?} | {} MB | {} |",
            candidate.entry.id,
            candidate.entry.kind,
            candidate.entry.approx_size_mb,
            if installed { "installed" } else { "to download" }
        );
        if !installed {
            total_mb += candidate.entry.approx_size_mb;
            to_install.push(candidate);
        }
    }
    if !with_large {
        let skipped: Vec<&str> = catalog::candidates()
            .into_iter()
            .filter(|c| c.large)
            .map(|c| c.entry.id)
            .collect();
        println!("\n(not included without --with-large: {})", skipped.join(", "));
    }
    println!(
        "\n{} to download, {:.1} GB in total.",
        to_install.len(),
        total_mb as f64 / 1024.0
    );

    if dry_run {
        println!("Dry run — nothing downloaded.");
        for candidate in &to_install {
            println!("  would fetch {}", candidate.entry.url);
        }
        return Ok(());
    }

    let mut failures = Vec::new();
    for (index, candidate) in to_install.iter().enumerate() {
        println!(
            "\n[{}/{}] {} ({} MB)",
            index + 1,
            to_install.len(),
            candidate.entry.label,
            candidate.entry.approx_size_mb
        );
        let last_printed = Cell::new(-1i64);
        let result = models_manager::install_entry(&bench_dir, &candidate.entry, &|p: InstallProgress| {
            // Print on every 5% step and on phase changes, on one line.
            let step = p.percent / 5;
            if step != last_printed.get() || p.phase != "downloading" {
                last_printed.set(step);
                eprint!(
                    "\r  {:<11} {:>3}%  {} / {} MB   ",
                    p.phase, p.percent, p.downloaded_mb, p.total_mb
                );
                let _ = std::io::stderr().flush();
            }
        });
        eprintln!();
        match result {
            Ok(()) => println!("  installed at {}", candidate.entry.path_in(&bench_dir).display()),
            Err(err) => {
                println!("  FAILED: {err}");
                failures.push(candidate.entry.id);
            }
        }
    }

    if failures.is_empty() {
        println!("\nAll requested candidates are installed.");
        Ok(())
    } else {
        Err(format!("{} candidate(s) failed to install: {}", failures.len(), failures.join(", ")))
    }
}
