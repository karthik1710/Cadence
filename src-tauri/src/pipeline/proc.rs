//! Running a native AI binary and measuring what it cost.
//!
//! Every stage of the pipeline shells out to a native tool, and §29 asks for
//! wall time *and* peak RAM per stage. Both come from here, so no stage has to
//! reimplement process handling — and so a stage's cost is measured on real
//! runs, not only inside the benchmark harness.

use crate::system;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// What one stage cost (§29).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StageMetrics {
    pub wall_ms: u64,
    /// Peak working set of the child process. `None` when the OS would not
    /// report it (or on a non-Windows build).
    pub peak_ram_mb: Option<u64>,
}

#[derive(Debug)]
pub struct Measured {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
    pub status: String,
    pub metrics: StageMetrics,
}

const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Runs `command` to completion, capturing its output and sampling its memory.
///
/// stdout and stderr are drained on their own threads: a long meeting makes
/// sherpa-onnx print thousands of segment lines, and leaving those in a pipe
/// while polling would deadlock as soon as the buffer filled.
pub fn run_measured(mut command: Command) -> Result<Measured, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let program = command.get_program().to_string_lossy().to_string();
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| format!("failed to launch {program}: {e}"))?;

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_reader = std::thread::spawn(move || drain(stdout_pipe));
    let stderr_reader = std::thread::spawn(move || drain(stderr_pipe));

    let mut peak_ram_mb = system::peak_working_set_mb(&child);
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None => {
                peak_ram_mb = max_option(peak_ram_mb, system::peak_working_set_mb(&child));
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    };
    // The process handle stays valid until `child` is dropped, so the final
    // peak — usually the highest — is still readable here.
    peak_ram_mb = max_option(peak_ram_mb, system::peak_working_set_mb(&child));

    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();

    Ok(Measured {
        stdout,
        stderr,
        success: status.success(),
        status: status.to_string(),
        metrics: StageMetrics {
            wall_ms: started.elapsed().as_millis() as u64,
            peak_ram_mb,
        },
    })
}

fn drain<R: Read>(pipe: Option<R>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buffer = Vec::new();
    let _ = pipe.read_to_end(&mut buffer);
    String::from_utf8_lossy(&buffer).to_string()
}

fn max_option(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_a_process_that_succeeds() {
        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            command.args(["/C", "echo hello"]);
        } else {
            command.args(["-c", "echo hello"]);
        }
        let measured = run_measured(command).unwrap();
        assert!(measured.success);
        assert!(measured.stdout.contains("hello"));
    }

    #[test]
    fn reports_a_process_that_fails() {
        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            command.args(["/C", "exit 3"]);
        } else {
            command.args(["-c", "exit 3"]);
        }
        let measured = run_measured(command).unwrap();
        assert!(!measured.success);
    }

    #[test]
    fn reports_a_missing_binary_clearly() {
        let err = run_measured(Command::new("cadence-no-such-binary")).unwrap_err();
        assert!(err.contains("failed to launch"));
    }

    #[test]
    fn keeps_the_larger_of_two_samples() {
        assert_eq!(max_option(Some(4), Some(9)), Some(9));
        assert_eq!(max_option(None, Some(9)), Some(9));
        assert_eq!(max_option(Some(4), None), Some(4));
        assert_eq!(max_option(None, None), None);
    }
}
