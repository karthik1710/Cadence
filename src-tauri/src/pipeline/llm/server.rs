//! Lifecycle for one local `llama-server.exe` process.
//!
//! The server is started when MoM generation begins and killed when it ends —
//! it never outlives a processing run, so a finished meeting leaves no model
//! resident in RAM. That matters most on the machines this tiering exists for:
//! on 16 GB, several gigabytes held by an idle inference server is the
//! difference between a responsive laptop and a swapping one.

use crate::pipeline::http;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Loading multi-gigabyte weights from a cold page cache on a laptop HDD/SSD
/// can genuinely take a couple of minutes.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

pub struct LlamaServer {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl LlamaServer {
    /// Spawns the server and blocks until it reports healthy.
    pub fn start(
        exe: &Path,
        model: &Path,
        context_tokens: usize,
        threads: usize,
        log_path: &Path,
    ) -> Result<Self, String> {
        if !exe.exists() {
            return Err(format!(
                "llama-server.exe is not installed at {}. Install the llama.cpp runtime from Settings → Models.",
                exe.display()
            ));
        }
        if !model.exists() {
            return Err(format!(
                "the selected language model is not installed at {}. Install it from Settings → Models.",
                model.display()
            ));
        }

        let port = http::free_local_port()?;
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        // Truncated per run: this is a diagnostic tail, not an audit trail, and
        // it must not grow without bound. llama.cpp does not log prompt or
        // response text at its default verbosity (§34).
        let log = std::fs::File::create(log_path)
            .map_err(|e| format!("could not open {}: {e}", log_path.display()))?;
        let log_err = log.try_clone().map_err(|e| e.to_string())?;

        let child = Command::new(exe)
            .arg("--model")
            .arg(model)
            // Loopback only: the inference server is never reachable from
            // another machine, even on a shared network.
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(context_tokens.to_string())
            .arg("--threads")
            .arg(threads.to_string())
            // Apply the model's own chat template, so instruct models are
            // prompted the way they were trained.
            .arg("--jinja")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err))
            .spawn()
            .map_err(|e| format!("failed to launch llama-server.exe: {e}"))?;

        let mut server = LlamaServer {
            child,
            port,
            log_path: log_path.to_path_buf(),
        };
        server.wait_until_healthy()?;
        Ok(server)
    }

    fn wait_until_healthy(&mut self) -> Result<(), String> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!(
                    "the local model server exited during startup ({status}). {}",
                    self.log_tail()
                ));
            }
            if let Ok(response) = http::request(
                self.port,
                "GET",
                "/health",
                None,
                Duration::from_secs(5),
            ) {
                if response.is_success() {
                    return Ok(());
                }
            }
            if started.elapsed() > STARTUP_TIMEOUT {
                return Err(format!(
                    "the local model server did not become ready within {}s. {}",
                    STARTUP_TIMEOUT.as_secs(),
                    self.log_tail()
                ));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Peak working set of the inference process, in MB — the §4 memory
    /// budget's only real measurement, as opposed to the pre-flight estimate.
    pub fn peak_ram_mb(&self) -> Option<u64> {
        crate::system::peak_working_set_mb(&self.child)
    }

    /// Last few lines of the server log, for error messages. Kept short: the
    /// user needs the cause, not a dump.
    pub fn log_tail(&self) -> String {
        match std::fs::read_to_string(&self.log_path) {
            Ok(text) => {
                let tail: Vec<&str> = text.lines().rev().take(6).collect();
                let tail: Vec<&str> = tail.into_iter().rev().collect();
                format!("Server log: {}", tail.join(" | "))
            }
            Err(_) => format!("See {}.", self.log_path.display()),
        }
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LlamaServer {
    /// The process must not outlive the run under any exit path — including a
    /// pipeline error or a panic mid-generation.
    fn drop(&mut self) {
        self.stop();
    }
}
