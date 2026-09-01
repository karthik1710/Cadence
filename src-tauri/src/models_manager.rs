//! Phase 13 Model Manager (master prompt §27): the catalog of installable
//! native runtimes and model weights, plus install / remove / verify.
//!
//! Two rules shape this module:
//!
//! * **Model files are never bundled into the installer** (§27/§35). They are
//!   downloaded once, into `<app_data>/models`, and the app runs fully offline
//!   afterwards (§28). Downloading happens only here, only when the user asks
//!   for it — never during recording, processing, or MoM generation.
//! * **Nothing links an HTTP client.** Downloads shell out to `curl.exe` and
//!   extraction to `tar.exe`, both shipped with Windows 10 1803+ and Windows
//!   11. That keeps the crate's dependency graph free of network stacks, which
//!   is the cheap structural check that no cloud path has crept back in.
//!
//! Integrity: the catalog carries no hard-coded digests, because inventing
//! digests that were never verified against the real files would be worse than
//! having none. Instead this is trust-on-first-install — the SHA-256 of every
//! installed file is computed at install time and recorded in
//! `models/manifest.json`, and [`verify`] recomputes it. That detects
//! corruption, truncation, and post-install tampering, but not a compromised
//! first download. See docs/privacy.md.

use crate::system::MemoryProfile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    /// A native binary + its DLLs (whisper.cpp, llama.cpp, sherpa-onnx).
    Runtime,
    Stt,
    Llm,
    Diarization,
    /// Optional speech enhancement (§11 audio preprocessing).
    Denoise,
    /// Optional voice activity detection (§11).
    Vad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archive {
    /// `.zip` and `.tar.bz2` alike — Windows' bundled bsdtar reads both.
    Tar,
}

/// One installable item. URLs are all that ever need editing if an upstream
/// release moves; nothing else in the app hard-codes a download location.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub kind: ModelKind,
    pub label: &'static str,
    /// Directory under `<app_data>/models` this installs into.
    pub subdir: &'static str,
    pub url: &'static str,
    /// Name to save the download as.
    pub download_name: &'static str,
    pub archive: Option<Archive>,
    /// Path, relative to `subdir`, that must exist once install succeeds.
    pub expect_relative: &'static str,
    /// Rough on-disk size, for the pre-flight disk/RAM check and the UI. Not
    /// an integrity check — the real size is recorded at install time.
    pub approx_size_mb: u64,
    pub notes: &'static str,
}

impl CatalogEntry {
    /// Where this entry's primary file lands.
    pub fn path_in(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(self.subdir).join(self.expect_relative)
    }

    pub fn is_installed(&self, models_dir: &Path) -> bool {
        self.path_in(models_dir).exists()
    }

    /// Estimated peak process RAM for an LLM entry at `context_tokens`
    /// (§4/§30 "memory usage"). Weights are memory-mapped but resident in
    /// practice, and the KV cache is budgeted at ~0.15 MB per token, which is
    /// the right order for the Qwen3 family's grouped-query attention. This is
    /// a pre-flight sanity check, not a measurement — real numbers come from
    /// `bench_cli`.
    pub fn estimated_llm_ram_mb(&self, context_tokens: usize) -> u64 {
        let kv_mb = (context_tokens as f64 * 0.15).round() as u64;
        // ~600 MB of llama.cpp runtime, compute buffers and process overhead.
        self.approx_size_mb + kv_mb + 600
    }
}

/// Every runtime and model Cadence knows how to install.
pub fn catalog() -> &'static [CatalogEntry] {
    &[
        CatalogEntry {
            id: "runtime-whisper",
            kind: ModelKind::Runtime,
            label: "whisper.cpp (speech-to-text runtime)",
            subdir: "bin",
            url: "https://github.com/ggml-org/whisper.cpp/releases/download/b4938/whisper-bin-x64.zip",
            download_name: "whisper-bin-x64.zip",
            archive: Some(Archive::Tar),
            expect_relative: "whisper-cli.exe",
            approx_size_mb: 30,
            notes: "CPU build. Required for transcription.",
        },
        CatalogEntry {
            id: "runtime-llama",
            kind: ModelKind::Runtime,
            label: "llama.cpp (meeting-intelligence runtime)",
            subdir: "bin",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10717/llama-b10717-bin-win-cpu-x64.zip",
            download_name: "llama-bin-win-cpu-x64.zip",
            archive: Some(Archive::Tar),
            expect_relative: "llama-server.exe",
            approx_size_mb: 40,
            notes: "CPU build. Required for minutes-of-meeting generation.",
        },
        CatalogEntry {
            id: "runtime-sherpa",
            kind: ModelKind::Runtime,
            label: "sherpa-onnx (speaker-diarization runtime)",
            subdir: "bin",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2",
            download_name: "sherpa-onnx-win-x64.tar.bz2",
            archive: Some(Archive::Tar),
            expect_relative: "sherpa-onnx-offline-speaker-diarization.exe",
            approx_size_mb: 60,
            notes: "CPU build. Required for speaker detection.",
        },
        CatalogEntry {
            id: "whisper-base",
            kind: ModelKind::Stt,
            label: "Whisper base (multilingual)",
            subdir: "stt",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            download_name: "ggml-base.bin",
            archive: None,
            expect_relative: "ggml-base.bin",
            approx_size_mb: 148,
            notes: "Fastest, least accurate. Default on machines under 12 GB RAM.",
        },
        CatalogEntry {
            id: "whisper-small",
            kind: ModelKind::Stt,
            label: "Whisper small (multilingual)",
            subdir: "stt",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            download_name: "ggml-small.bin",
            archive: None,
            expect_relative: "ggml-small.bin",
            approx_size_mb: 488,
            notes: "Default. Balance of accuracy and speed.",
        },
        CatalogEntry {
            id: "whisper-medium",
            kind: ModelKind::Stt,
            label: "Whisper medium (multilingual)",
            subdir: "stt",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            download_name: "ggml-medium.bin",
            archive: None,
            expect_relative: "ggml-medium.bin",
            approx_size_mb: 1530,
            notes: "Most accurate candidate; roughly 3x slower than small. Benchmark before adopting (§6).",
        },
        CatalogEntry {
            id: "qwen3-4b-q4",
            kind: ModelKind::Llm,
            label: "Qwen3 4B (Q4_K_M)",
            subdir: "llm",
            url: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
            download_name: "Qwen3-4B-Q4_K_M.gguf",
            archive: None,
            expect_relative: "Qwen3-4B-Q4_K_M.gguf",
            approx_size_mb: 2500,
            notes: "Default on 16 GB machines. Multilingual, ~3.5 GB peak at 4K context.",
        },
        CatalogEntry {
            id: "qwen3-8b-q4",
            kind: ModelKind::Llm,
            label: "Qwen3 8B (Q4_K_M)",
            subdir: "llm",
            url: "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            download_name: "Qwen3-8B-Q4_K_M.gguf",
            archive: None,
            expect_relative: "Qwen3-8B-Q4_K_M.gguf",
            approx_size_mb: 5030,
            notes: "Default on 32 GB machines. Quality candidate (§5).",
        },
        CatalogEntry {
            id: "gemma3-4b-q4",
            kind: ModelKind::Llm,
            label: "Gemma 3 4B Instruct (Q4_K_M)",
            subdir: "llm",
            url: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
            download_name: "gemma-3-4b-it-Q4_K_M.gguf",
            archive: None,
            expect_relative: "gemma-3-4b-it-Q4_K_M.gguf",
            approx_size_mb: 2490,
            notes: "Multilingual comparison candidate (§5). Optional — install only to benchmark.",
        },
        CatalogEntry {
            id: "diarization-segmentation",
            kind: ModelKind::Diarization,
            label: "pyannote segmentation 3.0 (ONNX)",
            subdir: "diarization",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            download_name: "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            archive: Some(Archive::Tar),
            expect_relative: "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
            approx_size_mb: 6,
            notes: "Required for speaker detection.",
        },
        CatalogEntry {
            id: "diarization-embedding",
            kind: ModelKind::Diarization,
            label: "WeSpeaker CAM++ speaker embeddings (ONNX)",
            subdir: "diarization",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
            download_name: "wespeaker_en_voxceleb_CAM++.onnx",
            archive: None,
            expect_relative: "wespeaker_en_voxceleb_CAM++.onnx",
            approx_size_mb: 28,
            notes: "Required for speaker detection.",
        },
        // The two entries below are optional on purpose: they improve audio
        // quality and processing time, and the app must still transcribe a
        // meeting end to end without either of them (§33).
        CatalogEntry {
            id: "denoise-gtcrn",
            kind: ModelKind::Denoise,
            label: "GTCRN speech denoiser (ONNX)",
            subdir: "denoise",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
            download_name: "gtcrn_simple.onnx",
            archive: None,
            expect_relative: "gtcrn_simple.onnx",
            approx_size_mb: 1,
            notes: "Optional. Suppresses room noise (fans, keyboards, corridor noise) before transcription. Runs on the sherpa-onnx runtime you already have.",
        },
        CatalogEntry {
            id: "vad-silero",
            kind: ModelKind::Vad,
            label: "Silero voice activity detection (ONNX)",
            subdir: "vad",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
            download_name: "silero_vad.onnx",
            archive: None,
            expect_relative: "silero_vad.onnx",
            approx_size_mb: 1,
            notes: "Optional. Measures how much of a recording is speech, and can skip silence to shorten processing time.",
        },
    ]
}

pub fn entry(id: &str) -> Option<&'static CatalogEntry> {
    catalog().iter().find(|e| e.id == id)
}

/// The ids a machine on `profile` needs before it can process a meeting
/// end to end. Used by the first-run setup screen and `check_models`.
pub fn required_ids(profile: MemoryProfile) -> Vec<&'static str> {
    vec![
        "runtime-whisper",
        "runtime-llama",
        "runtime-sherpa",
        profile.default_stt_model(),
        profile.default_llm_model(),
        "diarization-segmentation",
        "diarization-embedding",
    ]
}

// ---------------------------------------------------------------- manifest

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledRecord {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub installed_at_ms: i64,
    pub source_url: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub entries: BTreeMap<String, InstalledRecord>,
}

fn manifest_path(models_dir: &Path) -> PathBuf {
    models_dir.join("manifest.json")
}

pub fn load_manifest(models_dir: &Path) -> Manifest {
    std::fs::read_to_string(manifest_path(models_dir))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

fn save_manifest(models_dir: &Path, manifest: &Manifest) -> Result<(), String> {
    std::fs::create_dir_all(models_dir).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
    std::fs::write(manifest_path(models_dir), json).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ status

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    pub label: String,
    pub kind: ModelKind,
    pub installed: bool,
    pub path: String,
    pub approx_size_mb: u64,
    pub actual_size_mb: u64,
    pub notes: String,
    /// True when this id is one the current machine needs (§27 health check).
    pub required: bool,
    /// SHA-256 recorded at install time, if any.
    pub recorded_sha256: Option<String>,
}

pub fn status_all(models_dir: &Path, profile: MemoryProfile) -> Vec<ModelStatus> {
    let manifest = load_manifest(models_dir);
    let required = required_ids(profile);
    catalog()
        .iter()
        .map(|e| {
            let path = e.path_in(models_dir);
            let actual = std::fs::metadata(&path)
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0);
            ModelStatus {
                id: e.id.to_string(),
                label: e.label.to_string(),
                kind: e.kind,
                installed: path.exists(),
                path: path.to_string_lossy().to_string(),
                approx_size_mb: e.approx_size_mb,
                actual_size_mb: actual,
                notes: e.notes.to_string(),
                required: required.contains(&e.id),
                recorded_sha256: manifest.entries.get(e.id).map(|r| r.sha256.clone()),
            }
        })
        .collect()
}

// ----------------------------------------------------------------- install

/// Live progress for one install, polled by the Settings screen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallProgress {
    pub id: String,
    /// "downloading" | "extracting" | "verifying" | "done" | "failed"
    pub phase: String,
    pub percent: i64,
    pub downloaded_mb: u64,
    pub total_mb: u64,
    pub error: Option<String>,
}

impl InstallProgress {
    pub fn new(id: &str, phase: &str, percent: i64) -> Self {
        Self {
            id: id.to_string(),
            phase: phase.to_string(),
            percent,
            downloaded_mb: 0,
            total_mb: 0,
            error: None,
        }
    }
}

/// Downloads and installs one catalog entry. `on_progress` is called from this
/// thread; callers run it off the UI thread.
///
/// The temp directory is removed on every exit path, success or failure — a
/// model whose download or extraction fails must not leave a partial,
/// multi-hundred-megabyte archive behind under `models/_tmp`.
pub fn install(
    models_dir: &Path,
    id: &str,
    on_progress: &dyn Fn(InstallProgress),
) -> Result<(), String> {
    let entry = entry(id).ok_or_else(|| format!("unknown model id: {id}"))?;
    let tmp_dir = models_dir.join("_tmp").join(entry.id);
    let result = install_inner(models_dir, entry, &tmp_dir, on_progress);
    let _ = std::fs::remove_dir_all(&tmp_dir);
    result
}

fn install_inner(
    models_dir: &Path,
    entry: &CatalogEntry,
    tmp_dir: &Path,
    on_progress: &dyn Fn(InstallProgress),
) -> Result<(), String> {
    let dest_dir = models_dir.join(entry.subdir);
    std::fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(tmp_dir).map_err(|e| e.to_string())?;

    let download_path = tmp_dir.join(entry.download_name);
    on_progress(InstallProgress::new(entry.id, "downloading", 0));
    download(entry, &download_path, on_progress)?;

    if entry.archive.is_some() {
        on_progress(InstallProgress::new(entry.id, "extracting", 90));
        let extract_dir = tmp_dir.join("extract");
        let _ = std::fs::remove_dir_all(&extract_dir);
        std::fs::create_dir_all(&extract_dir).map_err(|e| e.to_string())?;
        extract(&download_path, &extract_dir)?;
        match entry.kind {
            // Runtimes are flattened: the app calls `<models>/bin/<tool>.exe`
            // and the DLLs must sit beside it.
            ModelKind::Runtime => copy_binaries(&extract_dir, &dest_dir)?,
            _ => copy_tree(&extract_dir, &dest_dir)?,
        }
    } else {
        let final_path = dest_dir.join(entry.expect_relative);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        move_file(&download_path, &final_path)?;
    }

    let installed = entry.path_in(models_dir);
    if !installed.exists() {
        return Err(format!(
            "install finished but {} is missing — the download may be for a different release",
            installed.display()
        ));
    }

    on_progress(InstallProgress::new(entry.id, "verifying", 95));
    let bytes = std::fs::metadata(&installed).map_err(|e| e.to_string())?.len();
    let digest = sha256_file(&installed)?;
    let mut manifest = load_manifest(models_dir);
    manifest.entries.insert(
        entry.id.to_string(),
        InstalledRecord {
            file: installed.to_string_lossy().to_string(),
            bytes,
            sha256: digest,
            installed_at_ms: now_ms(),
            source_url: entry.url.to_string(),
        },
    );
    save_manifest(models_dir, &manifest)?;

    on_progress(InstallProgress::new(entry.id, "done", 100));
    Ok(())
}

/// Shells out to Windows' bundled `curl.exe`. Progress comes from watching the
/// output file grow rather than from parsing curl's progress meter, which is
/// terminal-formatted and not meant to be machine-read.
fn download(
    entry: &CatalogEntry,
    dest: &Path,
    on_progress: &dyn Fn(InstallProgress),
) -> Result<(), String> {
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    let mut child = Command::new("curl.exe")
        .arg("-L")
        .arg("--fail")
        .arg("--silent")
        .arg("--show-error")
        .arg("-o")
        .arg(dest)
        .arg(entry.url)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            format!("could not launch curl.exe (needed to download models): {e}")
        })?;

    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => {
                if status.success() {
                    return Ok(());
                }
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                let _ = std::fs::remove_file(dest);
                return Err(format!(
                    "download of {} failed ({}). {}",
                    entry.label,
                    status,
                    stderr.trim()
                ));
            }
            None => {
                let downloaded_mb = std::fs::metadata(dest)
                    .map(|m| m.len() / (1024 * 1024))
                    .unwrap_or(0);
                let percent = if entry.approx_size_mb > 0 {
                    ((downloaded_mb * 90) / entry.approx_size_mb.max(1)).min(90) as i64
                } else {
                    0
                };
                on_progress(InstallProgress {
                    id: entry.id.to_string(),
                    phase: "downloading".to_string(),
                    percent,
                    downloaded_mb,
                    total_mb: entry.approx_size_mb,
                    error: None,
                });
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
}

/// Windows' bundled bsdtar reads `.zip` and `.tar.bz2` alike, so one call
/// covers every archive in the catalog.
fn extract(archive: &Path, dest: &Path) -> Result<(), String> {
    let output = Command::new("tar.exe")
        .arg("-xf")
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .output()
        .map_err(|e| format!("could not launch tar.exe (needed to unpack models): {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "unpacking {} failed: {}",
            archive.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

/// Flattens every `.exe` / `.dll` in a release archive into `<models>/bin`.
fn copy_binaries(from: &Path, to: &Path) -> Result<(), String> {
    let mut copied = 0usize;
    visit_files(from, &mut |path| {
        let is_binary = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("exe") || e.eq_ignore_ascii_case("dll"))
            .unwrap_or(false);
        if !is_binary {
            return Ok(());
        }
        let name = path.file_name().ok_or("archive entry has no file name")?;
        std::fs::copy(path, to.join(name)).map_err(|e| e.to_string())?;
        copied += 1;
        Ok(())
    })?;
    if copied == 0 {
        return Err("archive contained no .exe/.dll files".to_string());
    }
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(from).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn visit_files(dir: &Path, f: &mut dyn FnMut(&Path) -> Result<(), String>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            visit_files(&path, f)?;
        } else {
            f(&path)?;
        }
    }
    Ok(())
}

/// `rename` first — a same-volume move is instant — falling back to copy when
/// the temp directory and the models directory are on different volumes.
fn move_file(from: &Path, to: &Path) -> Result<(), String> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(from);
    Ok(())
}

/// Removes an installed model. Refuses to remove something the current
/// machine's profile still needs, so a user cannot leave the app unable to
/// process a meeting by accident.
pub fn remove(models_dir: &Path, id: &str, profile: MemoryProfile) -> Result<(), String> {
    let entry = entry(id).ok_or_else(|| format!("unknown model id: {id}"))?;
    if required_ids(profile).contains(&entry.id) {
        return Err(format!(
            "{} is in use on this machine. Select a different model in Settings first.",
            entry.label
        ));
    }
    let path = entry.path_in(models_dir);
    if path.is_dir() {
        std::fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
    } else if path.exists() {
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    let mut manifest = load_manifest(models_dir);
    manifest.entries.remove(entry.id);
    save_manifest(models_dir, &manifest)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub id: String,
    pub ok: bool,
    pub message: String,
    pub sha256: Option<String>,
}

/// §27's "model health check": the file is present, non-empty, and still
/// hashes to what it hashed to at install time.
pub fn verify(models_dir: &Path, id: &str) -> Result<VerifyReport, String> {
    let entry = entry(id).ok_or_else(|| format!("unknown model id: {id}"))?;
    let path = entry.path_in(models_dir);
    if !path.exists() {
        return Ok(VerifyReport {
            id: id.to_string(),
            ok: false,
            message: "Not installed.".to_string(),
            sha256: None,
        });
    }
    let bytes = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
    if bytes == 0 {
        return Ok(VerifyReport {
            id: id.to_string(),
            ok: false,
            message: "File is empty — reinstall.".to_string(),
            sha256: None,
        });
    }
    let digest = sha256_file(&path)?;
    let recorded = load_manifest(models_dir)
        .entries
        .get(id)
        .map(|r| r.sha256.clone());

    let (ok, message) = match recorded {
        Some(expected) if expected == digest => (
            true,
            format!("OK — {} MB, digest unchanged since install.", bytes / (1024 * 1024)),
        ),
        Some(_) => (
            false,
            "File has changed since it was installed. Reinstall it.".to_string(),
        ),
        None => (
            true,
            format!(
                "Present ({} MB) but installed outside the app, so there is no recorded digest to compare against.",
                bytes / (1024 * 1024)
            ),
        ),
    };

    Ok(VerifyReport {
        id: id.to_string(),
        ok,
        message,
        sha256: Some(digest),
    })
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let read = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique() {
        let mut ids: Vec<&str> = catalog().iter().map(|e| e.id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }

    #[test]
    fn every_profile_default_exists_in_the_catalog() {
        for p in [
            MemoryProfile::Low,
            MemoryProfile::Balanced,
            MemoryProfile::High,
        ] {
            assert!(entry(p.default_llm_model()).is_some(), "{p:?} llm");
            assert!(entry(p.default_stt_model()).is_some(), "{p:?} stt");
            for id in required_ids(p) {
                assert!(entry(id).is_some(), "{id} missing from catalog");
            }
        }
    }

    #[test]
    fn the_16gb_default_llm_fits_its_budget() {
        let profile = MemoryProfile::Balanced;
        let llm = entry(profile.default_llm_model()).unwrap();
        let estimated = llm.estimated_llm_ram_mb(profile.default_llm_context());
        assert!(
            estimated <= profile.llm_ram_budget_mb(),
            "estimated {estimated} MB exceeds the {} MB budget for a 16 GB machine",
            profile.llm_ram_budget_mb()
        );
    }

    #[test]
    fn the_32gb_default_llm_fits_its_budget() {
        let profile = MemoryProfile::High;
        let llm = entry(profile.default_llm_model()).unwrap();
        assert!(llm.estimated_llm_ram_mb(profile.default_llm_context()) <= profile.llm_ram_budget_mb());
    }

    #[test]
    fn the_8b_model_is_rejected_on_a_16gb_machine_budget() {
        // The guard that makes the tiering meaningful: if this ever passes,
        // the Balanced budget has drifted up and 16 GB users will swap.
        let llm = entry("qwen3-8b-q4").unwrap();
        let profile = MemoryProfile::Balanced;
        assert!(llm.estimated_llm_ram_mb(profile.default_llm_context()) > profile.llm_ram_budget_mb());
    }

    #[test]
    fn the_audio_quality_models_stay_optional() {
        // If either of these ever lands in required_ids, a machine without an
        // internet connection to fetch them can no longer process a meeting.
        for id in ["denoise-gtcrn", "vad-silero"] {
            assert!(entry(id).is_some(), "{id} missing from catalog");
            for p in [
                MemoryProfile::Low,
                MemoryProfile::Balanced,
                MemoryProfile::High,
            ] {
                assert!(!required_ids(p).contains(&id), "{id} must stay optional on {p:?}");
            }
        }
    }

    #[test]
    fn an_optional_audio_model_can_be_removed_again() {
        let dir = std::env::temp_dir().join("cadence-remove-optional");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("vad")).unwrap();
        let path = entry("vad-silero").unwrap().path_in(&dir);
        std::fs::write(&path, b"stub").unwrap();
        remove(&dir, "vad-silero", MemoryProfile::Balanced).unwrap();
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hashes_a_known_value() {
        let dir = std::env::temp_dir().join("cadence-sha-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.txt");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&path);
    }
}
