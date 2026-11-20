//! Phase 13 Model Manager: the catalog of installable
//! native runtimes and model weights, plus install / remove / verify.
//!
//! Three rules shape this module:
//!
//! * **Model weights are never bundled into the installer**. They are
//!   downloaded once, into `<app_data>/models`, and the app runs fully offline
//!   afterwards. Downloading happens only here, only when the user asks
//!   for it — never during recording, processing, or MoM generation.
//! * **A release build never downloads an executable.** Its installer ships
//!   the three native runtimes, signed, in `<exe dir>/runtimes/<runtime>/`
//!   (`scripts/prepare-runtimes.ps1` fills them from pinned archives). A
//!   bundled runtime counts as installed, is never offered for download, and
//!   wins over any copy in `<app_data>/models/bin`. The runtime download path
//!   below survives for development builds that ship none.
//! * **Nothing links an HTTP client.** Downloads shell out to `curl.exe` and
//!   extraction to `tar.exe`, both shipped with Windows 10 1803+ and Windows
//!   11. That keeps the crate's dependency graph free of network stacks, which
//!   is the cheap structural check that no cloud path has crept back in.
//!
//! Integrity: the catalog carries no hard-coded digests for model data,
//! because inventing digests that were never verified against the real files
//! would be worse than having none. Downloads are trust-on-first-install — the
//! SHA-256 of every installed file (for a runtime, every file in its own
//! folder: the launcher and each DLL beside it) is computed at install time
//! and recorded in `models/manifest.json`, and [`verify`] recomputes it. That
//! detects corruption, truncation, and post-install tampering, but not a
//! compromised first download. A bundled runtime is checked instead against
//! the `runtimes/manifest.json` written at build time, after its archive was
//! checked against a pinned SHA-256. See docs/architecture.md.

use crate::system;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    /// A native binary + its DLLs (whisper.cpp, llama.cpp, sherpa-onnx).
    Runtime,
    Stt,
    Llm,
    Diarization,
    /// Speech enhancement (audio preprocessing).
    Denoise,
    /// Voice activity detection.
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
    /// Directory under `<app_data>/models` this installs into, `/`-separated.
    /// Each runtime has its own (`bin/whisper`, …): see [`Runtime`].
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

/// Joins a `/`-separated relative path one component at a time, so a path
/// shown to the user does not mix separators.
fn join_relative(base: &Path, relative: &str) -> PathBuf {
    relative
        .split('/')
        .filter(|part| !part.is_empty())
        .fold(base.to_path_buf(), |path, part| path.join(part))
}

impl CatalogEntry {
    /// The folder this entry installs into.
    pub fn dir_in(&self, models_dir: &Path) -> PathBuf {
        join_relative(models_dir, self.subdir)
    }

    /// Where an install puts this entry's primary file — always the current
    /// layout, whatever an earlier build left behind.
    pub fn own_path_in(&self, models_dir: &Path) -> PathBuf {
        join_relative(&self.dir_in(models_dir), self.expect_relative)
    }

    /// Where builds before the per-runtime folders put a runtime's launcher:
    /// straight into the shared `<models>/bin`. `None` for anything that is
    /// not a runtime.
    fn legacy_path_in(&self, models_dir: &Path) -> Option<PathBuf> {
        (self.kind == ModelKind::Runtime)
            .then(|| join_relative(&models_dir.join(RUNTIMES_DIR), self.expect_relative))
    }

    /// The runtime folder the installer shipped this entry in, when it is one
    /// of the catalog's runtimes and that folder holds its launcher.
    fn bundled_dir(&self, roots: Roots) -> Option<PathBuf> {
        Runtime::from_id(self.id)?.bundled_dir(roots.bundled)
    }

    /// True for a runtime still installed the old way — flattened into the
    /// shared `<models>/bin`, where whisper.cpp and llama.cpp overwrite each
    /// other's same-named `ggml*.dll` — and neither bundled nor in its own
    /// folder. It keeps working from there; a reinstall moves it to its own
    /// folder.
    pub fn uses_legacy_layout(&self, models_dir: &Path) -> bool {
        self.uses_legacy_layout_at(Roots::with_bundled(models_dir))
    }

    fn uses_legacy_layout_at(&self, roots: Roots) -> bool {
        self.bundled_dir(roots).is_none()
            && !self.own_path_in(roots.models).exists()
            && self
                .legacy_path_in(roots.models)
                .map(|legacy| legacy.exists())
                .unwrap_or(false)
    }

    /// Where this entry's primary file is. For a runtime, in order: the copy
    /// the installer shipped, its own folder under `<models>/bin`, or — for one
    /// an earlier build installed and nobody has reinstalled since — the
    /// shared `<models>/bin` (see [`Self::uses_legacy_layout`]).
    pub fn path_in(&self, models_dir: &Path) -> PathBuf {
        self.path_at(Roots::with_bundled(models_dir))
    }

    fn path_at(&self, roots: Roots) -> PathBuf {
        if let Some(dir) = self.bundled_dir(roots) {
            return join_relative(&dir, self.expect_relative);
        }
        match self.legacy_path_in(roots.models) {
            Some(legacy) if self.uses_legacy_layout_at(roots) => legacy,
            _ => self.own_path_in(roots.models),
        }
    }

    pub fn is_installed(&self, models_dir: &Path) -> bool {
        self.is_installed_at(Roots::with_bundled(models_dir))
    }

    fn is_installed_at(&self, roots: Roots) -> bool {
        self.path_at(roots).exists()
    }

    /// Estimated peak process RAM for an LLM entry at `context_tokens`
    /// ("memory usage"). Weights are memory-mapped but resident in
    /// practice. The KV cache is budgeted at ~0.08 MB per token: Qwen3-8B has
    /// 36 layers of 8 KV heads x 128 dims, ~0.14 MB per token at f16, and
    /// `llm::server` runs it at q8_0 (8.5 bits per value), which roughly
    /// halves that. On the pinned llama.cpp build, a same-shape Qwen3 model at
    /// 8K context measured ~540 MB less working set at q8_0 than at f16 — the
    /// size this arithmetic predicts. Still a pre-flight sanity check, not a
    /// measurement — real numbers come from `bench_cli`.
    pub fn estimated_llm_ram_mb(&self, context_tokens: usize) -> u64 {
        let kv_mb = (context_tokens as f64 * 0.08).round() as u64;
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
            subdir: "bin/whisper",
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
            subdir: "bin/llama",
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
            subdir: "bin/sherpa",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2",
            download_name: "sherpa-onnx-win-x64.tar.bz2",
            archive: Some(Archive::Tar),
            expect_relative: "sherpa-onnx-offline-speaker-diarization.exe",
            approx_size_mb: 60,
            notes: "CPU build. Required for speaker detection.",
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
            notes: "Required. Multilingual speech-to-text, including Tamil and code-switched speech.",
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
            // 5,027,783,488 bytes.
            approx_size_mb: 4795,
            notes: "Required. Writes the minutes of meeting. Multilingual; about 6 GB of RAM while it runs.",
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
        // The two entries below are about a megabyte each and part of the
        // installed set, because they are the cheapest voice/noise gain there
        // is. The pipeline still skips each gracefully when it is absent — a
        // deleted or corrupted file must never cost a meeting.
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
            notes: "Required. Suppresses room noise (fans, keyboards, corridor noise) before transcription. Runs on the sherpa-onnx runtime.",
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
            notes: "Required. Finds where people are speaking, so transcription blocks break in silence rather than mid-word.",
        },
    ]
}

pub fn entry(id: &str) -> Option<&'static CatalogEntry> {
    catalog().iter().find(|e| e.id == id)
}

// ---------------------------------------------------------------- runtimes

/// The folder under `<models>` that holds the runtimes, one subfolder each.
pub const RUNTIMES_DIR: &str = "bin";

/// The three native runtimes. Each lives in its own folder under
/// `<models>/bin` (`whisper`, `llama`, `sherpa`) because whisper.cpp and
/// llama.cpp both ship `ggml.dll`, `ggml-base.dll` and `ggml-cpu-*.dll`, built
/// from different ggml versions. Windows resolves an executable's DLLs from
/// its own folder first, so separate folders are what keep each tool on the
/// DLLs it was built against. Builds before this flattened all three into
/// `<models>/bin` itself, where whichever installed last won; that layout
/// still resolves (see [`runtime_dir`]) so nobody is forced to reinstall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    /// whisper.cpp: `whisper-cli.exe`, `whisper-server.exe`.
    Whisper,
    /// llama.cpp: `llama-server.exe`.
    Llama,
    /// sherpa-onnx: diarization, the GTCRN denoiser and the VAD.
    Sherpa,
}

impl Runtime {
    pub const ALL: [Runtime; 3] = [Runtime::Whisper, Runtime::Llama, Runtime::Sherpa];

    pub fn id(self) -> &'static str {
        match self {
            Runtime::Whisper => "runtime-whisper",
            Runtime::Llama => "runtime-llama",
            Runtime::Sherpa => "runtime-sherpa",
        }
    }

    /// The runtime's catalog entry. Every runtime is in the static catalog
    /// (`every_runtime_has_its_own_folder_in_the_catalog` pins it).
    pub fn entry(self) -> &'static CatalogEntry {
        entry(self.id()).expect("every runtime is in the catalog")
    }

    pub fn from_id(id: &str) -> Option<Runtime> {
        Runtime::ALL.into_iter().find(|runtime| runtime.id() == id)
    }

    /// The runtime's folder name — the last part of its `<models>/bin/…`
    /// folder, and its folder under the bundled `runtimes/` and key in the
    /// bundled `manifest.json` (`scripts/prepare-runtimes.ps1` uses the same
    /// names).
    pub fn folder(self) -> &'static str {
        match self {
            Runtime::Whisper => "whisper",
            Runtime::Llama => "llama",
            Runtime::Sherpa => "sherpa",
        }
    }

    /// The folder the installer shipped this runtime in, under
    /// `bundled_root`, when its launcher is there. Decided on the launcher,
    /// like the legacy fallback, so a runtime's executables and DLLs always
    /// come from one folder together.
    fn bundled_dir(self, bundled_root: Option<&Path>) -> Option<PathBuf> {
        let dir = bundled_root?.join(self.folder());
        dir.join(self.entry().expect_relative).is_file().then_some(dir)
    }
}

// --------------------------------------------------------- bundled runtimes

/// The folder, beside the app executable, that a release installer puts the
/// runtimes in: `runtimes/<whisper|llama|sherpa>/` plus `manifest.json`.
/// Tauri installs bundle resources next to the executable on Windows, and
/// tauri-build copies them next to `target/<profile>/app.exe` in a dev build,
/// so `pipeline_cli` and `bench_cli` — built into the same folder — resolve
/// the same copy.
pub const BUNDLED_RUNTIMES_DIR: &str = "runtimes";

/// Written by `scripts/prepare-runtimes.ps1` after signing: per runtime
/// folder, `{ "version": …, "files": { "<name>": "<sha256>" } }`.
const BUNDLED_MANIFEST: &str = "manifest.json";

static BUNDLED_ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Where bundled runtimes would be for this process: `<exe dir>/runtimes`.
/// Whether a given runtime is actually there is checked on each use, on its
/// launcher. `None` under `cargo test`, so unit tests never depend on what
/// happens to sit next to the test binary; they inject a root through
/// [`Roots`] instead.
pub fn bundled_runtimes_root() -> Option<&'static Path> {
    BUNDLED_ROOT.get_or_init(locate_bundled_root).as_deref()
}

#[cfg(not(test))]
fn locate_bundled_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join(BUNDLED_RUNTIMES_DIR))
}

#[cfg(test)]
fn locate_bundled_root() -> Option<PathBuf> {
    None
}

/// The two places a runtime can come from: the models folder under app data
/// and, optionally, the bundled `runtimes/` folder. Every public function
/// that takes a `models_dir` uses the process's [`bundled_runtimes_root`];
/// tests build a `Roots` with a temporary one.
#[derive(Debug, Clone, Copy)]
struct Roots<'a> {
    models: &'a Path,
    bundled: Option<&'a Path>,
}

impl<'a> Roots<'a> {
    fn with_bundled(models: &'a Path) -> Self {
        Self { models, bundled: bundled_runtimes_root() }
    }
}

/// The folder a runtime's executables run from, in order of precedence: the
/// copy the installer shipped (`<exe dir>/runtimes/<runtime>`), its own
/// folder under app data (`<models>/bin/<runtime>`), or — if it is still
/// installed the old, flat way and has not been reinstalled since —
/// `<models>/bin`.
///
/// The choice is made per runtime, on its launcher, never per file — so a
/// runtime's executables and DLLs always come from one folder together.
/// When no location has it, this is the own folder, so a "not installed
/// at …" message points where an install will put it.
pub fn runtime_dir(models_dir: &Path, runtime: Runtime) -> PathBuf {
    runtime_dir_at(Roots::with_bundled(models_dir), runtime)
}

fn runtime_dir_at(roots: Roots, runtime: Runtime) -> PathBuf {
    let entry = runtime.entry();
    if let Some(bundled) = runtime.bundled_dir(roots.bundled) {
        bundled
    } else if entry.uses_legacy_layout_at(roots) {
        roots.models.join(RUNTIMES_DIR)
    } else {
        entry.dir_in(roots.models)
    }
}

/// [`runtime_dir`] against an explicit bundled root rather than the process's
/// own — for `pipeline::paths`' tests.
pub fn runtime_dir_with_bundled(models_dir: &Path, bundled_root: Option<&Path>, runtime: Runtime) -> PathBuf {
    runtime_dir_at(Roots { models: models_dir, bundled: bundled_root }, runtime)
}

/// One executable of a runtime, resolved through [`runtime_dir`]. Everything
/// that launches a native tool goes through this.
pub fn runtime_binary(models_dir: &Path, runtime: Runtime, exe: &str) -> PathBuf {
    runtime_dir(models_dir, runtime).join(exe)
}

/// Why `id` must not be downloaded, if it must not: a runtime the installer
/// shipped is repaired by reinstalling Cadence, never by fetching an
/// executable into app data.
pub fn install_refusal(id: &str) -> Option<String> {
    install_refusal_with(bundled_runtimes_root(), id)
}

fn install_refusal_with(bundled_root: Option<&Path>, id: &str) -> Option<String> {
    let entry = entry(id)?;
    Runtime::from_id(entry.id)?.bundled_dir(bundled_root).map(|_| {
        format!(
            "{} is included with Cadence — reinstall Cadence to repair it.",
            entry.label
        )
    })
}

/// The one model set every machine installs at first-run setup, in install
/// order: runtimes first, then the small models, and the multi-gigabyte LLM
/// last, so a failure on the biggest download cannot hold up the rest.
pub fn required_ids() -> Vec<&'static str> {
    vec![
        "runtime-whisper",
        "runtime-sherpa",
        "runtime-llama",
        "diarization-segmentation",
        "diarization-embedding",
        "denoise-gtcrn",
        "vad-silero",
        system::STT_MODEL_ID,
        system::LLM_MODEL_ID,
    ]
}

/// Required ids not yet installed under `models_dir`, in install order. A
/// runtime the installer shipped is installed, so on a release build this is
/// model data only, and first-run setup downloads no executable.
pub fn missing_required_ids(models_dir: &Path) -> Vec<&'static str> {
    missing_required_ids_at(Roots::with_bundled(models_dir))
}

fn missing_required_ids_at(roots: Roots) -> Vec<&'static str> {
    required_ids()
        .into_iter()
        .filter(|id| entry(id).map(|e| !e.is_installed_at(roots)).unwrap_or(true))
        .collect()
}

/// Approximate total download, in MB, still needed to complete the required
/// set — the number the first-run consent screen shows before anything is
/// fetched.
pub fn required_download_mb(models_dir: &Path) -> u64 {
    required_download_mb_at(Roots::with_bundled(models_dir))
}

fn required_download_mb_at(roots: Roots) -> u64 {
    missing_required_ids_at(roots)
        .iter()
        .filter_map(|id| entry(id))
        .map(|e| e.approx_size_mb)
        .sum()
}

// ---------------------------------------------------------------- manifest

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledRecord {
    pub file: String,
    pub bytes: u64,
    /// SHA-256 of `file` — for a runtime, its launcher `.exe` only.
    pub sha256: String,
    pub installed_at_ms: i64,
    pub source_url: String,
    /// For a runtime installed into its own folder: every file in that folder
    /// (`/`-separated, relative to it) and its SHA-256 at install time, so
    /// [`verify`] checks the DLLs the launcher loads, not the launcher alone.
    /// Empty for everything else, and for runtimes installed before this was
    /// recorded; omitted from the JSON when empty, so those records read and
    /// write exactly as before.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub file_digests: BTreeMap<String, String>,
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

/// Writes the manifest, first pointing every record's informational `file`
/// path at the current models directory.
///
/// Builds from before the project rename recorded paths under
/// `%APPDATA%\com.echomind.app\…`, and nothing rewrote them, so anyone reading
/// the manifest to debug an install was sent to the wrong folder. Only `file`
/// changes, and only when it lies outside `models_dir`: lookups go by id and
/// SHA-256, so verification behaves exactly as before.
fn save_manifest(models_dir: &Path, manifest: &Manifest) -> Result<(), String> {
    std::fs::create_dir_all(models_dir).map_err(|e| e.to_string())?;
    let mut manifest = manifest.clone();
    refresh_recorded_paths(models_dir, &mut manifest);
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(manifest_path(models_dir), json).map_err(|e| e.to_string())
}

fn refresh_recorded_paths(models_dir: &Path, manifest: &mut Manifest) {
    for record in manifest.entries.values_mut() {
        if Path::new(&record.file).starts_with(models_dir) {
            continue;
        }
        if let Some(relative) = relative_in_models(models_dir, &record.file) {
            record.file = models_dir.join(relative).to_string_lossy().into_owned();
        }
    }
}

/// The part of a recorded path below a models directory — `llm/x.gguf` out of
/// `…\com.echomind.app\models\llm\x.gguf` — found by stripping `models_dir`
/// itself or, failing that, everything up to the last component named like
/// `models_dir`'s own folder. `None` unless what is left is one or more plain
/// names, so a `..` or a rooted tail can never come out of it.
///
/// Used only to rewrite informational paths and to attach a manifest id to a
/// file found on disk — never to decide what to delete.
fn relative_in_models(models_dir: &Path, recorded: &str) -> Option<PathBuf> {
    use std::path::Component;
    let recorded = Path::new(recorded);
    let tail: Vec<Component> = match recorded.strip_prefix(models_dir) {
        Ok(rest) => rest.components().collect(),
        Err(_) => {
            let anchor = models_dir.file_name()?;
            let components: Vec<Component> = recorded.components().collect();
            let at = components.iter().rposition(|component| {
                matches!(component, Component::Normal(name) if name.eq_ignore_ascii_case(anchor))
            })?;
            components[at + 1..].to_vec()
        }
    };
    if tail.is_empty() || !tail.iter().all(|c| matches!(c, Component::Normal(_))) {
        return None;
    }
    Some(tail.iter().collect())
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
    /// True when this id is part of the required set (health check).
    pub required: bool,
    /// SHA-256 recorded at install time, if any. For a bundled runtime, its
    /// launcher's digest in the bundled `manifest.json`.
    pub recorded_sha256: Option<String>,
    /// A runtime the installer shipped ("Included with Cadence"): never
    /// downloaded or reinstalled from the app, only verified.
    #[serde(default)]
    pub bundled: bool,
}

/// Whole megabytes, rounded up, so a non-empty file never reads as "0 MB".
fn mb_ceil(bytes: u64) -> u64 {
    bytes.div_ceil(1024 * 1024)
}

/// What an installed entry occupies on disk, in whole megabytes (rounded up).
///
/// A runtime is its launcher `.exe` plus the DLLs beside it, so one bundled
/// with the app or in its own folder is sized as that whole folder. One still
/// on the old flat layout falls back to the catalog estimate: its launcher
/// alone would understate it by orders of magnitude, and the shared `bin/`
/// folder holds every other runtime's files too.
fn actual_size_mb(roots: Roots, e: &CatalogEntry) -> u64 {
    let path = e.path_at(roots);
    let Ok(meta) = std::fs::metadata(&path) else {
        return 0;
    };
    if e.kind != ModelKind::Runtime {
        return mb_ceil(meta.len());
    }
    if let Some(bundled) = e.bundled_dir(roots) {
        return mb_ceil(size_on_disk(&bundled));
    }
    if e.uses_legacy_layout_at(roots) {
        return e.approx_size_mb;
    }
    mb_ceil(size_on_disk(&e.dir_in(roots.models)))
}

pub fn status_all(models_dir: &Path) -> Vec<ModelStatus> {
    status_all_at(Roots::with_bundled(models_dir))
}

fn status_all_at(roots: Roots) -> Vec<ModelStatus> {
    let manifest = load_manifest(roots.models);
    let bundled_manifest = roots
        .bundled
        .and_then(|root| load_bundled_manifest(root).ok().flatten());
    let required = required_ids();
    catalog()
        .iter()
        .map(|e| {
            let path = e.path_at(roots);
            let bundled = e.bundled_dir(roots).is_some();
            let recorded_sha256 = if bundled {
                Runtime::from_id(e.id).and_then(|runtime| {
                    bundled_manifest
                        .as_ref()?
                        .get(runtime.folder())?
                        .files
                        .get(e.expect_relative)
                        .cloned()
                })
            } else {
                manifest.entries.get(e.id).map(|r| r.sha256.clone())
            };
            ModelStatus {
                id: e.id.to_string(),
                label: e.label.to_string(),
                kind: e.kind,
                installed: path.exists(),
                path: path.to_string_lossy().to_string(),
                approx_size_mb: e.approx_size_mb,
                actual_size_mb: actual_size_mb(roots, e),
                notes: e.notes.to_string(),
                required: required.contains(&e.id),
                recorded_sha256,
                bundled,
            }
        })
        .collect()
}

// ----------------------------------------------------------------- install

/// Live progress for one install, polled by the Settings screen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallProgress {
    pub id: String,
    /// "queued" | "downloading" | "extracting" | "verifying" | "done" | "failed"
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
    install_with_bundled(models_dir, bundled_runtimes_root(), id, on_progress)
}

/// [`install`], refusing — before anything is downloaded — a runtime that
/// `bundled_root` holds (see [`install_refusal`]).
fn install_with_bundled(
    models_dir: &Path,
    bundled_root: Option<&Path>,
    id: &str,
    on_progress: &dyn Fn(InstallProgress),
) -> Result<(), String> {
    let entry = entry(id).ok_or_else(|| format!("unknown model id: {id}"))?;
    if let Some(refusal) = install_refusal_with(bundled_root, id) {
        return Err(refusal);
    }
    install_entry(models_dir, entry, on_progress)
}

/// Installs an entry that need not be in [`catalog`], with the same download,
/// extraction, temp-dir cleanup and manifest recording as [`install`].
///
/// Exists for `bench_cli setup`, whose comparison candidates are deliberately
/// kept out of the app's catalog (and so out of Settings, the setup screen and
/// [`required_ids`]). It installs them under a separate `models_dir` —
/// `<app_data>/models/bench` — so they get their own `manifest.json` and can
/// never be mistaken for the model set the app runs.
pub fn install_entry(
    models_dir: &Path,
    entry: &CatalogEntry,
    on_progress: &dyn Fn(InstallProgress),
) -> Result<(), String> {
    let tmp_dir = models_dir.join(TMP_DIR).join(entry.id);
    let result = install_inner(models_dir, entry, &tmp_dir, on_progress);
    remove_scratch(models_dir, entry.id);
    result
}

/// Install scratch, under `<models>`: one subfolder per id being installed.
const TMP_DIR: &str = "_tmp";

/// Removes one install's scratch folder, then `<models>/_tmp` itself if that
/// left it empty. Only ever `remove_dir` on the parent — it fails harmlessly
/// when another id's scratch is still inside, which must never be swept away.
fn remove_scratch(models_dir: &Path, id: &str) {
    let tmp_root = models_dir.join(TMP_DIR);
    let _ = std::fs::remove_dir_all(tmp_root.join(id));
    let _ = std::fs::remove_dir(&tmp_root);
}

fn install_inner(
    models_dir: &Path,
    entry: &CatalogEntry,
    tmp_dir: &Path,
    on_progress: &dyn Fn(InstallProgress),
) -> Result<(), String> {
    let dest_dir = entry.dir_in(models_dir);
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
        place_extracted(entry, &extract_dir, &dest_dir)?;
    } else {
        let final_path = entry.own_path_in(models_dir);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        move_file(&download_path, &final_path)?;
    }

    on_progress(InstallProgress::new(entry.id, "verifying", 95));
    record_install(models_dir, entry)?;

    on_progress(InstallProgress::new(entry.id, "done", 100));
    Ok(())
}

/// Records what an install just put on disk in the manifest: the primary
/// file's size and SHA-256 and, for a runtime, the SHA-256 of every file in its
/// own folder. Runs on the install worker thread, never the UI thread.
fn record_install(models_dir: &Path, entry: &CatalogEntry) -> Result<(), String> {
    // The own path, not `path_in`: a runtime still on the legacy flat layout
    // must not make an install that never produced its launcher look done.
    let installed = entry.own_path_in(models_dir);
    if !installed.exists() {
        return Err(format!(
            "install finished but {} is missing — the download may be for a different release",
            installed.display()
        ));
    }

    let bytes = std::fs::metadata(&installed)
        .map_err(|e| e.to_string())?
        .len();
    let digest = sha256_file(&installed)?;
    let file_digests = if entry.kind == ModelKind::Runtime {
        digest_folder(&entry.dir_in(models_dir))?
    } else {
        BTreeMap::new()
    };
    let mut manifest = load_manifest(models_dir);
    manifest.entries.insert(
        entry.id.to_string(),
        InstalledRecord {
            file: installed.to_string_lossy().to_string(),
            bytes,
            sha256: digest,
            installed_at_ms: now_ms(),
            source_url: entry.url.to_string(),
            file_digests,
        },
    );
    save_manifest(models_dir, &manifest)
}

/// Every file under `dir`, keyed by its `/`-separated path relative to `dir`.
fn files_in(dir: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut files = BTreeMap::new();
    visit_files(dir, &mut |path| {
        let relative = path.strip_prefix(dir).map_err(|e| e.to_string())?;
        files.insert(key_of(relative), path.to_path_buf());
        Ok(())
    })?;
    Ok(files)
}

/// The SHA-256 of every file under `dir`, keyed as [`files_in`] keys them.
fn digest_folder(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    files_in(dir)?
        .into_iter()
        .map(|(name, path)| {
            let digest = sha256_file(&path).map_err(|e| format!("could not hash {name}: {e}"))?;
            Ok((name, digest))
        })
        .collect()
}

/// `.exe` or `.dll` — what a runtime folder exists to hold, and what Windows
/// will load from it.
fn is_executable_or_dll(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("exe") || e.eq_ignore_ascii_case("dll"))
        .unwrap_or(false)
}

/// Keep local curl configuration from weakening TLS or adding extra transfers,
/// and reject protocol downgrades for both the initial URL and redirects.
fn download_command(entry: &CatalogEntry, dest: &Path) -> Command {
    let mut command = Command::new("curl.exe");
    command
        // curl only skips its default configuration when this is first.
        .arg("--disable")
        .arg("--proto")
        .arg("=https")
        .arg("--proto-redir")
        .arg("=https")
        .arg("-L")
        .arg("--fail")
        .arg("--silent")
        .arg("--show-error")
        .arg("-o")
        .arg(dest)
        .arg(entry.url)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    crate::system::hide_console(&mut command);
    command
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
    let mut command = download_command(entry, dest);
    let mut child = command
        .spawn()
        .map_err(|e| format!("could not launch curl.exe (needed to download models): {e}"))?;

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
    let mut command = Command::new("tar.exe");
    command.arg("-xf").arg(archive).arg("-C").arg(dest);
    crate::system::hide_console(&mut command);
    let output = command
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

/// Moves an unpacked archive into the entry's folder. Runtimes are flattened
/// into their own folder (`<models>/bin/<runtime>`), because the app calls
/// `<tool>.exe` there and its DLLs must sit beside it; everything else keeps
/// the archive's layout.
fn place_extracted(entry: &CatalogEntry, extract_dir: &Path, dest_dir: &Path) -> Result<(), String> {
    match entry.kind {
        ModelKind::Runtime => replace_runtime_dir(extract_dir, dest_dir),
        _ => {
            std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
            copy_tree(extract_dir, dest_dir)
        }
    }
}

/// A runtime is one release's set of executables and DLLs, so a reinstall
/// *replaces* its folder rather than copying over it. Copying over would leave
/// any file the new release no longer ships beside the ones it does — a DLL
/// from one release loaded by an exe from another — and the full-folder digest
/// recorded next would then vouch for it.
///
/// The new copy is staged in a sibling `<name>.new` folder, the current folder
/// is renamed aside, and the staged one renamed into place, so the installed
/// runtime keeps working until the new one is complete. If the swap cannot be
/// made — on Windows, typically because something still has a file in the
/// folder open — the old folder is put back and the install fails visibly.
fn replace_runtime_dir(extract_dir: &Path, dest_dir: &Path) -> Result<(), String> {
    let sibling = |suffix: &str| {
        let mut name = dest_dir.file_name().unwrap_or_default().to_os_string();
        name.push(suffix);
        dest_dir.with_file_name(name)
    };
    let staging = sibling(".new");
    let previous = sibling(".old");
    // Leftovers from an interrupted earlier attempt.
    let _ = std::fs::remove_dir_all(&staging);
    let _ = std::fs::remove_dir_all(&previous);

    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
    if let Err(err) = copy_binaries(extract_dir, &staging) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(err);
    }

    let had_previous = dest_dir.exists();
    if had_previous {
        if let Err(err) = std::fs::rename(dest_dir, &previous) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(format!(
                "could not replace {}: {err}. Something may still be using it — restart Cadence and try again.",
                dest_dir.display()
            ));
        }
    }
    if let Err(err) = std::fs::rename(&staging, dest_dir) {
        if had_previous {
            let _ = std::fs::rename(&previous, dest_dir);
        }
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!("could not move the new runtime into {}: {err}", dest_dir.display()));
    }
    // Best effort: a leftover `.old` folder is unused and is cleared by the
    // next install of this runtime.
    let _ = std::fs::remove_dir_all(&previous);
    Ok(())
}

/// Flattens every `.exe` / `.dll` in a release archive into `to` — one
/// runtime's own folder — returning the file names it copied (sorted,
/// deduplicated).
fn copy_binaries(from: &Path, to: &Path) -> Result<Vec<String>, String> {
    let mut copied = Vec::new();
    visit_files(from, &mut |path| {
        if !is_executable_or_dll(path) {
            return Ok(());
        }
        let name = path.file_name().ok_or("archive entry has no file name")?;
        std::fs::copy(path, to.join(name)).map_err(|e| e.to_string())?;
        copied.push(name.to_string_lossy().into_owned());
        Ok(())
    })?;
    if copied.is_empty() {
        return Err("archive contained no .exe/.dll files".to_string());
    }
    copied.sort_unstable();
    copied.dedup();
    Ok(copied)
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

/// Removes an installed model. Refuses to remove anything in the required
/// set, so a user cannot leave the app unable to process a meeting by
/// accident. Every catalog entry is currently required; this stays general so
/// an optional entry added later is removable without touching it.
pub fn remove(models_dir: &Path, id: &str) -> Result<(), String> {
    let entry = entry(id).ok_or_else(|| format!("unknown model id: {id}"))?;
    if required_ids().contains(&entry.id) {
        return Err(format!(
            "{} is part of the model set Cadence needs to process meetings, so it cannot be removed. Use Verify to check it, or reinstall it if it is damaged.",
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

// ---------------------------------------------------------- retired models

/// An entry an earlier build's catalog shipped and this one no longer does.
/// Kept only so a leftover file can be named properly ("Qwen3-4B", not
/// `Qwen3-4B-Q4_K_M.gguf`); nothing installs or runs these.
struct RetiredEntry {
    id: &'static str,
    label: &'static str,
    subdir: &'static str,
    file: &'static str,
}

/// The per-tier alternatives dropped when every machine moved to one model
/// set (commit b136c25).
const RETIRED: &[RetiredEntry] = &[
    RetiredEntry { id: "qwen3-4b-q4", label: "Qwen3-4B", subdir: "llm", file: "Qwen3-4B-Q4_K_M.gguf" },
    RetiredEntry { id: "gemma3-4b-q4", label: "Gemma 3 4B", subdir: "llm", file: "gemma-3-4b-it-Q4_K_M.gguf" },
    RetiredEntry { id: "whisper-base", label: "Whisper base", subdir: "stt", file: "ggml-base.bin" },
    RetiredEntry { id: "whisper-medium", label: "Whisper medium", subdir: "stt", file: "ggml-medium.bin" },
];

/// Folders under the models directory that are never searched for retired
/// models: `bin` holds the runtimes and their DLLs, which the catalog does not
/// list one by one (its old flat files are handled as one group instead, see
/// [`legacy_runtime_files`]); `bench` belongs to `bench_cli`'s comparison
/// candidates; `_tmp` is install scratch.
const NOT_SCANNED: &[&str] = &["bin", "bench", "_tmp"];

/// The handle for the one grouped item that stands for every file still
/// sitting directly in `<models>/bin` from the old flat runtime layout. `*`
/// cannot appear in a Windows file name, so it never collides with a real
/// item's key.
pub const LEGACY_RUNTIME_FILES_KEY: &str = "bin/*";

/// The handle for the one grouped item offered instead of
/// [`LEGACY_RUNTIME_FILES_KEY`] once every runtime runs from the copy the
/// installer shipped: the downloaded runtime folders under `<models>/bin`
/// *and* any old flat files there, none of which can run any more. At most
/// one of the two keys is ever listed.
pub const UNUSED_RUNTIMES_KEY: &str = "bin/**";

/// Something under the models directory that the current catalog does not
/// own: a model an earlier build installed, or a file put there by hand.
/// Reported so the user can see it and remove it; never removed on its own.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetiredModel {
    /// Where it sits under the models directory, `/`-separated
    /// (`llm/Qwen3-4B-Q4_K_M.gguf`). The handle [`remove_retired`] takes.
    pub key: String,
    /// The catalog id an earlier build installed it under, when known.
    pub id: Option<String>,
    pub label: String,
    pub path: String,
    /// Rounded up, so a file under a megabyte does not read as "0 MB".
    pub size_mb: u64,
    /// True when an earlier Cadence build installed it (a manifest record or
    /// a retired catalog file name); false for a file nobody recognises.
    pub from_previous_version: bool,
}

/// Every catalog folder that can hold a model, plus any a retired entry used.
fn scanned_subdirs() -> Vec<&'static str> {
    let mut dirs: Vec<&'static str> = catalog()
        .iter()
        .map(|e| e.subdir)
        .chain(RETIRED.iter().map(|r| r.subdir))
        // By top-level folder, so `bin/whisper` is skipped along with `bin`.
        .filter(|d| !NOT_SCANNED.iter().any(|skip| d.split('/').next() == Some(*skip)))
        .collect();
    dirs.sort_unstable();
    dirs.dedup();
    dirs
}

/// True when `name`, directly inside `subdir`, is (or contains) a catalog
/// entry's file — e.g. `sherpa-onnx-pyannote-segmentation-3-0` in
/// `diarization`. Case-insensitive, as Windows paths are.
fn owned_by_catalog(subdir: &str, name: &str) -> bool {
    catalog().iter().any(|e| {
        e.subdir.eq_ignore_ascii_case(subdir)
            && e.expect_relative
                .split('/')
                .next()
                .map(|top| top.eq_ignore_ascii_case(name))
                .unwrap_or(false)
    })
}

fn key_of(relative: &Path) -> String {
    relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Bytes on disk, without following links.
fn size_on_disk(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| size_on_disk(&entry.path()))
                .sum()
        })
        .unwrap_or(0)
}

/// True when `dir` is a real folder directly inside `models_dir` — not a link
/// or junction, and not somewhere else once every link along the way is
/// resolved.
fn is_real_child_dir(models_dir: &Path, dir: &Path) -> bool {
    let Ok(meta) = std::fs::symlink_metadata(dir) else {
        return false;
    };
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return false;
    }
    match (std::fs::canonicalize(models_dir), std::fs::canonicalize(dir)) {
        (Ok(root), Ok(resolved)) => {
            resolved.parent() == Some(root.as_path())
                && resolved.file_name().map(|n| n.to_string_lossy().to_lowercase())
                    == dir.file_name().map(|n| n.to_string_lossy().to_lowercase())
        }
        _ => false,
    }
}

/// True when every runtime runs from the copy the installer shipped, so
/// nothing under `<models>/bin` can run any more.
fn every_runtime_bundled(roots: Roots) -> bool {
    Runtime::ALL
        .iter()
        .all(|runtime| runtime.bundled_dir(roots.bundled).is_some())
}

/// The files the old flat runtime layout left directly in `<models>/bin`,
/// listed only once nothing can run from there any more: every runtime is
/// installed in its own folder, or every runtime is bundled. Until then this
/// is empty, so a file a runtime still uses is never offered for deletion.
///
/// Plain files directly inside `bin/` only — never a per-runtime folder or
/// anything in one, never a link, and nothing at all when `bin` is itself a
/// link or junction.
fn legacy_runtime_files(roots: Roots) -> Vec<PathBuf> {
    let models_dir = roots.models;
    let every_runtime_moved = Runtime::ALL
        .iter()
        .all(|runtime| runtime.entry().own_path_in(models_dir).exists());
    let bin = models_dir.join(RUNTIMES_DIR);
    if !(every_runtime_moved || every_runtime_bundled(roots)) || !is_real_child_dir(models_dir, &bin) {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(&bin) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        // `DirEntry::file_type` does not follow links: a link is not a file.
        .filter(|item| item.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|item| item.path())
        .collect();
    files.sort();
    files
}

/// True when `inner` is `outer` or lies inside it, once both are resolved.
/// False when either cannot be resolved.
fn resolves_within(inner: &Path, outer: &Path) -> bool {
    match (std::fs::canonicalize(inner), std::fs::canonicalize(outer)) {
        (Ok(inner), Ok(outer)) => inner.starts_with(&outer),
        _ => false,
    }
}

/// The downloaded runtime folders under `<models>/bin` — `whisper`, `llama`,
/// `sherpa`, and any `.new`/`.old` an interrupted reinstall left — listed only
/// once every runtime runs from the copy the installer shipped. Real folders
/// directly inside a real `bin/` only: never a link or junction, never one
/// that resolves anywhere else, and never one that is, holds, or sits inside
/// the bundled folder itself.
fn unused_runtime_folders(roots: Roots) -> Vec<PathBuf> {
    let bin = roots.models.join(RUNTIMES_DIR);
    let Some(bundled) = roots.bundled else {
        return Vec::new();
    };
    if !every_runtime_bundled(roots) || !is_real_child_dir(roots.models, &bin) {
        return Vec::new();
    }
    let mut folders = Vec::new();
    for runtime in Runtime::ALL {
        for suffix in ["", ".new", ".old"] {
            let folder = bin.join(format!("{}{suffix}", runtime.folder()));
            if is_real_child_dir(&bin, &folder)
                && !resolves_within(&folder, bundled)
                && !resolves_within(bundled, &folder)
            {
                folders.push(folder);
            }
        }
    }
    folders
}

/// Lists what sits directly inside the model folders (`llm/`, `stt/`, …) that
/// the current catalog does not own, plus one grouped item for runtime files
/// nothing can run any more: keyed [`UNUSED_RUNTIMES_KEY`] — the downloaded
/// runtime folders and old flat files — once every runtime is bundled with
/// the app, or else [`LEGACY_RUNTIME_FILES_KEY`] — the old flat files — once
/// every runtime has its own folder. Never looks in `bench/` or `_tmp/`, nor
/// inside the per-runtime folders, nor at the bundled runtimes.
///
/// What is listed comes from the disk, not the manifest: a manifest record
/// only lends an id and a name to a file that is actually there, and a record
/// whose file is gone lists nothing.
pub fn find_retired(models_dir: &Path) -> Vec<RetiredModel> {
    find_retired_at(Roots::with_bundled(models_dir))
}

fn find_retired_at(roots: Roots) -> Vec<RetiredModel> {
    let models_dir = roots.models;
    // Non-catalog manifest records, keyed by where they would sit now.
    let recorded: Vec<(String, String)> = load_manifest(models_dir)
        .entries
        .iter()
        .filter(|(id, _)| entry(id).is_none())
        .filter_map(|(id, record)| {
            relative_in_models(models_dir, &record.file).map(|rel| (key_of(&rel), id.clone()))
        })
        .collect();

    let mut found = Vec::new();
    for subdir in scanned_subdirs() {
        let Ok(entries) = std::fs::read_dir(models_dir.join(subdir)) else {
            continue;
        };
        for item in entries.filter_map(Result::ok) {
            // A name that is not valid Unicode cannot round-trip through the
            // key the UI sends back, so it could never be removed; skip it.
            let Some(name) = item.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if owned_by_catalog(subdir, &name) {
                continue;
            }
            let key = format!("{subdir}/{name}");
            let known = RETIRED
                .iter()
                .find(|r| r.subdir == subdir && r.file.eq_ignore_ascii_case(&name));
            let id = recorded
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(&key))
                .map(|(_, id)| id.clone())
                .or_else(|| known.map(|r| r.id.to_string()));
            let label = known
                .or_else(|| RETIRED.iter().find(|r| Some(r.id) == id.as_deref()))
                .map(|r| r.label.to_string())
                .unwrap_or_else(|| name.clone());
            let path = item.path();
            found.push(RetiredModel {
                key,
                from_previous_version: id.is_some(),
                id,
                label,
                size_mb: size_on_disk(&path).div_ceil(1024 * 1024),
                path: path.to_string_lossy().into_owned(),
            });
        }
    }

    let legacy = legacy_runtime_files(roots);
    let folders = unused_runtime_folders(roots);
    if !legacy.is_empty() || !folders.is_empty() {
        let (key, label) = if every_runtime_bundled(roots) {
            (UNUSED_RUNTIMES_KEY, "downloaded runtimes no longer used")
        } else {
            (LEGACY_RUNTIME_FILES_KEY, "old shared runtime files")
        };
        found.push(RetiredModel {
            key: key.to_string(),
            id: None,
            label: label.to_string(),
            path: models_dir.join(RUNTIMES_DIR).to_string_lossy().into_owned(),
            size_mb: legacy
                .iter()
                .chain(folders.iter())
                .map(|item| size_on_disk(item))
                .sum::<u64>()
                .div_ceil(1024 * 1024),
            from_previous_version: true,
        });
    }

    found.sort_by(|a, b| a.key.cmp(&b.key));
    found
}

/// Deletes the old flat runtime files [`legacy_runtime_files`] lists, and
/// nothing else. Tries every file, so a retry after a failure finishes the
/// job rather than starting over.
fn remove_legacy_runtime_files(roots: Roots) -> Result<(), String> {
    let mut failed = Vec::new();
    for file in legacy_runtime_files(roots) {
        if let Err(e) = std::fs::remove_file(&file) {
            failed.push(format!("{}: {e}", file.display()));
        }
    }
    match failed.first() {
        None => Ok(()),
        Some(first) => Err(format!(
            "{} old runtime file(s) could not be removed, probably because one is still in use. Restart Cadence and try again. ({first})",
            failed.len()
        )),
    }
}

/// Deletes what [`UNUSED_RUNTIMES_KEY`] stands for — the downloaded runtime
/// folders [`unused_runtime_folders`] lists and the flat files
/// [`legacy_runtime_files`] lists — and drops the manifest records that
/// described the folders. Never the bundled copy, which those listings
/// exclude. Tries everything, so a retry finishes the job.
fn remove_unused_runtimes(roots: Roots) -> Result<(), String> {
    let mut failed = Vec::new();
    for folder in unused_runtime_folders(roots) {
        // `remove_dir_all` does not follow links or junctions inside the
        // folder; the folder itself was checked to be a real directory.
        if let Err(e) = std::fs::remove_dir_all(&folder) {
            failed.push(format!("{}: {e}", folder.display()));
        }
    }
    if let Err(e) = remove_legacy_runtime_files(roots) {
        failed.push(e);
    }

    let mut manifest = load_manifest(roots.models);
    let before = manifest.entries.len();
    manifest.entries.retain(|id, _| {
        Runtime::from_id(id)
            .map(|runtime| runtime.entry().dir_in(roots.models).exists())
            .unwrap_or(true)
    });
    if manifest.entries.len() != before {
        save_manifest(roots.models, &manifest)?;
    }

    match failed.first() {
        None => Ok(()),
        Some(first) => Err(format!(
            "{} downloaded runtime item(s) could not be removed, probably because a file is still in use. Restart Cadence and try again. ({first})",
            failed.len()
        )),
    }
}

/// Resolves a retired-model key to the path to delete, or refuses.
///
/// The key must name exactly one item directly inside a scanned model folder
/// (`llm/x.gguf`) — never `bin/`, `bench/` or `_tmp/`, never a catalog
/// entry's file, never a link — and once every link and junction along the
/// way is resolved, it must still be that same item inside `models_dir`. That
/// last check is what stops a junctioned folder or a crafted key from steering
/// a delete anywhere else.
fn confined_target(models_dir: &Path, key: &str) -> Result<PathBuf, String> {
    let refuse = || format!("{key} is not inside the models folder, so it was not removed.");
    let parts: Vec<&str> = key.split('/').collect();
    let [subdir, name] = parts.as_slice() else {
        return Err(refuse());
    };
    for part in [subdir, name] {
        if part.is_empty()
            || *part == "."
            || *part == ".."
            || part.contains(['\\', ':'])
            || part.chars().any(char::is_control)
        {
            return Err(refuse());
        }
    }
    if !scanned_subdirs().iter().any(|dir| dir == subdir) {
        return Err(refuse());
    }
    if owned_by_catalog(subdir, name) {
        return Err(format!(
            "{key} belongs to the model set Cadence needs, so it was not removed."
        ));
    }

    let candidate = models_dir.join(subdir).join(name);
    let meta = std::fs::symlink_metadata(&candidate).map_err(|e| e.to_string())?;
    if meta.file_type().is_symlink() {
        return Err(format!("{key} is a link, so it was not removed."));
    }
    let root = std::fs::canonicalize(models_dir).map_err(|e| e.to_string())?;
    let resolved = std::fs::canonicalize(&candidate).map_err(|e| e.to_string())?;
    let relative = resolved.strip_prefix(&root).map_err(|_| refuse())?;
    let resolved_parts: Vec<String> = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    match resolved_parts.as_slice() {
        [dir, file]
            if dir.eq_ignore_ascii_case(subdir)
                && file.eq_ignore_ascii_case(name)
                && !owned_by_catalog(dir, file) =>
        {
            Ok(resolved)
        }
        _ => Err(refuse()),
    }
}

/// Deletes one retired model the user has explicitly chosen to remove.
///
/// Only something [`find_retired`] currently lists can be removed, and the
/// path is rebuilt from the models directory — never taken from a manifest
/// record, whose `file` may still point at the pre-rename
/// `com.echomind.app` folder. Any manifest record that described it is
/// dropped. Callers must refuse while an install is running.
pub fn remove_retired(models_dir: &Path, key: &str) -> Result<RetiredModel, String> {
    remove_retired_at(Roots::with_bundled(models_dir), key)
}

fn remove_retired_at(roots: Roots, key: &str) -> Result<RetiredModel, String> {
    let models_dir = roots.models;
    let item = find_retired_at(roots)
        .into_iter()
        .find(|m| m.key == key)
        .ok_or_else(|| {
            format!("{key} is not a model Cadence has stopped using, so it was not removed.")
        })?;
    if item.key == LEGACY_RUNTIME_FILES_KEY {
        // Only listed once every runtime runs from its own folder, and only
        // ever deletes plain files directly in `bin/`. No manifest record
        // describes these files, so there is nothing to drop.
        remove_legacy_runtime_files(roots)?;
        return Ok(item);
    }
    if item.key == UNUSED_RUNTIMES_KEY {
        // Only listed once every runtime runs from the bundled copy.
        remove_unused_runtimes(roots)?;
        return Ok(item);
    }
    let target = confined_target(models_dir, &item.key)?;
    if target.is_dir() {
        std::fs::remove_dir_all(&target).map_err(|e| e.to_string())?;
    } else {
        std::fs::remove_file(&target).map_err(|e| e.to_string())?;
    }

    let mut manifest = load_manifest(models_dir);
    let before = manifest.entries.len();
    manifest.entries.retain(|id, record| {
        entry(id).is_some()
            || !relative_in_models(models_dir, &record.file)
                .map(|rel| key_of(&rel).eq_ignore_ascii_case(&item.key))
                .unwrap_or(false)
    });
    if manifest.entries.len() != before {
        save_manifest(models_dir, &manifest)?;
    }
    Ok(item)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub id: String,
    pub ok: bool,
    pub message: String,
    pub sha256: Option<String>,
}

/// the "model health check": the file is present, non-empty, and still
/// hashes to what it hashed to at install time.
pub fn verify(models_dir: &Path, id: &str) -> Result<VerifyReport, String> {
    verify_at(Roots::with_bundled(models_dir), id)
}

fn verify_at(roots: Roots, id: &str) -> Result<VerifyReport, String> {
    let models_dir = roots.models;
    let entry = entry(id).ok_or_else(|| format!("unknown model id: {id}"))?;
    if let (Some(runtime), Some(dir)) = (Runtime::from_id(id), entry.bundled_dir(roots)) {
        let root = dir.parent().unwrap_or(&dir).to_path_buf();
        return verify_bundled_runtime(entry, runtime, &dir, &root);
    }
    let path = entry.path_at(roots);
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
    let record = load_manifest(models_dir).entries.remove(id);
    let legacy = entry.uses_legacy_layout_at(roots);
    let in_own_folder = entry.kind == ModelKind::Runtime && !legacy;
    if in_own_folder {
        if let Some(record) = record.as_ref().filter(|r| !r.file_digests.is_empty()) {
            return verify_runtime_folder(models_dir, entry, record);
        }
    }

    let digest = sha256_file(&path)?;
    let recorded = record.as_ref().map(|r| r.sha256.clone());

    let (ok, mut message) = match recorded {
        Some(expected) if expected == digest => (
            true,
            format!("OK — {} MB, digest unchanged since install.", mb_ceil(bytes)),
        ),
        Some(_) => (
            false,
            "File has changed since it was installed. Reinstall it.".to_string(),
        ),
        None => (
            true,
            format!(
                "Present ({} MB) but installed outside the app, so there is no recorded digest to compare against.",
                mb_ceil(bytes)
            ),
        ),
    };
    if legacy {
        message.push_str(&format!(
            " It was installed by an earlier version into the shared {} folder, where it shares DLLs with the other runtimes. It still works from there; reinstalling it moves it into a folder of its own.",
            models_dir.join(RUNTIMES_DIR).display()
        ));
    } else if in_own_folder && record.is_some() {
        // Installed into its own folder by a build that recorded only the
        // launcher's digest: say plainly how little that covered.
        let dir = entry.dir_in(models_dir);
        let count = files_in(&dir)?.len();
        message.push_str(&format!(
            " Only the launcher, {}, could be checked — not the other {} file(s) in its {} MB folder, because it was installed before Cadence recorded a digest for every file. Reinstalling it records one for each.",
            entry.expect_relative,
            count.saturating_sub(1),
            mb_ceil(size_on_disk(&dir))
        ));
    }

    Ok(VerifyReport {
        id: id.to_string(),
        ok,
        message,
        sha256: Some(digest),
    })
}

/// At most this many file names in one verify message; the rest are counted.
const NAMED_FILES: usize = 3;

/// `a, b, c and 2 more` — a short list of offending files for a message.
fn some_names(names: &[&str]) -> String {
    let shown = names[..names.len().min(NAMED_FILES)].join(", ");
    match names.len().saturating_sub(NAMED_FILES) {
        0 => shown,
        more => format!("{shown} and {more} more"),
    }
}

/// `1 file` / `2 files`.
fn files_count(n: usize) -> String {
    if n == 1 { "1 file".to_string() } else { format!("{n} files") }
}

/// Verifies a runtime in its own folder against the per-file digests recorded
/// at install: fails if any recorded file is missing or changed, or if an
/// `.exe`/`.dll` appeared that was not there at install — a new binary beside
/// the launcher is exactly what Windows' DLL search order would load. Any
/// other new file is reported but does not fail the check.
fn verify_runtime_folder(
    models_dir: &Path,
    entry: &CatalogEntry,
    record: &InstalledRecord,
) -> Result<VerifyReport, String> {
    let dir = entry.dir_in(models_dir);
    let check = check_folder(&dir, &record.file_digests, entry.expect_relative)?;
    let ok = check.ok();
    let size_mb = mb_ceil(size_on_disk(&dir));
    let mut message = if ok {
        format!("OK — {}, {size_mb} MB, all unchanged since install.", files_count(check.recorded))
    } else {
        format!(
            "Runtime folder has changed since install: {}. Reinstall it.",
            check.problems().join("; ")
        )
    };
    message.push_str(&check.other_files_note());
    Ok(VerifyReport {
        id: entry.id.to_string(),
        ok,
        message,
        sha256: check.launcher_digest,
    })
}

/// A runtime folder compared file by file against the digests on record.
struct FolderCheck {
    recorded: usize,
    changed: Vec<String>,
    missing: Vec<String>,
    new_binaries: Vec<String>,
    new_other: Vec<String>,
    launcher_digest: Option<String>,
}

impl FolderCheck {
    /// Fails on a changed or missing file, or on a new `.exe`/`.dll` — a new
    /// binary beside the launcher is exactly what Windows' DLL search order
    /// would load. A new non-binary file does not fail it.
    fn ok(&self) -> bool {
        self.changed.is_empty() && self.missing.is_empty() && self.new_binaries.is_empty()
    }

    fn problems(&self) -> Vec<String> {
        let names = |list: &[String]| some_names(&list.iter().map(String::as_str).collect::<Vec<_>>());
        let mut problems = Vec::new();
        if !self.changed.is_empty() {
            problems.push(format!("{} of {} changed ({})", self.changed.len(), self.recorded, names(&self.changed)));
        }
        if !self.missing.is_empty() {
            problems.push(format!("{} missing ({})", files_count(self.missing.len()), names(&self.missing)));
        }
        if !self.new_binaries.is_empty() {
            problems.push(format!(
                "{} new executable(s) or DLL(s) not there at install ({})",
                self.new_binaries.len(),
                names(&self.new_binaries)
            ));
        }
        problems
    }

    /// The note appended when new non-binary files are present; empty if none.
    fn other_files_note(&self) -> String {
        if self.new_other.is_empty() {
            return String::new();
        }
        let (verb, pronoun, kind) = if self.new_other.len() == 1 {
            ("was", "it is", "an executable or DLL")
        } else {
            ("were", "they are", "executables or DLLs")
        };
        let names: Vec<&str> = self.new_other.iter().map(String::as_str).collect();
        format!(
            " The folder also holds {} that {verb} not there at install ({}); {pronoun} not {kind}, so this does not fail the check.",
            files_count(self.new_other.len()),
            some_names(&names)
        )
    }
}

fn check_folder(dir: &Path, recorded: &BTreeMap<String, String>, launcher: &str) -> Result<FolderCheck, String> {
    let present = files_in(dir)?;
    let mut check = FolderCheck {
        recorded: recorded.len(),
        changed: Vec::new(),
        missing: Vec::new(),
        new_binaries: Vec::new(),
        new_other: Vec::new(),
        launcher_digest: None,
    };
    for (name, expected) in recorded {
        let Some(path) = present.get(name) else {
            check.missing.push(name.clone());
            continue;
        };
        let digest = sha256_file(path).map_err(|e| format!("could not read {name}: {e}"))?;
        if digest != *expected {
            check.changed.push(name.clone());
        }
        if name == launcher {
            check.launcher_digest = Some(digest);
        }
    }
    for name in present.keys().filter(|name| !recorded.contains_key(*name)) {
        if is_executable_or_dll(Path::new(name)) {
            check.new_binaries.push(name.clone());
        } else {
            check.new_other.push(name.clone());
        }
    }
    Ok(check)
}

/// One runtime's record in the bundled `runtimes/manifest.json`. Extra fields
/// the build script writes (source URL, archive digest) are ignored here.
#[derive(Debug, Clone, Deserialize)]
struct BundledRecord {
    version: String,
    #[serde(default)]
    files: BTreeMap<String, String>,
}

/// The bundled manifest under `root`: `Ok(None)` when there is none (a
/// development build whose runtimes were put there by hand), `Err` when it is
/// there but unreadable — on a release install, a sign of damage.
fn load_bundled_manifest(root: &Path) -> Result<Option<BTreeMap<String, BundledRecord>>, String> {
    let path = root.join(BUNDLED_MANIFEST);
    match std::fs::read_to_string(&path) {
        Ok(raw) => serde_json::from_str(raw.trim_start_matches('\u{feff}'))
            .map(Some)
            .map_err(|e| e.to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Verifies a runtime the installer shipped against the digests
/// `scripts/prepare-runtimes.ps1` recorded for it at build time, after
/// signing. Same rules as an installed runtime's folder check; the remedy is
/// reinstalling Cadence, since the app never downloads an executable it ships.
fn verify_bundled_runtime(
    entry: &CatalogEntry,
    runtime: Runtime,
    dir: &Path,
    root: &Path,
) -> Result<VerifyReport, String> {
    let report = |ok: bool, message: String, sha256: Option<String>| VerifyReport {
        id: entry.id.to_string(),
        ok,
        message,
        sha256,
    };
    let size_mb = mb_ceil(size_on_disk(dir));
    let manifest = match load_bundled_manifest(root) {
        Ok(manifest) => manifest,
        Err(e) => {
            return Ok(report(
                false,
                format!(
                    "The copy included with Cadence ({}) could not be checked: its manifest {} is unreadable ({e}). Reinstall Cadence to repair it.",
                    dir.display(),
                    root.join(BUNDLED_MANIFEST).display()
                ),
                None,
            ))
        }
    };
    let launcher = sha256_file(&dir.join(entry.expect_relative)).ok();
    let Some(manifest) = manifest else {
        return Ok(report(
            true,
            format!(
                "Included with Cadence ({}, {size_mb} MB), but this build carries no runtime manifest — a development build — so its files cannot be checked against what was shipped.",
                dir.display()
            ),
            launcher,
        ));
    };
    let Some(record) = manifest.get(runtime.folder()) else {
        return Ok(report(
            false,
            format!(
                "The copy included with Cadence ({}) could not be checked: the manifest shipped with it lists no {} runtime. Reinstall Cadence to repair it.",
                dir.display(),
                runtime.folder()
            ),
            launcher,
        ));
    };

    let check = check_folder(dir, &record.files, entry.expect_relative)?;
    let ok = check.ok();
    let mut message = if ok {
        format!(
            "OK — {}, {size_mb} MB, all unchanged since this version of Cadence was built ({}). This is the copy included with Cadence, in {}.",
            files_count(check.recorded),
            record.version,
            dir.display()
        )
    } else {
        format!(
            "The runtime included with Cadence ({}) has changed since it was installed: {}. Reinstall Cadence to repair it.",
            dir.display(),
            check.problems().join("; ")
        )
    };
    message.push_str(&check.other_files_note());
    Ok(report(ok, message, check.launcher_digest.or(launcher)))
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

    struct TestModelsDir(PathBuf);

    impl TestModelsDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "cadence-model-integrity-{}-{stamp}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn record_model(&self) -> PathBuf {
            let model = entry("vad-silero").unwrap();
            let path = model.path_in(&self.0);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"synthetic installed model").unwrap();
            let mut manifest = Manifest::default();
            manifest.entries.insert(model.id.to_string(), InstalledRecord {
                file: path.to_string_lossy().into_owned(),
                bytes: std::fs::metadata(&path).unwrap().len(),
                sha256: sha256_file(&path).unwrap(),
                installed_at_ms: 0,
                source_url: model.url.to_string(),
                file_digests: BTreeMap::new(),
            });
            save_manifest(&self.0, &manifest).unwrap();
            path
        }
    }

    impl Drop for TestModelsDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn verify_accepts_unchanged_installed_model() {
        let dir = TestModelsDir::new();
        let path = dir.record_model();
        let report = verify(&dir.0, "vad-silero").unwrap();
        assert!(report.ok);
        assert_eq!(report.sha256, Some(sha256_file(&path).unwrap()));
        assert!(report.message.contains("digest unchanged"));
    }

    #[test]
    fn verify_detects_same_size_tampering_and_preserves_manifest() {
        let dir = TestModelsDir::new();
        let path = dir.record_model();
        let original_manifest = std::fs::read(manifest_path(&dir.0)).unwrap();
        let mut tampered = std::fs::read(&path).unwrap();
        tampered[0] ^= 1;
        std::fs::write(&path, &tampered).unwrap();
        let report = verify(&dir.0, "vad-silero").unwrap();
        assert!(!report.ok);
        assert!(report.message.contains("changed since"));
        assert_eq!(report.sha256, Some(sha256_file(&path).unwrap()));
        assert_eq!(std::fs::read(&path).unwrap(), tampered);
        assert_eq!(std::fs::read(manifest_path(&dir.0)).unwrap(), original_manifest);
    }

    #[test]
    fn verify_reports_missing_file_despite_install_record() {
        let dir = TestModelsDir::new();
        let path = dir.record_model();
        std::fs::remove_file(path).unwrap();
        let report = verify(&dir.0, "vad-silero").unwrap();
        assert!(!report.ok);
        assert_eq!(report.sha256, None);
        assert_eq!(report.message, "Not installed.");
        assert!(load_manifest(&dir.0).entries.contains_key("vad-silero"));
    }

    #[test]
    fn verify_rejects_truncated_and_empty_installed_models() {
        let dir = TestModelsDir::new();
        let path = dir.record_model();
        std::fs::write(&path, b"synthetic").unwrap();
        let truncated = verify(&dir.0, "vad-silero").unwrap();
        assert!(!truncated.ok);
        assert!(truncated.sha256.is_some());
        std::fs::write(&path, b"").unwrap();
        let empty = verify(&dir.0, "vad-silero").unwrap();
        assert!(!empty.ok);
        assert_eq!(empty.sha256, None);
        assert!(empty.message.contains("empty"));
    }

    #[test]
    fn verify_keeps_manual_install_compatibility_without_manifest() {
        let dir = TestModelsDir::new();
        let path = dir.record_model();
        std::fs::remove_file(manifest_path(&dir.0)).unwrap();
        let report = verify(&dir.0, "vad-silero").unwrap();
        assert!(report.ok);
        assert_eq!(report.sha256, Some(sha256_file(&path).unwrap()));
        assert!(report.message.contains("no recorded digest"));
        assert!(!manifest_path(&dir.0).exists());
    }

    #[test]
    fn downloads_ignore_curlrc_and_require_https_including_redirects() {
        let dest = Path::new("models with spaces/vad/silero_vad.onnx");
        for model in catalog() {
            assert!(model.url.starts_with("https://"), "{}", model.id);
            let command = download_command(model, dest);
            assert_eq!(command.get_program(), "curl.exe");
            let args: Vec<_> = command.get_args().collect();
            let expected: Vec<&std::ffi::OsStr> = vec![
                "--disable".as_ref(),
                "--proto".as_ref(),
                "=https".as_ref(),
                "--proto-redir".as_ref(),
                "=https".as_ref(),
                "-L".as_ref(),
                "--fail".as_ref(),
                "--silent".as_ref(),
                "--show-error".as_ref(),
                "-o".as_ref(),
                dest.as_os_str(),
                model.url.as_ref(),
            ];
            assert_eq!(args, expected, "{}", model.id);
        }
    }

    #[test]
    fn download_destinations_are_single_literal_arguments() {
        // Inspect without starting curl: punctuation must never go through a shell.
        for destination in [
            "models with spaces/model.onnx",
            "தமிழ்/模型/silero.onnx",
            "models/$(echo injected); & echo injected | model.onnx",
            "--config=untrusted-curlrc",
            "models/\"quoted\"/%TEMP%/model.onnx",
        ] {
            let dest = Path::new(destination);
            for model in catalog() {
                let command = download_command(model, dest);
                assert_eq!(command.get_program(), "curl.exe");
                let args: Vec<_> = command.get_args().collect();
                assert_eq!(args.len(), 12);
                assert_eq!(args[9], "-o");
                assert_eq!(args[10], dest.as_os_str());
                assert_eq!(args[11], model.url);
            }
        }
    }

    #[test]
    fn catalog_ids_are_unique() {
        let mut ids: Vec<&str> = catalog().iter().map(|e| e.id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }

    #[test]
    fn every_required_id_exists_in_the_catalog() {
        for id in required_ids() {
            assert!(entry(id).is_some(), "{id} missing from catalog");
        }
        assert_eq!(entry(system::LLM_MODEL_ID).unwrap().kind, ModelKind::Llm);
        assert_eq!(entry(system::STT_MODEL_ID).unwrap().kind, ModelKind::Stt);
    }

    #[test]
    fn the_catalog_holds_exactly_one_llm_and_one_stt_model() {
        let count = |kind| catalog().iter().filter(|e| e.kind == kind).count();
        assert_eq!(count(ModelKind::Llm), 1);
        assert_eq!(count(ModelKind::Stt), 1);
    }

    #[test]
    fn the_llm_fits_its_budget_on_a_16gb_machine() {
        let llm = entry(system::LLM_MODEL_ID).unwrap();
        let estimated = llm.estimated_llm_ram_mb(system::LLM_CONTEXT_TOKENS);
        assert!(
            estimated <= system::LLM_RAM_BUDGET_MB,
            "estimated {estimated} MB exceeds the {} MB budget",
            system::LLM_RAM_BUDGET_MB
        );
    }

    #[test]
    fn the_audio_quality_models_are_part_of_the_required_set() {
        // About a megabyte each and the cheapest voice/noise gain available,
        // so first-run setup installs them. The pipeline still skips each
        // when absent (see pipeline::denoise and pipeline::vad).
        for id in ["denoise-gtcrn", "vad-silero"] {
            assert!(entry(id).is_some(), "{id} missing from catalog");
            assert!(required_ids().contains(&id), "{id} must be required");
        }
    }

    #[test]
    fn runtimes_install_first_and_the_llm_last() {
        let ids = required_ids();
        assert_eq!(ids.last(), Some(&system::LLM_MODEL_ID));
        let runtimes = ids
            .iter()
            .take_while(|id| entry(id).unwrap().kind == ModelKind::Runtime)
            .count();
        assert_eq!(runtimes, 3);
    }

    #[test]
    fn a_required_model_cannot_be_removed() {
        let dir = std::env::temp_dir().join("cadence-remove-required");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("vad")).unwrap();
        let path = entry("vad-silero").unwrap().path_in(&dir);
        std::fs::write(&path, b"stub").unwrap();
        assert!(remove(&dir, "vad-silero").is_err());
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_download_total_counts_only_what_is_missing() {
        let dir = std::env::temp_dir().join("cadence-download-total");
        let _ = std::fs::remove_dir_all(&dir);
        let everything: u64 = required_ids()
            .iter()
            .map(|id| entry(id).unwrap().approx_size_mb)
            .sum();
        assert_eq!(required_download_mb(&dir), everything);

        let llm = entry(system::LLM_MODEL_ID).unwrap();
        std::fs::create_dir_all(dir.join(llm.subdir)).unwrap();
        std::fs::write(llm.path_in(&dir), b"stub").unwrap();
        assert_eq!(required_download_mb(&dir), everything - llm.approx_size_mb);
        assert!(!missing_required_ids(&dir).contains(&system::LLM_MODEL_ID));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `<tmp>/<unique>/models`, so the folder is named like the real one and
    /// a stale `…\com.echomind.app\models\…` path resolves against it.
    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cadence-retired-{}-{sequence}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(path.join("models")).unwrap();
            Self(path)
        }

        fn models(&self) -> PathBuf {
            self.0.join("models")
        }

        fn write(&self, relative: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
            path
        }

        /// A record as a pre-rename build wrote it.
        fn record(&self, manifest: &mut Manifest, id: &str, file: &str) {
            manifest.entries.insert(id.to_string(), InstalledRecord {
                file: file.to_string(),
                bytes: 4,
                sha256: format!("sha-of-{id}"),
                installed_at_ms: 1,
                source_url: "https://example.invalid/".to_string(),
                file_digests: BTreeMap::new(),
            });
        }
    }

    fn status_of(models_dir: &Path, id: &str) -> ModelStatus {
        status_all(models_dir).into_iter().find(|s| s.id == id).unwrap()
    }

    #[test]
    fn a_sub_megabyte_model_reports_one_mb_not_zero() {
        let root = TestRoot::new();
        root.write("models/vad/silero_vad.onnx", &vec![7u8; 600 * 1024]);
        let status = status_of(&root.models(), "vad-silero");
        assert!(status.installed);
        assert_eq!(status.actual_size_mb, 1);
        assert_eq!(mb_ceil(0), 0);
        assert_eq!(mb_ceil(1024 * 1024), 1);
        assert_eq!(mb_ceil(1024 * 1024 + 1), 2);
        assert_eq!(status_of(&root.models(), "denoise-gtcrn").actual_size_mb, 0, "not installed");
    }

    #[test]
    fn every_runtime_has_its_own_folder_in_the_catalog() {
        let mut dirs = Vec::new();
        for runtime in Runtime::ALL {
            let e = runtime.entry();
            assert_eq!(e.kind, ModelKind::Runtime, "{}", e.id);
            let (top, own) = e.subdir.split_once('/').expect("a runtime folder under bin/");
            assert_eq!(top, RUNTIMES_DIR, "{}", e.id);
            assert!(!own.is_empty() && !own.contains('/'), "{}", e.id);
            dirs.push(own);
        }
        dirs.sort_unstable();
        dirs.dedup();
        assert_eq!(dirs.len(), Runtime::ALL.len(), "two runtimes share a folder");
        let runtimes_in_catalog = catalog().iter().filter(|e| e.kind == ModelKind::Runtime).count();
        assert_eq!(runtimes_in_catalog, Runtime::ALL.len());
    }

    /// Unpacks a synthetic release into `extract/<id>` the way `tar.exe`
    /// would, nested a level deep as the real archives are.
    fn fake_release(root: &TestRoot, id: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let extract = root.0.join("extract").join(id);
        for (name, bytes) in files {
            root.write(&format!("extract/{id}/release/{name}"), bytes);
        }
        extract
    }

    #[test]
    fn each_runtime_installs_into_its_own_folder_and_keeps_its_own_ggml() {
        let root = TestRoot::new();
        let models = root.models();
        let whisper = Runtime::Whisper.entry();
        let llama = Runtime::Llama.entry();

        let from = fake_release(&root, "w", &[("whisper-cli.exe", b"whisper exe"), ("ggml.dll", b"whisper ggml")]);
        place_extracted(whisper, &from, &whisper.dir_in(&models)).unwrap();
        let from = fake_release(&root, "l", &[("llama-server.exe", b"llama exe"), ("ggml.dll", b"llama ggml")]);
        place_extracted(llama, &from, &llama.dir_in(&models)).unwrap();

        let whisper_ggml = runtime_binary(&models, Runtime::Whisper, "ggml.dll");
        let llama_ggml = runtime_binary(&models, Runtime::Llama, "ggml.dll");
        assert_eq!(whisper_ggml, models.join("bin").join("whisper").join("ggml.dll"));
        assert_eq!(llama_ggml, models.join("bin").join("llama").join("ggml.dll"));
        assert_eq!(std::fs::read(&whisper_ggml).unwrap(), b"whisper ggml");
        assert_eq!(std::fs::read(&llama_ggml).unwrap(), b"llama ggml");
        // Nothing lands in the shared folder itself.
        assert!(!models.join("bin").join("ggml.dll").exists());
        assert!(!models.join("bin").join("whisper-cli.exe").exists());

        // Reinstalling one runtime leaves the other's same-named DLL alone.
        let from = fake_release(&root, "w2", &[("whisper-cli.exe", b"whisper exe 2"), ("ggml.dll", b"whisper ggml 2")]);
        place_extracted(whisper, &from, &whisper.dir_in(&models)).unwrap();
        assert_eq!(std::fs::read(&whisper_ggml).unwrap(), b"whisper ggml 2");
        assert_eq!(std::fs::read(&llama_ggml).unwrap(), b"llama ggml");

        for runtime in [Runtime::Whisper, Runtime::Llama] {
            let e = runtime.entry();
            assert!(e.is_installed(&models), "{}", e.id);
            assert!(!e.uses_legacy_layout(&models), "{}", e.id);
            assert_eq!(e.path_in(&models), e.own_path_in(&models));
        }
        assert!(!Runtime::Sherpa.entry().is_installed(&models));
    }

    #[test]
    fn a_runtime_reinstall_replaces_its_folder_instead_of_copying_over_it() {
        let root = TestRoot::new();
        let models = root.models();
        let whisper = Runtime::Whisper.entry();
        let dir = whisper.dir_in(&models);

        let from = fake_release(&root, "old", &[("whisper-cli.exe", b"v1"), ("ggml-old.dll", b"only in v1")]);
        place_extracted(whisper, &from, &dir).unwrap();
        let from = fake_release(&root, "new", &[("whisper-cli.exe", b"v2"), ("ggml.dll", b"v2 ggml")]);
        place_extracted(whisper, &from, &dir).unwrap();

        assert_eq!(std::fs::read(dir.join("whisper-cli.exe")).unwrap(), b"v2");
        assert!(dir.join("ggml.dll").exists());
        // A file the new release does not ship must not survive beside it.
        assert!(!dir.join("ggml-old.dll").exists());
        // No staging or backup folder is left behind.
        let bin = models.join("bin");
        assert!(!bin.join("whisper.new").exists());
        assert!(!bin.join("whisper.old").exists());
    }

    #[test]
    fn a_failed_runtime_reinstall_leaves_the_working_copy_untouched() {
        let root = TestRoot::new();
        let models = root.models();
        let whisper = Runtime::Whisper.entry();
        let dir = whisper.dir_in(&models);

        let from = fake_release(&root, "good", &[("whisper-cli.exe", b"working"), ("ggml.dll", b"working ggml")]);
        place_extracted(whisper, &from, &dir).unwrap();
        // An archive with no binaries in it fails before anything is swapped.
        let from = fake_release(&root, "bad", &[("README.md", b"no binaries here")]);
        assert!(place_extracted(whisper, &from, &dir).is_err());

        assert_eq!(std::fs::read(dir.join("whisper-cli.exe")).unwrap(), b"working");
        assert_eq!(std::fs::read(dir.join("ggml.dll")).unwrap(), b"working ggml");
        assert!(!models.join("bin").join("whisper.new").exists());
    }

    #[test]
    fn a_legacy_flat_runtime_still_resolves_and_counts_as_installed() {
        let root = TestRoot::new();
        let models = root.models();
        let flat = models.join("bin");
        root.write("models/bin/whisper-cli.exe", b"old flat whisper");
        root.write("models/bin/ggml.dll", b"whichever installed last");
        let whisper = Runtime::Whisper.entry();

        assert!(whisper.uses_legacy_layout(&models));
        assert!(whisper.is_installed(&models));
        assert_eq!(whisper.path_in(&models), flat.join("whisper-cli.exe"));
        assert_eq!(runtime_dir(&models, Runtime::Whisper), flat);
        // Every executable of the runtime comes from the same folder.
        assert_eq!(
            runtime_binary(&models, Runtime::Whisper, "whisper-server.exe"),
            flat.join("whisper-server.exe")
        );
        assert!(!missing_required_ids(&models).contains(&"runtime-whisper"));

        let status = status_of(&models, "runtime-whisper");
        assert!(status.installed);
        assert_eq!(PathBuf::from(&status.path), flat.join("whisper-cli.exe"));
        assert_eq!(status.actual_size_mb, whisper.approx_size_mb, "legacy layout falls back to the estimate");

        let report = verify(&models, "runtime-whisper").unwrap();
        assert!(report.ok, "{}", report.message);
        assert!(report.message.contains("shares DLLs"), "{}", report.message);
        assert!(report.message.contains("reinstalling it moves it"), "{}", report.message);

        // A runtime in neither layout is not installed, and points at its own
        // folder, where an install would put it.
        let llama = Runtime::Llama.entry();
        assert!(!llama.is_installed(&models));
        assert!(!llama.uses_legacy_layout(&models));
        assert_eq!(runtime_dir(&models, Runtime::Llama), flat.join("llama"));
    }

    #[test]
    fn the_own_folder_wins_when_both_layouts_exist() {
        let root = TestRoot::new();
        let models = root.models();
        root.write("models/bin/sherpa-onnx-offline-speaker-diarization.exe", b"old");
        root.write("models/bin/sherpa/sherpa-onnx-offline-speaker-diarization.exe", b"new");
        let sherpa = Runtime::Sherpa.entry();
        assert!(!sherpa.uses_legacy_layout(&models));
        assert_eq!(sherpa.path_in(&models), models.join("bin").join("sherpa").join("sherpa-onnx-offline-speaker-diarization.exe"));
        assert_eq!(
            runtime_binary(&models, Runtime::Sherpa, "sherpa-onnx-vad.exe"),
            models.join("bin").join("sherpa").join("sherpa-onnx-vad.exe")
        );
        let report = verify(&models, "runtime-sherpa").unwrap();
        assert!(report.ok);
        assert!(!report.message.contains("shares DLLs"), "{}", report.message);
    }

    #[test]
    fn a_runtime_in_its_own_folder_is_sized_as_that_folder() {
        let root = TestRoot::new();
        let mb = 1024 * 1024;
        root.write("models/bin/whisper/whisper-cli.exe", &vec![0u8; 300 * 1024]);
        root.write("models/bin/whisper/whisper.dll", &vec![0u8; 2 * mb]);
        root.write("models/bin/whisper/ggml-cpu-haswell.dll", &vec![0u8; mb]);
        // Neither the shared folder nor another runtime's folder counts.
        root.write("models/bin/onnxruntime.dll", &vec![0u8; 16 * mb]);
        root.write("models/bin/llama/ggml-cpu-haswell.dll", &vec![0u8; 5 * mb]);
        let status = status_of(&root.models(), "runtime-whisper");
        assert!(status.installed);
        // 300 KB + 2 MB + 1 MB, rounded up.
        assert_eq!(status.actual_size_mb, 4);
        // llama's folder has no launcher: not installed, so no size.
        assert_eq!(status_of(&root.models(), "runtime-llama").actual_size_mb, 0);
    }

    #[test]
    fn a_manifest_with_the_dropped_runtime_files_field_still_loads() {
        let root = TestRoot::new();
        std::fs::write(
            manifest_path(&root.models()),
            r#"{"entries":{"runtime-llama":{"file":"x","bytes":8,"sha256":"s","installed_at_ms":1,"source_url":"https://example.invalid/","runtime_files":["llama-server.exe"]}}}"#,
        )
        .unwrap();
        let manifest = load_manifest(&root.models());
        assert_eq!(manifest.entries["runtime-llama"].sha256, "s");
        assert!(manifest.entries["runtime-llama"].file_digests.is_empty());
    }

    #[test]
    fn a_record_without_file_digests_round_trips_without_the_field() {
        let root = TestRoot::new();
        let raw = r#"{"entries":{"vad-silero":{"file":"x","bytes":8,"sha256":"s","installed_at_ms":1,"source_url":"https://example.invalid/"}}}"#;
        std::fs::write(manifest_path(&root.models()), raw).unwrap();
        let manifest = load_manifest(&root.models());
        assert!(manifest.entries["vad-silero"].file_digests.is_empty());
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(!json.contains("file_digests"), "{json}");
        assert_eq!(json, raw);
    }

    /// A runtime installed the current way — synthetic release unpacked into
    /// its own folder and recorded as `install` records it.
    fn installed_whisper(root: &TestRoot) -> PathBuf {
        let models = root.models();
        let whisper = Runtime::Whisper.entry();
        let from = fake_release(root, "whisper", &[
            ("whisper-cli.exe", b"whisper launcher"),
            ("whisper.dll", b"whisper library"),
            ("ggml.dll", b"whisper ggml"),
            ("ggml-base.dll", b"whisper ggml base"),
        ]);
        place_extracted(whisper, &from, &whisper.dir_in(&models)).unwrap();
        record_install(&models, whisper).unwrap();
        whisper.dir_in(&models)
    }

    #[test]
    fn a_runtime_install_records_a_digest_for_every_file_in_its_folder() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        let record = &load_manifest(&root.models()).entries["runtime-whisper"];
        let names: Vec<&str> = record.file_digests.keys().map(String::as_str).collect();
        assert_eq!(names, ["ggml-base.dll", "ggml.dll", "whisper-cli.exe", "whisper.dll"]);
        for (name, digest) in &record.file_digests {
            assert_eq!(digest, &sha256_file(&dir.join(name)).unwrap(), "{name}");
        }
        // The single launcher digest is still recorded as before.
        assert_eq!(record.sha256, sha256_file(&dir.join("whisper-cli.exe")).unwrap());
        assert!(std::fs::read_to_string(manifest_path(&root.models())).unwrap().contains("file_digests"));
    }

    #[test]
    fn a_non_runtime_install_records_no_file_map() {
        let root = TestRoot::new();
        root.write("models/vad/silero_vad.onnx", b"synthetic vad");
        record_install(&root.models(), entry("vad-silero").unwrap()).unwrap();
        assert!(load_manifest(&root.models()).entries["vad-silero"].file_digests.is_empty());
        assert!(!std::fs::read_to_string(manifest_path(&root.models())).unwrap().contains("file_digests"));
    }

    #[test]
    fn verify_checks_every_file_of_an_untouched_runtime() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(report.ok, "{}", report.message);
        assert_eq!(report.message, "OK — 4 files, 1 MB, all unchanged since install.");
        assert_eq!(report.sha256, Some(sha256_file(&dir.join("whisper-cli.exe")).unwrap()));
    }

    #[test]
    fn verify_fails_when_a_runtime_dll_changes() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        std::fs::write(dir.join("ggml.dll"), b"whisper ggmL").unwrap();
        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(!report.ok);
        assert_eq!(
            report.message,
            "Runtime folder has changed since install: 1 of 4 changed (ggml.dll). Reinstall it."
        );
        // The launcher itself is untouched, and still reported.
        assert_eq!(report.sha256, Some(sha256_file(&dir.join("whisper-cli.exe")).unwrap()));
    }

    #[test]
    fn verify_fails_when_a_recorded_runtime_file_is_deleted() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        std::fs::remove_file(dir.join("whisper.dll")).unwrap();
        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(!report.ok);
        assert_eq!(
            report.message,
            "Runtime folder has changed since install: 1 file missing (whisper.dll). Reinstall it."
        );
    }

    #[test]
    fn verify_fails_when_a_new_dll_or_exe_appears_in_a_runtime_folder() {
        for planted in ["version.dll", "helper.EXE"] {
            let root = TestRoot::new();
            let dir = installed_whisper(&root);
            std::fs::write(dir.join(planted), b"planted").unwrap();
            let report = verify(&root.models(), "runtime-whisper").unwrap();
            assert!(!report.ok, "{planted}");
            assert_eq!(
                report.message,
                format!("Runtime folder has changed since install: 1 new executable(s) or DLL(s) not there at install ({planted}). Reinstall it.")
            );
        }
    }

    #[test]
    fn verify_names_a_few_offenders_and_counts_the_rest() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        for name in ["ggml-base.dll", "ggml.dll", "whisper-cli.exe", "whisper.dll"] {
            std::fs::write(dir.join(name), b"tampered").unwrap();
        }
        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(!report.ok);
        assert!(
            report.message.contains("4 of 4 changed (ggml-base.dll, ggml.dll, whisper-cli.exe and 1 more)"),
            "{}",
            report.message
        );
    }

    #[test]
    fn verify_reports_but_passes_a_new_non_binary_file_in_a_runtime_folder() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        std::fs::write(dir.join("notes.txt"), b"left by someone").unwrap();
        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(report.ok, "{}", report.message);
        assert_eq!(
            report.message,
            "OK — 4 files, 1 MB, all unchanged since install. The folder also holds 1 file that was not there at install (notes.txt); it is not an executable or DLL, so this does not fail the check."
        );
    }

    #[test]
    fn a_runtime_recorded_before_file_maps_falls_back_to_its_launcher_with_a_note() {
        let root = TestRoot::new();
        let dir = installed_whisper(&root);
        // Strip the map, as a build before this change would have written it.
        let mut manifest = load_manifest(&root.models());
        manifest.entries.get_mut("runtime-whisper").unwrap().file_digests.clear();
        save_manifest(&root.models(), &manifest).unwrap();

        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(report.ok, "{}", report.message);
        assert_eq!(
            report.message,
            "OK — 1 MB, digest unchanged since install. Only the launcher, whisper-cli.exe, could be checked — not the other 3 file(s) in its 1 MB folder, because it was installed before Cadence recorded a digest for every file. Reinstalling it records one for each."
        );
        // A changed DLL goes unseen this way — which is why the note is there.
        std::fs::write(dir.join("ggml.dll"), b"tampered").unwrap();
        assert!(verify(&root.models(), "runtime-whisper").unwrap().ok);
        // A changed launcher is still caught.
        std::fs::write(dir.join("whisper-cli.exe"), b"tampered").unwrap();
        let report = verify(&root.models(), "runtime-whisper").unwrap();
        assert!(!report.ok);
        assert!(report.message.starts_with("File has changed since it was installed."), "{}", report.message);
    }

    #[test]
    fn install_scratch_parent_is_removed_only_when_empty() {
        let root = TestRoot::new();
        let models = root.models();
        root.write("models/_tmp/runtime-whisper/extract/whisper-cli.exe", b"scratch");
        remove_scratch(&models, "runtime-whisper");
        assert!(!models.join("_tmp").exists(), "an empty _tmp is removed");

        root.write("models/_tmp/runtime-whisper/download.zip", b"scratch");
        root.write("models/_tmp/qwen3-8b-q4/partial.gguf", b"another install's download");
        remove_scratch(&models, "runtime-whisper");
        assert!(!models.join("_tmp/runtime-whisper").exists());
        assert!(models.join("_tmp/qwen3-8b-q4/partial.gguf").exists(), "another id's scratch is kept");

        // Nothing to remove is not an error, and leaves the models folder alone.
        remove_scratch(&models, "runtime-llama");
        assert!(models.exists());
    }

    #[test]
    fn copy_binaries_reports_the_flattened_names_it_copied() {
        let root = TestRoot::new();
        root.write("archive/release/bin/whisper-cli.exe", b"exe");
        root.write("archive/release/bin/ggml.dll", b"dll");
        root.write("archive/release/lib/ggml.dll", b"dll again");
        root.write("archive/release/README.md", b"not a binary");
        let dest = Runtime::Whisper.entry().dir_in(&root.models());
        std::fs::create_dir_all(&dest).unwrap();
        let names = copy_binaries(&root.0.join("archive"), &dest).unwrap();
        assert_eq!(names, ["ggml.dll", "whisper-cli.exe"]);
        assert!(!dest.join("README.md").exists());
    }

    fn legacy_keys(models: &Path) -> Vec<String> {
        find_retired(models)
            .into_iter()
            .filter(|m| m.key == LEGACY_RUNTIME_FILES_KEY)
            .map(|m| m.key)
            .collect()
    }

    #[test]
    fn the_old_flat_files_are_offered_only_once_every_runtime_has_its_own_folder() {
        let root = TestRoot::new();
        let models = root.models();
        let mb = 1024 * 1024;
        // An upgraded machine: the old flat layout, all three runtimes in it.
        root.write("models/bin/whisper-cli.exe", b"w");
        root.write("models/bin/llama-server.exe", b"l");
        root.write("models/bin/sherpa-onnx-offline-speaker-diarization.exe", b"s");
        root.write("models/bin/ggml.dll", &vec![0u8; 2 * mb]);
        assert!(legacy_keys(&models).is_empty());

        // Two reinstalled, one still running from the flat folder: its files
        // there are still in use, so nothing is offered or removable.
        root.write("models/bin/whisper/whisper-cli.exe", b"w");
        root.write("models/bin/whisper/ggml.dll", b"whisper ggml");
        root.write("models/bin/llama/llama-server.exe", b"l");
        assert!(Runtime::Sherpa.entry().uses_legacy_layout(&models));
        assert!(legacy_keys(&models).is_empty());
        assert!(remove_retired(&models, LEGACY_RUNTIME_FILES_KEY).is_err());
        assert!(models.join("bin/ggml.dll").exists());

        // A runtime not installed anywhere also holds it back.
        std::fs::remove_file(models.join("bin/sherpa-onnx-offline-speaker-diarization.exe")).unwrap();
        assert!(legacy_keys(&models).is_empty());
        root.write("models/bin/sherpa-onnx-offline-speaker-diarization.exe", b"s");

        // The last one reinstalled: one grouped item for every flat file.
        root.write("models/bin/sherpa/sherpa-onnx-offline-speaker-diarization.exe", b"s");
        root.write("models/bin/sherpa/onnxruntime.dll", b"ort");
        root.write("models/bin/stray-folder/nested.dll", b"not top-level");
        let found = find_retired(&models);
        let item = found.iter().find(|m| m.key == LEGACY_RUNTIME_FILES_KEY).expect("grouped item");
        assert_eq!(found.iter().filter(|m| m.key.starts_with("bin")).count(), 1, "one item, not one per file");
        assert!(item.from_previous_version);
        assert_eq!(item.size_mb, 3, "2 MB + three tiny launchers, rounded up");
        assert_eq!(PathBuf::from(&item.path), models.join("bin"));

        let removed = remove_retired(&models, LEGACY_RUNTIME_FILES_KEY).unwrap();
        assert_eq!(removed.key, LEGACY_RUNTIME_FILES_KEY);
        for flat in ["whisper-cli.exe", "llama-server.exe", "sherpa-onnx-offline-speaker-diarization.exe", "ggml.dll"] {
            assert!(!models.join("bin").join(flat).exists(), "{flat} survived");
        }
        // Every per-runtime folder, and anything else that is not a top-level
        // file, is untouched.
        assert_eq!(std::fs::read(models.join("bin/whisper/ggml.dll")).unwrap(), b"whisper ggml");
        assert!(models.join("bin/sherpa/onnxruntime.dll").exists());
        assert!(models.join("bin/stray-folder/nested.dll").exists());
        for runtime in Runtime::ALL {
            assert!(runtime.entry().is_installed(&models));
            assert!(!runtime.entry().uses_legacy_layout(&models));
        }
        assert!(legacy_keys(&models).is_empty(), "nothing left to offer");
        assert!(remove_retired(&models, LEGACY_RUNTIME_FILES_KEY).is_err());
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // ------------------------------------------------- bundled runtimes

    /// What an installer would put beside `Cadence.exe`: `install/runtimes/` with
    /// each listed runtime's launcher, its other executables and a DLL, plus
    /// the build-time `manifest.json` covering them.
    fn bundled_install(root: &TestRoot, runtimes: &[Runtime]) -> PathBuf {
        let bundled = root.0.join("install").join(BUNDLED_RUNTIMES_DIR);
        let mut manifest = serde_json::Map::new();
        for runtime in runtimes {
            let folder = runtime.folder();
            let mut files = serde_json::Map::new();
            for name in [runtime.entry().expect_relative, "ggml.dll"] {
                let bytes = format!("bundled {folder} {name}");
                let path = root.write(&format!("install/runtimes/{folder}/{name}"), bytes.as_bytes());
                files.insert(name.to_string(), sha256_file(&path).unwrap().into());
            }
            manifest.insert(
                folder.to_string(),
                serde_json::json!({ "version": format!("{folder} test build"), "files": files }),
            );
        }
        std::fs::write(
            bundled.join(BUNDLED_MANIFEST),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        bundled
    }

    fn roots<'a>(models: &'a Path, bundled: &'a Path) -> Roots<'a> {
        Roots { models, bundled: Some(bundled) }
    }

    #[test]
    fn runtime_folder_names_match_their_app_data_folders() {
        for runtime in Runtime::ALL {
            assert_eq!(runtime.entry().subdir, format!("{RUNTIMES_DIR}/{}", runtime.folder()));
            assert_eq!(Runtime::from_id(runtime.id()), Some(runtime));
        }
        assert_eq!(Runtime::from_id("vad-silero"), None);
    }

    #[test]
    fn a_bundled_runtime_takes_precedence_over_its_own_folder_and_the_legacy_one() {
        let root = TestRoot::new();
        let models = root.models();
        let bin = models.join("bin");
        // whisper.cpp in all three places; llama.cpp in its own folder and
        // the flat one; sherpa-onnx only in the flat one.
        root.write("models/bin/whisper-cli.exe", b"legacy");
        root.write("models/bin/whisper/whisper-cli.exe", b"own");
        root.write("models/bin/llama-server.exe", b"legacy");
        root.write("models/bin/llama/llama-server.exe", b"own");
        root.write("models/bin/sherpa-onnx-offline-speaker-diarization.exe", b"legacy");
        let bundled = bundled_install(&root, &[Runtime::Whisper]);
        let at = roots(&models, &bundled);

        assert_eq!(runtime_dir_at(at, Runtime::Whisper), bundled.join("whisper"));
        assert_eq!(runtime_dir_at(at, Runtime::Llama), bin.join("llama"));
        assert_eq!(runtime_dir_at(at, Runtime::Sherpa), bin);
        assert_eq!(
            runtime_dir_with_bundled(&models, Some(&bundled), Runtime::Whisper),
            bundled.join("whisper")
        );
        let whisper = Runtime::Whisper.entry();
        assert_eq!(whisper.path_at(at), bundled.join("whisper").join("whisper-cli.exe"));
        assert!(!whisper.uses_legacy_layout_at(at));
        assert!(Runtime::Sherpa.entry().uses_legacy_layout_at(at));

        // A bundled folder without its launcher does not count: the app-data
        // copy is used, never a mix of the two.
        std::fs::remove_file(bundled.join("whisper").join("whisper-cli.exe")).unwrap();
        assert_eq!(runtime_dir_at(at, Runtime::Whisper), bin.join("whisper"));
        // No bundled root at all behaves exactly as before.
        root.write("install/runtimes/whisper/whisper-cli.exe", b"back");
        let no_bundle = Roots { models: &models, bundled: None };
        assert_eq!(runtime_dir_at(no_bundle, Runtime::Whisper), bin.join("whisper"));
    }

    #[test]
    fn a_bundled_runtime_is_installed_and_never_a_missing_download() {
        let root = TestRoot::new();
        let models = root.models();
        let bundled = bundled_install(&root, &Runtime::ALL);
        let at = roots(&models, &bundled);

        let missing = missing_required_ids_at(at);
        for runtime in Runtime::ALL {
            assert!(runtime.entry().is_installed_at(at));
            assert!(!missing.contains(&runtime.id()), "{} still offered for download", runtime.id());
        }
        // First-run setup is model data only.
        assert!(missing.iter().all(|id| entry(id).unwrap().kind != ModelKind::Runtime));
        let runtimes_mb: u64 = Runtime::ALL.iter().map(|r| r.entry().approx_size_mb).sum();
        let everything: u64 = required_ids().iter().map(|id| entry(id).unwrap().approx_size_mb).sum();
        assert_eq!(required_download_mb_at(at), everything - runtimes_mb);

        let status = status_all_at(at);
        let whisper = status.iter().find(|s| s.id == "runtime-whisper").unwrap();
        assert!(whisper.installed && whisper.bundled);
        assert_eq!(PathBuf::from(&whisper.path), bundled.join("whisper").join("whisper-cli.exe"));
        assert_eq!(whisper.actual_size_mb, 1, "sized as the bundled folder");
        assert_eq!(
            whisper.recorded_sha256,
            Some(sha256_file(&bundled.join("whisper").join("whisper-cli.exe")).unwrap())
        );
        let vad = status.iter().find(|s| s.id == "vad-silero").unwrap();
        assert!(!vad.bundled && !vad.installed);
    }

    #[test]
    fn installing_a_bundled_runtime_is_refused_before_any_download() {
        let root = TestRoot::new();
        let models = root.models();
        let bundled = bundled_install(&root, &[Runtime::Whisper]);
        let calls = std::cell::Cell::new(0);
        let err = install_with_bundled(&models, Some(&bundled), "runtime-whisper", &|_| calls.set(calls.get() + 1))
            .unwrap_err();
        assert_eq!(
            err,
            "whisper.cpp (speech-to-text runtime) is included with Cadence — reinstall Cadence to repair it."
        );
        assert_eq!(calls.get(), 0, "no progress, so nothing started");
        assert!(!models.join("_tmp").exists());
        assert!(!models.join("bin").exists());
        // Only the bundled runtime is refused; the rest stay installable.
        assert!(install_refusal_with(Some(&bundled), "runtime-llama").is_none());
        assert!(install_refusal_with(Some(&bundled), "vad-silero").is_none());
        assert!(install_refusal_with(None, "runtime-whisper").is_none());
    }

    #[test]
    fn verify_checks_a_bundled_runtime_against_its_manifest() {
        let root = TestRoot::new();
        let models = root.models();
        let bundled = bundled_install(&root, &[Runtime::Whisper]);
        let at = roots(&models, &bundled);
        let dir = bundled.join("whisper");
        let report = verify_at(at, "runtime-whisper").unwrap();
        assert!(report.ok, "{}", report.message);
        assert_eq!(
            report.message,
            format!(
                "OK — 2 files, 1 MB, all unchanged since this version of Cadence was built (whisper test build). This is the copy included with Cadence, in {}.",
                dir.display()
            )
        );
        assert_eq!(report.sha256, Some(sha256_file(&dir.join("whisper-cli.exe")).unwrap()));
        // An app-data copy of the same runtime is not what is checked.
        root.write("models/bin/whisper/whisper-cli.exe", b"downloaded");
        assert!(verify_at(at, "runtime-whisper").unwrap().message.contains("included with Cadence"));
    }

    #[test]
    fn verify_fails_a_changed_missing_or_planted_bundled_file() {
        let changed = |dir: &Path| std::fs::write(dir.join("ggml.dll"), b"tampered").unwrap();
        let missing = |dir: &Path| std::fs::remove_file(dir.join("ggml.dll")).unwrap();
        let planted = |dir: &Path| std::fs::write(dir.join("version.dll"), b"planted").unwrap();
        let cases: [(&dyn Fn(&Path), &str); 3] = [
            (&changed, "1 of 2 changed (ggml.dll)"),
            (&missing, "1 file missing (ggml.dll)"),
            (&planted, "1 new executable(s) or DLL(s) not there at install (version.dll)"),
        ];
        for (damage, problem) in cases {
            let root = TestRoot::new();
            let models = root.models();
            let bundled = bundled_install(&root, &[Runtime::Whisper]);
            let dir = bundled.join("whisper");
            damage(&dir);
            let report = verify_at(roots(&models, &bundled), "runtime-whisper").unwrap();
            assert!(!report.ok, "{problem}");
            assert_eq!(
                report.message,
                format!(
                    "The runtime included with Cadence ({}) has changed since it was installed: {problem}. Reinstall Cadence to repair it.",
                    dir.display()
                )
            );
        }
    }

    #[test]
    fn verify_passes_but_notes_a_new_non_binary_file_in_a_bundled_runtime() {
        let root = TestRoot::new();
        let models = root.models();
        let bundled = bundled_install(&root, &[Runtime::Whisper]);
        std::fs::write(bundled.join("whisper").join("notes.txt"), b"x").unwrap();
        let report = verify_at(roots(&models, &bundled), "runtime-whisper").unwrap();
        assert!(report.ok, "{}", report.message);
        assert!(report.message.ends_with("The folder also holds 1 file that was not there at install (notes.txt); it is not an executable or DLL, so this does not fail the check."), "{}", report.message);
    }

    #[test]
    fn verify_says_so_when_a_bundled_runtime_has_no_usable_manifest() {
        let root = TestRoot::new();
        let models = root.models();
        let bundled = bundled_install(&root, &[Runtime::Whisper]);
        let at = roots(&models, &bundled);
        let manifest = bundled.join(BUNDLED_MANIFEST);

        // No manifest at all: a development build. Present, and said plainly
        // that nothing could be checked.
        std::fs::remove_file(&manifest).unwrap();
        let report = verify_at(at, "runtime-whisper").unwrap();
        assert!(report.ok);
        assert!(report.message.contains("carries no runtime manifest"), "{}", report.message);
        assert!(report.message.contains("cannot be checked"), "{}", report.message);
        assert!(!report.message.contains("unchanged"), "{}", report.message);

        // A manifest that does not list this runtime, or cannot be read, is
        // a failed check on a release install.
        std::fs::write(&manifest, r#"{"llama":{"version":"x","files":{}}}"#).unwrap();
        let report = verify_at(at, "runtime-whisper").unwrap();
        assert!(!report.ok);
        assert!(report.message.contains("lists no whisper runtime"), "{}", report.message);
        std::fs::write(&manifest, b"not json").unwrap();
        let report = verify_at(at, "runtime-whisper").unwrap();
        assert!(!report.ok);
        assert!(report.message.contains("is unreadable"), "{}", report.message);
        assert!(report.message.ends_with("Reinstall Cadence to repair it."), "{}", report.message);
    }

    #[test]
    fn the_bundled_manifest_tolerates_a_bom_and_extra_fields() {
        let root = TestRoot::new();
        let bundled = root.0.join("runtimes");
        std::fs::create_dir_all(&bundled).unwrap();
        std::fs::write(
            bundled.join(BUNDLED_MANIFEST),
            "\u{feff}{\"sherpa\":{\"version\":\"v\",\"source_url\":\"u\",\"archive_sha256\":\"a\",\"files\":{\"x.dll\":\"d\"}}}",
        )
        .unwrap();
        let manifest = load_bundled_manifest(&bundled).unwrap().unwrap();
        assert_eq!(manifest["sherpa"].files["x.dll"], "d");
        assert!(load_bundled_manifest(&root.0.join("nowhere")).unwrap().is_none());
    }

    /// An upgraded machine: all three runtimes downloaded into their own
    /// folders, leftover flat files, a stale `.old`, and manifest records.
    fn machine_with_downloaded_runtimes(root: &TestRoot) {
        for runtime in Runtime::ALL {
            let folder = runtime.folder();
            root.write(&format!("models/bin/{folder}/{}", runtime.entry().expect_relative), b"downloaded");
            root.write(&format!("models/bin/{folder}/ggml.dll"), &vec![0u8; 1024 * 1024]);
        }
        root.write("models/bin/whisper.old/whisper-cli.exe", b"interrupted reinstall");
        root.write("models/bin/ggml-cpu-haswell.dll", b"old flat file");
        root.write("models/bin/stray-folder/nested.dll", b"not ours to judge");
        let mut manifest = Manifest::default();
        for runtime in Runtime::ALL {
            root.record(&mut manifest, runtime.id(), &root.models().join("bin").to_string_lossy());
        }
        root.record(&mut manifest, "vad-silero", "x");
        save_manifest(&root.models(), &manifest).unwrap();
    }

    fn runtime_items(found: &[RetiredModel]) -> Vec<&str> {
        found.iter().filter(|m| m.key.starts_with("bin")).map(|m| m.key.as_str()).collect()
    }

    #[test]
    fn downloaded_runtimes_become_one_removable_item_only_when_all_three_are_bundled() {
        let root = TestRoot::new();
        let models = root.models();
        machine_with_downloaded_runtimes(&root);

        // Two of three bundled: llama.cpp still runs from app data, so the
        // folders are not offered; the old flat files are, as before.
        let bundled = bundled_install(&root, &[Runtime::Whisper, Runtime::Sherpa]);
        let at = roots(&models, &bundled);
        assert_eq!(runtime_items(&find_retired_at(at)), [LEGACY_RUNTIME_FILES_KEY]);
        assert!(remove_retired_at(at, UNUSED_RUNTIMES_KEY).is_err());
        assert!(models.join("bin/llama/llama-server.exe").exists());

        // All three bundled: one grouped item covering folders and flat files.
        let bundled = bundled_install(&root, &Runtime::ALL);
        let at = roots(&models, &bundled);
        let found = find_retired_at(at);
        assert_eq!(runtime_items(&found), [UNUSED_RUNTIMES_KEY], "one item, not one per folder");
        let item = found.iter().find(|m| m.key == UNUSED_RUNTIMES_KEY).unwrap();
        assert_eq!(item.label, "downloaded runtimes no longer used");
        assert!(item.from_previous_version);
        assert_eq!(item.size_mb, 4, "3 MB of DLLs plus small files, rounded up");
        assert_eq!(PathBuf::from(&item.path), models.join("bin"));
        assert!(remove_retired_at(at, LEGACY_RUNTIME_FILES_KEY).is_err(), "only one key is offered");

        let bundled_before: Vec<(String, Vec<u8>)> = files_in(&bundled)
            .unwrap()
            .into_iter()
            .map(|(name, path)| (name, std::fs::read(path).unwrap()))
            .collect();
        let removed = remove_retired_at(at, UNUSED_RUNTIMES_KEY).unwrap();
        assert_eq!(removed.key, UNUSED_RUNTIMES_KEY);
        for gone in ["whisper", "llama", "sherpa", "whisper.old", "ggml-cpu-haswell.dll"] {
            assert!(!models.join("bin").join(gone).exists(), "{gone} survived");
        }
        // Confined: what is not a known runtime folder stays, and the bundled
        // copy is byte-for-byte untouched.
        assert!(models.join("bin/stray-folder/nested.dll").exists());
        let bundled_after: Vec<(String, Vec<u8>)> = files_in(&bundled)
            .unwrap()
            .into_iter()
            .map(|(name, path)| (name, std::fs::read(path).unwrap()))
            .collect();
        assert_eq!(bundled_before, bundled_after);
        for runtime in Runtime::ALL {
            assert!(runtime.entry().is_installed_at(at), "{} lost", runtime.id());
        }
        // Records for the deleted folders are dropped; the rest are kept.
        let manifest = load_manifest(&models);
        assert!(Runtime::ALL.iter().all(|r| !manifest.entries.contains_key(r.id())));
        assert!(manifest.entries.contains_key("vad-silero"));
        assert!(runtime_items(&find_retired_at(at)).is_empty(), "nothing left to offer");
    }

    #[test]
    fn unused_runtime_removal_never_touches_a_bundled_folder_inside_bin() {
        // A pathological layout: the "bundled" root is the app-data bin
        // folder itself. The runtime folders are then the bundled copies and
        // must never be listed.
        let root = TestRoot::new();
        let models = root.models();
        machine_with_downloaded_runtimes(&root);
        let bundled = models.join("bin");
        let at = roots(&models, &bundled);
        assert!(every_runtime_bundled(at));
        assert!(unused_runtime_folders(at).is_empty());
        if let Some(item) = find_retired_at(at).into_iter().find(|m| m.key == UNUSED_RUNTIMES_KEY) {
            remove_retired_at(at, &item.key).unwrap();
        }
        for runtime in Runtime::ALL {
            assert!(bundled.join(runtime.folder()).join(runtime.entry().expect_relative).exists());
        }
    }

    #[cfg(windows)]
    #[test]
    fn unused_runtime_removal_does_not_follow_a_junctioned_runtime_folder() {
        let root = TestRoot::new();
        let models = root.models();
        let bundled = bundled_install(&root, &Runtime::ALL);
        let outside = root.write("outside/keep.dll", b"must survive");
        std::fs::create_dir_all(models.join("bin")).unwrap();
        let link = models.join("bin").join("llama");
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(outside.parent().unwrap())
            .output()
            .unwrap();
        if !status.status.success() {
            return; // junctions unavailable here; the symlink check is covered by is_real_child_dir
        }
        let at = roots(&models, &bundled);
        assert!(!unused_runtime_folders(at).contains(&link));
        if let Some(item) = find_retired_at(at).into_iter().find(|m| m.key == UNUSED_RUNTIMES_KEY) {
            let _ = remove_retired_at(at, &item.key);
        }
        assert_eq!(std::fs::read(&outside).unwrap(), b"must survive");
        let _ = std::fs::remove_dir(&link);
    }

    const STALE: &str = r"C:\Users\someone\AppData\Roaming\com.echomind.app\models";

    /// The upgraded dev machine from roadmap risk 17, in miniature.
    fn upgraded_machine() -> TestRoot {
        let root = TestRoot::new();
        root.write("models/llm/Qwen3-4B-Q4_K_M.gguf", b"old llm");
        root.write("models/stt/ggml-small.bin", b"catalog stt");
        root.write("models/vad/silero_vad.onnx", b"catalog vad");
        root.write("models/vad/ggml-silero-v5.1.2.bin", b"hand-placed");
        root.write("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", b"seg");
        root.write("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/README.md", b"readme");
        root.write("models/bin/ggml-cpu-haswell.dll", b"runtime dll");
        root.write("models/bench/llm/Qwen3-4B-Q4_K_M.gguf", b"bench candidate");
        root.write("models/_tmp/whisper-small/partial.bin", b"partial");
        let mut manifest = Manifest::default();
        root.record(&mut manifest, "qwen3-4b-q4", &format!(r"{STALE}\llm\Qwen3-4B-Q4_K_M.gguf"));
        root.record(&mut manifest, "whisper-small", &format!(r"{STALE}\stt\ggml-small.bin"));
        std::fs::write(
            manifest_path(&root.models()),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        root
    }

    #[test]
    fn retired_models_are_found_on_disk_and_named_from_the_manifest() {
        let root = upgraded_machine();
        let found = find_retired(&root.models());
        let keys: Vec<&str> = found.iter().map(|m| m.key.as_str()).collect();
        assert_eq!(keys, ["llm/Qwen3-4B-Q4_K_M.gguf", "vad/ggml-silero-v5.1.2.bin"]);

        let qwen = &found[0];
        assert_eq!(qwen.id.as_deref(), Some("qwen3-4b-q4"));
        assert_eq!(qwen.label, "Qwen3-4B");
        assert!(qwen.from_previous_version);
        assert_eq!(qwen.size_mb, 1, "a few bytes still rounds up to 1 MB");
        assert!(Path::new(&qwen.path).starts_with(root.models()));

        let unknown = &found[1];
        assert_eq!(unknown.id, None);
        assert_eq!(unknown.label, "ggml-silero-v5.1.2.bin");
        assert!(!unknown.from_previous_version);
    }

    #[test]
    fn a_manifest_id_names_a_file_no_retired_entry_lists() {
        let root = TestRoot::new();
        root.write("models/llm/some-old-model.gguf", b"x");
        let mut manifest = Manifest::default();
        root.record(&mut manifest, "some-old-model", &format!(r"{STALE}\llm\some-old-model.gguf"));
        save_manifest(&root.models(), &manifest).unwrap();
        let found = find_retired(&root.models());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id.as_deref(), Some("some-old-model"));
        assert!(found[0].from_previous_version);
    }

    #[test]
    fn a_manifest_record_alone_lists_nothing() {
        // The record points outside the models folder and nothing is on disk
        // inside it: there is nothing to report and nothing to delete.
        let root = TestRoot::new();
        let outside = root.write("com.echomind.app/models/llm/Qwen3-4B-Q4_K_M.gguf", b"elsewhere");
        let mut manifest = Manifest::default();
        root.record(&mut manifest, "qwen3-4b-q4", &outside.to_string_lossy());
        save_manifest(&root.models(), &manifest).unwrap();
        assert!(find_retired(&root.models()).is_empty());
        assert!(remove_retired(&root.models(), "llm/Qwen3-4B-Q4_K_M.gguf").is_err());
        assert!(outside.exists());
    }

    #[test]
    fn removing_a_retired_model_deletes_only_it_and_its_record() {
        let root = upgraded_machine();
        let models = root.models();
        let removed = remove_retired(&models, "llm/Qwen3-4B-Q4_K_M.gguf").unwrap();
        assert_eq!(removed.id.as_deref(), Some("qwen3-4b-q4"));
        assert!(!models.join("llm/Qwen3-4B-Q4_K_M.gguf").exists());
        // The llm folder itself stays, as do the catalog models, the bench
        // harness's copy of the same file name, and everything else.
        assert!(models.join("llm").is_dir());
        assert!(models.join("stt/ggml-small.bin").exists());
        assert!(models.join("vad/silero_vad.onnx").exists());
        assert!(models.join("vad/ggml-silero-v5.1.2.bin").exists());
        assert!(models.join("bench/llm/Qwen3-4B-Q4_K_M.gguf").exists());
        assert!(models.join("bin/ggml-cpu-haswell.dll").exists());

        let manifest = load_manifest(&models);
        assert!(!manifest.entries.contains_key("qwen3-4b-q4"));
        assert!(manifest.entries.contains_key("whisper-small"));
        let keys: Vec<String> = find_retired(&models).into_iter().map(|m| m.key).collect();
        assert_eq!(keys, ["vad/ggml-silero-v5.1.2.bin"]);
    }

    #[test]
    fn a_retired_folder_is_removed_whole() {
        let root = TestRoot::new();
        root.write("models/diarization/old-segmentation/model.onnx", b"a");
        root.write("models/diarization/old-segmentation/nested/extra.onnx", b"b");
        let models = root.models();
        assert_eq!(find_retired(&models).len(), 1);
        remove_retired(&models, "diarization/old-segmentation").unwrap();
        assert!(!models.join("diarization/old-segmentation").exists());
        assert!(models.join("diarization").is_dir());
    }

    #[test]
    fn retired_removal_refuses_catalog_bench_runtime_and_escaping_keys() {
        let root = upgraded_machine();
        let models = root.models();
        let outside = root.write("outside.txt", b"not a model");
        for key in [
            "stt/ggml-small.bin",
            "vad/silero_vad.onnx",
            "diarization/sherpa-onnx-pyannote-segmentation-3-0",
            "bench/llm",
            "bench/llm/Qwen3-4B-Q4_K_M.gguf",
            "bin/ggml-cpu-haswell.dll",
            // Not offered here: no runtime has its own folder yet.
            LEGACY_RUNTIME_FILES_KEY,
            "bin/whisper",
            "bin",
            "_tmp/whisper-small",
            "manifest.json",
            "../outside.txt",
            "llm/../../outside.txt",
            "llm/..",
            "llm/.",
            "llm/",
            "/llm/Qwen3-4B-Q4_K_M.gguf",
            r"llm\..\..\outside.txt",
            "C:/outside.txt",
            "llm/C:outside.txt",
            "",
        ] {
            assert!(remove_retired(&models, key).is_err(), "{key:?} was accepted");
            assert!(confined_target(&models, key).is_err(), "{key:?} resolved");
        }
        assert!(outside.exists());
        assert!(models.join("stt/ggml-small.bin").exists());
        assert!(models.join("vad/silero_vad.onnx").exists());
        assert!(models.join("bench/llm/Qwen3-4B-Q4_K_M.gguf").exists());
        assert!(models.join("bin/ggml-cpu-haswell.dll").exists());
        assert!(models.join("_tmp/whisper-small/partial.bin").exists());
        assert!(models.join("llm/Qwen3-4B-Q4_K_M.gguf").exists());
        assert!(manifest_path(&models).exists());
    }

    #[test]
    fn the_confinement_guard_accepts_only_a_direct_child_of_a_model_folder() {
        let root = upgraded_machine();
        let models = root.models();
        let target = confined_target(&models, "llm/Qwen3-4B-Q4_K_M.gguf").unwrap();
        assert!(target.starts_with(std::fs::canonicalize(&models).unwrap()));
        assert!(target.ends_with("llm/Qwen3-4B-Q4_K_M.gguf"));
        // Present, but in a folder the catalog has never used.
        root.write("models/elsewhere/file.bin", b"x");
        assert!(confined_target(&models, "elsewhere/file.bin").is_err());
        // Missing.
        assert!(confined_target(&models, "llm/missing.gguf").is_err());
    }

    #[test]
    fn bench_bin_and_tmp_are_never_scanned() {
        let dirs = scanned_subdirs();
        for skipped in NOT_SCANNED {
            assert!(!dirs.contains(skipped), "{skipped} is scanned");
        }
        assert!(dirs.contains(&"llm") && dirs.contains(&"stt"));
        assert!(!dirs.iter().any(|d| d.starts_with("bin")), "a runtime folder is scanned: {dirs:?}");
    }

    #[test]
    fn no_retired_entry_collides_with_the_catalog() {
        for retired in RETIRED {
            assert!(entry(retired.id).is_none(), "{} is back in the catalog", retired.id);
            assert!(!owned_by_catalog(retired.subdir, retired.file), "{}", retired.file);
        }
    }

    #[test]
    fn saving_the_manifest_points_stale_paths_at_the_current_models_folder() {
        let root = upgraded_machine();
        let models = root.models();
        let mut manifest = load_manifest(&models);
        let current = models.join("vad").join("silero_vad.onnx");
        root.record(&mut manifest, "vad-silero", &current.to_string_lossy());
        root.record(&mut manifest, "odd", r"C:\nowhere\file.bin");
        root.record(
            &mut manifest,
            "diarization-segmentation",
            &format!(r"{STALE}\diarization\sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
        );
        save_manifest(&models, &manifest).unwrap();

        let saved = load_manifest(&models);
        let file = |id: &str| PathBuf::from(&saved.entries[id].file);
        assert_eq!(file("whisper-small"), models.join("stt").join("ggml-small.bin"));
        assert_eq!(file("qwen3-4b-q4"), models.join("llm").join("Qwen3-4B-Q4_K_M.gguf"));
        assert_eq!(
            file("diarization-segmentation"),
            models
                .join("diarization")
                .join("sherpa-onnx-pyannote-segmentation-3-0")
                .join("model.onnx")
        );
        assert_eq!(file("vad-silero"), current);
        // No `models` folder in the path: nothing to anchor on, left alone.
        assert_eq!(saved.entries["odd"].file, r"C:\nowhere\file.bin");
        // Only `file` changes; verification still compares the same digest.
        assert_eq!(saved.entries["whisper-small"].sha256, "sha-of-whisper-small");
        assert_eq!(saved.entries.len(), manifest.entries.len());
    }

    #[test]
    fn relative_paths_never_climb_out_of_the_models_folder() {
        let models = Path::new(r"C:\data\models");
        let rel = |p: &str| relative_in_models(models, p);
        assert_eq!(rel(r"C:\data\models\llm\x.gguf"), Some(PathBuf::from("llm").join("x.gguf")));
        assert_eq!(rel(r"D:\old\Models\stt\y.bin"), Some(PathBuf::from("stt").join("y.bin")));
        assert_eq!(rel(r"C:\old\models\llm\..\..\escape"), None);
        assert_eq!(rel(r"C:\old\models"), None);
        assert_eq!(rel(r"C:\data\models"), None);
        assert_eq!(rel(r"C:\no\anchor\here.bin"), None);
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
