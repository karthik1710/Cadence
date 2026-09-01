# Cadence — Architecture

How the app is put together, why each major decision was made, and how meeting data is handled. Model choices and measured numbers live in [benchmarks.md](benchmarks.md); status, open risks, and backlog live in [roadmap.md](roadmap.md).

## Shape of the system

One Tauri desktop app at the repo root. No second machine, no networked backend, no client/server split — recording, STT, diarization, alignment, and MoM generation are all local processes orchestrated from Rust in a single binary.

```text
Cadence/
├── src/                  # React UI
├── src-tauri/src/        # Rust: audio capture, pipeline orchestration, storage
├── prompts/              # versioned LLM prompt templates (compiled in via include_str!)
└── docs/                 # architecture.md, benchmarks.md, roadmap.md
```

### Two decisions everything else follows from

**Tauri + React + TypeScript**, over WPF/WinUI. The dev machine already had Node and a Rust toolchain but no .NET SDK, so this started immediately. Windows audio access (device enumeration, level metering, disconnect handling) goes through Rust `cpal` rather than a JS audio API, keeping WASAPI-level control — see `audio.rs`, whose recording stream stays pinned to one dedicated OS thread for its whole lifetime because cpal's WASAPI backend is COM apartment-threaded.

**Rust-native AI runtime, no Python**. Rust spawns and drives native binaries directly: whisper.cpp for STT, sherpa-onnx for diarization, llama.cpp for the LLM. This is what satisfies "the user needs no Python/Node/Docker/WSL" *architecturally*, instead of by freezing and bundling a Python runtime. Model weights (GGUF, ONNX/ggml) install as data files after setup and are never bundled into the installer.

## The processing pipeline

`src-tauri/src/pipeline/`, orchestrated by `pipeline::run`:

```text
original.wav → preprocess (16kHz mono) → diarize → STT → align → transcript.json → MoM → mom.json
```

| Module | Role |
|---|---|
| `paths.rs` | Resolves installed binaries/weights under `<app_data>/models`, failing with a list of what's missing rather than a crash |
| `preprocess.rs` | Downmix + resample to 16kHz mono `processed.wav`; never touches `original.wav` |
| `diarization.rs` | Spawns sherpa-onnx's offline speaker-diarization CLI, parses its stdout intervals |
| `stt.rs` | Spawns whisper.cpp `whisper-cli.exe`, parses its JSON output |
| `codeswitch.rs` | Per-block transcription for mixed English/Tamil audio, and the timestamp remapping back onto the original timeline |
| `align.rs` | Merges STT segments with diarization intervals into the speaker-labelled transcript |
| `chunk.rs` | Speaker-aware transcript chunking for MoM generation |
| `mom.rs` | Extraction → synthesis → validate/repair → grounding |
| `prompts.rs` | Versioned prompt loading, with an app-data override for iterating without a rebuild |
| `llm/` | `LlmProvider` trait, `LlamaCppProvider` driving `llama-server.exe`, and the loopback HTTP client |
| `models.rs` | Serde types for every artifact, plus MoM schema validation |
| `proc.rs` | `run_measured`: wall time + peak RAM for every native stage, on every real run — not just benchmarks |

Each stage writes its raw output (`diarization.json`, `stt.json`) before the next runs, kept separate from the derived `transcript.json`, so a late failure never costs earlier work.

**Alignment attributes per STT segment by maximum temporal overlap** — a segment containing a speaker change goes wholly to whoever holds most of it, because whisper.cpp's JSON carries no word-level timestamps to split on. `--max-len 80` keeps segments short enough that this is a small error, but it is a real limitation: a fast two-way exchange inside one segment collapses onto the dominant speaker. Segments with no overlapping interval fall back to the nearest speaker; with no diarization at all they become `SPEAKER_UNKNOWN` rather than being dropped. Consecutive same-speaker segments less than 1s apart merge into one readable turn. Stable ids (`SPEAKER_00`) and presentation labels (`Person 1`, first-appearance order) are both assigned here.

`src-tauri/src/bin/pipeline_cli.rs` runs the whole thing headlessly:

```powershell
cd src-tauri
cargo run --bin pipeline_cli -- <audio.wav> [out-dir]
```

### MoM generation

```text
transcript → speaker-aware chunks → per-chunk extraction →
merged facts → (condense if needed) → synthesis → validate/repair → grounding → mom.json
```

**Chunking** never splits a speaker turn mid-sentence, and overlaps one turn at each boundary so a decision straddling a cut is visible on both sides. `chunk::max_chunk_chars` sizes the budget pessimistically (~2 chars/token) because Tamil tokenizes denser than English and a code-switched meeting is a mix of both.

**Extraction** runs one call per chunk (`prompts/extraction_v1.txt`) asking only for facts with timestamps — never prose. Results merge with dedup, keeping the earliest timestamp for anything appearing in overlapping chunks. **Synthesis** (`prompts/synthesis_v1.txt`) turns merged facts into the `Mom` structure; if they still overflow the context window, `mom::condense` synthesizes in batches and folds results back (bounded to 2 rounds), with `trim_to_budget` as a last resort rather than failing outright.

**Validate/repair**: malformed JSON goes through `prompts/repair_json_v1.txt` and is re-parsed once, rather than being accepted as-is or silently discarded. `MomMeta.repaired` records whether that happened, per meeting.

**Grounding** (`mom::ground`) is what makes the output trustworthy rather than merely well-formed. It runs *after* validation and overwrites everything the model is not allowed to invent:

- `participants` comes entirely from the transcript's real speaker labels — never from model output.
- Timestamps outside the recording's actual duration (5s slack for rounding) are **dropped, not clamped**: a wrong evidence link is worse than a missing one.
- Every `owner`/`speaker` is matched case-insensitively against real transcript labels; an invented name becomes `null`. This is where V1's "never identify real people" rule is enforced at the data layer rather than by prompt convention.
- An `action_items[].confidence` outside `high|medium|low` is downgraded to `low`.

Prompts are versioned plain text under `prompts/`, compiled in via `include_str!` so a packaged install has no loose files to lose. Their version strings land in every `mom.json`'s `meta` block alongside model/quantization/runtime, so any minutes can be traced to exactly what produced them.

`LlamaCppProvider` starts `llama-server.exe` when MoM generation begins and kills it via `Drop` when it ends, so no model stays resident between meetings — on a 16 GB machine that's the difference between an idle laptop and a swapping one. `MockLlm` is the scripted stand-in for tests and `--dry-run`.

## Machine memory tiering

The reference machine has 32 GB, but 16 GB laptop support became a hard requirement afterwards: the same build has to behave sensibly on both with no user configuration.

`system::HostInfo::detect()` reads total/available RAM and core count via `GlobalMemoryStatusEx` (Windows API, no crate). `MemoryProfile::for_total_ram_mb` buckets it:

| Profile | Threshold | Default STT | Default LLM | LLM context | LLM RAM budget |
|---|---|---|---|---|---|
| `Low` | < 12 GB | whisper-base | qwen3-4b-q4 | 2048 | 3,500 MB |
| `Balanced` | 12–24 GB (16 GB laptop) | whisper-small | qwen3-4b-q4 | 4096 | 6,000 MB |
| `High` | ≥ 24 GB (32 GB reference) | whisper-small | qwen3-8b-q4 | 8192 | 10,000 MB |

Thresholds sit below nominal sizes (24 not 32, 12 not 16) because Windows reports *usable* physical memory — a 16 GB laptop with shared iGPU memory commonly reports ~15.7 GB.

Every model/thread/context field in `settings::Settings` is `Option<...>`, where `None` means "let the profile decide." The `Settings::effective_*` methods are the single place a preference resolves against the detected machine; `PipelineConfig::from_settings` and `paths::ModelPaths` build on them rather than re-deriving defaults. **Nothing downstream hard-codes a model id or thread count** — a 16 GB and a 32 GB machine are expected to select different models by design, and `RunMetrics`/`bench_cli` record which models a run actually used so their numbers are never silently compared.

Thread caps are per stage, not one shared number: `MAX_STT_THREADS` 24, `MAX_DIARIZATION_THREADS` 8, `MAX_LLM_THREADS` 16. The reasoning is measured — see [benchmarks.md](benchmarks.md).

Before starting `llama-server.exe`, `preflight_memory` checks the selected model's estimated RAM (weights + ~0.15 MB/token KV cache + ~600 MB overhead) against actually-free RAM, reserving a 1.5 GB slice for Windows/webview/file cache. A model that won't fit is refused *before* it OOMs mid-generation, with a message naming what to change in Settings. This is an estimate-based guard, not a hard allocator limit (llama.cpp has none), and it deliberately blocks only models exceeding *available RAM* — not ones merely over the tier's budget that still physically fit.

## Model Manager

`models_manager.rs` is the catalog-driven install/remove/verify system that replaced "run `scripts/setup-models.ps1` and hope" (the script remains a dev-time fallback provisioning the same layout).

`catalog()` is a static list of every installable item — three runtimes (whisper.cpp, llama.cpp, sherpa-onnx) plus weights (three Whisper sizes, Qwen3-4B/8B Q4_K_M, Gemma3-4B Q4_K_M, two diarization ONNX models). Each entry carries its download URL, install path, and the file that must exist for it to count as installed. `required_ids(profile)` derives which subset a given `MemoryProfile` needs, which is how `ModelPaths::missing` and the first-run screen name exactly what's absent instead of a generic "models missing".

`remove()` refuses to delete a model the current profile still needs, so a user can't accidentally leave the app unable to process a meeting.

Downloads shell out to Windows' bundled `curl.exe` and `tar.exe` rather than linking an HTTP crate — see [Privacy and data handling](#privacy-and-data-handling) for why that's structural rather than convenient.

**Integrity is trust-on-first-install.** `install()` computes each file's SHA-256 right after installing and records it in `models/manifest.json`; `verify()` (Settings → Models → health check) recomputes and compares. That catches corruption, truncation, and post-install tampering — but the catalog ships **no pinned, independently-sourced hashes**, so it cannot detect a compromised upstream release or a download that was already bad when the hash was taken. TLS via `curl.exe` covers transit; nothing here independently attests to what a file *should* be. Inventing digests never checked against real upstream files would be worse than having none, so this gap is written down rather than papered over.

## Storage and history

`storage.rs` owns the on-disk layout:

```text
<app_data>/
├── config/settings.json
├── models/                    catalog installs + manifest.json (SHA-256s)
├── database/meetings.db       SQLite index for history: metadata only
├── logs/
└── meetings/<id>/
    ├── metadata.json          authoritative record for this meeting
    ├── original.wav           never modified, never deleted automatically
    ├── processed.wav
    ├── diarization.json       raw stage output
    ├── stt.json               raw stage output
    ├── transcript.json        derived
    ├── mom.json               derived
    ├── metrics.json           per-stage wall time + peak RAM
    └── exports/
```

`metadata.json` is the source of truth; the SQLite `meetings` table is a **rebuildable index** over it, holding only what the history list needs to sort and filter — never full transcript or MoM content. Every metadata write goes through `write_metadata`, which also upserts the index row; a failed upsert is logged, not propagated, because the meeting is already safe on disk. `list_meetings` reads the index, but falls back to `scan_meetings` (walking `meetings/*/metadata.json`) and rebuilds it if the index is empty or unusable — **no meeting can be lost to a corrupted database file.** `recover_interrupted` uses the same scan at startup to mark meetings left mid-recording or mid-processing as failed-but-recoverable, with a message explaining what's still safe.

`delete_meeting` is the only path that removes a recording, and only on explicit user action: the whole `meetings/<id>/` directory plus the SQLite row, in one call. Because the index is a rebuildable cache, there is no secondary copy left behind to forget about.

## Tauri command surface

`processing.rs` exposes the pipeline to the UI: `start_processing` (spawns on a worker thread, returns immediately), `get_processing_status` (polled for stage and progress), `regenerate_mom` (the retry path when only the LLM stage failed — re-uses the saved `transcript.json`), and `check_models` (so the UI warns about missing models *before* a meeting is recorded, not after). `library.rs` adds history/transcript/MoM/export/rename/delete; `configuration.rs` adds host info, settings, and Model Manager commands. Full registration list in `lib.rs`.

Processing status is in memory only (`Mutex<HashMap<meeting_id, MeetingStatus>>`). A restart loses progress but never artifacts — every stage's output is already on disk, and re-running is the recovery path.

## Export

`export.rs` renders Markdown and JSON (PDF/DOCX are out of V1 scope), written into the meeting's own `exports/` directory so deleting the meeting deletes them too. Both apply user speaker renames consistently: a rename is keyed by the stable id (`SPEAKER_00`) and applied to the presentation label wherever it appears, so it can never land on the transcript but not the MoM. Markdown carries a meeting-info header, participants, summary, topics/decisions/questions/risks/discussion each with their evidence, action items with owner/deadline/confidence, and the full transcript; JSON is the same content structured for reuse, tagged `"generatedBy": "Cadence (local processing, no cloud services)"`.

## Privacy and data handling

The privacy model is architectural, not a policy statement: the app cannot send meeting content anywhere because the code paths to do so do not exist in the shipped binary.

**What is captured.** Everything listed in the `meetings/<id>/` layout above — raw audio, per-stage output, derived transcript and minutes, metadata, and per-run metrics (numbers only, no meeting content). None of it is created, read, or transmitted by anything other than the app running on the same machine.

**No telemetry.** No analytics SDK, no crash reporter, no update-check call, anywhere in the pipeline or UI.

**No cloud STT or LLM.** The crate links no HTTP client for AI processing, and `cargo tree | grep reqwest` returning nothing is the check — deliberately kept binary, so "no cloud path snuck in" is a one-line answer rather than something re-argued by reading every module. The old networked-backend client and its UI backend-URL field were deleted in Phase 8.

Exactly two places in the crate contain anything network-shaped, and neither is a general-purpose HTTP client:

- **`models_manager.rs`** shells out to `curl.exe` and `tar.exe` to fetch runtimes and weights from public release URLs. This runs *only* on an explicit install click in Settings → Models (or `setup-models.ps1` at dev time) — never during recording, processing, or MoM generation. It downloads *into* the app; there is no code path from a meeting's audio, transcript, or MoM to this module at all. Shelling out to tools Windows already ships keeps the `cargo tree` check meaningful, which is the actual reason for it.
- **`pipeline/llm/http.rs`** is a hand-rolled HTTP/1.1 client talking to the local `llama-server.exe` over loopback — llama.cpp's own IPC mechanism, the same role stdin/stdout plays for whisper.cpp. `request()` takes a **port, not a host or URL**, and hard-codes `127.0.0.1`, so no parameter or code path can point it off-machine even by mistake. It is hand-rolled rather than `reqwest` for the same reason as above: adding a real HTTP crate would defeat the check, for a need a ~250-line loopback client already meets.

Once models are installed the app is designed to work end to end with the network disconnected. Today that is an architectural property — nothing in the pipeline makes an outbound call — not something a test verifies continuously; an automated offline-mode test is still open.

**What's logged**: meeting id, processing stage, model used, timings, memory, errors. **Not** full transcript text, raw audio, or MoM content. A deeper debug mode exists (`Settings.debug_logging`, off by default) and turning it on is a deliberate, visible choice. One named exception: `logs/llama-server.log` is a diagnostic tail, truncated every run, used only to surface the last few lines if the model server fails to start. llama.cpp does not log prompt or response text at default verbosity, so it holds no meeting content in normal operation.

**Deletion** is user-initiated only — no retention limit, no background cleanup. See `delete_meeting` above.

## Status

Fully local, single-machine, verified end to end from recording through speaker-labelled transcript and MoM generation, with no network path for AI processing.

**Done:** audio capture (1), alignment (4), headless pipeline runner (5), Tauri integration (8), MoM generation (7), transcript/MoM UI (9/10), SQLite history with filesystem rebuild (11), Markdown/JSON export (12), Model Manager (13), and memory tiering (cross-cutting, not a numbered phase).

**Written but never executed:** `bench_cli` (Phases 2/3/6/14). The harness is new; the results are not. STT model size, the diarization cluster threshold, and the memory-tier default LLMs are all provisional — see [benchmarks.md](benchmarks.md) for exactly what is and isn't measured, and [roadmap.md](roadmap.md) for full phase status.
