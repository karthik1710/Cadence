# EchoMind — Architecture

## Repository layout

Single Tauri desktop app, living at the repo root — no second machine, no networked backend:

```text
EchoMind/
├── src/                  # React UI
├── src-tauri/src/        # Rust: audio capture, AI pipeline orchestration
├── docs/                 # architecture.md, models.md, privacy.md, performance.md, model-benchmark.md
├── meeting-ai-app-PLAN.md
└── README.md
```

A JSON-schema-style contract for meeting metadata/transcript/MoM, if wanted for validation, belongs somewhere like `schemas/` at the repo root, since Rust and TypeScript are the only two consumers, both in this repo. Left as an open call for Phase 1, not decided here.

## Windows client tech decision

**Tauri + React + TypeScript**, over the WPF/WinUI alternatives listed in the master prompt.

Rationale: the dev machine has Node.js, npm, and a Rust toolchain (cargo/rustc) already installed, but no .NET SDK. Tauri lets development start immediately without installing a new SDK. Native Windows audio device access (mic selection, level metering, disconnect handling) goes through a Rust backend command (the `cpal` crate) rather than a JS audio API, keeping WASAPI-level control.

## AI runtime decision

**Rust-native orchestration, no Python.** Rust/Tauri spawns and drives local native binaries/libraries directly — whisper.cpp (or sherpa-onnx's Whisper ONNX runner) for STT, llama.cpp for the LLM, an ONNX-runtime-based library for diarization/VAD. See `meeting-ai-app-PLAN.md` §19 for the full rationale: this was chosen specifically so the "user needs no Python/Node/Docker/WSL" packaging requirement (§35) is satisfied architecturally, rather than by freezing and bundling a Python runtime. Model weights (GGUF, ONNX/ggml) are installed as data files post-install, never bundled into the installer.

## Local processing pipeline

Lives in `src-tauri/src/pipeline/`, orchestrated by `pipeline::run` (master prompt §11):

```text
original.wav → preprocess (16kHz mono) → diarize → STT → align → transcript.json
```

| Module | Role |
|---|---|
| `paths.rs` | Resolves the installed binaries/model weights under `<app_data_dir>/models`, failing with a listing of what's missing rather than a crash (§33) |
| `preprocess.rs` | Downmix + linear resample to 16kHz mono `processed.wav`; never touches `original.wav` (§10) |
| `diarization.rs` | Spawns sherpa-onnx's offline speaker-diarization CLI, parses its stdout intervals |
| `stt.rs` | Spawns whisper.cpp `whisper-cli.exe`, parses its JSON output |
| `align.rs` | Merges STT segments with diarization intervals into the canonical speaker-labelled transcript (§13) |
| `models.rs` | Serde types for every artifact, plus MoM schema validation (§15/§16) |
| `proc.rs` | `run_measured`: runs a native stage's process and captures wall time + peak RAM (§29) uniformly, so every real meeting — not only benchmark runs — produces a data point |
| `chunk.rs` | Speaker-aware transcript chunking for MoM generation (§14/§18) |
| `mom.rs` | The MoM pipeline: extraction → synthesis → validate/repair → grounding (§14–§18); see below |
| `prompts.rs` | Versioned prompt template loading with an app-data override (§17) |
| `llm/` | `LlmProvider` trait, `LlamaCppProvider` (drives `llama-server.exe`), and the hand-rolled loopback HTTP client (§5/§19); see below |

Each stage writes its raw output (`diarization.json`, `stt.json`) before the next stage runs, kept separate from the derived `transcript.json`, so a later failure never costs earlier work.

**Alignment approach.** Attribution is per STT segment by maximum temporal overlap — a segment containing a speaker change goes wholly to whoever holds most of it, because whisper.cpp's JSON output carries no word-level timestamps to split on. `--max-len 80` keeps segments short enough that this is a small error. Segments with no overlapping diarization interval (diarization skips non-speech) fall back to the nearest speaker; with no diarization at all they become `SPEAKER_UNKNOWN` rather than being dropped. Consecutive same-speaker segments less than 1s apart merge into one readable turn. Stable ids (`SPEAKER_00`) and presentation labels (`Person 1`, first-appearance order) are both set here (§7).

`src-tauri/src/bin/pipeline_cli.rs` runs the whole thing headlessly for benchmarking and re-runs on test recordings, before Phase 8 wires it into the UI:

```powershell
cd src-tauri
cargo run --bin pipeline_cli -- <audio.wav> [out-dir]
```

`scripts/setup-models.ps1` installs the binaries and weights the pipeline expects into that same `<app_data_dir>/models` layout — dev-time stand-in for the Phase 13 Model Manager.

## Tauri command surface

`src-tauri/src/processing.rs` exposes the pipeline to the UI: `start_processing` (spawns the pipeline on a worker thread and returns immediately), `get_processing_status` (polled for the §12 stage and progress), `regenerate_mom` (the retry path when only the LLM stage failed — `pipeline::run_mom_only`, re-uses the saved `transcript.json`), and `check_models` (lets the UI warn about missing models *before* a meeting is recorded rather than after). `library.rs` adds the history/transcript/MoM/export/rename/delete commands, and `configuration.rs` adds host info, settings, and the Model Manager commands. The full list is registered in `src-tauri/src/lib.rs`.

Processing status lives in memory only — a `Mutex<HashMap<meeting_id, MeetingStatus>>`. An app restart loses progress but never artifacts: every stage's output is already on disk, and re-running the pipeline is the recovery path. On startup, `storage::recover_interrupted` marks any meeting that was mid-recording or mid-processing when the app last closed as failed-but-recoverable (§33), before the UI reads it.

## Machine memory tiering (16 GB laptop support)

The master prompt's reference machine has 32 GB of RAM (§1, §4), but supporting 16 GB laptops became a hard requirement after the fact — the same build has to behave sensibly on both without the user configuring anything. This is what `src-tauri/src/system.rs` and `src-tauri/src/settings.rs` exist for.

`system::HostInfo::detect()` reads total/available physical RAM and logical core count via `GlobalMemoryStatusEx` (Windows API, no crate). `system::MemoryProfile::for_total_ram_mb` buckets that into three tiers:

| Profile | Threshold | Default STT | Default LLM | LLM context | LLM RAM budget |
|---|---|---|---|---|---|
| `Low` | < 12 GB | whisper-base | qwen3-4b-q4 | 2048 | 3,500 MB |
| `Balanced` | 12–24 GB (the 16 GB laptop case) | whisper-small | qwen3-4b-q4 | 4096 | 6,000 MB |
| `High` | ≥ 24 GB (the 32 GB reference machine) | whisper-small | qwen3-8b-q4 | 8192 | 10,000 MB |

Thresholds sit below the nominal sizes (24 GB not 32, 12 GB not 16) because Windows reports *usable* physical memory — a 16 GB laptop with shared iGPU memory commonly reports ~15.7 GB.

`settings::Settings` layers user overrides on top: every model/thread/context field is `Option<...>`, and `None` means "let the profile decide." `Settings::effective_*` methods (`effective_profile`, `effective_stt_model`, `effective_llm_model`, `effective_llm_context`, `effective_stt_threads`, `effective_diarization_threads`, `effective_llm_threads`) are the single place that resolves a preference against the detected machine — `pipeline::PipelineConfig::from_settings` and `pipeline::paths::ModelPaths` both build on these rather than re-deriving defaults. Thread scaling follows the same per-stage caps as the 32 GB reference machine (`MAX_STT_THREADS` 24, `MAX_DIARIZATION_THREADS` 8, `MAX_LLM_THREADS` 16) but naturally comes out lower on a machine with fewer cores.

Before starting `llama-server.exe`, `pipeline::llm::LlamaCppProvider::start` runs a pre-flight check (`preflight_memory`) against the selected model's estimated RAM (`CatalogEntry::estimated_llm_ram_mb`, weights + ~0.15 MB/token KV cache + ~600 MB runtime overhead) versus the host's actually-free RAM, with a 1.5 GB headroom slice for Windows/webview/file cache. A model that will not fit is refused before it OOMs mid-generation, with a message naming what to change in Settings — this is an estimate-based guard, not a hard allocator limit (llama.cpp has none), and it deliberately does not block a model that exceeds the *tier's* budget but still physically fits, only one that exceeds *available RAM*.

This tiering is why nothing else in the pipeline hard-codes "32 GB": `ModelPaths` resolves to a different LLM/STT pair per machine transparently (§5's swappable-provider requirement), and `RunMetrics`/`bench_cli` record which models a given run actually used, so a 16 GB and a 32 GB machine's numbers are never silently compared as if they used the same model.

## Model Manager (§27)

`src-tauri/src/models_manager.rs` is the catalog-driven install/remove/verify system that replaces "run `scripts/setup-models.ps1` and hope" as the primary path (the script remains as a dev-time fallback that provisions the same `<app_data_dir>/models` layout).

**Catalog.** `catalog()` is a static list of every installable item — three runtimes (whisper.cpp, llama.cpp, sherpa-onnx) and their model weights (three Whisper sizes, Qwen3-4B/8B Q4_K_M, Gemma3-4B Q4_K_M, the two diarization ONNX models). Each entry carries its download URL, where it installs under `<app_data_dir>/models`, and the file that must exist for it to count as installed. `required_ids(profile)` derives which subset a given `MemoryProfile` actually needs, which is what `pipeline::paths::ModelPaths::missing` and the first-run setup screen use to say exactly what's absent rather than a generic "models missing" error (§33).

**No HTTP client dependency.** Downloads shell out to `curl.exe` and archive extraction to `tar.exe` (bsdtar) — both ship with Windows 10 1803+/11, so nothing new has to be installed. This is a deliberate structural choice, not a convenience: EchoMind's core invariant is that the crate links no HTTP/network stack for anything AI-related (`cargo tree | grep reqwest` finding nothing), and that invariant is cheap to check specifically because it's binary — either some crate pulls in an HTTP implementation or none does. Shelling out to Windows' own CLI tools for a manual, user-initiated, one-time model download keeps that check meaningful. `pipeline/llm/http.rs` follows the same logic in the opposite direction: it needs to talk to `llama-server.exe` over HTTP as an IPC mechanism (the same role stdin/stdout plays for whisper.cpp and sherpa-onnx), so rather than pull in `reqwest` it hand-rolls a client that can only ever address `127.0.0.1:<port>` — the host isn't even a parameter, only the port is, so there's no code path that could point it anywhere else even by mistake.

**Integrity: trust-on-first-install.** The catalog carries no pinned SHA-256 hashes — inventing digests that were never checked against the real upstream files would be worse than having none. Instead, `install()` computes the SHA-256 of every file right after installing it and records it in `<app_data_dir>/models/manifest.json`. `verify()` recomputes and compares: it catches corruption, truncation, or post-install tampering, but it cannot detect a first download that was already compromised (e.g. a poisoned release asset or a MITM'd `curl.exe` run, though TLS covers the latter). This is documented, not hidden — see `docs/privacy.md`. `remove()` refuses to delete a model the current profile still needs, so a user cannot leave the app unable to process a meeting by accident.

## MoM generation pipeline (§14–§18)

`src-tauri/src/pipeline/mom.rs`, driven by `pipeline::run`'s `generate_minutes` step:

```text
transcript → speaker-aware chunks → per-chunk extraction →
merged facts → (condense if needed) → synthesis → validate/repair →
grounding → mom.json
```

**Chunking** (`pipeline/chunk.rs`) splits the transcript so it fits the model's context window, never splitting a speaker turn mid-sentence and overlapping one turn at each boundary so a decision that straddles a cut is visible on both sides. `chunk::max_chunk_chars` sizes the budget pessimistically (~2 chars/token) because Tamil text tokenizes denser than English and a code-switched meeting (§8) is a mix of both.

**Extraction** runs one LLM call per chunk against `prompts/extraction_v1.txt`, asking only for facts (topics, decisions, action items, open questions, risks, notes) with timestamps — never prose minutes. Per-chunk extractions are merged with dedup (`mom::merge`, `dedupe_evidenced`/`dedupe_actions`), keeping the earliest timestamp for anything that shows up in more than one overlapping chunk.

**Synthesis** turns the merged facts into the actual `Mom` structure (summary, participants, etc.) via `prompts/synthesis_v1.txt`. If the merged facts are still too large for the context window, `mom::condense` runs synthesis over batches and folds the results back in (bounded to 2 rounds), and as a last resort `trim_to_budget` truncates rather than failing outright.

**Validate/repair** (§16): the model's JSON is parsed and schema-validated (`Mom::validate`); malformed output is sent through `prompts/repair_json_v1.txt` and re-parsed once rather than accepted as-is or silently discarded. `MomMeta.repaired` records whether that happened, so it's visible per meeting, not just in aggregate.

**Grounding** (`mom::ground`) is what makes the result trustworthy rather than merely well-formed: it runs *after* validation and rewrites everything the model is not allowed to invent.
- `participants` is overwritten entirely from the transcript's real speaker labels — never taken from the model's output.
- Every timestamp is checked against the meeting's actual duration (`normalize_timestamp`, with a 5s slack for rounding) and dropped — not clamped, not kept — if it falls outside the recording, because a wrong evidence link is worse than a missing one.
- Every `owner`/`speaker` name is matched case-insensitively against the transcript's real presentation labels (`match_label`); a name the model invented (a guessed real name, a label from nowhere) becomes `null` rather than being shown as fact — this is also how V1's "never identify real people" rule (§7) is enforced at the data layer, not just by convention in the prompt.
- An `action_items[].confidence` value outside `high|medium|low` is downgraded to `low` rather than passed through.

Prompts are versioned plain text under `prompts/` at the repo root (`extraction_v1.txt`, `synthesis_v1.txt`, `repair_json_v1.txt`), compiled in via `include_str!` so a packaged install has no loose files to lose, with an optional override at `<app_data_dir>/prompts/<version>.txt` for iterating without a rebuild. The version strings are recorded in every `mom.json`'s `meta` block alongside the model/quantization/runtime, so any generated minutes can be traced back to exactly what produced them (§17).

The LLM itself runs behind the `LlmProvider` trait (`pipeline/llm/mod.rs`); `LlamaCppProvider` drives a per-run `llama-server.exe` (`pipeline/llm/server.rs`) over the loopback HTTP client described above, and `MockLlm` is a scripted stand-in the test suite and CLI `--dry-run` paths use instead. The server is started when MoM generation begins and killed (via `Drop`) when it ends, so no model stays resident in RAM between meetings — the difference on a 16 GB machine between an idle laptop and a swapping one.

## Storage and history (§21, §25)

`src-tauri/src/storage.rs` owns the on-disk layout:

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
    ├── metrics.json           per-stage wall time + peak RAM (§29)
    └── exports/
```

`metadata.json` is the source of truth for a meeting; the SQLite `meetings` table is a **rebuildable index** over it, holding only what the history list needs to sort and filter (title, notes, timestamps, duration, state, has-transcript/has-mom flags) — never full transcript or MoM content (§21: "do not duplicate full transcript data unnecessarily"). Every metadata write goes through `write_metadata`, which also upserts the index row; if that upsert fails it's logged, not propagated, because the meeting itself is already safe on disk. `list_meetings` reads from the index normally, but if the index comes back empty or unusable it falls back to `scan_meetings` (walking `meetings/*/metadata.json` directly) and rebuilds the index from what it finds — no meeting can be lost to a corrupted or missing database file. `recover_interrupted` uses the same scan to catch meetings left `Recording` or mid-processing-state when the app last closed, and marks them `Failed` with a message explaining what's still safe (§33).

`delete_meeting` is the only path that ever removes a recording, only on an explicit user action (§10): it removes the whole `meetings/<id>/` directory (audio, transcript, MoM, exports — everything) and the corresponding SQLite row.

## Export (§26)

`src-tauri/src/export.rs` renders Markdown and JSON (PDF/DOCX remain out of V1 scope, per the plan). Both formats are written into the meeting's own `meetings/<id>/exports/` directory, so deleting the meeting deletes its exports too, and both apply any user-assigned speaker renames (§23) consistently across the transcript and the minutes — a rename is keyed by the stable speaker id (`SPEAKER_00`) and applied to the presentation label (`Person 1`) wherever it appears, so it can never apply to the transcript and not the MoM or vice versa. Markdown includes a meeting-info header, participants, executive summary, topics/decisions/questions/risks/discussion (each with its evidence — timestamp and/or speaker where the grounding step kept one), action items with owner/deadline/confidence, and the full transcript; JSON is the same content structured for programmatic reuse, tagged `"generatedBy": "EchoMind (local processing, no cloud services)"`.

## Status

Fully local, single-machine architecture, verified end-to-end for recording through speaker-labelled transcript and MoM generation, with no network path for AI processing — the crate links no HTTP client for that purpose (the old `backend.rs` and the UI's backend-URL field were deleted in Phase 8; the model-download and loopback-LLM code paths described above are the only network-adjacent code in the app, and neither is a general-purpose HTTP client).

**Done:** audio capture (1), alignment (4), headless pipeline runner (5), Tauri integration (8), MoM generation (7), transcript/MoM UI wired to the backend commands (9/10), SQLite-backed history with filesystem rebuild (11), Markdown/JSON export (12), catalog-driven Model Manager (13), and the machine-memory-tiering system described above (a cross-cutting addition, not a numbered phase).

**Infrastructure that exists but has not been run:** `src-tauri/src/bin/bench_cli.rs` (Phases 2/3/6/14's benchmark harness — `stt`/`diarize`/`llm`/`full` subcommands) is written and unit-tested at the level its component functions allow, but **no one has executed it against real 30/45/60-minute or Tamil/Tanglish audio yet**. Say this plainly: Phase 2 (STT benchmark) and Phase 3 (diarization benchmark) results still do not exist, and Phase 6 (LLM benchmark) results still do not exist either — the harness that will produce them is new, not the results themselves. See `docs/model-benchmark.md`.

**Still unvalidated for the same reason:** the STT model size (`whisper-small`), the diarization cluster threshold (0.8), and the memory-tier default LLMs (Qwen3-4B on 16 GB, Qwen3-8B on 32 GB) are all provisional choices per §5/§6/§7 — none has a real benchmark behind it yet, only the single unlabelled 57.4s sample recorded before this round of work. See `docs/models.md` and `docs/performance.md`. See `meeting-ai-app-PLAN.md` §38 for the full phase status.
