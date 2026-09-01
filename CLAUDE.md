# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cadence is a local-first meeting transcription system: a single Windows desktop app (Tauri 2 + React 19 + TypeScript + Rust) that records a physical meeting from a USB/laptop mic and, entirely on that same machine, runs speech-to-text, speaker diarization, timestamp/speaker alignment, and LLM-based meeting-minutes (MoM) generation. There is no separate backend and no second machine — the app owns the full pipeline and all meeting data locally.

Hard constraints: **no cloud STT, no cloud LLM, no external AI APIs, no telemetry, no networked backend.** The AI runtime is Rust-orchestrated native binaries/libraries only (whisper.cpp, llama.cpp, an ONNX-based diarization/VAD library) — no Python or PyTorch ships in the app. After models are installed, the app must run with the Internet fully disconnected. V1 targets post-meeting processing (5–10 min latency acceptable), not real-time.

There is no separate spec document — the specification is the code plus three living docs under `docs/`, which should be updated as decisions change: `docs/architecture.md` (how the app is built, plus the privacy/data-handling model), `docs/benchmarks.md` (model choices, measured performance, what is *not* yet measured, and how to run the `bench_cli` harness below), and `docs/roadmap.md` (status, open risks, backlog). Read the relevant one before starting work in its area.

Rust unit tests exist under `src-tauri/src/pipeline/` (`cargo test`). There is no frontend test suite.

## How to work in this repo

**Delegate multi-file work to subagents by default.** Anything that touches more
than a couple of files — a feature spanning Rust and the UI, a redesign, a docs
sweep, a cross-cutting refactor — goes to the Agent tool with a self-contained
prompt (file list, constraints, and the verification commands it must run),
rather than being edited file-by-file in the main thread. Reserve direct edits
for genuinely small, single-file changes.

Two rules that matter when doing this, both learned the hard way here:

* **Give each concurrent agent a disjoint file list.** Two agents editing
  `Cargo.toml`/`lib.rs` at once cost a broken build and a killed dev server. If
  a second agent needs a file the first owns, either sequence them or tell the
  second to define its code and hand back the wiring (a `#[tauri::command]`
  compiles fine before it is registered).
* **Verify their work yourself; do not take the report on trust.** Re-run
  `cargo build && cargo test` and `npx tsc --noEmit && npm run build` after an
  agent lands. An agent has reported "tests pass" here while two were failing,
  and another left nine dead-code warnings behind because it could not finish
  its own wiring.

## Commands

### Windows client (repo root)

```powershell
npm install                 # at repo root
npm run tauri dev           # run the full desktop app (spawns Vite on :1420 + Rust)
npm run dev                 # Vite frontend only (no Tauri APIs available)
npm run build               # tsc typecheck + vite build
npm run tauri build         # production Windows bundle
cargo build                 # in src-tauri/ — Rust-only build
cargo test                  # in src-tauri/ — Rust tests

# headless pipeline run (audio → preprocess → diarize → STT → align → MoM), in src-tauri/
cargo run --bin pipeline_cli -- <audio.wav> [out-dir]

# benchmark harness (STT / diarization / LLM / full pipeline), in src-tauri/
# prints Markdown tables meant to be pasted into docs/benchmarks.md
cargo run --release --bin bench_cli -- <stt|diarize|llm|full> <input> [options]

.\scripts\setup-models.ps1   # dev-time fallback; the in-app Model Manager (models_manager.rs) is the primary install path now
```

There is no ESLint / Prettier / JS test runner configured. `npm run build` (via `tsc`) is the only frontend check.

## Architecture notes that span files

**Single-process local pipeline.** Recording, STT, diarization, alignment, and MoM generation all run as local processes/libraries orchestrated from Rust — no HTTP contract, no client/server split, no shared job queue across machines. The old networked-backend client (`backend.rs`) and the UI's backend-URL field were deleted in Phase 8; [src-tauri/src/processing.rs](src-tauri/src/processing.rs) (`start_processing`/`regenerate_mom`/`get_processing_status`), [src-tauri/src/library.rs](src-tauri/src/library.rs) (history/transcript/MoM/export/delete), and [src-tauri/src/configuration.rs](src-tauri/src/configuration.rs) (settings/`check_models`/Model Manager) replaced them with local commands — full list in `lib.rs`. The crate links no HTTP client for any of this — keep it that way; `cargo tree | grep reqwest` returning nothing is a cheap check that nothing has crept back in. (The two narrow exceptions, neither a general HTTP client: `models_manager.rs`'s `curl.exe`/`tar.exe` shellouts for one-time model downloads, and `pipeline/llm/http.rs`'s hand-rolled loopback-only client driving `llama-server.exe`. See `docs/architecture.md`.)

**Windows audio is Rust, not Web Audio.** [src-tauri/src/audio.rs](src-tauri/src/audio.rs) uses `cpal`. The recording stream runs on its own dedicated OS thread for its whole lifetime because cpal's WASAPI backend is COM apartment-threaded — the stream is never moved across threads; only control channels + an atomic level meter cross the boundary. Writes 16-bit PCM WAV via `hound` to `app_data_dir/meetings/<meeting-id>/original.wav`. This module is fully local already and needs no changes for the pipeline to build on top of it.

**Model abstraction.** STT/diarization/LLM providers stay swappable behind clean interfaces, so models can be changed after benchmarking without touching the UI or pipeline glue.

**Machine memory tiering — do not assume 32 GB.** [src-tauri/src/system.rs](src-tauri/src/system.rs) detects total RAM and buckets the machine into a `MemoryProfile` (`Low` < 12 GB, `Balanced` 12–24 GB — the 16 GB laptop case, `High` ≥ 24 GB — the reference machine), each with its own default STT/LLM model, LLM context size, and LLM RAM budget. [src-tauri/src/settings.rs](src-tauri/src/settings.rs)'s `Settings::effective_*` methods resolve a user's optional overrides against that profile, and everything downstream (`PipelineConfig`, `ModelPaths`, thread counts) reads through those methods rather than hard-coding a model id or thread count. A 16 GB and a 32 GB machine are expected to select different default models by design — check `effective_llm_model()`/`effective_stt_model()` before assuming which one is in play.

## Project-specific gotchas

- **whisper.cpp required flags.** Once whisper.cpp is wired up locally, always pass `--max-len 80 --split-on-word --max-context 0`. These are not tuning knobs — without them whisper.cpp silently collapsed a real ~13s span of speech into a short wrong sentence on actual meeting audio. Re-verify they hold on any new (esp. Tanglish) test audio. See `docs/benchmarks.md`.
- **Never ship a native CLI's default thread count.** whisper.cpp defaults to 4 threads and sherpa-onnx to 1; on the target i9-13950HX that costs ~4× on the whole pipeline. `PipelineConfig` sets them per stage (STT scales to 24, diarization peaks at 8 and regresses past it). See `docs/benchmarks.md`.
- **sherpa-onnx's default cluster threshold over-splits speakers.** Its default (0.5) turned one dominant voice into 8 speaker ids on real audio; Cadence defaults to 0.8. Provisional — it rests on one unlabelled sample, so re-test before trusting it on multi-speaker meetings. See `docs/benchmarks.md`.
- **Language is `auto` / multilingual.** Meetings mix English, Tamil, and code-switched "Tanglish" (language changes mid-sentence). Never hard-code a single STT language.
- **AI runtime is Rust-native, not Python.** Do not reach for Python libraries (faster-whisper, pyannote-audio, etc.) for pipeline stages — the shipped app must not depend on a Python/PyTorch runtime. Use native binaries/libraries (whisper.cpp, llama.cpp, ONNX-runtime-based diarization).
- **Never silently lose a meeting.** Every failure path (mic disconnect, STT crash, invalid LLM JSON) must surface as a visible, recoverable error and preserve raw artifacts (`original.wav`, `stt.json`, `diarization.json` kept separate from derived `transcript.json`).
- **The crate has two binaries.** `pipeline_cli` sits alongside the `app` binary, so `Cargo.toml` needs `default-run = "app"` — without it `tauri dev`'s bare `cargo run` refuses to start.
