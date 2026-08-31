# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EchoMind is a local-first meeting transcription system: a single Windows desktop app (Tauri 2 + React 19 + TypeScript + Rust) that records a physical meeting from a USB/laptop mic and, entirely on that same machine, runs speech-to-text, speaker diarization, timestamp/speaker alignment, and LLM-based meeting-minutes (MoM) generation. There is no separate backend and no second machine — the app owns the full pipeline and all meeting data locally.

Hard constraints: **no cloud STT, no cloud LLM, no external AI APIs, no telemetry, no networked backend.** The AI runtime is Rust-orchestrated native binaries/libraries only (whisper.cpp, llama.cpp, an ONNX-based diarization/VAD library) — no Python or PyTorch ships in the app. After models are installed, the app must run with the Internet fully disconnected. V1 targets post-meeting processing (5–10 min latency acceptable), not real-time.

`meeting-ai-app-PLAN.md` at the repo root is the authoritative spec — phased (Phase 0…15), with per-phase doc deliverables under `docs/`. Read the relevant plan section before implementing a phase. `docs/architecture.md` and `docs/models.md` record decisions already made and should be updated as architecture changes.

Not a git repository yet. No test suite exists yet.

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
```

There is no ESLint / Prettier / JS test runner configured. `npm run build` (via `tsc`) is the only frontend check.

## Architecture notes that span files

**Single-process local pipeline.** Recording, STT, diarization, alignment, and MoM generation all run as local processes/libraries orchestrated from Rust — no HTTP contract, no client/server split, no shared job queue across machines. `src-tauri/src/backend.rs` currently contains a networked-backend HTTP client (`upload_and_process`/`get_meeting_status`/`get_transcript`) that no longer has anything to call — it is dead code, superseded by local pipeline commands as each processing stage is implemented (master prompt §38 Phase 8). Don't extend it; replace it when that phase lands.

**Windows audio is Rust, not Web Audio.** [src-tauri/src/audio.rs](src-tauri/src/audio.rs) uses `cpal`. The recording stream runs on its own dedicated OS thread for its whole lifetime because cpal's WASAPI backend is COM apartment-threaded — the stream is never moved across threads; only control channels + an atomic level meter cross the boundary. Writes 16-bit PCM WAV via `hound` to `app_data_dir/meetings/<meeting-id>/original.wav`. This module is fully local already and needs no changes for the pipeline to build on top of it.

**Model abstraction.** The master prompt (§5–§7, §19) wants STT/diarization/LLM providers swappable behind clean interfaces so models can be changed after benchmarking, without touching the UI or pipeline glue.

## Project-specific gotchas

- **whisper.cpp required flags.** Once whisper.cpp is wired up locally, always pass `--max-len 80 --split-on-word --max-context 0`. These are not tuning knobs — without them whisper.cpp silently collapsed a real ~13s span of speech into a short wrong sentence on actual meeting audio. Re-verify they hold on any new (esp. Tanglish) test audio. See `docs/models.md`.
- **Language is `auto` / multilingual.** Meetings mix English, Tamil, and code-switched "Tanglish" (language changes mid-sentence). Never hard-code a single STT language.
- **AI runtime is Rust-native, not Python.** Do not reach for Python libraries (faster-whisper, pyannote-audio, etc.) for pipeline stages — the shipped app must not depend on a Python/PyTorch runtime. Use native binaries/libraries (whisper.cpp, llama.cpp, ONNX-runtime-based diarization) per master prompt §19.
- **Never silently lose a meeting.** Every failure path (mic disconnect, STT crash, invalid LLM JSON) must surface as a visible, recoverable error and preserve raw artifacts (`original.wav`, `stt.json`, `diarization.json` kept separate from derived `transcript.json`).
- The default-backend-URL field in [src/App.tsx](src/App.tsx) (`DEFAULT_BACKEND_URL`) targets a networked backend that no longer exists and will be removed along with `backend.rs` once local pipeline commands replace the upload/poll flow.
