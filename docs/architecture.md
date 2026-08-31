# EchoMind — Architecture

## Repository layout

Single Tauri desktop app, living at the repo root — no second machine, no networked backend:

```text
EchoMind/
├── src/                  # React UI
├── src-tauri/src/        # Rust: audio capture, AI pipeline orchestration
├── docs/                 # architecture.md, models.md, privacy.md, performance.md
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

Each stage writes its raw output (`diarization.json`, `stt.json`) before the next stage runs, kept separate from the derived `transcript.json`, so a later failure never costs earlier work.

**Alignment approach.** Attribution is per STT segment by maximum temporal overlap — a segment containing a speaker change goes wholly to whoever holds most of it, because whisper.cpp's JSON output carries no word-level timestamps to split on. `--max-len 80` keeps segments short enough that this is a small error. Segments with no overlapping diarization interval (diarization skips non-speech) fall back to the nearest speaker; with no diarization at all they become `SPEAKER_UNKNOWN` rather than being dropped. Consecutive same-speaker segments less than 1s apart merge into one readable turn. Stable ids (`SPEAKER_00`) and presentation labels (`Person 1`, first-appearance order) are both set here (§7).

`src-tauri/src/bin/pipeline_cli.rs` runs the whole thing headlessly for benchmarking and re-runs on test recordings, before Phase 8 wires it into the UI:

```powershell
cd src-tauri
cargo run --bin pipeline_cli -- <audio.wav> [out-dir]
```

`scripts/setup-models.ps1` installs the binaries and weights the pipeline expects into that same `<app_data_dir>/models` layout — dev-time stand-in for the Phase 13 Model Manager.

## Status

Fully local, single-machine architecture. Done: audio capture (Phase 1), preprocessing/diarization/STT/alignment and the headless pipeline runner (Phases 4–5). Not yet started: STT and diarization benchmarks on Tamil/code-switched audio (Phases 2–3), LLM benchmark and MoM generation (Phases 6–7), and Tauri/UI integration (Phase 8 onward). `backend.rs` and the UI's backend-URL upload/poll flow are dead code from the superseded networked-backend design, removed when Phase 8 lands.
