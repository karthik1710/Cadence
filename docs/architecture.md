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

## Status

Fully local, single-machine architecture. Not yet started: Windows-laptop hardware/toolchain discovery (master prompt Phase 0), and the STT/diarization/alignment/LLM pipeline (Phases 1–7).
