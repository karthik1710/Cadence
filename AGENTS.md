# AGENTS.md

Standing brief for any coding agent (Claude Code, Codex, or similar) working in this repository. Tool-specific details (exact dev commands, file-level gotchas) live in [CLAUDE.md](CLAUDE.md); this file is the tool-agnostic rule set.

## Project

Cadence is a local-first meeting transcription and MoM (minutes of meeting) app for Windows: a single Tauri + React/TypeScript + Rust desktop app that records, transcribes, diarizes, and summarizes meetings entirely on the laptop. No cloud STT, no cloud LLM, no external AI APIs, no telemetry, no networked backend. [meeting-ai-app-PLAN.md](meeting-ai-app-PLAN.md) is the authoritative spec — re-read the relevant section before starting a phase, and report progress against it at the end of one.

Using a cloud coding assistant (this one included) during development does not violate the runtime's offline requirement — the constraint is on the shipped app, not the development process.

## Rules

1. Inspect before modifying — read the current code and the relevant plan section first.
2. Work phase-by-phase (master prompt §38); don't implement later-phase or V2 features early.
3. Don't skip the STT / diarization / LLM benchmarks — don't permanently select a model before benchmark results exist.
4. Keep STT, diarization, and LLM providers swappable behind clean interfaces.
5. Preserve raw artifacts (`original.wav`, `stt.json`, `diarization.json`) separately from derived data (`transcript.json`, `mom.json`) — never let a later pipeline stage's failure destroy an earlier stage's output.
6. Never add a cloud runtime dependency, and never upload meeting content anywhere.
7. The AI runtime is Rust-native: whisper.cpp, llama.cpp, and an ONNX-based diarization/VAD library, orchestrated from Rust. No Python or PyTorch ships in the app (master prompt §19).
8. Validate all LLM JSON output against a typed schema; repair/retry invalid output rather than accepting it.
9. Ground every generated decision/action item in transcript evidence — never invent decisions, owners, or deadlines; use `null` and preserve uncertainty when evidence is thin (master prompt §16).
10. Keep prompts versioned under `prompts/`, not hard-coded in UI components.
11. Keep UI (React) separate from AI/business logic.
12. Stay within the memory budget: LLM process RAM ≤ 15 GB hard limit, ≤ 10 GB preferred (master prompt §4).
13. Document architecture decisions as they're made — update `docs/architecture.md` and `docs/models.md`, not just code comments.
14. Add tests alongside each major feature.

## End-of-phase report

- Implementation summary
- Files changed
- Tests run
- Benchmark results (if applicable)
- Known issues
- Next phase
