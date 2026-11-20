# AGENTS.md

Standing brief for any coding agent (Claude Code, Codex, or similar) working in this repository. Tool-specific details (exact dev commands, file-level gotchas) live in [CLAUDE.md](CLAUDE.md); this file is the tool-agnostic rule set.

## Project

Cadence is a local-first meeting transcription and MoM (minutes of meeting) app for Windows: a single Tauri + React/TypeScript + Rust desktop app that records, transcribes, diarizes, and summarizes meetings entirely on the laptop. No cloud STT, no cloud LLM, no external AI APIs, no telemetry, no networked backend. There is no separate spec document: the specification is the code plus three living docs under `docs/` — [architecture.md](docs/architecture.md), [benchmarks.md](docs/benchmarks.md), and [roadmap.md](docs/roadmap.md). Read the relevant one before starting work in its area, and update it when a decision changes.

Using a cloud coding assistant (this one included) during development does not violate the runtime's offline requirement — the constraint is on the shipped app, not the development process.

## Rules

1. Inspect before modifying — read the current code and the relevant plan section first.
2. Work incrementally against `docs/roadmap.md`; do not implement backlog or out-of-V1 features early.
3. Don't skip the STT / diarization / LLM benchmarks — don't permanently select a model before benchmark results exist.
4. Keep STT, diarization, and LLM providers swappable behind clean interfaces.
5. Preserve raw artifacts (`original.wav`, `stt.json`, `diarization.json`) separately from derived data (`transcript.json`, `mom.json`) — never let a later pipeline stage's failure destroy an earlier stage's output.
6. Never add a cloud runtime dependency, and never upload meeting content anywhere.
7. The AI runtime is Rust-native: whisper.cpp, llama.cpp, and an ONNX-based diarization/VAD library, orchestrated from Rust. No Python or PyTorch ships in the app.
8. Validate all LLM JSON output against a typed schema; repair/retry invalid output rather than accepting it.
9. Ground every generated decision/action item in transcript evidence — never invent decisions, owners, or deadlines; use `null` and preserve uncertainty when evidence is thin.
10. Keep prompts versioned under `prompts/`, not hard-coded in UI components.
11. Keep UI (React) separate from AI/business logic.
12. Stay within the memory budget: one model set for every machine, sized for the 16 GB minimum — the LLM's advisory budget is 7,000 MB, and minutes are refused (never the transcript) when free RAM is short. Never reintroduce per-machine model tiers; if the model does not fit, replace the single model.
13. Document architecture decisions as they're made — update `docs/architecture.md` and `docs/benchmarks.md`, not just code comments.
14. Add tests alongside each major feature.
15. Keep each native runtime (whisper.cpp, llama.cpp, sherpa-onnx) in its own folder with the DLLs it shipped with. Never put two runtimes' binaries in one folder: they ship same-named DLLs from different versions, and a shared folder silently runs one tool on the other's libraries.
16. Model downloads happen only on an explicit user action, one at a time across the whole app, and never while a meeting is recording or processing.
17. A release build never downloads an executable. The native runtimes ship inside the signed installer, built from upstream archives pinned by SHA-256 (`scripts/prepare-runtimes.ps1`); first-run setup downloads model data only. Never weaken a pin to make a mismatch pass. See [docs/release.md](docs/release.md).

## End-of-phase report

- Implementation summary
- Files changed
- Tests run
- Benchmark results (if applicable)
- Known issues
- Next phase
