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
npm run tauri build         # production Windows bundle (unsigned test build; warns per unsigned file)
npm run release             # same, but CADENCE_REQUIRE_SIGNING=1: fails unless every binary is signed (docs/release.md)
.\scripts\prepare-runtimes.ps1   # fills vendor/runtimes/ from pinned archives; runs automatically as beforeBundleCommand
cargo build                 # in src-tauri/ — Rust-only build
cargo test                  # in src-tauri/ — Rust tests

# headless pipeline run (audio → preprocess → diarize → STT → align → MoM), in src-tauri/
cargo run --bin pipeline_cli -- <audio.wav> [out-dir]

# benchmark harness (STT / diarization / LLM / full pipeline), in src-tauri/
# prints Markdown tables meant to be pasted into docs/benchmarks.md
cargo run --release --bin bench_cli -- <stt|diarize|llm|full> <input> [options]

# baseline regression test: real YouTube clips (tests/baseline/clips.json) scored against their captions
.\scripts\fetch-baseline-clips.ps1                          # once: caches clips in bench-runs/baseline-cache (gitignored, copyrighted, never commit)
cargo run --release --bin bench_cli -- baseline [--clips id,..] [--update]   # in src-tauri/; exit 1 = regression; --update only for an intentional improvement

.\scripts\setup-models.ps1   # dev-time fallback fetching the same required set; the in-app first-run setup / Model Manager (models_manager.rs) is the primary install path
```

There is no ESLint / Prettier / JS test runner configured. `npm run build` (via `tsc`) is the only frontend check.

Run `bench_cli baseline` before and after any change to the audio front end, STT, diarization or alignment, and treat a REGRESSED row as a failing test (details in `docs/benchmarks.md`, "Baseline regression test").

## Architecture notes that span files

**Single-process local pipeline.** Recording, STT, diarization, alignment, and MoM generation all run as local processes/libraries orchestrated from Rust — no HTTP contract, no client/server split, no shared job queue across machines. The old networked-backend client (`backend.rs`) and the UI's backend-URL field were deleted in Phase 8; [src-tauri/src/processing.rs](src-tauri/src/processing.rs) (`start_processing`/`regenerate_mom`/`get_processing_status`), [src-tauri/src/library.rs](src-tauri/src/library.rs) (history/transcript/MoM/export/delete), and [src-tauri/src/configuration.rs](src-tauri/src/configuration.rs) (settings/`check_models`/Model Manager) replaced them with local commands — full list in `lib.rs`. The crate links no HTTP client for any of this — keep it that way; `cargo tree | grep reqwest` returning nothing is a cheap check that nothing has crept back in. (The two narrow exceptions, neither a general HTTP client: `models_manager.rs`'s `curl.exe`/`tar.exe` shellouts for one-time model downloads — model data only in a release build, whose runtimes ship inside the installer — and `pipeline/llm/http.rs`'s hand-rolled loopback-only client driving `llama-server.exe`. See `docs/architecture.md`.)

**Windows audio is Rust, not Web Audio.** [src-tauri/src/audio.rs](src-tauri/src/audio.rs) uses `cpal`. The recording stream runs on its own dedicated OS thread for its whole lifetime because cpal's WASAPI backend is COM apartment-threaded — the stream is never moved across threads; only control channels + an atomic level meter cross the boundary. Writes 16-bit PCM WAV via `hound` to `app_data_dir/meetings/<meeting-id>/original.wav`. This module is fully local already and needs no changes for the pipeline to build on top of it.

**Model abstraction.** STT/diarization/LLM providers stay swappable behind clean interfaces, so models can be changed after benchmarking without touching the UI or pipeline glue.

**One model set, 16 GB minimum — no hardware tiers.** Every machine runs the same models: Whisper small and Qwen3-8B Q4_K_M at 8K context. The constants live in [src-tauri/src/system.rs](src-tauri/src/system.rs) (`STT_MODEL_ID`, `LLM_MODEL_ID`, `LLM_CONTEXT_TOKENS`, `LLM_RAM_BUDGET_MB`, `MIN_TOTAL_RAM_MB` = 15,000, since Windows reports a 16 GB laptop as ~15.7 GB). `HostInfo::meets_minimum()` is advisory — it warns, never blocks recording. The old `MemoryProfile` (Low/Balanced/High) and the per-tier alternatives (Qwen3-4B, Gemma3-4B, whisper-base/medium) are gone; do not reintroduce tiers — if the 8B does not fit 16 GB, swap the single model. `Settings::effective_stt_model()`/`effective_llm_model()` now just return the constants; thread counts are still per machine via `effective_*_threads`. The 8B has never run on any machine; on the one real 16 GB laptop tried (2026-11-19, ordinary apps open) the LLM RAM preflight refused it — 4.9–6.1 GB free against the 7,550 MB it wants. Whether it fits once started is unmeasured and is the top open risk (`docs/roadmap.md` risk 14).

## Project-specific gotchas

- **whisper.cpp required flags.** Once whisper.cpp is wired up locally, always pass `--max-len 80 --split-on-word --max-context 0`. These are not tuning knobs — without them whisper.cpp silently collapsed a real ~13s span of speech into a short wrong sentence on actual meeting audio. Re-verify they hold on any new (esp. Tanglish) test audio. See `docs/benchmarks.md`.
- **Never ship a native CLI's default thread count.** whisper.cpp defaults to 4 threads and sherpa-onnx to 1; on the target i9-13950HX that costs ~4× on the whole pipeline. `PipelineConfig` sets them per stage (STT scales to 24, diarization peaks at 8 and regresses past it). See `docs/benchmarks.md`.
- **The cluster threshold has no single safe default yet.** sherpa-onnx's own default (0.5) shattered one noisy dominant voice into 8 speaker ids, so Cadence shipped 0.8 — which then collapsed a clean two-speaker sample into one id (0.55+ all do, on that sample). Cadence now ships 0.5 back, because a full collapse erases speaker attribution entirely while over-splitting at least preserves it. Both failure modes are real; re-test on any new multi-speaker audio and nudge `cluster_threshold` in Settings if it misclassifies. See `docs/benchmarks.md`.
- **Language is `auto` / multilingual.** Meetings mix English, Tamil, and code-switched "Tanglish" (language changes mid-sentence). Never hard-code a single STT language.
- **AI runtime is Rust-native, not Python.** Do not reach for Python libraries (faster-whisper, pyannote-audio, etc.) for pipeline stages — the shipped app must not depend on a Python/PyTorch runtime. Use native binaries/libraries (whisper.cpp, llama.cpp, ONNX-runtime-based diarization).
- **Never silently lose a meeting.** Every failure path (mic disconnect, STT crash, invalid LLM JSON) must surface as a visible, recoverable error and preserve raw artifacts (`original.wav`, `stt.json`, `diarization.json` kept separate from derived `transcript.json`).
- **The crate has two binaries.** `pipeline_cli` sits alongside the `app` binary, so `Cargo.toml` needs `default-run = "app"` — without it `tauri dev`'s bare `cargo run` refuses to start. Both CLIs carry `required-features = ["dev-cli"]` (a default feature) so `tauri build` leaves them out of the installer, and every folder under `src/bin/` must be a binary's own folder (`bench_cli/main.rs` + `bench/`): Tauri treats any entry there as a binary to bundle, and a stray module folder fails the bundle with "failed to run powershell" while signing a nonexistent exe.
- **DSP is hand-written here, on purpose.** `pipeline/spectrum.rs` (radix-2 FFT, STFT), `preprocess.rs`'s biquad and `enhance.rs`'s compressor are all written out rather than pulled from a crate. One filter or one FFT is ~80 lines against a dependency with its own API-stability risk, in a project whose whole point is shipping no runtime it did not choose. Do not "fix" this by adding `rustfft` or a DSP crate.
- **The classifiers are deliberately not models.** `noise_class.rs` (what kind of background noise) and `topic.rs` (what kind of meeting) are signal processing and word counting. That is a design decision, not a shortcut: both are deterministic, auditable, need no download, work on a fresh install, and — for `topic.rs` — cannot classify the same meeting differently from one run (or one model swap) to the next, which reusing the LLM would. `redact.rs` ends with the same argument for why its Tier 3 ONNX classifier is not built yet.
- **Redaction annotates, never edits.** `redact.rs` produces `flags` on each transcript turn; `transcript.json` always holds what was actually said, and masking happens at render and export time. This keeps the raw-artifact rule intact and makes a wrong flag reversible. `Flag.start_char`/`end_char` are **Unicode code point offsets** — the frontend must slice with `Array.from(text)`, because these meetings contain Tamil and `String.slice` would misplace every offset after the first non-BMP-adjacent character.
- **The shipped lexicon carries no slurs, and the Tamil lists are empty on purpose.** `assets/lexicon/en.txt` is English profanity only: a public repository should not contain a slur list, so the `slur` category is wired end to end and organisations populate it from `%APPDATA%\com.cadence.app\lexicon\`. `ta.txt`/`ta-latn.txt` ship empty because romanized Tamil has no standard orthography and no Tanglish audio has ever been through this pipeline — a guessed variant list would have unknown recall *and* unknown precision, and false positives in a circulated record are the expensive failure. Tests use a synthetic fixture lexicon; do not commit real slurs to fixtures either.
- **Audio front-end order is load-bearing:** DC removal → high-pass → noise reduction → leveler → compressor → normalize. Noise reduction precedes both gain stages so neither pumps up room tone, and so the leveler's speech/pause decision is made against a clean floor. The leveler (`enhance::level`) is slow, speech-gated, phrase-length gain that closes the steady near/far-talker gap the compressor cannot see; it precedes the compressor so the compressor only catches the fast peaks left over. The compressor precedes normalize so normalize still sets the final level and holds the peak ceiling — which is also why the leveler steers towards the recording's own mean speech level, not an absolute target. GTCRN *supersedes* the model-free reducer — never run both.
- **Leveler, compressor and model-free noise reduction all default on.** Noise reduction is on as the GTCRN fallback (it only runs when GTCRN does not); its full-recording STFT against the 10-minute target is an accepted, unmeasured cost. The leveler has no Settings toggle yet and runs unconditionally. Every one of their defaults is validated on synthetic signals only — see `docs/benchmarks.md`.
- **First-run setup is sequential and consent-gated on purpose.** `install_required_models` and `src/features/setup/SetupView.tsx` download the whole required set (now including GTCRN and Silero VAD; model data only when the runtimes are bundled) one item at a time on one thread, only after an explicit click over a visible list. Commit 59ecece removed an earlier one-click batch installer because back-to-back `curl.exe`/`tar.exe` downloads read as dropper behaviour to endpoint security; do not make setup auto-start or parallelise downloads. Whether EDR still flags it is untested. Recording must never wait on it ("Record without setting up").
- **Installs are one at a time app-wide, and never during a meeting.** `configuration.rs`'s `install_blocker` is the single guard for setup, Install and Reinstall: the Model Manager and setup share one progress map, so a second download is refused while any item is queued/downloading/extracting/verifying, and every install is refused while a meeting is recording (dropped audio) or processing (a runtime reinstall would overwrite DLLs the pipeline has open). Route any new install path through it.
- **Each runtime lives in its own folder — never flatten them into one again.** `<models>/bin/whisper`, `bin/llama`, `bin/sherpa`; resolve executables only via `models_manager::runtime_dir`/`runtime_binary`. whisper.cpp and llama.cpp both ship `ggml*.dll` from different ggml versions; the old shared `bin/` let the last install win, and whisper.cpp ran on llama.cpp's ggml by ABI luck. Every STT timing before 2026-11-19 was measured that way, and isolating cost ~32% STT time (`docs/benchmarks.md`). A legacy flat runtime still resolves from `bin/` (decided per runtime, on its launcher) until reinstalled; Reinstall swaps the whole folder (`.new` → rename, `.old` kept until it succeeds), and Verify checks a digest for every file in it. `scripts/setup-models.ps1` uses the same layout.
- **Runtimes are bundled; a release build never downloads an executable.** The installer ships whisper.cpp, llama.cpp and sherpa-onnx in `runtimes\<whisper|llama|sherpa>\` beside the exe, filled at build time by `scripts/prepare-runtimes.ps1` (Tauri `beforeBundleCommand`). It pins each release archive by URL **and SHA-256** (hard fail on mismatch), keeps every DLL plus only the exes the code launches (`whisper-cli`, `whisper-server`, `llama-server`, sherpa diarization/denoiser/vad — add any new one to its `Launch` list), signs them via `scripts/sign.ps1`, *then* writes `runtimes/manifest.json`. Resolution precedence is bundled → `<models>/bin/<rt>` → legacy flat `bin/`, all in `models_manager::runtime_dir`; the bundled root is `None` under `cargo test`, so tests inject one through `Roots`. A bundled runtime is never in `missing_required_ids`, `install()` refuses it, and Verify checks it against the bundled manifest. Never re-add a runtime download to setup for release builds, never edit a pin to make a mismatch pass, and change a runtime version in the script and the catalog together. Only `vendor/runtimes/README.md` is committed (Tauri fails the build if the resource dir is missing). After filling or emptying `vendor/runtimes`, touch `src-tauri/tauri.conf.json` so tauri-build re-copies it into `target/<profile>/runtimes`. See `docs/release.md`.
- **The whisper.cpp runtime (b4938) is due an upgrade.** Its bundled ggml is older and slower than the one it used to borrow; b5454 is the candidate, gated on the owner's sign-off (unpinned upstream hash), a `bench_cli baseline` re-run, and re-verifying the required whisper flags and `--output-json-full`. Do not "fix" the speed by sharing DLLs again. See `docs/roadmap.md` backlog.
