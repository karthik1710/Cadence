# Model Benchmark Results

This is the results doc for master prompt Phases 2 (STT), 3 (diarization), 6 (LLM), and 14 (full pipeline). `src-tauri/src/bin/bench_cli.rs` is the tool that fills it in — it runs the relevant pipeline stage(s) on a real audio file (or, for the LLM mode, a real `transcript.json`) and prints a Markdown table shaped to paste directly into the matching section below.

None of the sections below have real results yet. `bench_cli` was written and unit-tested at the level `cargo test` reaches, but has never actually been run — the environment that built it had no Rust toolchain available to compile and execute it. This doc exists so that gap is visible and has a clear, exact path to being closed, rather than the benchmark doc simply not existing (as it did before this pass — see `meeting-ai-app-PLAN.md` §38, Phase 0).

Small-sample numbers from a single unlabelled 57.4s recording already exist in `docs/performance.md` and `docs/models.md` — those predate `bench_cli` and were collected by hand. They are real data points, but they are not a substitute for the runs below: no 30/45/60-minute recording and no Tamil/Tanglish sample has been tested at all.

## How to run this

From `src-tauri/`, against a release build (`--release` matters — debug-build orchestration overhead is negligible next to the native binaries either way, but there's no reason not to use the fast build once you have a toolchain):

```powershell
cargo build --release --bin bench_cli

# Phase 2 — STT: compares Whisper model sizes and thread counts
.\target\release\bench_cli.exe stt <audio.wav> [--models whisper-base,whisper-small,whisper-medium] [--threads 8,24]

# Phase 3 — diarization: sweeps the sherpa-onnx cluster threshold
.\target\release\bench_cli.exe diarize <audio.wav> [--thresholds 0.5,0.6,0.7,0.8,0.9] [--threads 8]

# Phase 6 — LLM: compares candidate models on one real transcript
.\target\release\bench_cli.exe llm <transcript.json> [--models qwen3-4b-q4,qwen3-8b-q4,gemma3-4b-q4]

# Phase 14 — full pipeline: one end-to-end run with the §29 budget check
.\target\release\bench_cli.exe full <audio.wav>
```

`<transcript.json>` for the `llm` mode can come from a prior `stt`/`diarize` run through the full pipeline, or from `pipeline_cli`'s output. Every mode needs the relevant models already installed (Settings → Models in the app, or `scripts/setup-models.ps1`); an uninstalled model prints `_not installed_` in its row rather than failing the whole run.

Required test material, per §29/§8: at minimum one 30-minute, one 45-minute, and one 60-minute recording, plus at least one recording with genuine Tamil/English code-switching (Tanglish) — mid-sentence language changes, not just Tamil and English in separate sentences. For diarization, recordings with known, labelled speaker counts (3, 5, 10 speakers) are needed to judge the cluster-threshold sweep against ground truth rather than by eye.

---

## Phase 2 — STT benchmark

_Not yet run — no Rust toolchain was available in the environment that built this harness. Run `bench_cli stt <audio.wav>` for each required recording (30/45/60-minute, plus the Tanglish sample) and paste the Markdown table output here, one table per recording. Note by hand which model's output was most accurate on each — `bench_cli` writes each candidate's raw transcript to a temp directory for exactly this comparison, since WER scoring needs a human reference transcript this tool does not generate._

## Phase 3 — diarization benchmark

_Not yet run — no Rust toolchain was available in the environment that built this harness. Run `bench_cli diarize <audio.wav>` against labelled 3/5/10-speaker recordings and paste the Markdown table output here. Record the real speaker count next to each table so the reported speaker-id count can be checked against ground truth._

## Phase 6 — LLM benchmark

_Not yet run — no Rust toolchain was available in the environment that built this harness. Run `bench_cli llm <transcript.json>` against a real transcript (ideally one containing Tamil/Tanglish content) and paste the Markdown table output here. Score each candidate's generated `mom.json` (written alongside the table, per model) against §30's rubric — factual accuracy, action-item/decision quality, Tamil handling, hallucination resistance — before choosing a default. Any model whose peak RAM exceeds the §4 15 GB hard limit is rejected outright regardless of quality._

## Phase 14 — full pipeline benchmark

_Not yet run — no Rust toolchain was available in the environment that built this harness. Run `bench_cli full <audio.wav>` on each required recording and paste the Markdown table output here, including the §29 budget-check line (MET / MISSED by Ns) that the tool prints automatically. This is the number that determines whether the §29 target (≤10 minutes processing for a 30–60 minute meeting, ≤15 minutes acceptable during development) is actually met._
