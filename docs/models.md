# Model Selection Notes

Living document — updated as benchmarks (master prompt Phases 2, 3, 6) produce results on the Windows laptop. Candidate lists live in the master prompt itself — see `meeting-ai-app-PLAN.md` §5 (LLM), §6 (STT), §7 (diarization) — and get filled in here as real benchmark results land.

## LLM (MoM generation)

**No benchmark has been run yet.** Benchmarking *infrastructure* now exists — `cargo run --release --bin bench_cli -- llm <transcript.json> [--models qwen3-4b-q4,qwen3-8b-q4,gemma3-4b-q4]` runs `mom::generate` against each candidate on an identical transcript and prints a Markdown table (wall time, peak RAM, JSON validity/repair rate, and counts of decisions/actions/questions/risks per §30's rubric). It has not been executed: this environment has no Rust toolchain available to compile and run it, so the code is written and unit-tested at the level `cargo test` reaches (mocked `LlmProvider`, no real model load) but has never been run end to end against a real `llama-server.exe`. Do not read anything below as a benchmark result — it is the catalog EchoMind ships with, not a ranking.

**Catalog** (`models_manager::catalog()`), all GGUF via llama.cpp:

| id | Model | Notes |
|---|---|---|
| `qwen3-4b-q4` | Qwen3 4B Q4_K_M | ~2.5 GB weights. Default on `Balanced` (16 GB) and `Low` machines. |
| `qwen3-8b-q4` | Qwen3 8B Q4_K_M | ~5 GB weights. Default on `High` (32 GB reference machine). |
| `gemma3-4b-q4` | Gemma 3 4B Instruct Q4_K_M | ~2.5 GB weights. Multilingual comparison candidate (§5); install-only-to-benchmark, not a default for any tier. |

**Memory-tiering defaults are provisional, not benchmarked.** `MemoryProfile::default_llm_model()` (`src-tauri/src/system.rs`) currently returns `qwen3-4b-q4` for 16 GB machines and `qwen3-8b-q4` for 32 GB machines. This split exists to keep the 8B model's ~5 GB weights + KV cache away from a 16 GB laptop's tighter RAM budget (`llm_ram_budget_mb`: 6,000 MB on `Balanced` vs. 10,000 MB on `High`), not because the 4B has been shown to produce acceptable minutes on this project's audio. §5 is explicit that no model may be permanently selected before benchmarking against this project's own meeting data — these are the shipped defaults precisely because *something* has to run out of the box, not a conclusion from Phase 6. Re-evaluate once `bench_cli llm` has actually been run, including whether Gemma3-4B outperforms Qwen3-4B on Tamil/Tanglish content at the 16 GB tier.

## STT

**Language requirement:** meetings mix English, Tamil, and code-switched Tanglish (language changes mid-sentence) — the model must handle this without assuming a single language per session.

**Candidates:** whisper.cpp and sherpa-onnx's Whisper ONNX runner (see master prompt §6) — not faster-whisper, which depends on Python/CTranslate2 and conflicts with the Rust-native runtime decision (§19).

**Known whisper.cpp gotcha — required flags, not optional tuning:** with whisper.cpp's default decoding settings, a genuine ~13-second span of conversational speech was silently collapsed into a single short, unrelated sentence. Root cause: whisper.cpp can carry a bad decode's text forward as context for the next segment, compounding the failure, and without a max segment length it will gamble a long span on one decode attempt instead of re-segmenting. **Fix:** always pass `--max-len 80 --split-on-word --max-context 0`. Re-verify this holds once whisper.cpp is wired into the local Rust pipeline, especially on longer, multi-speaker, and Tanglish test audio.

**Known limitation to test for, not assume away:** all ASR models (Whisper included) make more errors right at a language-switch boundary than within a monolingual stretch — there's no clean fix, only benchmarking against real Tanglish audio to see how bad it is. Include "code-switched Tamil/English" as an explicit case in the master prompt's §29 benchmark dataset (test D).

**First real run (2026-08-31).** whisper.cpp `ggml-small.bin` through the Rust pipeline on a 57.4s English laptop-mic recording: 6.2s at 24 threads (0.11x realtime) — see `docs/performance.md`. Language auto-detect picked `en` at p=0.57, a low confidence worth watching once Tamil/Tanglish audio is tested. The `--max-len 80 --split-on-word --max-context 0` flags were in place and no content collapse recurred. Still missing: any WER measurement, any Tamil or code-switched sample, and anything longer than a minute.

**Model size not yet justified.** `small` was chosen by the setup script, not by benchmark. §6 wants `base`/`medium` compared against it on this project's own audio before committing.

**Benchmark harness now exists, not yet run.** `cargo run --release --bin bench_cli -- stt <audio.wav> [--models whisper-base,whisper-small,whisper-medium] [--threads 8,24]` runs each model/thread combination through `stt::transcribe` and prints a Markdown table (wall time, realtime factor, peak RAM, segment/character counts), plus writes each model's raw transcript to a temp dir for manual accuracy comparison — WER needs a human reference transcript, which this tool does not generate. Same caveat as the LLM harness above: this was written and never executed, because no Rust toolchain was available in the environment that built it. Phase 2 still has no results.

## Diarization

Running: sherpa-onnx offline speaker diarization (pyannote segmentation 3.0 + WeSpeaker CAM++ embeddings, both ONNX), invoked from Rust — see master prompt §7. Native `pyannote-audio` (Python/PyTorch) was rejected to keep the shipped runtime Python-free per §19; ONNX-ported quality vs. native pyannote-audio remains the open risk.

**Known gotcha — the default cluster threshold over-splits.** sherpa-onnx clusters speakers by a distance threshold (`--clustering.cluster-threshold`, default 0.5) when the speaker count isn't given up front. On the 57.4s test recording that default produced **8 speaker ids for what is essentially one dominant voice plus a faint second one**. Sweep on that sample:

| threshold | speaker ids | dominant voice |
|---|---|---|
| 0.5 (default) | 8 | fragmented across many ids |
| 0.6 | 5 | fragmented |
| 0.7 | 4 | consolidated (36.5s in one id) |
| 0.8 | 4 | consolidated, second voice still separate (4.6s) |
| 0.9 | 2 | second voice largely absorbed into the first |

EchoMind therefore ships **0.8** as its default (`DEFAULT_CLUSTER_THRESHOLD` in `pipeline/mod.rs`), not sherpa-onnx's 0.5. This is provisional and rests on a single unlabelled sample: it establishes only that 0.5 is wrong here. Phase 3 needs labelled 3/5/10-speaker recordings — with real headcounts — to settle it, and should also test `--clustering.num-clusters` for the case where the organizer knows the headcount.

**Known limitation.** Alignment attributes each STT segment to whichever speaker holds most of it (see `docs/architecture.md`), so a fast two-way exchange inside one whisper segment collapses onto the dominant speaker. On the test recording a 30s stretch of back-and-forth rendered as one Person 1 turn. Fixing it properly needs word-level timestamps from STT or finer diarization segments — evaluate during Phase 3/4 refinement, not by tuning the merge gap.

**Benchmark harness now exists, not yet run.** `cargo run --release --bin bench_cli -- diarize <audio.wav> [--thresholds 0.5,0.6,0.7,0.8,0.9] [--threads 8]` runs `diarization::diarize` at each threshold and prints a Markdown table (wall time, realtime factor, peak RAM, speaker count, segment count, and the dominant speaker's share of total speech), for comparing against a recording's real headcount (§7). Same caveat as above: written and unit-scaffolded, never executed against real audio in this environment. Phase 3 still has no results beyond the single 57.4s unlabelled sample already in this file.
