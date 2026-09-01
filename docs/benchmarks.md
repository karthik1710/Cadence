# Cadence — Models, Performance, and Benchmarks

Which models Cadence runs, what has actually been measured, what has not, and how to close the gap. Design decisions live in [architecture.md](architecture.md); status and backlog live in [roadmap.md](roadmap.md).

## Read this first: what is and isn't measured

**Two real measurements exist.** A 57.4-second English recording (laptop mic, two speakers, one heavily dominant, unlabelled) produced the thread-scaling and end-to-end numbers below. A second, 26-second two-speaker recording (also unlabelled beyond "two people") was run through `bench_cli diarize` after the default cluster threshold was found to be silently merging real speakers — see the Diarization section below. That's still it.

**Nothing else has been benchmarked.** `bench_cli` — the harness for Phases 2 (STT), 3 (diarization), 6 (LLM), and 14 (full pipeline) — was written and unit-tested to the level `cargo test` reaches, and Phase 3 (diarization) has now been run against real audio twice; the rest **has never been executed against real audio**. There are no 30/45/60-minute results, no Tamil or code-switched results, no WER, no labelled-speaker-count results, and no LLM quality scores.

**So every model choice below is provisional.** The whisper model size, the diarization cluster threshold, and the memory-tier default LLMs are what ships because *something* has to run out of the box (project rule: no model is permanently selected before it has been benchmarked on this project\x27s own audio). None is a conclusion. Nothing in this document should be read as a ranking.

## Measured: the one sample

Target machine: i9-13950HX, 32 GB RAM, 32 logical threads, iGPU unused (CPU-only). Target: a 30–60 minute meeting processes in ≤ 10 minutes (≤ 15 acceptable during development).

### Thread scaling (2026-08-31)

Both native CLIs default to far fewer threads than this machine has — whisper.cpp to 4, sherpa-onnx to 1. **Shipping those defaults was costing roughly 4× on the whole pipeline.**

| whisper.cpp `-t` (`ggml-small.bin`) | Wall | Realtime |  | sherpa-onnx diarization threads | Wall | Realtime |
|---|---|---|---|---|---|---|
| 4 (default) | 20.3s | 0.35x |  | 1 (default) | 10.2s | 0.18x |
| 8 | 15.2s | 0.27x |  | 4 | 6.2s | 0.11x |
| 16 | 7.2s | 0.12x |  | 8 | 5.2s | 0.09x |
| 24 | 6.2s | 0.11x |  | 16 | 6.1s | 0.11x |

STT keeps gaining out to 24 threads; **diarization peaks at 8 and regresses past it.** Hence per-stage caps in `PipelineConfig` (`MAX_STT_THREADS` 24, `MAX_DIARIZATION_THREADS` 8) rather than one shared count.

### End-to-end

`pipeline_cli` on the same sample, debug build (orchestration is process-spawning and JSON parsing; the work happens in native binaries, which are release builds either way):

| Config | Wall | Realtime |
|---|---|---|
| Stock CLI defaults (4 / 1 threads) | 44.1s | 0.78x |
| Tuned (24 / 8 threads) | 14.4s | 0.25x |

Tuned stage split: preprocess ~1.2s, diarization ~7.6s, STT ~5.6s, alignment negligible.

**Against that target:** 0.25x realtime puts a 60-minute meeting near 15 minutes — outside the 10-minute target, inside the 15-minute dev threshold, and that is *before* LLM generation is added. Caveats cut both ways: some of the 14.4s is fixed model-load overhead that amortizes over a long recording, but a 60× longer input has never been run. Treat as "plausibly close, unproven."

Levers not yet tried, roughly by expected value: a longer real recording (to see how much cost is fixed overhead); whisper.cpp `-p` alongside `-t`; a smaller/quantized STT model if `small` is the bottleneck at scale (trades accuracy — score it before adopting); Intel SYCL/iGPU offload (optional — CPU stays the baseline).

**Not measured at all:** peak RAM, peak CPU, and thermals per stage (all three are required); LLM generation speed and memory; anything longer than a minute; anything not English.

### Self-measurement is built in

Every real pipeline run measures itself — not just benchmark runs. `pipeline::proc::run_measured` wraps each native stage in wall-clock timing plus peak-working-set sampling (`K32GetProcessMemoryInfo`, polled every 250 ms, with a final sample before the handle closes). `pipeline::run` collects these into `RunMetrics` — per-stage wall time and peak RAM for diarization and STT, plus preprocess/align/MoM timing and the total — and writes `metrics.json` beside the meeting's other artifacts. `RunMetrics::realtime_factor()` reports total processing as a multiple of meeting length, the unit the 10-minute target is stated in (0.167x ≈ 10 minutes for a 60-minute meeting).

## Model choices (all provisional)

### STT — whisper.cpp

Candidates are whisper.cpp and sherpa-onnx's Whisper ONNX runner. Not faster-whisper: it depends on Python/CTranslate2, which conflicts with the Rust-native runtime decision.

**Required flags, not tuning knobs: `--max-len 80 --split-on-word --max-context 0`.** With whisper.cpp's defaults, a genuine ~13-second span of conversational speech was silently collapsed into one short, unrelated sentence. Cause: whisper.cpp carries a bad decode's text forward as context for the next segment, compounding the failure, and with no max segment length it gambles a long span on a single decode instead of re-segmenting. Re-verify these hold on any new test audio, especially longer, multi-speaker, and Tanglish material.

**First real run (2026-08-31).** `ggml-small.bin` on the 57.4s English sample: 6.2s at 24 threads (0.11x). Language auto-detect picked `en` at p=0.57 — low confidence worth watching once Tamil audio is tested. The required flags were in place and no collapse recurred. **`small` was chosen by the setup script, not by benchmark**; $1 still need to be compared against it on this project's own audio.

### Code-switching, English/Tamil

**`--language auto` does not solve this.** whisper.cpp's `auto` detects a language *once*, from the opening window, and applies it to the whole file. In a Tanglish meeting that locks the decoder to whichever language came first and degrades the other for the rest: Tamil comes back romanized or dropped when locked to `en`; English garbled or transliterated when locked to `ta`. The setting looked like it handled code-switching. It did not.

`Settings.stt_language` now selects a strategy:

| value | behaviour |
|---|---|
| `code-switched` (**default**) | VAD splits the recording into blocks, each transcribed with its own language detection, then reassembled onto the original timeline. Each turn carries the language whisper detected for it. |
| `auto` | the old single-detection path — cheapest, right for a monolingual meeting |
| `ta` / `en` / any ISO code | forced, passed straight through |

`code-switched` is the default because these meetings *are* mixed; defaulting to a mode that mishandles that was wrong. **`--translate` is never passed in any mode** — the transcript must preserve what was actually said; translation is the MoM stage's job, for its own output only.

Every timestamp whisper returns is relative to its block, while the transcript, alignment, evidence links, and click-to-play all index the original recording. `pipeline/codeswitch.rs` remaps them, covered by 23 unit tests including non-first blocks, boundary segments, VAD gaps, and out-of-range clamping — a silent error here would misplace every evidence link in the app, so it is tested rather than assumed. On the rendering side, `"Nirmala UI"` is explicit in the CSS font stacks: Segoe UI carries no Tamil glyphs, so mixed lines were falling back implicitly and rendering Tamil and Latin at different baselines.

**Known limitation to test for, not assume away:** every ASR model makes more errors at a language-switch boundary than inside a monolingual stretch. There is no clean fix, only measurement.

**This is implementation, not proof — no Tamil or code-switched audio has been through this pipeline.** To close that gap, transcribe one recording three ways and compare by hand:

```powershell
cd src-tauri
cargo run --release --bin pipeline_cli -- <tanglish.wav> out-cs   --language code-switched
cargo run --release --bin pipeline_cli -- <tanglish.wav> out-auto --language auto
cargo run --release --bin pipeline_cli -- <tanglish.wav> out-ta   --language ta
```

Look for: does `code-switched` keep Tamil in Tamil script *and* English in Latin within the same turn, where the single-language runs mangle one? Are the per-turn language tags plausible at switch points? Does processing still fit the processing budget with blocks transcribed separately? Record answers here — a WER number against a hand-corrected reference is better still.

### Diarization — sherpa-onnx

sherpa-onnx offline speaker diarization (pyannote segmentation 3.0 + WeSpeaker CAM++ embeddings, both ONNX), invoked from Rust. Native `pyannote-audio` was rejected to keep the runtime Python-free; ONNX-ported quality versus native pyannote remains an open risk.

**No single cluster threshold has worked on both samples tried so far.** sherpa-onnx clusters by distance threshold (`--clustering.cluster-threshold`, default 0.5) when speaker count isn't given up front. On the original 57.4s sample, that default produced **8 speaker ids for essentially one dominant voice plus a faint second** — badly over-split:

| threshold | speaker ids | dominant voice |
|---|---|---|
| 0.5 (sherpa-onnx default) | 8 | fragmented across many ids |
| 0.6 | 5 | fragmented |
| 0.7 | 4 | consolidated (36.5s in one id) |
| 0.8 | 4 | consolidated, second voice still separate (4.6s) |
| 0.9 | 2 | second voice largely absorbed into the first |

Raising the default to 0.8 fixed that sample. It then shipped unexamined against any *other* sample until a second, cleaner 26-second two-speaker recording surfaced the opposite failure — 0.8 silently merged both speakers into one id, which erases speaker attribution outright rather than just fragmenting it:

| threshold | speakers | segments | longest speaker share |
|---|---|---|---|
| 0.4 | 2 | 7 | 68% |
| 0.45 | 2 | 7 | 68% |
| 0.5 | 2 | 7 | 68% |
| 0.55 | 1 | 4 | 100% |
| 0.6 – 0.8 | 1 | 4 | 100% |

(`bench_cli diarize <wav> --thresholds 0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8`.) The cliff between correct (≤0.5) and fully collapsed (≥0.55) is sharp and sits well inside the "consolidated" range the first sample called good.

**Cadence now ships 0.5** (`DEFAULT_CLUSTER_THRESHOLD` in `settings.rs`) — back to sherpa-onnx's own default. The reasoning: a full collapse (0.8, second sample) destroys per-speaker attribution completely, while over-splitting (0.5, first sample) at least leaves every speaker distinguishable under some id, so a human skimming the transcript can still tell people apart even if the labels are noisier than they should be. **This is a judgment call between two failure modes on two small samples, not a resolved question.** Phase 3 still needs labelled 3/5/10-speaker recordings with real headcounts to settle it properly, and should test `--clustering.num-clusters` for the case where the organizer knows the headcount. Until then, the Settings screen's "Speaker cluster threshold" field is the escape hatch for a recording that lands on the wrong side of the cliff — raise it if one person is being split across ids, lower it if two people are being merged into one.

**Alignment limitation.** Attribution is per STT segment by maximum overlap, so a fast two-way exchange inside one whisper segment collapses onto the dominant speaker — on the 57.4s sample, a 30s back-and-forth rendered as one Person 1 turn. The 26s sample shows the same shape: whisper.cpp split it into 5 well-formed turns, but because diarization at 0.8 returned only one speaker id for the whole clip, every turn aligned to it regardless — so this limitation and the clustering threshold compound rather than being independent bugs. Fixing alignment properly needs word-level STT timestamps or finer diarization segments; evaluate during Phase 3/4 refinement, not by tuning the merge gap.

### LLM — llama.cpp

Catalog (`models_manager::catalog()`), all GGUF:

| id | Model | Notes |
|---|---|---|
| `qwen3-4b-q4` | Qwen3 4B Q4_K_M | ~2.5 GB. Default on `Balanced` (16 GB) and `Low`. |
| `qwen3-8b-q4` | Qwen3 8B Q4_K_M | ~5 GB. Default on `High` (32 GB reference). |
| `gemma3-4b-q4` | Gemma 3 4B Instruct Q4_K_M | ~2.5 GB. Multilingual comparison candidate; install-to-benchmark, not a default. |

The 4B/8B split by memory tier exists to keep the 8B's ~5 GB weights plus KV cache away from a 16 GB laptop's budget (6,000 MB on `Balanced` vs 10,000 MB on `High`) — **not** because 4B has been shown to produce acceptable minutes on this project's audio. Re-evaluate once `bench_cli llm` has run, including whether Gemma3-4B beats Qwen3-4B on Tamil/Tanglish at the 16 GB tier.

## Running the benchmarks

From `src-tauri/`, release build:

```powershell
cargo build --release --bin bench_cli

# Phase 2 — STT: model sizes × thread counts
.\target\release\bench_cli.exe stt <audio.wav> [--models whisper-base,whisper-small,whisper-medium] [--threads 8,24]

# Phase 3 — diarization: cluster-threshold sweep
.\target\release\bench_cli.exe diarize <audio.wav> [--thresholds 0.5,0.6,0.7,0.8,0.9] [--threads 8]

# Phase 6 — LLM: candidates on one real transcript
.\target\release\bench_cli.exe llm <transcript.json> [--models qwen3-4b-q4,qwen3-8b-q4,gemma3-4b-q4]

# Phase 14 — full pipeline, with the 10-minute budget check
.\target\release\bench_cli.exe full <audio.wav>
```

Each mode prints a Markdown table shaped to paste straight into the results section below. `stt` also writes each candidate's raw transcript to a temp dir for manual accuracy comparison — **WER needs a human reference transcript, which this tool does not generate.** `llm` writes each model's `mom.json` for scoring and reports peak RAM against the 15 GB hard limit. `full` computes the budget check automatically (`audio_ms × 0.167` vs actual, reported MET or MISSED-by-N-seconds).

`<transcript.json>` can come from any prior `pipeline_cli` run. Every mode needs the relevant models installed (Settings → Models, or `scripts/setup-models.ps1`); an uninstalled model prints `_not installed_` in its row rather than failing the run.

**Test material required**: at minimum one 30-minute, one 45-minute, and one 60-minute recording, plus at least one with genuine Tanglish — mid-sentence language changes, not Tamil and English in separate sentences. For diarization, recordings with known labelled speaker counts (3, 5, 10) are needed to judge the threshold sweep against ground truth rather than by eye.

## Results

Paste `bench_cli` output here as it lands, one table per recording.

### Phase 2 — STT

_Not yet run._ Note by hand which model's output was most accurate on each recording.

### Phase 3 — Diarization

_Not yet run._ Record the real speaker count next to each table so reported ids can be checked against ground truth.

### Phase 6 — LLM

_Not yet run._ Score each candidate's `mom.json` against the quality rubric — factual accuracy, action-item/decision quality, Tamil handling, hallucination resistance — before choosing a default. Any model whose peak RAM exceeds the 15 GB hard limit is rejected outright regardless of quality.

### Phase 14 — Full pipeline

_Not yet run._ Include the MET / MISSED-by-Ns budget line. This is the number that decides whether the 10-minute target is actually met.
