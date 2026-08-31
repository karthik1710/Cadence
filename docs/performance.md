# Performance

Measurements from the target machine (i9-13950HX, 32 GB RAM, 32 logical threads, Intel UHD iGPU unused — CPU-only). Targets come from `meeting-ai-app-PLAN.md` §29: a 30–60 minute meeting must process in ≤ 10 minutes (≤ 15 minutes acceptable during development).

## Sample

One 57.4s English test recording (`meetings/5de8d0c8-.../original.wav`, laptop mic, two speakers with one heavily dominant). This is *not* a §29 benchmark set — no 30/45/60-minute recording and no Tamil/code-switched sample exists yet, so every number below is a small-sample extrapolation.

## Thread scaling (2026-08-31)

Both native CLIs default to far fewer threads than this machine has: whisper.cpp to 4, sherpa-onnx to 1. Leaving those defaults alone was costing roughly 4× on the whole pipeline.

**whisper.cpp `whisper-cli.exe`, `ggml-small.bin`:**

| `-t` | Wall | Realtime factor |
|---|---|---|
| 4 (default) | 20.3s | 0.35x |
| 8 | 15.2s | 0.27x |
| 16 | 7.2s | 0.12x |
| 24 | 6.2s | 0.11x |

**sherpa-onnx offline speaker diarization:**

| threads | Wall | Realtime factor |
|---|---|---|
| 1 (default) | 10.2s | 0.18x |
| 4 | 6.2s | 0.11x |
| 8 | 5.2s | 0.09x |
| 16 | 6.1s | 0.11x |

STT keeps gaining out to 24 threads; diarization peaks at 8 and regresses past it. Hence the per-stage caps in `PipelineConfig` (`MAX_STT_THREADS` 24, `MAX_DIARIZATION_THREADS` 8) rather than one shared thread count.

## End-to-end pipeline

`cargo run --bin pipeline_cli` on the same 57.4s sample, debug build (the Rust orchestration is process-spawning and JSON parsing; the work happens in the native binaries, which are release builds either way):

| Config | Wall | Realtime factor |
|---|---|---|
| Stock CLI defaults (4 / 1 threads) | 44.1s | 0.78x |
| Tuned (24 / 8 threads) | 14.4s | 0.25x |

Stage split at the tuned setting: preprocess ~1.2s, diarization ~7.6s, STT ~5.6s, alignment negligible.

**Extrapolation to the §29 target.** At 0.25x realtime a 60-minute meeting lands around 15 minutes — outside the 10-minute target, inside the 15-minute dev threshold, and that is *before* LLM MoM generation (Phases 6–7) is added. Caveats in both directions: some of the 14.4s is fixed model-load overhead that amortizes over a long recording, but a 60× longer input has not actually been measured. Treat this as "plausibly close, unproven."

Levers not yet tried, in rough order of expected value:
- A longer real recording, to see how much of the current cost is fixed overhead.
- whisper.cpp `-p` (parallel processors) alongside `-t`.
- A smaller/quantized STT model, if `small` proves to be the bottleneck at scale — trades accuracy, so benchmark quality with it (§30).
- Intel SYCL/iGPU offload (§31) — optional, CPU stays the baseline.

## Not yet measured

Peak RAM, peak CPU, and thermals per stage (§29 asks for all three); LLM generation speed and memory (Phase 6); anything on 30/45/60-minute or Tamil/code-switched audio.
