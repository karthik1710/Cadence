# Cadence — Models, Performance, and Benchmarks

Which models Cadence runs, what has actually been measured, what has not, and how to close the gap. Design decisions live in [architecture.md](architecture.md); status and backlog live in [roadmap.md](roadmap.md).

## Read this first: what is and isn't measured

**Two real recordings have been measured, plus one LLM memory number.** A 57.4-second English recording (laptop mic, two speakers, one heavily dominant, unlabelled) produced the thread-scaling and end-to-end numbers below. A second, 26-second two-speaker recording (also unlabelled beyond "two people") was run through `bench_cli diarize` after the default cluster threshold was found to be silently merging real speakers — see the Diarization section below. Separately, llama-server's working set was measured with and without a q8_0 KV cache — on Qwen3-**4B**, not the 8B that now ships (see [LLM](#llm--llamacpp)). Since 2026-10-05 there is more, all of it English: two caption-referenced public clips (the [baseline set](#baseline-regression-test-baseline)) have been through the shipped pipeline up to the transcript, and on 2026-10-06 a 16 GB hybrid laptop (i7-1260P) was swept for thread scaling on STT, diarization and — on a Qwen3-4B stand-in — the LLM ([below](#thread-scaling-on-a-hybrid-laptop-cpu-i7-1260p-2026-10-06)). On 2026-11-19 the same laptop gave the first in-app run (a 28 s recording, through to the transcript) and a controlled STT A/B that shows **every STT timing in this document before that date was measured with whisper.cpp loading llama.cpp's ggml DLLs by accident**, a configuration that no longer ships and was ~32% faster than the one that does ([below](#runtime-isolation-and-the-first-in-app-run-2026-11-19)).

**The top unmeasured risk is the LLM itself.** Cadence now runs Qwen3-8B Q4_K_M on every machine, with a 16 GB minimum. **The 8B has never been run on any machine here**, and nobody has measured on a real 16 GB machine its peak RAM, its generation time for a 60-minute meeting, or whether it fits alongside Windows, the webview and the page cache at all. The whole single-model-set decision rests on an estimate (~6 GB at 8K context, against a 7,000 MB advisory budget). The one real data point so far is a refusal: on a 16 GB laptop with ordinary apps open, `preflight_memory` found 4.9–6.1 GB free against the 7.55 GB it needs and did not start the model ([below](#runtime-isolation-and-the-first-in-app-run-2026-11-19)).

**Nothing else has been benchmarked.** The newer `bench_cli setup`/`compare` modes and the [bench recording guide](bench-recording-guide.md) exist to close this gap on one purpose-made recording, which has not been made yet (`compare` has run only on the public council clip). The original `bench_cli` — the harness for Phases 2 (STT), 3 (diarization), 6 (LLM), and 14 (full pipeline) — was written and unit-tested to the level `cargo test` reaches; Phase 3 (diarization) has now been run against real audio several times and Phase 2 (STT) as a thread sweep, while `llm` and `full` **have never been executed against real audio**. There are no 30/45/60-minute meeting results, no Tamil or code-switched results, no labelled-speaker-count results, and no LLM quality scores. **WER exists for two English clips only**: 16.7% on the council news clip, against non-verbatim live captions and so an upper bound, and 2.8% on the TED talk, against professional captions. That says `whisper-small` is not broken on English; it says nothing about Tamil or Tanglish.

**The audio and classifier stages are entirely unmeasured.** The leveler, the compressor, the model-free noise reduction, the noise classifier, the audio quality score, the unparliamentary-language lexicon and the meeting-type classifier all ship with numbers in them, and **not one of those numbers came from a measurement**. They have unit tests and no evidence. [Unmeasured: the audio and classifier defaults](#unmeasured-the-audio-and-classifier-defaults) below lists each one and what would settle it.

**The two newest pipeline changes are unmeasured in the same way.** Word-level alignment splitting was verified against real whisper token output (single-voice probes) and against constructed diarization intervals — never against sherpa-onnx's real intervals on a genuine fast exchange, which is the case it exists for. And masking the MoM model's input has never been run on a meeting containing flagged language, so what it costs the minutes is unknown. Both are written up below: [Alignment limitation](#diarization--sherpa-onnx) and [Masking the model's input](#masking-the-models-input).

**So every model choice below is provisional.** The whisper model size, the diarization cluster threshold, and Qwen3-8B are what ships because *something* has to run out of the box (project rule: no model is permanently selected before it has been benchmarked on this project's own audio). None is a conclusion. Nothing in this document should be read as a ranking.

## Measured: the one sample

Security hardening on 2026-10-05 adds path-validation and downloader argument regression tests and a production CSP check. No AI models, prompts, or signal-processing defaults change, so no new model benchmark result is claimed. A real desktop recording/playback/processing smoke test remains necessary to verify runtime behavior under the CSP.

Follow-up regression coverage includes Windows reserved-device variants, Unicode paths, root/artifact/export junctions, model tampering/missing/truncated files, profiling without raw-file modification, development-only CSP allowances, and atomic preservation of existing recordings. These are correctness tests on synthetic artifacts, not AI performance benchmarks.

Target machine: i9-13950HX, 32 GB RAM, 32 logical threads, iGPU unused (CPU-only). Target: a 30–60 minute meeting processes in ≤ 10 minutes (≤ 15 acceptable during development).

### Thread scaling (2026-08-31)

Both native CLIs default to far fewer threads than this machine has — whisper.cpp to 4, sherpa-onnx to 1. **Shipping those defaults was costing roughly 4× on the whole pipeline.**

*Caveat added 2026-11-19:* these runs predate the per-runtime folders, when whisper.cpp and llama.cpp were flattened into one `bin/` and shared whichever `ggml*.dll` was installed last. Which ggml whisper.cpp loaded on this machine was not recorded. On the i7-1260P it was llama.cpp's, and that was worth ~32% of STT time ([below](#runtime-isolation-and-the-first-in-app-run-2026-11-19)), so treat the absolute STT walls here as possibly flattered. The thread-scaling shape is what these numbers were used for.

| whisper.cpp `-t` (`ggml-small.bin`) | Wall | Realtime |  | sherpa-onnx diarization threads | Wall | Realtime |
|---|---|---|---|---|---|---|
| 4 (default) | 20.3s | 0.35x |  | 1 (default) | 10.2s | 0.18x |
| 8 | 15.2s | 0.27x |  | 4 | 6.2s | 0.11x |
| 16 | 7.2s | 0.12x |  | 8 | 5.2s | 0.09x |
| 24 | 6.2s | 0.11x |  | 16 | 6.1s | 0.11x |

STT keeps gaining out to 24 threads; **diarization peaks at 8 and regresses past it.** Hence per-stage caps in `PipelineConfig` (`MAX_STT_THREADS` 24, `MAX_DIARIZATION_THREADS` 8) rather than one shared count. **The STT half of that does not carry to laptop parts** — on a 16 GB i7-1260P, STT is flat from 4 to 16 threads ([below](#thread-scaling-on-a-hybrid-laptop-cpu-i7-1260p-2026-10-06)). The LLM default is no longer `logical − 2` (cap 16) everywhere: on a hybrid CPU it is the P-cores' logical count, floored at 4 and capped at min(logical, 16) — 8 on the i7-1260P, 16 on this i9 (unchanged) — while a non-hybrid CPU keeps the old rule (`settings::default_llm_threads`).

### End-to-end

`pipeline_cli` on the same sample, debug build (orchestration is process-spawning and JSON parsing; the work happens in native binaries, which are release builds either way):

| Config | Wall | Realtime |
|---|---|---|
| Stock CLI defaults (4 / 1 threads) | 44.1s | 0.78x |
| Tuned (24 / 8 threads) | 14.4s | 0.25x |

Tuned stage split: preprocess ~1.2s, diarization ~7.6s, STT ~5.6s, alignment negligible. (STT here carries the same shared-`bin/` caveat as the thread-scaling table above.)

**Against that target:** 0.25x realtime puts a 60-minute meeting near 15 minutes — outside the 10-minute target, inside the 15-minute dev threshold, and that is *before* LLM generation is added. Caveats cut both ways: some of the 14.4s is fixed model-load overhead that amortizes over a long recording, but a 60× longer input has never been run. Treat as "plausibly close, unproven."

Levers not yet tried, roughly by expected value: a longer real recording (to see how much cost is fixed overhead); whisper.cpp `-p` alongside `-t`; a smaller/quantized STT model if `small` is the bottleneck at scale (trades accuracy — score it before adopting); Intel SYCL/iGPU offload (optional — CPU stays the baseline).

**Not measured at all:** peak RAM, peak CPU, and thermals per stage (all three are required); the shipped LLM's generation speed and memory (the only LLM numbers are the KV-cache comparison below and the hybrid-laptop sweep, both on a different model); anything not English. Everything in this subsection is from the 32 GB reference machine, which is no longer the minimum; the 16 GB numbers are in the next one.

### Thread scaling on a hybrid laptop CPU (i7-1260P, 2026-10-06)

**Machine:** i7-1260P — hybrid, 4 P-cores with SMT plus 8 E-cores, 16 logical — with 16 GB, i.e. the minimum machine, and the same box as the 2026-10-05 baseline runs. Quiet machine (nothing else running), AC power, Windows Balanced plan. **Input:** a 2:00 excerpt of the TED [baseline](#baseline-regression-test-baseline) clip. Repeated runs are listed where they were made; the spread between them is the noise floor, and it is not small.

*Caveat added 2026-11-19:* every STT figure in this subsection (and the 11:44 TED run quoted at its end) was measured with whisper.cpp b4938 running on llama.cpp b10717's ggml DLLs, an accident of the old shared `bin/` folder. The shipped layout gives whisper.cpp its own, older ggml, which the [controlled A/B below](#runtime-isolation-and-the-first-in-app-run-2026-11-19) found ~32% slower. The flat thread-scaling shape is not expected to change; the absolute STT walls would be higher today. Diarization and the LLM ran on their own runtimes' DLLs either way.

**STT** (`whisper-small`, `bench_cli stt --languages auto`):

| threads | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| wall | 50.7s | 53.9s | 50.6s | 49.6s | 54.5s |
| realtime | 0.42x | 0.45x | 0.42x | 0.41x | 0.45x |

**Flat.** The split says why: the encoder does scale a little (23.4 s at 4 threads, 21.0 s at 16), but beam-search decoding took ~25 s at *every* thread count — 2,671 decoder steps at ~9.4 ms each, serial by construction. Decode is half of STT on this part and threads cannot touch it; the lever is the decode strategy (beam size 1, gated on baseline WER — [roadmap.md](roadmap.md) backlog), not the thread count.

**Diarization** (sherpa-onnx, threshold 0.5):

| threads | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| wall (each run) | 18.3s | 21.1s | 16.0 / 21.6 / 19.8s | 21.1 / 20.3s | 20.8 / 20.6 / 20.8s | 23.9 / 23.6s | 34.6 / 31.1s |

Flat from 2 to 8 within run-to-run noise (three runs at 4 threads span 16.0–21.6 s), slower past 8, and ~1.6× slower at 16. The existing cap of 8 is right here too; nothing changed.

**LLM — on a stand-in.** The shipped Qwen3-8B was **not installed on this machine** (risk 17 in [roadmap.md](roadmap.md)) and is **not measured**; these are Qwen3-4B Q4_K_M, with flash attention and q8_0 KV. First, `llama-bench` at empty context:

| threads | 4 | 6 | 8 | 12 | 14 |
|---|---|---|---|---|---|
| prefill (pp512), tok/s | 18.6 | 20.1 | 20.7 | 21.6 | 21.6 |
| generation (tg64), tok/s | 8.3 | 7.2 | 6.6 | 5.6 | 5.6 |

Pinned, generation only: 4 threads on P-cores 10.4 ± 3.1 tok/s, 4 on E-cores 5.2, 8 on the P-cores plus their SMT siblings 7.6. At empty context fewer threads generate faster and the E-cores are half the speed — consistent with llama.cpp's per-layer barrier waiting on the slowest core in the pool.

Empty context is not the pipeline's workload, so the deciding run was `llama-server` with the pipeline's own flags, a 3,162-token prompt and 200 generated tokens:

| threads | prefill, tok/s | generation, tok/s | wall |
|---|---|---|---|
| 4 | 8.0 / 4.8 | 3.0 / 4.1 | 462 / 709s |
| 6 | 10.4 | 3.2 | 368s |
| **8 (P-core threads)** | **12.3 / 13.3** | **4.6 / 4.7** | **300 / 281s** |
| 14 (old `logical − 2` default) | 10.8 | 3.0 | 361s |

**At depth, 8 — the P-cores' logical count — wins on both prefill and generation**, and 4 is both slower and erratic (its two runs differ by 1.5× in wall time). That is the basis of the hybrid rule in `settings::default_llm_threads`: P-core logical threads, floored at 4, capped at min(logical, 16). The floor of 4, for parts with one or two P-cores, is **unmeasured**.

**The conclusion that matters: thread count is not the lever on laptop-class hardware, and the i9's "STT scales to 24" does not carry over.** On this part STT is flat 4–16, diarization flat to 8 and worse beyond, and the LLM best at the P-core count.

**Prefill is the minutes bottleneck.** At ~12 tok/s at depth, an ~5K-token chunk costs 6–7 minutes of prefill per call before a single token is generated — and that is the 4B; the 8B has roughly twice the weights and has not been run. On 16 GB laptop-class hardware it is minutes generation, not STT, that puts a 60-minute meeting far outside the 5–10 minute target. An earlier run of the full 11:44 TED clip on this machine, before the thread change, agrees: the pipeline up to the transcript took 600.7 s (0.85x realtime, STT alone 0.57x), and minutes with Qwen3-4B at 14 threads took **7,042 s** — a figure contaminated by builds running concurrently, so an overstatement, but not by a factor that rescues the target.

### Runtime isolation and the first in-app run (2026-11-19)

**Machine:** the same i7-1260P (16 logical, 16,008 MB), Windows 11.

**What changed.** The three runtimes used to be flattened into one shared `<models>/bin/`. whisper.cpp (b4938) and llama.cpp (b10717) both ship `ggml.dll`, `ggml-base.dll` and `ggml-cpu-*.dll`, built from different ggml versions, so whichever installed last won. On this machine `whisper.dll` (2026-08-20) had been running against llama.cpp's ggml (2026-08-31), working by ABI luck. Each runtime now installs into its own folder ([architecture.md](architecture.md#runtime-layout)). After reinstalling on this machine: `bin\whisper` 38 files / 21 MB, `bin\llama` 50 files / 44 MB, `bin\sherpa` 24 files / 36 MB, and whisper.cpp's `ggml.dll` (67,584 bytes) is no longer llama.cpp's (86,016 bytes).

**What it cost: a controlled A/B.** The 2:15 council clip (`rXRqgTI7LgU`), the same `whisper-cli.exe` b4938 and `ggml-small.bin`, flags `--language auto --max-len 80 --split-on-word --max-context 0 --beam-size 5 --best-of 5 -t 16`, runs alternated between the two layouts:

| whisper.cpp loading | run 1 | run 2 |
|---|---|---|
| llama.cpp b10717's ggml (old shared `bin/`, by accident) | 56.2 s | 55.2 s |
| its own b4938 ggml (per-runtime folder, shipped) | 73.6 s | 73.9 s |

**Isolation costs about 32% of STT time on this CPU**, and the transcripts were byte-identical, so it is pure speed. The cause is the ggml whisper.cpp now loads: the one bundled with b4938 is older and slower than the one it had been borrowing. Two runs per side is a small sample, but the gap is ~18 s against a within-side spread of ~1 s. The fix that keeps isolation is a newer whisper.cpp build with a current ggml ([roadmap.md](roadmap.md) backlog, gated on the owner's sign-off and a baseline re-run). Every STT timing in this document dated before 2026-11-19 was taken on the borrowed ggml, and is annotated where it appears.

**`bench_cli baseline` after isolation: PASS** (report `bench-runs/20261119-174323-baseline/report.md`, STT language `code-switched`, threads STT 16 / diarization 8). The `expected` blocks were deliberately **not** updated: `--update` is for an intentional improvement, and this is neither an improvement nor a change to accuracy.

| clip | WER (expected) | S / D / I | speakers (bound) | longest repeat | RTF (expected, info only) |
|---|---|---|---|---|---|
| council `rXRqgTI7LgU` | 16.4% (16.7%, ±2 pp) | 26/7/23 | 6 (2–12) | 2 | 1.44x (0.80x) |
| TED `R1vskiVDwl4` | 2.7% (2.8%) | 18/8/26 | 7 (1–2; known failure, expected 8) | 2 | 1.48x (1.19x) |

| clip | preprocess | denoise | VAD | diarization | STT | total (excl. minutes) |
|---|---|---|---|---|---|---|
| council | 12.3 s | 11.8 s | 1.5 s | 30.8 s | 149.9 s | 194.9 s |
| TED | 102.5 s | 99.1 s | 4.3 s | 141.9 s | 786.3 s (587.1 s recorded 2026-10-06) | 1036.2 s |

Read the timings with the run conditions in mind: the app was running in dev mode alongside it and only ~6 GB was free, so the absolute numbers are noisy, and the A/B above is the controlled figure. The TED STT rose by ~34% against its recorded value, which is consistent with the A/B but not independent evidence for it. WER and repeat runs held. The council clip now reports 6 speakers, against 10 recorded on 2026-10-05 (before the island fix) and 9 counted after it on 2026-10-06 ([roadmap.md](roadmap.md) risk 12). The step from 9 to 6 is inside the clip's bound and was not investigated; nothing in the runtime change is known to explain it, so do not read it as an improvement.

**The first in-app run.** Under `npm run tauri dev`, driven through the UI: a 28 s recording from the laptop's built-in Intel mic array went preprocess → GTCRN → Silero VAD → diarization → STT → align → classification → redaction scan to a transcript in ~45 s, and ~58 s on a re-run (wall times read from the app, not from `metrics.json`; one short, quiet, single-voice recording, so they say the path works, not how fast a meeting is).

**The LLM was refused, not measured.** Both times, `preflight_memory` refused Qwen3-8B: 4,921 MB free, then 6,079 MB, against the 7,550 MB it needs (6,050 MB estimated for the model at 8,192 context, plus 1,500 MB headroom). Settings showed 1.0 GB free at one point. **No LLM timing or peak-RAM figure exists for the 8B**; the only llama.cpp binary exercised from its own folder was `llama-server.exe --version` (build 10717). What the run does establish is that a 16 GB laptop with ordinary apps open did not have the free RAM the check asks for ([roadmap.md](roadmap.md) risk 14).

### Self-measurement is built in

Every real pipeline run measures itself — not just benchmark runs. `pipeline::proc::run_measured` wraps each native stage in wall-clock timing plus peak-working-set sampling (`K32GetProcessMemoryInfo`, polled every 250 ms, with a final sample before the handle closes). `pipeline::run` collects these into `RunMetrics` — per-stage wall time and peak RAM for diarization and STT, plus preprocess/align/MoM timing and the total — and writes `metrics.json` beside the meeting's other artifacts. `RunMetrics::realtime_factor()` reports total processing as a multiple of meeting length, the unit the 10-minute target is stated in (0.167x ≈ 10 minutes for a 60-minute meeting).

## Model choices (all provisional)

### STT — whisper.cpp

Candidates are whisper.cpp and sherpa-onnx's Whisper ONNX runner. Not faster-whisper: it depends on Python/CTranslate2, which conflicts with the Rust-native runtime decision.

**Required flags, not tuning knobs: `--max-len 80 --split-on-word --max-context 0`.** With whisper.cpp's defaults, a genuine ~13-second span of conversational speech was silently collapsed into one short, unrelated sentence. Cause: whisper.cpp carries a bad decode's text forward as context for the next segment, compounding the failure, and with no max segment length it gambles a long span on a single decode instead of re-segmenting. Re-verify these hold on any new test audio, especially longer, multi-speaker, and Tanglish material.

**First real run (2026-08-31).** `ggml-small.bin` on the 57.4s English sample: 6.2s at 24 threads (0.11x; measured under the old shared `bin/` layout, see the caveat on the thread-scaling table). Language auto-detect picked `en` at p=0.57 — low confidence worth watching once Tamil audio is tested. The required flags were in place and no collapse recurred. **`small` was chosen by the setup script, not by benchmark**, and it is now the only STT model in the catalog — `base` and `medium` were removed with the memory tiers. `bench_cli compare` now runs `whisper-medium` and `whisper-large-v3-turbo` (q5_0 file) alongside it as bench-only candidates. Adopting one would still mean adding it to the app catalog and pointing `system::STT_MODEL_ID` at it.

**Word timestamps are on, and they do not disturb the required flags.** `--output-json-full` (`-ojf`) is *additive* to `--output-json` — spelling verified against the vendored binary's own `--help`, not assumed — so `--max-len 80 --split-on-word --max-context 0` are passed exactly as before and the segmentation behaviour above is unchanged. Its cost has not been measured: it makes the JSON file larger and nothing has timed a run with and without it. A build too old for the flag bails on the argument before loading a model, and `stt.rs` re-runs once without it, which is a second process start rather than a second decode. The code-switched path gets the same data from `whisper-server`'s `verbose_json`, so both STT strategies populate `SttSegment.words` — on a server build old enough to omit the field, transcription is unaffected and only the word-level split is lost.

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

**A third sweep (2026-10-06) reconfirms there is no single default.** The council news clip from the [baseline set](#baseline-regression-test-baseline) — anchor, reporter and several council-room voices, true count plausibly 6–9 — swept and re-aligned with the current `align` (island absorption and `drop_nested` included). "≥5%" counts ids holding at least 5% of the talk time, which separates real voices from fragments:

| threshold | raw clusters | raw ids ≥5% | transcript speakers | transcript ids ≥5% | turns |
|---|---|---|---|---|---|
| 0.5 (shipped) | 10 | 6 | 9 | 7 | 29 |
| 0.6 | 8 | 6 | 7 | 6 | 25 |
| 0.7 | 6 | 4 | 5 | 5 | 22 |
| 0.8 | 5 | 3 | 4 | 4 | 17 |
| 0.9 | 4 | 3 | 3 | 3 | 18 |

0.5–0.6 lands in the plausible range on this clip; 0.7 and up falls below it, so — if that range is right, which nothing has labelled — merges real voices. Against that, the 2:00 TED excerpt — one speaker — gave 3 clusters at 0.5, one holding 87% of the talk: over-split, as on the first sample. So the three samples now pull both ways at once — a many-voice room wants 0.5–0.6, a single talker wants higher, and the clean two-speaker sample collapses above 0.5. **No threshold serves all of them; 0.5 stays as the less destructive failure.** The fix is not another number but more information: an expected-speakers hint (sherpa-onnx's `--clustering.num-clusters`) or a second pass that merges small clusters into their nearest large one ([roadmap.md](roadmap.md) backlog).

**Alignment limitation — addressed in code, unverified on real multi-speaker audio.** Attribution was per STT segment by maximum overlap, so a fast two-way exchange inside one whisper segment collapsed onto the dominant speaker — on the 57.4s sample, a 30s back-and-forth rendered as one Person 1 turn. The 26s sample shows the same shape: whisper.cpp split it into 5 well-formed turns, but because diarization at 0.8 returned only one speaker id for the whole clip, every turn aligned to it regardless — so this limitation and the clustering threshold compound rather than being independent bugs.

`align::speaker_pieces` now cuts a segment at a word boundary when the words inside it disagree about the speaker (three conditions, in [architecture.md](architecture.md)). **That has not been tested against real multi-speaker audio.** What was verified is narrower and worth stating exactly: word assembly was checked against real `whisper-cli.exe --output-json-full` output — single-voice probes of the vendored binaries, which is where the BPE splitting, the `[_BEG_]`/`[_TT_220]` markers and the punctuation stamped 390 ms after its word all came from — and the splitting itself against constructed diarization intervals in unit tests. Nobody has confirmed that sherpa-onnx's *real* intervals produce sensible word-level runs on a genuine fast exchange, which is the case the whole thing exists for. Neither sample above has been re-run since the split landed, and the 57.4s one is the cheapest check going: its 30s back-and-forth rendering as a single Person 1 turn is exactly the failure this is meant to fix, so re-running it and reading the result is a few minutes of work nobody has spent yet. Note the confound if it is tried on the 26s sample instead — at 0.55 and above that clip collapses to one speaker id, and with one id there is nothing to split on regardless of how good the word timings are.

**No minimum-run-length heuristic was added — deliberately, and it is the first thing to watch.** `speaker_pieces` will emit a run of a single word if that is what the intervals say, with no floor beneath it. The judgement was that suppressing short runs would guess at a problem nobody has observed yet, and would trade a visible wrong label for a silent one; that judgement is untested, and the code carries no comment defending it, so treat it as a decision to revisit rather than a settled one. If real intervals turn out to jitter at boundaries, a run-length floor is the obvious lever, and it should be sized against measured jitter rather than against this paragraph. **What to measure:** run one recording of a genuine two-way exchange through `pipeline_cli`, count how many segments split, and read the split turns against the audio — a split landing one word early is fine, a one-word run assigned to the wrong person is the failure mode.

**Update (2026-10-06): real intervals did jitter, and the fix went upstream of the split** (`absorb_islands` and `drop_nested`, [architecture.md](architecture.md)). Measured on the council clip by reading the transcript against the audio, at 0.5:

| council clip | turns | speakers | turns < 2 s | mid-sentence speaker flips |
|---|---|---|---|---|
| shipped front end, before → after | 44 → 29 | 10 → 9 | 25 → 9 | 29 → 18 |
| raw front end, before → after | 33 → 19 | — | — | 20 → 11 |

Better, not fixed: 18 mid-sentence flips remain, and the clip still reports more speakers than it plausibly has, which the threshold sweep above bears on more than alignment does. The known cost is that a gapless back-channel under 1.5 s is absorbed into the surrounding speaker. A genuine fast *exchange* has still not been through the split — a news report is mostly long turns.

Finer diarization segments remain the other half of the fix, and the clustering threshold above still gates all of it: no amount of word-level precision helps when diarization returns one id for two people.

### LLM — llama.cpp

One model, on every machine (`system::LLM_MODEL_ID`):

| id | Model | Notes |
|---|---|---|
| `qwen3-8b-q4` | Qwen3 8B Q4_K_M | ~4.8 GB of weights (5,027,783,488 bytes). 8,192-token context, q8_0 KV cache. Estimated ~6 GB of process RAM; advisory budget 7,000 MB. |

The former 4B/8B split by memory tier is gone, and Gemma3-4B survives only as a bench-only `compare` candidate (see [Comparing candidates](#comparing-candidates-setup-and-compare)). The minimum machine is now 16 GB and every machine runs the 8B. That choice rests on the 8B being the best-quality model that *should* fit 16 GB — **not** on it having been shown to fit, or to produce acceptable minutes on this project's audio. Neither has been checked.

**Measured: q8_0 KV cache (the one real LLM number).** llama-server now runs with `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0` (spellings verified against the pinned b10717 binary's `--help`). On Qwen3-**4B** Q4_K_M at 8,192 context, working set dropped from **3,604 MB (f16 KV) to 3,064 MB (q8_0 KV), ~540 MB less**. The 4B and 8B share a KV-cache shape (the code comment in `pipeline/llm/server.rs` calls it "a same-shape Qwen3 model"), so a similar saving is expected on the 8B — expected, not measured. The RAM estimate's KV coefficient in `CatalogEntry::estimated_llm_ram_mb` was lowered from 0.15 to 0.08 MB/token to match. **Unmeasured:** whether q8_0 KV changes minutes quality at all — compare `mom.json` from the same transcript with and without it.

**Unmeasured, and the top open risk: Qwen3-8B on a 16 GB machine.** What to measure, on a real 16 GB laptop with a normal set of background apps open:

- **Peak RAM** of `llama-server.exe` through a full MoM run (`bench_cli llm` reports it), against the ~6 GB estimate and the 7,000 MB budget.
- **Generation time for a 60-minute meeting**, and what it does to the 10-minute processing target — the target was already at risk before the LLM stage existed.
- **Whether it fits alongside Windows at all**: does `preflight_memory` (which wants ~7.5 GB *free*) pass on a typical 16 GB machine, does the system page heavily during generation, and does the rest of the machine stay usable. *Partly answered (2026-11-19), and not in its favour:* on the i7-1260P with ordinary apps open it did not pass — 4,921 and 6,079 MB free against 7,550 MB needed ([above](#runtime-isolation-and-the-first-in-app-run-2026-11-19)). Whether it passes once a browser is closed, and what the model then actually uses, is still unmeasured.

If it does not fit, the fallback is not a return to tiers but a smaller single model for everyone; re-evaluating newer ~8B models is on the [roadmap.md](roadmap.md) backlog.

## Unmeasured: the audio and classifier defaults

Everything in this section is implemented, unit-tested, and **has never been run against real meeting audio**. `bench_cli compare`'s front-end variants (`raw`, `shipped`, `no-leveler`, `nr-instead-of-gtcrn`, `gtcrn-only`) now exist to measure the audio front end ([below](#comparing-candidates-setup-and-compare)), but they have not been run either. Nothing in `bench_cli` sweeps an individual default such as the compressor threshold, and the classifiers still have no bench mode. The defaults below were chosen by reasoning about the failure mode each stage is meant to fix; that is a different thing from evidence, and this section exists so the two are not confused later.

The unit tests behind these stages assert *mechanism*, not quality — that the compressor reduces crest factor on a synthetic signal, that the classifier finds a 50 Hz tone that was deliberately added, that the lexicon matches a synthetic fixture list. None of them says a default is right on a real recording.

### The audio front end

| Default | Value | Why that value, and what would settle it |
|---|---|---|
| Compressor threshold | −24 dBFS | Deliberately gentle: this is an ASR front end, not a broadcast chain. **Unvalidated.** Settle it by processing the same meeting at several thresholds and comparing STT output against a hand-corrected reference — the number that matters is WER on the *far* speaker, since lifting them is the entire point. |
| Compressor ratio | 2.5:1 | Same reasoning, same gap. Too high flattens the meeting and makes a quiet room breathe; nothing has been measured to say where that starts on this material. |
| Knee / attack / release | 6 dB / 5 ms / 120 ms | Attack slow enough not to chew a consonant's onset, release slow enough not to pump — standard speech-chain reasoning, **not** tuned on this project's audio. Watch for pumping on a recording with long silences between turns. |
| Auto-makeup cap | 12 dB, ceiling −1 dBFS | A guard against a mostly-silent recording having its room tone lifted to speech level. Whether 12 dB is the right cap is untested; the ceiling is a correctness constraint rather than a tuning knob. |
| Noise-reduction strength | 0.5 | Maps onto over-subtraction and spectral floor together, so a user turns one knob rather than three. **Unvalidated**, and the risk is asymmetric: too much spectral subtraction damages speech before it sounds obviously wrong, so the number to watch is `speech_level_delta_db` in `NoiseReductionReport` — a run that quietened the voice rather than the room should be visible there. |
| Quiet-frame share for the noise profile | 30% | Assumes a meeting has pauses spread through it. A recording with almost no silence gives a profile built from speech frames; the report's `frames` count and the noise-floor delta are the check, but nothing has been checked. |
| Noise reduction on by default | on (GTCRN fallback only) | Runs only when GTCRN does not. Its cost is discussed below; its benefit is as unmeasured as its strength. |

**The leveler** (`enhance::LevelerOptions`) runs unconditionally, with no Settings toggle yet, and every default was validated **only on synthetic signals** — a constructed near/pause/far/pause/near table, silence, and an empty buffer. It reports `LevelerReport` into `metrics.json` and `pipeline_cli` prints it; the number to read is `speech_spread_before_db` → `speech_spread_after_db`.

| Default | Value | Why that value, and what would settle it |
|---|---|---|
| Max boost / max cut | ±12 dB | 24 dB of range for an assumed 20–26 dB table, split between cutting the near end and lifting the far one. Every dB of boost is a dB of that talker's room noise. **What to measure:** WER on the *far* talker with the leveler on and off — that is the whole case for the stage. |
| Window | 400 ms | Long enough to span the gaps between words, so a level is a phrase's rather than a syllable's. Unvalidated on real speech rate. |
| Speech margin | 10 dB over the floor | Less than that is taken to be a cough, a keyboard, or the room. **What to measure:** keyboard typing and HVAC misclassified as speech (and so boosted), and whether the floor estimate — read from the quietest windows — survives crosstalk, where there are few truly quiet windows to read it from. |
| Reference | mean speech-window level | Relative, not an absolute target, so normalize still owns the final level. Whether the mean is the right centre when one talker dominates the meeting is untested. |
| Rise / fall / lookahead | 250 / 100 / 300 ms | Down faster than up so a loud talker after a quiet one is not let through at the quiet one's boost; lookahead so the boost lands before the first syllable. Reasoning, not measurement. |
| Hold / pause release | 250 / 300 ms | A pause holds the last talker's gain, then relaxes to 0 dB. **What to measure:** whisper hallucinations in long pauses after a far talker — the pause is where whisper invents text, and it is the last place to add gain. |
| Interaction with the compressor | leveler first | The compressor should only catch fast peaks afterwards. **What to measure:** compressor gain reduction (`CompressorReport`) with the leveler on and off; if the compressor now does almost nothing, one of the two defaults is wrong. |
| Added time | O(n), plus sorting one level per 50 ms hop | **What to measure:** its wall time as a share of `preprocess_ms` on a 60-minute input. Expected to be negligible; that expectation is itself unmeasured. |

**Noise reduction now defaults on, and its processing cost is accepted rather than measured.** It is a full analysis/synthesis STFT over the entire recording, plus two average-spectrum passes for the before/after report — a fixed multiple of the recording length, on top of every stage that already exists. It used to be opt-in for exactly that reason: open risk 4 in [roadmap.md](roadmap.md) is that the one measured sample already extrapolates to roughly 15 minutes for a 60-minute meeting against a 10-minute target, and that was before the LLM stage existed. It is now on as the fallback for when GTCRN cannot run (GTCRN still supersedes it — never both), so on a machine with the full required set it normally costs nothing; on one without GTCRN, or where GTCRN fails and the chain is re-run, it costs its full STFT. **What to measure:** wall time of `enhance::reduce_noise` on 30/45/60-minute inputs as a share of total pipeline time, alongside whether the WER it buys is worth it. The compressor and the leveler are exempt from this concern for a structural reason rather than an optimistic one — both are O(n) with no transform — but "costs nothing measurable" is itself an unmeasured claim and should appear in `metrics.json` timings once a real run happens.

### The noise classifier

Every threshold in `noise_class.rs` is written as two readable numbers — where a finding starts, and where it stops being ambiguous — precisely so they can be moved once there is evidence. There is none yet. The ones most likely to be wrong:

| Threshold | Value | What it risks |
|---|---|---|
| Quiet-room gate | speech-to-noise ≥ 30 dB suppresses all spectral findings | Set too high, a genuinely noisy room reports nothing; too low, every recording gets a horoscope. Untested on any real room. |
| Reverberation | envelope decay > 300 ms | This is an envelope-decay estimate, **not a calibrated RT60** — a real one needs an impulse or interrupted noise source and a calibrated room. Whether 300 ms corresponds to a room that actually degrades transcription is unknown. |
| Impulsive | > 8 transients/minute | The transient test is deliberately narrow (a frame that jumps and has fallen back by the next), so ordinary speech should not register. "Should not" is the claim; a real typing-heavy meeting is the test. |
| Rumble / fan / hiss / babble band fractions | 0.30 / 0.30 / 0.20 / 0.45 | Band-share cutoffs on the room tone. The babble test additionally needs low flatness and some quiet share, to keep hiss out of it; whether that separates real corridor speech from real hiss is unverified. |
| Mains hum | harmonic score ≥ 2.0, 50 vs 60 Hz by score | A 2048-sample frame gives 7.8 Hz bins, which is enough to separate 50 from 60 Hz in principle. Never checked against a recording with a known ground loop. |

**How to settle these honestly**: they need recordings whose noise cause is *known* — a deliberately hummed recording, one made next to an air-conditioning vent, one in a bare-walled room, one with someone typing. That is a small amount of recording work and it is the only thing that turns these from guesses into thresholds. Precision matters more than recall here: advice that names the wrong cause sends a user to fix the wrong thing.

### The audio quality score

The 0–100 score and its `excellent | good | fair | poor` grade are computed from measurements that are themselves sound (sample rate, silence share, clipping percentage, peak level, speech-band SNR, and the noise findings). **The arithmetic that turns them into one number is not calibrated against anything** — neither the size of each deduction nor the 85/70/50 grade cutoffs. No study relates a Cadence score to WER, so a "fair" grade does not currently mean a measurable amount of transcription accuracy. Its honest use today is comparative — this recording is worse than that one, and here is which measurement says so — not predictive.

**What would settle it:** score a set of recordings, transcribe them, and check whether the score orders them the same way WER does. If it does not, the deductions are wrong and should move. Until that exists, treat the individual measurements and warnings as the product and the single number as a summary of them.

### Meeting-type classification

The cue phrases and their weights in `topic.rs` (longer phrases weighted higher, because they are less likely to be incidental), the square-root damping by turn count, the structural multipliers, and the score floor below which the answer is `general` — **all chosen by reasoning, none measured.**

There is no ground truth to measure against yet: that needs a set of recordings whose type is known and agreed in advance, scored for how often the classifier agrees. Two properties make that cheaper than it sounds — the classifier is deterministic, so a fixed corpus gives a stable number, and it emits its `alternatives` and `signals` alongside the label, so a disagreement can be diagnosed rather than just counted.

**Known and stated rather than measured**: type cues are English, so a Tamil-language meeting falls through to `general` rather than being mislabelled. That is the intended behaviour, not a bug to be tuned around, and it cannot be improved without the Tamil audio that open risk 3 is about. Topic extraction is language-agnostic term frequency and has no such limitation.

### The lexicon

Recall and precision of the shipped English list are **unknown**, and cannot be stated without transcripts containing real profanity to check against. Two specific gaps are recorded in [roadmap.md](roadmap.md) rather than here, because they are scope decisions rather than measurement gaps: the list ships with no slurs by design, and the two Tamil lists ship empty on purpose.

The Tier 2 defaults have the same status as everything else above: edit distance ≤ 1 for terms of 5+ characters, gated to `severe` terms, off by default. The gating and the length floor exist because a shorter term is one edit from half the language — reasoning, again, not a false-positive rate anyone has counted.

### Masking the model's input

`mom::mask_for_grounding` masks flagged spans *before* the transcript is chunked, so under `mask` or `redact` the extraction model reads `b****` where a heated exchange actually happened. That is the intended trade — minutes are the artefact that circulates, and a model that never sees the word cannot quote it back — but **it costs the model signal, and nobody has measured how much.**

The concrete worry: extraction is asked for the decision or the point being made, not the language it was made in, and a masked token still marks where the heat was. Whether the shipped model actually reads it that way, or whether a run of asterisks derails a chunk into hedged, contentless facts, is unknown. Two things sit between the risk and the output — the v2 prompts tell the model to treat `f***` and `[expunged]` as deliberate masking to carry through rather than guess at, and `flag` (the default mode) hides nothing, so the common case is unaffected — but neither has been run against a real model.

**Nothing has been through this path.** No meeting audio containing flagged language has ever been recorded, transcribed and summarised here, so there is no output to judge and no comparison to make. **What would settle it:** take one transcript that genuinely contains flagged language, generate minutes under `off`, `mask` and `redact` with the same model, and read the three side by side. The questions are whether the masked runs materially change which decisions and action items are extracted, whether any masked span leaks back into the minutes unmasked, and whether the model ever "corrects" a mask into a guess — the last being the failure the v2 prompt language is aimed at. Until that exists, `mask` and `redact` are policies whose effect on minutes quality is unknown, which is one more reason `flag` is the default.

## Running the benchmarks

From `src-tauri/`, release build:

```powershell
cargo build --release --bin bench_cli

# Phase 2 — STT: the shipped model (whisper-small) × thread counts × language strategies
.\target\release\bench_cli.exe stt <audio.wav> [--threads 8,24] [--languages auto,code-switched,ta]

# Phase 3 — diarization: cluster-threshold sweep
.\target\release\bench_cli.exe diarize <audio.wav> [--thresholds 0.5,0.6,0.7,0.8,0.9] [--threads 8]

# Phase 6 — LLM: the shipped model (qwen3-8b-q4) on one real transcript
.\target\release\bench_cli.exe llm <transcript.json>

# Phase 14 — full pipeline, with the 10-minute budget check
.\target\release\bench_cli.exe full <audio.wav>
```

These four modes have no `--models` flag: every machine runs the same STT and LLM (`app_lib::system`), so `stt` and `llm` measure only that one model. `compare` (below) is where candidates are compared.

Each of these four modes prints a Markdown table shaped to paste straight into the results section below. `stt` also writes each run's raw transcript to a temp dir for manual accuracy comparison — **WER needs a human reference transcript, which this tool does not generate.** `llm` writes the model's `mom.json` for scoring and reports peak RAM against the 7,000 MB LLM budget (`system::LLM_RAM_BUDGET_MB`) — the bar that matters on a 16 GB minimum machine, together with whether the run fits on a real 16 GB box at all. `full` computes the budget check automatically (`audio_ms × 0.167` vs actual, reported MET or MISSED-by-N-seconds).

`<transcript.json>` can come from any prior `pipeline_cli` run. Every mode needs the relevant models installed (the first-run setup screen, Settings → Models, or `scripts/setup-models.ps1`); in `stt`, a missing model prints `_not installed_` in its row rather than failing the run, while `llm` fails with the reason.

### Comparing candidates: `setup` and `compare`

**How to make the recording this is meant for, and the exact commands to run on it: [bench-recording-guide.md](bench-recording-guide.md).**

Two more modes compare the shipped set against candidates on one real recording. Neither has been run against real audio yet, so this section describes the tool, not any results. The flags below describe the harness as designed. If one errors, check the usage line `bench_cli` prints.

```powershell
# install bench-only candidate models (the only step that touches the network)
cargo run --release --bin bench_cli -- setup [--with-large] [--dry-run]

# one recording through the variant matrix
cargo run --release --bin bench_cli -- compare <audio.wav | meeting-id> [--out <dir>] [--quick]
    [--variants a,b,...] [--reference <ref.txt>] [--reference-range mm:ss-mm:ss]
    [--llm-cleanup] [--llm-file <model.gguf>[,...]]
```

**`setup`** installs candidates through the app's own installer (`models_manager::install_entry`) into a separate tree, `%APPDATA%\com.cadence.app\models\bench\`, with its own manifest. They are not in `models_manager::catalog()`, so they never appear in Settings, the setup screen or `required_ids()`, and the shipped model set is untouched. It skips anything already installed. `--dry-run` prints the table and total size without downloading.

| Kind | Baseline (shipped) | Bench candidates (`bench/catalog.rs`) |
|---|---|---|
| STT | `whisper-small` | `whisper-medium`, `whisper-large-v3-turbo` (q5_0 file) |
| LLM | `qwen3-8b-q4` (Q4_K_M) | `gemma3-4b-q4`, `qwen35-4b-q4`, `gemma4-e2b-q4`, `llama3.1-8b-q4`, `gemma4-e4b-qat-q4`, `qwen35-9b-q4`, `qwen3-8b-q5`; `--with-large` adds `gemma4-12b-qat-q4` and `gemma3-12b-q4`, quality-ceiling references that are **over the 16 GB budget by design** and could not ship |
| Denoiser | GTCRN | `denoise-dpdfnet4` (DPDFNet-4, ONNX, 16 kHz) |

`setup` installs the LLMs smallest first. `--dry-run` totals at the time of writing: 29.8 GB for the default set (plus the 12 MB DPDFNet-4 file) and 43.1 GB with `--with-large`.

**LLM candidates and why each is there.** All Gemma 4 and Qwen3.5 files are Apache-2.0. **None of these has been benchmarked yet**, on any recording. The reasons below are why each is worth a run, not results.

| id | file | why |
|---|---|---|
| `gemma3-4b-q4` | Gemma 3 4B, Q4_K_M (2.4 GB) | Half the shipped model's RAM. |
| `qwen35-4b-q4` | Qwen3.5 4B, Q4_K_M, unsloth conversion (2.6 GB) | The speed option from the Qwen3.5 family. |
| `gemma4-e2b-q4` | Gemma 4 E2B, Q4_K_M, unsloth conversion (3.0 GB) | Fast floor and reference point for the Gemma 4 family. |
| `llama3.1-8b-q4` | Llama 3.1 8B, Q4_K_M (4.7 GB) | Same size class as the shipped model, different family. |
| `gemma4-e4b-qat-q4` | Gemma 4 E4B, Google's official QAT Q4_0 (4.9 GB) | ~4.5B effective parameters (8B counting per-layer embeddings) with a small sliding-window KV cache, and by far the most token-efficient on Tamil script (below). |
| `qwen35-9b-q4` | Qwen3.5 9B, Q4_K_M, LM Studio community conversion (5.4 GB) | The shipped model's successor family (hybrid Gated DeltaNet + attention, arch `qwen35`). Qwen publishes no official GGUF. It thinks by default. The provider's `enable_thinking: false` turns that off. |
| `qwen3-8b-q5` | Qwen3 8B, Q5_K_M (5.6 GB) | The shipped model at a higher-precision quantisation. |
| `gemma4-12b-qat-q4` | Gemma 4 12B, official QAT Q4_0 (6.5 GB), `--with-large` | Gemma 4 quality ceiling. The weights alone fit under 7,000 MB, but not once the KV cache and runtime are added. |
| `gemma3-12b-q4` | Gemma 3 12B, Q4_K_M (6.8 GB), `--with-large` | Gemma 3 quality ceiling. |

*Tamil tokenization (why Gemma 4 is on the list).* One Tamil-script sentence came to 146 tokens under Qwen3-8B's tokenizer, 57 under Qwen3.5 and 28 under Gemma 4. Romanized Tanglish came out roughly equal across all three. This is **a single-sentence, indicative measurement**, not a corpus figure. It matters because every Tamil-script token spent on the transcript is a token not available for the rest of the meeting inside a fixed context. Whether it changes minutes quality is exactly what `compare` has to show.

*Runtime compatibility with the pinned llama.cpp (b10717, commit `a32af33de`).* These were checked without downloading any model. The checks used the installed binaries, the llama.cpp source at tag `b10717`, and the first 24 MB (the GGUF header) of each candidate file.

- **Architectures are supported.** `src/llama-arch.cpp` at b10717 maps `LLM_ARCH_QWEN35` to `"qwen35"` and `LLM_ARCH_GEMMA4` to `"gemma4"`. The installed `llama.dll` contains both strings, plus `llama_model_gemma4` and `llama_model_qwen35`. The GGUF headers declare `general.architecture` = `gemma4` for all three Gemma 4 files and `qwen35` for both Qwen3.5 files. `compare` therefore needs no skip rule for them. A runtime that did not know an arch would fail at load, and that failure is reported as that variant's row. It would not abort the run.
- **Thinking stays off with the existing request.** The server maps `chat_template_kwargs.enable_thinking: false` onto the template for every model. Gemma 4's embedded template defaults `enable_thinking` to false and only emits `<|think|>` when it is true. The 12B template also pre-fills an empty `<|channel>thought\n<channel|>` when it is false. Both Qwen3.5 templates emit the empty `<think>\n\n</think>` block when it is false. The 4B's unsloth template is actually off unless `enable_thinking` is explicitly true. llama-server's Gemma 4 handler parses any `<|channel>thought … <channel|>` block into `reasoning_content`, and the provider reads only `content`. **No per-model tweak was added.**
- **Flash attention + q8_0 KV work unchanged.** The CPU backend's `supports_op` accepts `FLASH_ATTN_EXT` unconditionally, and its kernel (`ggml-cpu/ops.cpp`) is generic over head size (`DK`/`DV` read from the tensors, with no fixed list). The only relevant check in `llama-context.cpp` is that the q8_0 block size (32) divides every layer's K/V head dim. Gemma 4's 256 (sliding) and 512 (global) do, and Gemma 4 requires K and V head dims to be equal, which they are. The shipped `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0` is kept for every candidate, with no fallback. If a real run ever proves this wrong, the server fails at startup with a logged reason, and `compare` records it as that candidate's failed row.

**`compare`** takes a WAV path or a meeting id. A meeting id resolves to `%APPDATA%\com.cadence.app\meetings\<id>\original.wav`, which is only ever read, and the tool refuses to write an output dir under the meetings directory. Every variant runs the real pipeline code (`pipeline::run` with an adjusted `PipelineConfig` or STT model path; `mom::generate` for minutes), so the required whisper flags, code-switched language handling, VAD, diarization and alignment are the app's. It uses shipped defaults, not the user's `settings.json`.

| Group | Variants (names usable with `--variants`) |
|---|---|
| Front end (STT fixed to `whisper-small`) | `raw` (resample to 16 kHz mono only), `shipped` (the default chain: GTCRN + leveler + compressor + normalize), `no-leveler`, `nr-instead-of-gtcrn` (model-free noise reduction in place of GTCRN), `gtcrn-only` (DC + high-pass + GTCRN, no level stages), `dpdfnet4` (default chain with DPDFNet-4 in place of GTCRN, composed bench-side because the pipeline has no switch for it) |
| STT (shipped front end) | `whisper-small`, `whisper-medium`, `whisper-large-v3-turbo` (q5_0 file) |
| LLM (minutes from the `shipped` transcript) | `qwen3-8b-q4` plus every installed LLM candidate; `--llm-file` adds an ad-hoc GGUF as `file-<stem>` |

- No flags: every variant; one that isn't installed is reported as skipped with the `setup` command that would install it.
- `--quick`: `shipped` + `whisper-small` + `qwen3-8b-q4` only, as a sanity check. It cannot be combined with `--variants`.
- `--variants`: any subset of the names above. Choosing any STT or LLM variant pulls in `shipped`, because the baseline STT row *is* the shipped run and minutes are generated from its transcript.
- `--reference <ref.txt>` computes WER (S/D/I in brackets) for every transcript variant against a plain-text file of what was said. `--reference-range mm:ss-mm:ss` (or `h:mm:ss`) limits the hypothesis to words whose midpoint falls inside the range, using word timestamps where whisper gave them. The range requires `--reference`. Normalisation is lowercase, strip punctuation and apostrophes, and split on whitespace. It is hand-written so Tamil combining marks survive. It does **not** reconcile Tamil script against romanized Tamil, the spelling variants of romanized Tamil, or `42` against `forty-two`, so on Tanglish material WER compares variants against each other better than it measures absolute accuracy.
- `--out <dir>`: defaults to `bench-runs/<UTC-timestamp>-<stem>/` at the repo root.

Output: `report.md` (front-end table with noise floor, speech level, SNR and the leveler's near/far spread before → after, plus words, speakers, hallucination signals and WER; STT table with per-stage timings and realtime factor; LLM table with load and generate time, peak `llama-server` RAM against the 7,000 MB budget (flagged **OVER** past it), chunk/repair counts and decision/action/question/risk counts). Alongside it: `transcript-<variant>.txt` (`[mm:ss] Person N: text`) and `.json`, `mom-<model>.md` and `.json`, and a `runs/` dir that keeps each variant's `processed.wav` for listening when two transcripts disagree. The hallucination signals are cheap proxies, not detection: the longest repeated n-gram run and the count of STT segments that repeat a recent one. Minutes quality is not scored by the tool. Read each `mom-<model>.md` against what was actually decided.

**`--llm-cleanup` is experimental and bench-only.** Each LLM is also asked to proofread the shipped transcript's *text* (`prompts/transcript_cleanup_v1.txt`: fix obvious ASR errors, punctuation and casing, never add, remove or translate content, and never convert between Tamil script and romanized Tamil). It writes `cleanup-<model>.txt` and a word diff `cleanup-<model>.diff.txt`, and reports tokens changed (every edit, casing included) and **content words changed** (after WER normalisation). The second number is the one to watch, since a high value is exactly the failure this pass must not have. To be plain about what it is: **an LLM is a text model and cannot clean audio.** It sees only whisper's words, so any "correction" is a guess about what was said, and it can turn an accurate transcript of an unusual phrase into a plausible wrong one. Nothing in the app uses it, and it never replaces or writes `transcript.json`. Its purpose is to measure whether such a pass would be worth building. The evidence that would settle it is WER against a reference before and after cleanup, and **the harness does not compute that yet**. `--reference` WER covers the transcript variants only, so for now the cleaned text has to be checked against the reference by hand.

`bench-runs/` holds transcripts and processed audio of real meetings. It is in `.gitignore`; never force-add it.

**Test material required**: at minimum one 30-minute, one 45-minute, and one 60-minute recording, plus at least one with genuine Tanglish — mid-sentence language changes, not Tamil and English in separate sentences. For diarization, recordings with known labelled speaker counts (3, 5, 10) are needed to judge the threshold sweep against ground truth rather than by eye, plus at least one with a genuine fast two-way exchange — interruptions and overlapping turns, not polite alternation — since that is the only material the word-level alignment split can be judged on. For MoM masking, one transcript containing real flagged language, which the shipped lexicon cannot be exercised against without. For the noise classifier, a handful of short recordings with a *known* cause — a ground loop, an air-conditioning vent, a bare-walled room, someone typing — are worth more than any amount of ordinary meeting audio, because they are the only way to tell a correct finding from a plausible one.

### Baseline regression test: `baseline`

A repeatable check that a change to the audio front end, STT, diarization or alignment did not make transcripts worse. A fixed set of public YouTube clips with human captions, listed in [`tests/baseline/clips.json`](../tests/baseline/clips.json), runs through the shipped pipeline (shipped front end, `whisper-small`, shipped diarization, shipped defaults rather than the user's `settings.json`). Each transcript is scored against its captions and against the numbers recorded for that clip.

| Clip | Length | Why it is in the set | Captions |
|---|---|---|---|
| `rXRqgTI7LgU`: KIII 3 News, Alice city council meeting | 2:15 | Real council-room audio with several distant talkers, cut with studio narration. Exercises far-field STT and diarization on many voices. | Broadcast live captions: ALL CAPS and **non-verbatim** (paraphrased, words dropped). Its WER is therefore an **upper bound** on the real error rate. It is useful for catching regressions but is not an accuracy figure. |
| `R1vskiVDwl4`: TED, Celeste Headlee, "10 ways to have a better conversation" | 11:44 | One practised speaker for ~12 minutes, with laughter and applause. Exercises long-form STT, whisper looping, and diarization **over-splitting a single talker**, the known failure mode at the shipped 0.5 threshold. Its speaker bound is 1..2 so that over-splitting fails the gate. | Professional captions, close to verbatim. `(Laughter)`/`(Applause)` annotations are dropped. |

Recorded so far: clip 1, hand-measured with `compare` on 2026-10-05: **WER 16.7%** (S27/D7/I23 over 342 caption words), 358 words, 10 speakers, longest repeat 2, no repeated segments, 0.80x RTF. The raw front end scored 16.1% with 9 speakers on the same clip, so on this clip the shipped chain is not a measurable accuracy gain. Clip 2 was recorded on 2026-10-06 (copied by hand from that day's `baseline` run on the i7-1260P): **WER 2.8%** (S20/D8/I26) against the professional captions, **8 speakers** for a single talker — listed under the clip's `known_failures`, since it fails the 1..2 bound the clip exists to enforce — and RTF 1.19x. An earlier full run of it, before the island fix and the LLM thread change, took 600.7 s to the transcript (0.85x realtime, STT 0.57x), then 7,042 s for minutes with Qwen3-4B (not the shipped 8B) at 14 threads, contaminated by concurrent builds. Every STT and RTF figure in this paragraph predates the per-runtime folders and was measured on the borrowed llama.cpp ggml. The latest run (2026-11-19, PASS) is [above](#runtime-isolation-and-the-first-in-app-run-2026-11-19).

```powershell
# once, and again only with -Force: portable yt-dlp + ffmpeg, then each clip's 16 kHz WAV and VTT
powershell -ExecutionPolicy Bypass -File scripts\fetch-baseline-clips.ps1    # needs node.js on PATH

# in src-tauri/
cargo run --release --bin bench_cli -- baseline [--clips id,..] [--cache <dir>] [--out <dir>] [--with-mom] [--update]
```

- **The audio is copyrighted YouTube content.** The fetch script caches it in `bench-runs/baseline-cache/` (gitignored) and nowhere else. Never commit it, and never put caption text into a test fixture. `bench_cli` itself only reads the cache and has no network code. A clip whose cache files are missing is an error that names the fetch script.
- **Reference text** comes from the VTT, parsed in Rust (`bench/baseline.rs`, `vtt_to_reference`). The parser drops the header, timing lines, inline tags, `>>` speaker marks and `(…)`/`[…]` annotations, and removes consecutive duplicate lines left by rolling captions. It is written to `<out>/<id>.ref.txt` so a WER can be checked by hand. WER uses the same normalisation as `compare`. If a clip's caption word count differs from the recorded one, the report says so, because a WER against a different reference is not comparable.
- **Gates.** WER is gated *relative* to the recorded value, within `tolerance.wer_pp` (2 pp) either way: worse is REGRESSED, better is IMPROVED, and no recorded value is NEW. Distinct speakers, the longest repeat run and the number of repeated STT segments are gated against *absolute* bounds per clip, because each one encodes a failure mode rather than a score. With `--with-mom`, a minutes run that fails also fails the gate. Chunk failures are reported, and minutes are skipped if the shipped LLM is not installed. **RTF and stage timings are reported only, never gated.** They depend on the machine and on whatever else is running, so do not run a baseline alongside another bench run. Exit code 1 means a gated metric regressed or a clip's pipeline failed.
- **`--update`** rewrites each run clip's `expected` block in `clips.json` with the measured values, the date, the git HEAD and the machine. Use it **only when a change is an intentional improvement** (or to record a clip's first baseline), and commit the manifest change along with the change that caused it. Under `--update`, WER changes are accepted and absolute gates still fail the run.
- Output: `report.md` (the per-clip `metric | expected | actual | Δ | status` table, stage timings, pipeline warnings), `<id>.ref.txt`, and `<id>/` with the transcript (`transcript.txt`/`.json`) and the pipeline's own artifacts (`processed.wav`, `stt.json`, `diarization.json`, `metrics.json`, plus `mom.json`/`mom.md` with `--with-mom`).

Run it before and after any change to the audio front end, STT, diarization or alignment.

## Results

Paste `bench_cli` output here as it lands, one table per recording.

### Phase 2 — STT

_Not yet run._ Note by hand how accurate the output was on each recording, per language strategy.

### Phase 3 — Diarization

_Not yet run._ Record the real speaker count next to each table so reported ids can be checked against ground truth. `bench_cli diarize` sweeps the threshold only — the word-level alignment split has no bench mode, so judge it from a `pipeline_cli` run's `transcript.json` against the audio, noting how many segments split and whether any short run landed on the wrong speaker.

### Phase 6 — LLM

_Not yet run._ Score `mom.json` against the quality rubric — factual accuracy, action-item/decision quality, Tamil handling, hallucination resistance. Record the machine's total RAM beside every row: the first run that matters is Qwen3-8B on a 16 GB machine, and a model that does not fit there does not fit the product, whatever its quality.

### Phase 14 — Full pipeline

_Not yet run._ Include the MET / MISSED-by-Ns budget line. This is the number that decides whether the 10-minute target is actually met.

### Audio front end and classifiers

_Not yet run._ The front-end comparison comes from `bench_cli compare`'s front-end table: paste it here with the near/far WER from the two `--reference-range` runs in the [recording guide](bench-recording-guide.md). The classifiers still have no bench mode. For those, and for per-stage detail, numbers come from a real pipeline run: `PreprocessReport`'s leveler, compressor and noise-reduction reports and the stage timings in `metrics.json`, next to the transcript they produced. Record the recording's known noise cause beside each row, and say plainly whether the noise classifier named it.
