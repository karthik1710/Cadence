# Model Selection Notes

Living document — updated as benchmarks (master prompt Phases 2, 3, 6) produce results on the Windows laptop. Candidate lists live in the master prompt itself — see `meeting-ai-app-PLAN.md` §5 (LLM), §6 (STT), §7 (diarization) — and get filled in here as real benchmark results land.

## LLM (MoM generation)

No results yet. See master prompt §5 for the current candidate list (Qwen3-8B Q4, Qwen3-4B Q4, Gemma 3 4B Q4) and §30 for the scoring rubric. Runs via llama.cpp against locally-installed GGUF weights.

## STT

**Language requirement:** meetings mix English, Tamil, and code-switched Tanglish (language changes mid-sentence) — the model must handle this without assuming a single language per session.

**Candidates:** whisper.cpp and sherpa-onnx's Whisper ONNX runner (see master prompt §6) — not faster-whisper, which depends on Python/CTranslate2 and conflicts with the Rust-native runtime decision (§19).

**Known whisper.cpp gotcha — required flags, not optional tuning:** with whisper.cpp's default decoding settings, a genuine ~13-second span of conversational speech was silently collapsed into a single short, unrelated sentence. Root cause: whisper.cpp can carry a bad decode's text forward as context for the next segment, compounding the failure, and without a max segment length it will gamble a long span on one decode attempt instead of re-segmenting. **Fix:** always pass `--max-len 80 --split-on-word --max-context 0`. Re-verify this holds once whisper.cpp is wired into the local Rust pipeline, especially on longer, multi-speaker, and Tanglish test audio.

**Known limitation to test for, not assume away:** all ASR models (Whisper included) make more errors right at a language-switch boundary than within a monolingual stretch — there's no clean fix, only benchmarking against real Tanglish audio to see how bad it is. Include "code-switched Tamil/English" as an explicit case in the master prompt's §29 benchmark dataset (test D).

**First real run (2026-08-31).** whisper.cpp `ggml-small.bin` through the Rust pipeline on a 57.4s English laptop-mic recording: 6.2s at 24 threads (0.11x realtime) — see `docs/performance.md`. Language auto-detect picked `en` at p=0.57, a low confidence worth watching once Tamil/Tanglish audio is tested. The `--max-len 80 --split-on-word --max-context 0` flags were in place and no content collapse recurred. Still missing: any WER measurement, any Tamil or code-switched sample, and anything longer than a minute.

**Model size not yet justified.** `small` was chosen by the setup script, not by benchmark. §6 wants `base`/`medium` compared against it on this project's own audio before committing.

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
