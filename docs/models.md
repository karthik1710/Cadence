# Model Selection Notes

Living document — updated as benchmarks (master prompt Phases 2, 3, 6) produce results on the Windows laptop. Candidate lists live in the master prompt itself — see `meeting-ai-app-PLAN.md` §5 (LLM), §6 (STT), §7 (diarization) — and get filled in here as real benchmark results land.

## LLM (MoM generation)

No results yet. See master prompt §5 for the current candidate list (Qwen3-8B Q4, Qwen3-4B Q4, Gemma 3 4B Q4) and §30 for the scoring rubric. Runs via llama.cpp against locally-installed GGUF weights.

## STT

**Language requirement:** meetings mix English, Tamil, and code-switched Tanglish (language changes mid-sentence) — the model must handle this without assuming a single language per session.

**Candidates:** whisper.cpp and sherpa-onnx's Whisper ONNX runner (see master prompt §6) — not faster-whisper, which depends on Python/CTranslate2 and conflicts with the Rust-native runtime decision (§19).

**Known whisper.cpp gotcha — required flags, not optional tuning:** with whisper.cpp's default decoding settings, a genuine ~13-second span of conversational speech was silently collapsed into a single short, unrelated sentence. Root cause: whisper.cpp can carry a bad decode's text forward as context for the next segment, compounding the failure, and without a max segment length it will gamble a long span on one decode attempt instead of re-segmenting. **Fix:** always pass `--max-len 80 --split-on-word --max-context 0`. Re-verify this holds once whisper.cpp is wired into the local Rust pipeline, especially on longer, multi-speaker, and Tanglish test audio.

**Known limitation to test for, not assume away:** all ASR models (Whisper included) make more errors right at a language-switch boundary than within a monolingual stretch — there's no clean fix, only benchmarking against real Tanglish audio to see how bad it is. Include "code-switched Tamil/English" as an explicit case in the master prompt's §29 benchmark dataset (test D).

No RTF/WER benchmark on the target i9-13950HX yet — to be filled in once whisper.cpp/sherpa-onnx are wired into the Rust pipeline (multi-sample, including Tanglish audio).

## Diarization

Not yet benchmarked. Candidate: an ONNX-based offline diarization pipeline (sherpa-onnx class — pyannote segmentation + speaker-embedding models exported to ONNX, invoked from Rust) — see master prompt §7. Native `pyannote-audio` (Python/PyTorch) was considered and rejected to keep the shipped runtime Python-free per §19; ONNX-ported diarization quality vs. native pyannote-audio is the open risk to benchmark. To be filled in after the diarization benchmark (master prompt Phase 3).
