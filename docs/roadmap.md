# Cadence — Status, Risks, and Backlog

Where the project actually stands, what is known to be unverified, and what is proposed next. Design lives in [architecture.md](architecture.md); model choices and measurements live in [benchmarks.md](benchmarks.md).

## V1 scope

Tauri + React/TypeScript · Windows microphone enumeration · recording · local storage · local STT · local speaker diarization · English/Tamil code-switching · speaker-labelled transcript · local LLM · structured MoM · meeting history · in-meeting transcript search · timestamp-linked playback · Markdown/JSON export · model management · progress and status · failure recovery · no cloud runtime dependency.

**Explicitly out of V1:** cloud transcription · cloud LLM · real-time captions · face recognition · voice biometric identity · automatic real-name recognition · cross-meeting RAG/search · calendar integration · Teams/Zoom integration · mobile or web client · remote backend · multi-machine processing · cloud backup. Keep interfaces extensible, but do not build these early.

## Status

The recording → transcript → MoM path is implemented end to end, entirely on the laptop, including history, export, a catalog-driven Model Manager, and a memory-tiering system so one build adapts to a 16 GB laptop or a 32 GB machine. A recorded meeting produces `original.wav`, `processed.wav`, `diarization.json`, `stt.json`, `transcript.json`, `mom.json`, and `metrics.json` under `%APPDATA%\com.cadence.app\meetings\<id>\`, with no network involved.

| Area | State |
|---|---|
| Audio capture, alignment, headless pipeline runner | Done |
| Tauri integration, MoM generation, transcript/MoM UI | Done |
| History (SQLite + filesystem rebuild), Markdown/JSON export | Done |
| Model Manager (install/verify/remove, SHA-256) | Done |
| Memory tiering (cross-cutting, not a numbered phase) | Done |
| STT / diarization / LLM / full-pipeline benchmarks | **Harness written, never run** |
| Packaging and clean-install validation | Not started |

## Open risks

These are carried forward deliberately rather than closed by assertion.

1. **Compiles and unit-tests pass, but has not been exercised on a real meeting end to end.** `cargo build`, `cargo test` (240 passing), and `npm run build` all succeed. That is not the same as the pipeline having been run against real audio in the app — which remains unverified.
2. **No sized benchmark has ever been executed.** The whisper model size and the memory-tier default LLMs are unvalidated defaults. The diarization cluster threshold now has a small threshold-sweep result (two conflicting samples, `bench_cli diarize`) but still needs labelled 3/5/10-speaker recordings to settle for real. See [benchmarks.md](benchmarks.md).
3. **No Tamil or code-switched audio has been through the pipeline**, though mixed English/Tamil is a hard V1 requirement. This now blocks judging MoM quality on Tanglish content, not just STT accuracy.
4. **Processing time may miss its target.** The one small sample extrapolates to roughly 15 minutes for a 60-minute meeting, against a 10-minute target — and that was measured before the LLM stage existed.
5. **Model integrity is trust-on-first-install.** No pinned upstream hashes; a digest recorded at install time cannot detect a download that was already compromised. Worth hardening before packaging, especially on machines running enterprise EDR.
6. **Offline operation is an architectural property, not a tested one.** Nothing in the pipeline makes an outbound call, but no automated test yet proves the app works with the network pulled.

## Backlog

Proposed, not yet scoped into a phase. Nothing here is committed work.

### 1. Unparliamentary-language detection and masking

Detect profanity, abuse, and slurs in a transcript and mask them, the way an official record expunges unparliamentary words. Meeting minutes get circulated; a verbatim transcript of a heated discussion often should not be.

**The design decision that matters most: never destroy the text.** Detection produces *annotations*, not edits. Each transcript turn gains a `flags` array — `{ start_char, end_char, term_id, category, severity }` — and `transcript.json` keeps the true text. Masking is applied at render and export time. This keeps the project's raw-artifact rule intact, makes the policy adjustable without re-running STT, keeps evidence timestamps valid, and makes the whole thing reversible if it gets something wrong. `stt.json` is never touched.

**Implementation tiers**, in the order they should be built:

**Tier 1 — lexicon matcher.** A new `pipeline/redact.rs` doing Aho-Corasick multi-pattern matching over transcript turns (the `aho-corasick` crate is pure Rust, no new runtime, no network). Ship this first; it is deterministic, auditable, fast enough to be free, and needs no model.
- Normalize before matching: Unicode NFKC, lowercase, strip diacritics, collapse repeated characters (`fuuuck`), fold leetspeak (`@`→a, `0`→o, `1`/`!`→i, `$`→s), and strip separators inside a word (`f.u.c.k`, `f u c k`).
- Match on word boundaries, with an explicit allowlist, or the Scunthorpe problem will produce embarrassing false positives.
- Terms are **data, not code**: `assets/lexicon/{en,ta,ta-latn}.txt` compiled in via `include_str!`, with a user override at `<app_data>/lexicon/` — the same pattern the versioned prompts already use, so an organization can add its own terms without a rebuild.
- Every term carries a category (`profanity | slur | sexual | violent | insult`) and severity (`mild | strong | severe`). A slur and a mild expletive should not get identical handling.

**Tier 2 — tolerance for STT error.** Whisper mishears profanity often, and some Whisper builds sanitize it outright, so exact matching will under-detect. Add bounded fuzzy matching (Levenshtein ≤ 1 for terms of 5+ characters) plus a phonetic key — Double Metaphone for English, a transliteration-normalizing key for romanized Tamil. **Gate this to `severe` terms only**; applied broadly, false positives climb fast.

**Tier 3 — local classifier, only if Tiers 1–2 prove insufficient.** A small multilingual toxicity classifier exported to ONNX, run per turn. Architecturally near-free: the app already ships an ONNX runtime for diarization, so this adds no new runtime — only a catalog entry, a Model Manager row, a memory-tier decision, and a slice of the processing budget. It buys context a wordlist cannot have: an insult assembled from individually innocuous words, or a benign use of a flagged term.
- **Prefer ONNX over reusing the loaded llama.cpp model** for this. An LLM classification prompt needs no extra download, but it is slower per run and non-deterministic — wrong properties for something that must be auditable and reproducible.

**Surfaces that must respect the setting.** A mode in Settings, overridable per meeting: `off` · `flag` (detect and highlight, original still visible — the review mode) · `mask` (`f***`, keeps first character and length) · `redact` (`[expunged]`, hides length too).
- Transcript view and MoM evidence quotes.
- Transcript search — search the true text, display the masked form, or a flagged word becomes unfindable.
- **Export.** A Markdown or JSON export taken at `mask` must not contain the original text. This is the surface where getting it wrong actually leaks.
- MoM generation: grounding should mask flagged spans in evidence, and the extraction prompt should be told not to reproduce flagged language in summaries — which means a prompt version bump.

**Limitations to write down rather than discover later:**
- A lexicon has no context. Quotation, sarcasm, and reclaimed usage will be misclassified sometimes; that is the accepted cost of a deterministic Tier 1.
- **Romanized Tamil is the hard part and the main accuracy risk.** It has no standard orthography, so one word has many spellings and recall will be poor without a large hand-built variant list. Measuring this needs real Tanglish audio, which the project does not yet have — the same blocker as risk 3 above.
- Masking the transcript does **not** mask the audio. `original.wav` still contains what was said, and playback will play it. Say so in the UI rather than implying the meeting has been sanitized. A playback bleep is a possible later item.
- **Per-speaker profanity counts are deliberately out of scope.** The data would make it trivial; it would turn a transcription tool into a surveillance one, and it cuts against the existing "never identify real people" rule. Build only on an explicit, considered request.
- Tests must use a synthetic or hashed fixture lexicon. Do not commit real slurs to the repository.

### 2. Other candidates

| Item | Why |
|---|---|
| Automated offline-mode test | Closes open risk 6 — the offline guarantee is currently unverified by any test |
| Pinned upstream model hashes | Closes open risk 5, if a maintainable source of truth for them can be found |
| Word-level STT timestamps | Would fix alignment collapsing fast two-way exchanges onto one speaker — the known diarization/alignment limitation |
| PDF / DOCX export | Named as a follow-on to the Markdown/JSON exporter |
| Intel iGPU (SYCL) offload | Optional performance lever; CPU stays the baseline |
| Playback polish | Recording playback exists in the UI but was never built out fully |
| Packaging and clean-install validation | The last unstarted phase — installer on a clean Windows 11 machine, verified offline |
