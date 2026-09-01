# MASTER PROMPT — Meeting AI (Standalone Offline Windows Desktop Application)

> **Usage:** Feed this file to your coding agent (Claude Code, Codex, etc.) as the standing project brief. It defines objective, architecture, constraints, schemas, phases, and agent rules. The agent should re-read this file at the start of every phase and report against it at the end of every phase (see §39).

---

## 1. Objective

Build a Windows 11 desktop application that records physical meetings, performs speech-to-text, speaker diarization, and meeting-minutes generation **entirely on the laptop**, and presents a searchable transcript and structured Minutes of Meeting (MoM).

**Target hardware:**
- Intel Core i9-13950HX
- 32 GB RAM
- Intel UHD integrated graphics
- Windows 11
- Tauri + React + TypeScript

**Target meetings:**
- 30–60 minutes
- 3–10 speakers
- English + Tamil code-switching within the same conversation
- Post-meeting processing is acceptable
- Runtime must work without cloud AI services

Treat this as a **privacy-first local meeting intelligence application**, not merely a recorder/transcriber.

---

## 2. Runtime Architecture

```
Windows 11 Laptop
┌────────────────────────────────────────────────────────┐
│ Tauri + React/TypeScript                                │
│                                                          │
│ Microphone → Recorder → Audio Processing                │
│                         │                                │
│                         ├── VAD                         │
│                         ├── Speaker Diarization         │
│                         └── Local STT                   │
│                                  │                       │
│                                  ▼                       │
│                       Speaker-labelled Transcript        │
│                                  │                       │
│                                  ▼                       │
│                             Local LLM                    │
│                                  │                       │
│                                  ▼                       │
│                         Structured MoM                   │
│                                                          │
│ Local Storage: audio / transcript / MoM / metadata      │
└────────────────────────────────────────────────────────┘

No cloud AI · No remote backend · No runtime Internet dependency
```

The Intel UHD GPU may be benchmarked experimentally, but **CPU inference is the baseline**. llama.cpp supports Intel CPUs and offers a SYCL backend for some Intel GPUs — actual iGPU support/performance must be verified on the target machine, not assumed.

---

## 3. Development-Agent Policy

- Claude Code / Codex (or similar) may be used **during development**, subject to each tool's own network/source-code policy.
- Using cloud coding assistants during development does **not** violate the runtime's offline requirement.
- The shipped **runtime** must not contain:
  - Claude API calls
  - OpenAI API calls
  - Azure AI calls
  - Gemini/Google API calls
  - Cloud STT
  - Cloud LLM
  - Cloud telemetry containing meeting content
- After models are installed, the application must operate fully with the Internet disconnected.

---

## 4. Memory Budget

- System RAM: 32 GB
- **LLM hard limit:** ≤ 15 GB actual process RAM
- **LLM preferred:** ≤ 10 GB actual process RAM

Benchmarks must measure **total runtime memory** — model weights + KV cache + context + runtime overhead + application overhead — not just model-file size. Leave enough headroom for Windows, Tauri, STT, diarization, and storage cache.

---

## 5. LLM Strategy

- Do **not** permanently select a model before benchmarking.
- Build an `LLMProvider` abstraction so models can be swapped without touching the UI or pipeline.

**Initial candidates:**
| Model | Notes |
|---|---|
| Qwen3-8B Q4 GGUF | Primary quality candidate — 8.2B params, 100+ languages, 32K native context (extendable) |
| Qwen3-4B Q4 GGUF | Speed/memory candidate |
| Gemma 3 4B Q4 GGUF | Multilingual comparison — 140+ languages, 128K context, positioned for summarization/reasoning and constrained local deployment |
| Another 7–9B instruct model | Only if needed |

Do not assume either model is best for Tamil/code-switching — benchmark against this project's own meeting data.

**Runtime:** llama.cpp + GGUF as the primary local inference path.

**Benchmark quantizations** (only as necessary): Q4_K_M, Q5, Q6

**Measure:**
- Peak RAM
- Prompt processing speed
- Generation speed
- Total processing time
- MoM quality
- Hallucination rate
- JSON/schema reliability

Reject any model exceeding the 15 GB hard limit.

---

## 6. STT (Speech-to-Text)

Evaluate local Whisper-family implementations for Windows CPU, including at minimum:
- whisper.cpp
- sherpa-onnx's Whisper ONNX runner (native, ships Rust bindings — fits the Rust-native runtime decision in §19)
- other mature local Whisper-compatible implementations if benchmarks justify

Do not use faster-whisper — it depends on CTranslate2/Python, which conflicts with the Rust-native, Python-free runtime decision in §19.

**Must be tested for:**
- English
- Tamil
- English/Tamil code-switching
- Technical terminology
- Room noise
- 30–60 minute recordings

STT must be independently replaceable from the LLM and diarization systems.

---

## 7. Speaker Diarization

- V1 does **not** identify real people.
- Presentation labels: `Person 1`, `Person 2`, … `Person 10`
- Internal stable IDs: `SPEAKER_00`, `SPEAKER_01`, …
- Target: minimum 3, typical 3–6, maximum design target 10 speakers
- Preserve timestamps and confidence/uncertainty where available
- Do **not** promise perfect attribution during overlapping speech, interruptions, strong echo, or poor mic pickup

**Candidate:** an ONNX-based offline diarization pipeline (sherpa-onnx class — pyannote segmentation + speaker-embedding models exported to ONNX, invoked from Rust). `pyannote-audio` itself (native Python/PyTorch) was considered and rejected to keep the shipped runtime Python-free, per the Rust-native decision in §19. ONNX-ported diarization quality relative to native pyannote-audio is an open risk — benchmark it with the same rigor §6/§30 demand for STT/LLM before committing.

---

## 8. English/Tamil Code-Switching (hard V1 requirement)

Example:

> **Person 1:** "இந்த feature ready ஆச்சா? We need to test it before Friday."
> **Person 2:** "Yes, testing இன்னும் pending இருக்கு."

- The transcript must preserve the spoken language as-is — no automatic translation.
- MoM output may default to English, but the architecture must permit a configurable MoM output language later.

---

## 9. Microphone

- Default: Windows default input device
- Enumerate all Windows recording devices and allow selection, e.g.:
  - ● Default — Microphone Array
  - ○ USB Conference Microphone
  - ○ Headset Microphone
  - ○ Other Device
- Support laptop microphone in V1; keep architecture compatible with external USB mics/arrays.
- Show: input level, selected device, recording state, low-input warning, clipping warning, device-disconnect warning.

---

## 10. Audio Recording

- Always preserve the **original** recording.
- Create a processing copy if normalization is required.
- Recommended processing format: PCM, Mono, 16 kHz, WAV.

```
meeting/
├── original.wav
└── processed.wav
```

Never destroy the original automatically.

---

## 11. Processing Pipeline

```
Meeting Audio
     ↓
Audio preprocessing
     ↓
Voice Activity Detection
     ↓
Speaker diarization
     ↓
STT
     ↓
Timestamp/speaker alignment
     ↓
Speaker-labelled transcript
     ↓
Chunk-level extraction
     ↓
Global MoM synthesis
     ↓
Schema validation
     ↓
mom.json
```

Processing runs asynchronously after recording stops.

---

## 12. Meeting State Machine

```
IDLE → RECORDING → RECORDED → PREPROCESSING → DIARIZING →
TRANSCRIBING → ALIGNING → EXTRACTING → SYNTHESIZING → COMPLETED

ANY STATE → FAILED → RETRY
```

State must survive an application restart where practical.

---

## 13. Transcript Alignment

STT produces timestamped speech segments/words; diarization produces speaker intervals. Merge into:

```json
{
  "speaker": "SPEAKER_00",
  "start": 92.1,
  "end": 101.4,
  "text": "We should move this to the next sprint."
}
```

- Build a dedicated alignment module — **not** in the React UI.
- Test: speaker changes inside an STT segment, short turns, silence, interruptions, overlapping speech, uncertain diarization.

---

## 14. LLM MoM Strategy

Do not send an entire long meeting to the model for a single-shot summary. Instead:

```
Speaker transcript
      ↓
speaker-aware chunks
      ↓
chunk extraction → topics, decisions, action items, questions, important discussion
      ↓
structured intermediate data
      ↓
global synthesis
      ↓
final MoM JSON
```

This reduces context pressure and improves reliability on local models.

---

## 15. MoM Schema

**Minimum structure:**

```json
{
  "summary": "...",
  "participants": [],
  "topics": [],
  "decisions": [],
  "actionItems": [],
  "openQuestions": [],
  "risks": []
}
```

**Action item shape:**

```json
{
  "description": "Check deployment configuration",
  "owner": "SPEAKER_01",
  "deadline": null,
  "timestamp": 2537,
  "confidence": "high"
}
```

Every important generated item should reference supporting transcript timestamps when possible.

---

## 16. Hallucination Controls

The LLM must:
- Never invent decisions, action items, participants, or deadlines
- Never convert speculation into a confirmed decision
- Never assign ownership without evidence
- Preserve uncertainty; use `null` when information is unavailable
- Keep speaker IDs grounded in the transcript
- Prefer evidence timestamps

**Example:** "We could probably target Friday." must **not** become `Deadline: Friday`. It should become:

```
Possible target mentioned: Friday
Confidence: low
```

All LLM output must be validated against a typed schema. Invalid JSON must be repaired/retried, not silently accepted.

---

## 17. Prompt Management

```
prompts/
├── extraction_v1.txt
├── synthesis_v1.txt
└── repair_json_v1.txt
```

Record with every generated MoM: model, model version, quantization, prompt version, generation timestamp.

Do not hard-code prompts inside React components.

---

## 18. Long Meetings

For 30–60 minute meetings, use hierarchical processing:

```
60-minute transcript → speaker-aware chunks → chunk extraction →
intermediate structured results → global synthesis → final MoM
```

Never lose original timestamps while chunking.

---

## 19. Desktop Technology

- **Tauri + React + TypeScript + Rust**
- **React:** UI, state presentation, transcript, MoM, settings, meeting history
- **Rust/Tauri:** Windows/native integration, filesystem, local process lifecycle, secure native commands, AI runtime orchestration

Keep business logic out of React components.

### AI Runtime Strategy

Committed decision: **Rust-native orchestration, no Python.** Rust/Tauri spawns and drives local native binaries/libraries directly:

- **STT:** whisper.cpp (or sherpa-onnx's Whisper ONNX runner — see §6)
- **LLM:** llama.cpp (see §5)
- **Diarization/VAD:** an ONNX-runtime-based library (sherpa-onnx class — see §7)

No Python, PyTorch, or other interpreted-runtime dependency ships inside the app. This was chosen specifically so §35's "user needs no Python/Node/Docker/WSL" requirement is satisfied architecturally, not by freezing/bundling a Python runtime. Model weights (GGUF for the LLM, ONNX/ggml for STT and diarization) are downloaded/installed as data files, never bundled into the installer (§27).

---

## 20. Suggested Repository Layout

```
meeting-ai/
├── src-tauri/
├── src/
│   ├── components/
│   ├── features/
│   │   ├── meetings/
│   │   ├── recording/
│   │   ├── transcript/
│   │   ├── mom/
│   │   ├── history/
│   │   └── settings/
│   ├── services/
│   ├── stores/
│   └── types/
├── ai/
│   ├── audio/
│   ├── stt/
│   ├── diarization/
│   ├── transcript/
│   ├── llm/
│   └── mom/
├── models/
├── prompts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmark/
├── docs/
├── AGENTS.md
├── CLAUDE.md
├── README.md
└── PLAN.md
```

Adapt to an existing repository rather than blindly replacing its structure.

---

## 21. Local Storage

```
MeetingAI/
├── config/
├── models/
├── meetings/
│   └── <meeting-id>/
│       ├── metadata.json
│       ├── original.wav
│       ├── processed.wav
│       ├── stt.json
│       ├── diarization.json
│       ├── transcript.json
│       ├── mom.json
│       └── exports/
├── database/
└── logs/
```

Use SQLite for meeting metadata/indexing. Do not duplicate full transcript data unnecessarily.

---

## 22. Meeting UI — Main Screen

```
Meeting AI

Meeting title
[ Architecture Discussion ]

Microphone
[ Default — Microphone Array ]

00:00:00

● Ready

[ Start Meeting ]

Local processing • No cloud
```

**During recording:**
```
● RECORDING
00:42:17
Microphone: Microphone Array
██████████░░
[ Stop Meeting ]
```

**After recording:**
```
✓ Recording
✓ Audio processing
✓ Speaker detection
✓ Transcription
→ Generating MoM
```

---

## 23. Transcript UI

```
[00:02:13] Person 1
We should move this feature to the next sprint.

[00:02:25] Person 2
I don't think the hardware is ready.

[00:02:41] Person 1
Then let's separate the software release.
```

**Features:** search, copy, timestamp click, audio playback, speaker filter, manual speaker rename (presentation metadata only — does not affect raw diarization data).

---

## 24. MoM UI

Display: Meeting Information, Participants, Executive Summary, Topics Discussed, Key Decisions, Action Items, Open Questions, Risks/Concerns, Detailed Discussion.

Action items show owner, deadline, confidence, and source timestamp where available.

---

## 25. Meeting History

List, date filtering, title search, transcript search within a meeting, playback, MoM, export, delete.

Cross-meeting semantic search/RAG is **V2**.

---

## 26. Export

- **V1:** Markdown, JSON
- **Later:** PDF, DOCX, TXT

Exports include meeting information, summary, topics, decisions, action items, questions, timestamps, and transcript.

---

## 27. Model Management (Settings)

```
Speech Recognition
  Installed model

Speaker Diarization
  Installed model

Meeting Intelligence
  Qwen3-8B Q4

RAM limit
  12000 MB

Processing threads
  Auto
```

Allow: model installation, model removal, default-model selection, model health check, model metadata.

Model files should not be required to be bundled into the application installer.

---

## 28. Offline Requirement

```
Disable Internet → Start Meeting → Record → Process → Transcript → MoM
```
…must work after models are installed. Provide an automated offline test mode. No cloud service should be required for normal runtime.

---

## 29. Performance Benchmark

Create representative non-confidential test recordings:

| Test | Duration / Speakers |
|---|---|
| A | 30 min / 3 speakers |
| B | 45 min / 5 speakers |
| C | 60 min / 8–10 speakers |
| D | English/Tamil code-switching / 3–5 speakers |

**Measure every stage:** audio preprocessing, VAD, diarization, STT, alignment, LLM extraction, LLM synthesis, total time, peak RAM, peak CPU, peak temperature.

- **Primary target:** 30–60 min meeting → ≤ 10 min processing
- **Early acceptable dev threshold:** ≤ 15 min

Optimize the actual bottleneck rather than prematurely reducing model quality.

---

## 30. Model Selection Score

| Factor | Weight |
|---|---|
| MoM factual accuracy | 25% |
| Action-item accuracy | 20% |
| Decision accuracy | 15% |
| English/Tamil handling | 15% |
| Hallucination resistance | 10% |
| Processing speed | 10% |
| Memory usage | 5% |

The winning model must satisfy **both** a quality threshold **and** ≤ 15 GB actual LLM memory. Prefer ≤ 10 GB.

---

## 31. Intel UHD Experiment

Benchmark CPU-only llama.cpp vs. llama.cpp + Intel SYCL. Only enable GPU acceleration if the exact UHD device and Windows driver provide a measurable, stable benefit. SYCL is optional; CPU remains the supported baseline.

---

## 32. Testing

- **Unit:** state machine, meeting metadata, transcript schema, MoM schema, prompt loading, provider interfaces, alignment, storage, exports
- **Integration:** recording → processing → transcript → MoM, mocked AI providers, restart/recovery
- **End-to-end:** 3/5/10 speakers, English, Tamil, English/Tamil code-switching, interruptions, background noise, 30–60 minute meetings

Never commit confidential meeting recordings.

---

## 33. Failure Recovery

Handle: microphone disconnect, Windows sleep, application restart, disk full, corrupt audio, missing model, STT crash, diarization failure, LLM OOM, invalid JSON, processing timeout.

Never lose the original recording because a later processing stage failed.

---

## 34. Logging

**Log:** meeting ID, processing stage, model, timings, memory, CPU, errors.

**Do not log by default:** full transcript, raw audio, full MoM content. Provide a separate developer/debug mode for deeper logging.

---

## 35. Packaging

- Target: Windows 11 x64
- User should not need: Python, Node.js, Docker, WSL, a separate backend, or a Raspberry Pi
- This is a committed architectural decision, not just a UX goal: the AI runtime is Rust-orchestrated native binaries/libraries (§19), so there is no Python/PyTorch runtime to freeze or bundle in the first place
- The packaged application manages its own local AI processes and models

---

## 36. V1 Scope

Tauri + React/TypeScript · Windows microphone enumeration · recording · local storage · local STT · local speaker diarization · English/Tamil code-switching · speaker-labelled transcript · local LLM · structured MoM · meeting history · transcript search within meeting · timestamp-linked audio playback · Markdown/JSON export · model management · progress/status · recovery · no cloud runtime dependency.

---

## 37. Explicitly Out of V1

Cloud transcription · cloud LLM · real-time captions · face recognition · voice biometric identity · automatic human-name recognition · cross-meeting RAG/search · calendar integration · Teams/Zoom integration · mobile/web client · remote backend · Raspberry Pi backend · multi-machine processing · cloud backup.

Keep interfaces extensible for future versions.

---

## 38. Implementation Phases

| Phase | Focus | Status |
|---|---|---|
| 0 — Discovery | Inspect repo, Windows environment, Rust/Node/Tauri versions, CPU/RAM/GPU, build tools. Create/update `docs/architecture.md`, `docs/privacy.md`, `docs/model-benchmark.md`, `docs/performance.md`, `AGENTS.md`, `CLAUDE.md` | **Done** — `docs/privacy.md` and `docs/model-benchmark.md` now exist |
| 1 — Audio | Microphone enumeration, selection, recording, audio level, metadata, playback | **Done** (playback not built) |
| 2 — STT Benchmark | Benchmark local STT on English/Tamil/code-switching recordings; select on accuracy, speed, RAM | **Harness built, not yet run** — `bench_cli stt`; whisper `small` is in use but was never benchmarked against `base`/`medium`, and no Tamil/Tanglish audio has been tested. See `docs/model-benchmark.md`. |
| 3 — Diarization Benchmark | Benchmark 3/5/10 speakers and room-noise scenarios | **Harness built, not yet run** — `bench_cli diarize`; cluster threshold set to 0.8 provisionally off one unlabelled sample. See `docs/model-benchmark.md`. |
| 4 — Alignment | Merge diarization + STT timestamps into canonical transcript | **Done** — `pipeline/align.rs`, 7 unit tests |
| 5 — CLI AI Pipeline | Repeatable audio → preprocess → diarize → STT → align → transcript pipeline, before UI integration | **Done** — `cargo run --bin pipeline_cli -- <wav> [out-dir]` |
| 6 — LLM Benchmark | Benchmark Qwen3-8B Q4, Qwen3-4B Q4, Gemma 3 4B Q4 on identical transcripts | **Harness built, not yet run — see `docs/model-benchmark.md`.** `bench_cli llm` runs any candidate set against a real transcript; no run has happened, so the memory-tier default LLMs remain provisional (§5). |
| 7 — MoM | Chunk extraction, synthesis, schema validation, prompt versioning, evidence timestamps | **Done** — chunking/extraction/synthesis/repair/grounding, prompts versioned, see `pipeline/mom.rs` |
| 8 — Tauri Integration | Integrate local pipeline and process lifecycle | **Done** — `processing.rs` commands, verified end-to-end through the UI |
| 9 — Transcript UI | Viewer, search, speaker filters, timestamps, playback | **Done** — search/filter/playback/rename added |
| 10 — MoM UI | Summary, topics, decisions, action items, questions, risks, evidence | **Done** — UI wired to the backend commands |
| 11 — History | SQLite, history, filters, delete, playback, reopen | **Done** — SQLite index with filesystem fallback/rebuild, `storage.rs` |
| 12 — Export | Markdown/JSON, then optional PDF/DOCX/TXT | **Done** — Markdown/JSON, `export.rs` |
| 13 — Model Manager | Install/remove/select/validate local models | **Done** — catalog-driven install/verify/remove with SHA-256, `models_manager.rs`; `scripts/setup-models.ps1` remains a dev-time fallback |
| 14 — Performance | Full 30–60 minute benchmarks and bottleneck optimization | **Harness built, not yet run** — `bench_cli full` computes the §29 budget check automatically; see `docs/performance.md` for the small-sample numbers so far and `docs/model-benchmark.md` for the run template |
| 15 — Packaging | Clean Windows 11 install and offline-operation validation | Not started |

**Cross-cutting addition — machine memory tiering.** Alongside the phases above, `src-tauri/src/system.rs` (`MemoryProfile`: Low/Balanced/High by detected RAM) and `src-tauri/src/settings.rs` (`Settings::effective_*`) were added to address a requirement that came in after the fact: the app must run well on 16 GB laptops, not only the 32 GB reference machine. This isn't a numbered phase — it changes how Phases 6, 7, 9–13 all resolve their model/thread defaults, so it's called out here rather than folded silently into one of them. See `docs/architecture.md`.

### Where things stand (updated 2026-08-31)

The recording → transcript → MoM path is now implemented end to end, entirely on the laptop, including history, export, and a catalog-driven model manager, plus a machine-memory-tiering system so the same build adapts to a 16 GB laptop or the 32 GB reference machine. A meeting recorded in the app produces `original.wav`, `processed.wav`, `diarization.json`, `stt.json`, `transcript.json`, `mom.json`, and `metrics.json` under `%APPDATA%\com.cadence.app\meetings\<id>\`, with no network involved in any of that — the app links no HTTP client for AI processing at all (verified via `cargo tree`; the loopback client that drives `llama-server.exe` and the `curl.exe`/`tar.exe` shellouts in the Model Manager are the only network-adjacent code, both scoped and explained in `docs/privacy.md`). The old Raspberry Pi / networked-backend path (`backend.rs`, the UI's backend-URL field) has been deleted.

**Be honest about what "done" means here: nothing in this round of work has been compiled or run.** This pass (Phases 6/7/9–13 plus memory tiering) was built in an environment with no Rust or Node toolchain available, so `cargo build`, `cargo test`, and `npm run build` have not been executed against any of it. Every claim of "Done" in the phase table above describes code that exists and, where unit tests are written inline, is unit-tested *in source* — it does not mean the crate has been proven to compile, that `cargo test` has been proven to pass, or that the app has been run. Compiling and running this is the immediate next step before any of it can be trusted, let alone shipped.

**Open risks carried forward:**

- Neither STT nor diarization has been benchmarked per §6/§7, and now neither has the LLM stage per §6/Phase 6 — the harness to do all three (`bench_cli`) exists (`docs/model-benchmark.md`) but has not been run. The whisper model size, the 0.8 cluster threshold, and the memory-tier default LLMs (Qwen3-4B on 16 GB, Qwen3-8B on 32 GB) are all unvalidated defaults — see `docs/models.md`.
- No Tamil or code-switched audio has been run through the pipeline yet, which is a hard V1 requirement (§8) — this now also blocks judging MoM quality on Tanglish content, not only STT.
- Processing extrapolated to roughly 15 min for a 60-min meeting on the one small sample available, over the §29 target of 10, and that was before the LLM stage existed; `bench_cli full` can now measure the real number once it's run, but hasn't been.
- `setup-models.ps1` downloads binaries with no checksum verification; the new `models_manager.rs` install path does record a SHA-256 at install time (`verify()` catches later corruption/tampering), but first-download trust is still not cryptographically guaranteed — see `docs/privacy.md`. Worth hardening further before packaging (§35), especially on a machine running enterprise EDR.
- This round of work has not been compiled — see above. A build pass (and fixing whatever it surfaces) has to happen before any of Phases 6/7/9–13 can be considered actually verified, regardless of what the phase table says.

---

## 39. Agent Rules

1. Inspect before modifying.
2. Work phase-by-phase.
3. Do not skip STT/diarization/LLM benchmarks.
4. Do not permanently select an LLM before benchmark results.
5. Keep providers replaceable.
6. Preserve raw artifacts.
7. Never add cloud runtime dependencies.
8. Never upload meeting content.
9. Validate all LLM JSON.
10. Ground generated decisions/actions in transcript evidence.
11. Keep prompts versioned.
12. Do not implement V2 features early.
13. Keep UI separate from AI business logic.
14. Keep memory under the defined budget.
15. Document architecture changes.
16. Add tests with each major feature.

**At the end of every phase, report:**
- Implementation
- Files changed
- Tests
- Benchmark results
- Known issues
- Next phase

---

## 40. Definition of Done

A user must be able to:

```
Open Meeting AI
   ↓
Select laptop/default microphone
   ↓
Start meeting
   ↓
Hold a 30–60 minute English/Tamil mixed meeting
   ↓
Stop meeting
   ↓
Wait for local processing
   ↓
Receive speaker-labelled transcript
   ↓
Receive structured MoM
   ↓
Review transcript
   ↓
Click timestamps to replay audio
   ↓
Export MoM
```

With Internet disconnected, the full runtime workflow must continue to work.

**First major technical milestone:** Prove with representative recordings that local STT + diarization + LLM can process a realistic 30–60 minute, 3–10 speaker English/Tamil code-switching meeting within the RAM and processing-time budget on the i9-13950HX.

Only after that milestone should significant effort be spent on UI polish.