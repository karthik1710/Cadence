# Privacy

EchoMind's privacy model is architectural, not a policy statement: the app cannot send meeting content anywhere because the code paths to do so do not exist in the shipped binary. This doc says exactly what is captured, where it lives, what does and doesn't leave the machine, what's logged, how deletion works, and the honest limits of the model-integrity story. See `meeting-ai-app-PLAN.md` §1, §28, §33–35 for the requirements this implements.

## What is captured

For every recorded meeting, under `<app_data_dir>/meetings/<meeting-id>/` (see `docs/architecture.md` for the full layout):

- `original.wav` — the raw recording. Never modified, never deleted except by an explicit user delete (§10).
- `processed.wav` — a resampled (16kHz mono) copy used by the AI stages.
- `diarization.json`, `stt.json` — raw per-stage output.
- `transcript.json` — the speaker-labelled transcript derived from the above.
- `mom.json` — the generated minutes (summary, decisions, action items, open questions, risks), if MoM generation is enabled.
- `metadata.json` — title, notes, timestamps, state, speaker renames.
- `metrics.json` — per-stage wall time and peak RAM for that run (§29). Contains no meeting content, only numbers.
- `exports/` — any Markdown/JSON exports the user generated.

None of this is created, read, or transmitted by anything other than the app itself running on the same machine.

## It never leaves the machine

**No telemetry.** EchoMind does not phone home. There is no analytics SDK, crash reporter, or update-check network call in the pipeline or UI code.

**No cloud STT, no cloud LLM.** Speech-to-text (whisper.cpp), diarization (sherpa-onnx), and minutes generation (llama.cpp) are native processes spawned and driven locally. The crate that ships as the app links no HTTP client for any of this — `cargo tree | grep reqwest` (or any other HTTP-client crate) returns nothing, and that's checked precisely because it makes "no cloud path snuck in" a one-line, binary-answer check rather than something that has to be re-argued by reading every module.

The only two places anything resembling network code exists in the whole crate, and why neither is a leak:

- `src-tauri/src/models_manager.rs` shells out to Windows' bundled `curl.exe` (download) and `tar.exe` (extraction) to fetch native runtimes and model weights from their public release URLs (GitHub, Hugging Face). This runs **only** when a user explicitly clicks install in Settings → Models (or runs `scripts/setup-models.ps1` at dev time) — never during recording, processing, or MoM generation. It downloads binaries and weights *into* the app; it never uploads anything, and there's no code path from a meeting's audio/transcript/MoM to this module at all.
- `src-tauri/src/pipeline/llm/http.rs` is a hand-rolled HTTP/1.1 client used to talk to the local `llama-server.exe` process over a loopback socket — that's llama.cpp's own IPC mechanism, the same role stdin/stdout plays for whisper.cpp and sherpa-onnx. It is not a general-purpose HTTP client: `request()` takes a `port`, not a host or URL, and hard-codes `127.0.0.1` as the address, so there is no parameter or code path that could point it at a non-loopback host even by mistake. This is also *why* it's hand-rolled instead of pulling in `reqwest`: adding a real HTTP client crate would defeat the `cargo tree` check above, for a need (talking to a process on the same machine) that a 250-line loopback-only client already meets.

Once models are installed, the app is designed to keep working with the network disconnected end to end — record, process, transcribe, generate minutes (§28). An automated offline-mode test for this is still open (§28); today it is an architectural property (nothing in the pipeline makes an outbound call), not something continuously verified by a test.

## What's logged

Per §34: the app logs meeting id, processing stage, which model was used, timings, memory, and errors. It does **not** log full transcript text, raw audio, or full MoM content by default — a separate debug/developer logging mode exists for deeper diagnostics, and turning it on is a deliberate, visible choice (`Settings.debug_logging`, off by default), not something that happens quietly.

One exception worth naming precisely: `llama-server.exe`'s own log (`<app_data_dir>/logs/llama-server.log`) is a diagnostic tail, truncated on every run, used only to surface the last few lines in an error message if the model server fails to start or answers with an error. llama.cpp does not log prompt or response text at its default verbosity, so this file does not contain meeting content in normal operation.

## Deletion

`storage::delete_meeting` is the only code path that removes a recording, and it only runs on an explicit user action (§10) — there is no automatic retention limit or background cleanup. Deleting a meeting removes the entire `meetings/<id>/` directory (audio, transcript, MoM, metrics, exports — everything) and the meeting's row in the SQLite history index (`database/meetings.db`) in the same call. Because the SQLite index is a rebuildable cache over `metadata.json` (see `docs/architecture.md`), there is no secondary copy of a deleted meeting's data anywhere else in the app's storage to forget about.

## Model integrity

Model weights and native runtime binaries are downloaded once (via `models_manager.rs`, described above) into `<app_data_dir>/models`, and the app runs fully offline afterwards. Integrity is **trust-on-first-install**:

- At install time, the SHA-256 of the installed file is computed and recorded in `models/manifest.json`.
- `verify()` (Settings → Models → model health check, §27) recomputes the hash and compares it against what was recorded, catching corruption, truncated downloads, or the file having been modified after install.
- **What this does not cover:** the catalog ships no pinned, independently-sourced hashes for any model or runtime. A digest recorded at install time only proves the file hasn't changed *since that install* — it cannot detect a compromised upstream release or a tampered download that happened before the hash was ever taken. TLS (via `curl.exe`) protects the download in transit; nothing in this system independently attests to what the file *should* be.

This is a real, acknowledged gap rather than a solved problem — pinning hashes for a fast-moving catalog of externally-hosted GGUF/ONNX files that this project doesn't control the release process for is a tradeoff, and it's written down here rather than implied to be stronger than it is.
