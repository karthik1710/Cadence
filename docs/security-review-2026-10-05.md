# Security review — 2026-10-05

## Implementation summary

Fetched all configured remotes and pruned stale references. The clean current branch, `audio-enhancement-and-classifiers`, was already at upstream revision `9f65024`; the fast-forward merge made no changes.

Fixed unchecked meeting paths that allowed IPC callers to read, overwrite, process, or delete files outside the meeting directory. The shared resolver validates Windows-safe path components, rejects existing symlinks/junctions, and checks metadata identity. Recording, audio profiling, library, processing, export, and benchmark callers now use it. Raw artifact formats and valid meeting IDs are preserved.

Hardened model downloads: `--disable` is curl's first argument, and both initial and redirected transfers require HTTPS. Consent and sequential installation behavior stay the same.

Enabled a production CSP that restricts scripts to bundled content and connections to Tauri IPC, with explicit local audio playback allowances. Development has a separate Vite hot-reload allowance. Inline styles remain supported. References: [Tauri CSP](https://v2.tauri.app/security/csp/) and [curl flags](https://curl.se/docs/manpage.html).

## Files changed

- `src-tauri/src/storage.rs`: checked paths, metadata identity, regression tests.
- `src-tauri/src/audio.rs` and `src-tauri/src/pipeline/audio_profile.rs`: recording and profiling path validation.
- `src-tauri/src/library.rs`, `processing.rs`, `export.rs`, and `bin/bench/compare.rs`: checked resolver callers and export output link checks.
- `src-tauri/src/models_manager.rs`: HTTPS-only download command and argument regression test.
- `src-tauri/tauri.conf.json`: production and development CSP.
- `src-tauri/tests/offline_guarantee.rs`: CSP assertions and narrowly scoped internal-origin recognition.
- `docs/architecture.md`, `benchmarks.md`, `roadmap.md`, and this report: decisions, verification, and remaining risks.

## Tests run

- `npm run build`: passed (TypeScript and Vite production build).
- `cargo build --locked --offline`: passed after final code changes.
- `cargo test --locked --offline --quiet`: 384 library tests, 22 benchmark tests, and 13 integration tests passed (419 total).
- `npm audit --json`: zero advisories.
- Temporary `cargo-audit 0.22.2`, `audit --file src-tauri/Cargo.lock --json`: zero vulnerability-class findings against 1,290 RustSec advisories; database updated 2026-10-03.
- Windows `cargo tree --locked --offline -i glib` and `-i proc-macro-error`: neither is present in the active Windows graph.

## Benchmark results

No model or DSP choices changed. No new AI benchmark was run or claimed.

## Known issues

- RustSec reports unmaintained `unic-*` crates inherited through Tauri/urlpattern. There is no patched version reported; replacing those transitive APIs is outside a behavior-preserving security patch.
- The cross-platform lockfile also includes unmaintained `proc-macro-error` and the `glib 0.18.5` unsoundness advisory RUSTSEC-2024-0429. Neither is in the active Windows dependency graph; reconsider before supporting other platforms.
- Upstream model/runtime hashes remain unpinned; HTTPS and post-install hashing do not independently attest to upstream authenticity.
- Link checks cannot eliminate a malicious same-user process racing filesystem changes between validation and access.
- Automated checks passed, but a packaged desktop recording/playback/processing smoke test under the CSP has not been performed. Builds and tests do not guarantee the absence of every possible regression or vulnerability.

## Follow-up: code update and expanded tests

Recording creation now opens `original.wav` with atomic `create_new` semantics, refusing duplicate destinations without truncating earlier audio. Two tests verify that valid recorded samples and existing non-audio files remain intact.

Added 20 regression tests overall: six storage cases (reserved device families, controls/ADS/namespace paths, artifact preservation, Unicode round trips, root junctions and artifact junctions), four export/profiling cases, six downloader/model-integrity cases, two recording cases, and two CSP/asset-scope cases. The existing downloader policy check now covers every catalog entry, and export roundtrip checks verify stable output paths and unchanged raw transcript data.

Follow-up verification: `cargo test --locked --offline --quiet` passes 402 library tests, 22 benchmark tests, and 15 integration tests (439 total). `cargo build --locked --offline`, `npx tsc --noEmit`, and `npm run build` pass. No model or DSP choices changed and no new benchmark was run. Changed files for this follow-up: `audio.rs`, `storage.rs`, `models_manager.rs`, `export.rs`, `pipeline/audio_profile.rs`, `tests/offline_guarantee.rs`, and the four documentation files. Desktop smoke testing and the known issues above remain open.

### Next phase after this follow-up

Run the clean-install desktop smoke test, including local playback and a disconnected record → process → minutes cycle. Obtain independently verified upstream model/runtime hashes before packaging, and track Tauri's replacement of the unmaintained Unicode dependencies.
