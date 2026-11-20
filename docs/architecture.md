# Cadence — Architecture

How the app is put together, why each major decision was made, and how meeting data is handled. Model choices and measured numbers live in [benchmarks.md](benchmarks.md); status, open risks, and backlog live in [roadmap.md](roadmap.md).

## Shape of the system

One Tauri desktop app at the repo root. No second machine, no networked backend, no client/server split — recording, STT, diarization, alignment, and MoM generation are all local processes orchestrated from Rust in a single binary.

```text
Cadence/
├── src/                  # React UI
├── src-tauri/src/        # Rust: audio capture, pipeline orchestration, storage
├── src-tauri/tests/      # offline_guarantee.rs — structural tests for the no-network promise
├── prompts/              # versioned LLM prompt templates (compiled in via include_str!)
└── docs/                 # architecture.md, benchmarks.md, roadmap.md
```

### Two decisions everything else follows from

**Tauri + React + TypeScript**, over WPF/WinUI. The dev machine already had Node and a Rust toolchain but no .NET SDK, so this started immediately. Windows audio access (device enumeration, level metering, disconnect handling) goes through Rust `cpal` rather than a JS audio API, keeping WASAPI-level control — see `audio.rs`, whose recording stream stays pinned to one dedicated OS thread for its whole lifetime because cpal's WASAPI backend is COM apartment-threaded.

**Rust-native AI runtime, no Python**. Rust spawns and drives native binaries directly: whisper.cpp for STT, sherpa-onnx for diarization, llama.cpp for the LLM. This is what satisfies "the user needs no Python/Node/Docker/WSL" *architecturally*, instead of by freezing and bundling a Python runtime. Those three runtimes ship **inside the signed installer**, so an installed Cadence never downloads an executable. Model weights (GGUF, ONNX/ggml) install as data files after setup and are never bundled into the installer ([Runtime layout](#runtime-layout), [release.md](release.md)).

## The processing pipeline

`src-tauri/src/pipeline/`, orchestrated by `pipeline::run`:

```text
original.wav → preprocess (16kHz mono, GTCRN, level stages) → VAD → diarize → STT → align
             → redaction scan + classification → transcript.json → MoM → mom.json
```

| Module | Role |
|---|---|
| `paths.rs` | Resolves weights under `<app_data>/models`, and each runtime's executables from one folder — the copy bundled with the app, else that runtime's own folder under app data — via `models_manager::runtime_dir`. Fails with a list of what's missing rather than a crash |
| `preprocess.rs` | Downmix + resample to 16kHz mono `processed.wav`, and the front-end filter/gain chain; never touches `original.wav` |
| `spectrum.rs` | Hand-written radix-2 FFT, averaged spectra, and an analysis/synthesis STFT — the frequency-domain machinery the audio stages share |
| `enhance.rs` | Three model-free front-end stages: a speech-gated leveler, a soft-knee compressor, and spectral-subtraction noise reduction |
| `denoise.rs` | The GTCRN speech denoiser (sherpa-onnx). Part of the required set, but a missing or failing model is a logged skip, never a failure |
| `audio_profile.rs` | Measures `original.wav` — levels, clipping, speech-band SNR, a bucketed timeline, and a quality score with the warnings behind it |
| `noise_class.rs` | Names *what kind* of background noise a recording has, and what to do about each |
| `vad.rs` | Voice activity detection — the speech ratio, and the blocks `codeswitch.rs` transcribes |
| `diarization.rs` | Spawns sherpa-onnx's offline speaker-diarization CLI, parses its stdout intervals |
| `stt.rs` | Spawns whisper.cpp `whisper-cli.exe`, parses its JSON output, and assembles per-token offsets into word timestamps |
| `codeswitch.rs` | Per-block transcription for mixed English/Tamil audio, and the timestamp remapping back onto the original timeline |
| `align.rs` | Merges STT segments with diarization intervals into the speaker-labelled transcript, after absorbing short gapless diarization "islands", splitting a segment on a word boundary when the speaker changes inside it |
| `redact.rs` | Unparliamentary-language detection over transcript turns — annotations, never edits |
| `topic.rs` | Meeting type and topic tags from the transcript, lexical and deterministic |
| `chunk.rs` | Speaker-aware transcript chunking for MoM generation |
| `mom.rs` | Mask the input → extraction → synthesis → validate/repair → grounding |
| `prompts.rs` | Versioned prompt loading, with an app-data override for iterating without a rebuild |
| `llm/` | `LlmProvider` trait and `LlamaCppProvider` driving `llama-server.exe` |
| `http.rs` | The hand-rolled loopback-only HTTP/1.1 client, shared by `llama-server.exe` and `whisper-server.exe` |
| `models.rs` | Serde types for every artifact, plus MoM schema validation |
| `proc.rs` | `run_measured`: wall time + peak RAM for every native stage, on every real run — not just benchmarks |

Each stage writes its raw output (`diarization.json`, `stt.json`) before the next runs, kept separate from the derived `transcript.json`, so a late failure never costs earlier work.

**Alignment attributes per STT segment by maximum temporal overlap**, and cuts a segment on a word boundary when the words disagree about who is speaking. Whisper ends a segment where the audio and `--max-len` say to, not where the speaker changes, so a quick "yes — no, we can't" exchange routinely lands inside one segment; whole-segment attribution then hands all of it to whoever overlapped it most and the other speaker's words leave the record entirely.

`align::speaker_pieces` splits only when three conditions all hold, because a wrong split is worse than the coarse attribution it replaces:

1. **The segment carries word timestamps** (`SttSegment.words`, below). Without them there is no boundary to cut on.
2. **The words disagree about the speaker.** If they all agree, the piece is attributed by the *segment-level* answer, not the word-level one.
3. **The words reconstruct the segment's own text** (whitespace-insensitively). A split builds each piece's text from its words, so unless they spell out exactly what the segment says, splitting would drop or invent transcript text. Whisper emits token streams this does not survive — a dropped undated token, a script without spaces between words — and those stay whole.

The invariant that falls out of conditions 1 and 2 is the one to hold onto: **word timestamps can only ever split a segment; they can never quietly re-attribute one that stays whole.** A segment with no words takes exactly the path it always did, so an `stt.json` written before word timestamps existed, a `whisper-server` build too old to report token times, and any parse that gave up all behave as before. At the two ends of a split the segment's own bounds win, so the audio before the first word and after the last is not orphaned.

Everything else is unchanged. Segments with no overlapping interval fall back to the nearest speaker; with no diarization at all they become `SPEAKER_UNKNOWN` rather than being dropped. Consecutive same-speaker pieces less than 1s apart merge into one readable turn. Stable ids (`SPEAKER_00`) and presentation labels (`Person 1`, first-appearance order) are both assigned here.

**Diarization islands are absorbed before any of that** (`align::absorb_islands`). The first real multi-speaker audio through the pipeline — a 2:15 council-meeting news clip from the caption-referenced baseline ([benchmarks.md](benchmarks.md#baseline-regression-test-baseline)) — showed sherpa-onnx's pyannote segmentation emitting back-to-back ~1 s windows, with no gap between them but different speaker ids, inside one continuous voice. The word-level split did exactly what it was built to do with those intervals and flipped the speaker every 1–2 s: one anchor's opening line came out Person 1 → 2 → 3 → 4 → 2, and the clip reported 10 speakers. This is the boundary jitter [roadmap.md](roadmap.md) risk 12 said to watch for, and it is fixed on the diarization side rather than with a minimum-run floor on the split, because the intervals were wrong before any word met them.

An island is a segment shorter than 1.5 s that touches a neighbour on *both* sides (gap ≤ 250 ms); it is relabelled. Islands whose two neighbours agree are resolved first, since that is the unambiguous case; otherwise the shortest remaining island takes its longer neighbour's speaker, and the pass repeats until nothing changes. Before that, `drop_nested` removes any segment shorter than 1.5 s that lies wholly inside an earlier one: sherpa-onnx reports overlapped speech as a second segment nested in the first (0.3–0.5 s slivers on the council clip), and since its neighbours in start order are not its neighbours in time the island test cannot see it — but its container holds the audio on both sides, which is an island whose neighbours agree, so dropping it hands the span to the container. The input is sorted by start, longest first, then speaker id, so the result never depends on the order sherpa-onnx wrote the segments in. A short segment with a pause on either side — a genuine "yes" between turns — is not an island and keeps its own speaker. The source of truth for the numbers is `ISLAND_MAX_MS` / `TOUCH_GAP_MS` in `align.rs`. Only the derived transcript changes: `diarization.json` stays what sherpa-onnx emitted, per the raw-artifact rule, so a different rule can be tried by re-running alignment over the same file. The accepted cost is that a real back-channel dropped into another speaker's run with no gap on either side looks identical to an island and is absorbed into the surrounding speaker.

**Word timestamps come from both STT paths.** `whisper-cli.exe` is passed `--output-json-full` (`-ojf`), which *adds* a per-token `tokens` array to `--output-json` and leaves the load-bearing `--max-len 80 --split-on-word --max-context 0` untouched; `whisper-server`'s `verbose_json` carries the same per-token timing under a `words` key — absent from older builds, so it is parsed as optional: a server that cannot report token times still transcribes the meeting, it just cannot split a turn mid-segment. Despite that name the server's entries are tokens too — a probe of the vendored binaries returned `" di"`, `"ar"`, `"ization"` as three separate "words" from both — so both paths share `stt::assemble_words`, which joins BPE pieces (whisper marks a word start with a leading space), drops whisper's own `[_BEG_]`/`[_TT_220]` markers, and attaches punctuation to the preceding word's *text but not its span*, since a `;` stamped 390 ms after the word it ends would move the very boundary alignment is looking for. `assemble_words` never fails: a stream it cannot make sense of yields fewer words or none, and alignment falls back to whole-segment attribution. A whisper build too old to know `-ojf` rejects the argument before it loads a model, so `stt.rs` detects that specific stderr (`unknown argument: --output-json-full`) and re-runs once without the flag rather than costing a meeting for the sake of a timing refinement; every other failure is reported on the first run.

`src-tauri/src/bin/pipeline_cli.rs` runs the whole thing headlessly:

```powershell
cd src-tauri
cargo run --bin pipeline_cli -- <audio.wav> [out-dir]
```

### The audio front end

`preprocess.rs` builds `processed.wav` from a copy of `original.wav`, which is only ever read. `apply_chain` runs, in this order:

```text
DC removal → high-pass → noise reduction → leveler → compressor → normalize
```

The order is load-bearing. Noise reduction runs **before** the leveler and the compressor, because both raise whatever is quiet and would otherwise raise hiss, and because the leveler's speech/pause decision is made against a lower, cleaner floor. The leveler runs **before** the compressor so the compressor only catches the fast peaks the slow leveler cannot follow, instead of fighting it over the same phrase-length level. The compressor runs **before** normalize so normalize is still the single stage that sets the final level and holds the peak ceiling.

Three of those stages live in `enhance.rs` and need no model at all:

- **The leveler** (`enhance::level`, `LevelerOptions`, `LevelerReport`). The compressor alone cannot close the near/far gap: it works on peaks over milliseconds, so a far talker who is *consistently* ~20 dB down sits under its threshold the whole time and is never touched. The leveler measures 400 ms windows, reads the noise floor from the quietest ones, and treats a window more than 10 dB above it as speech. Each speech window is steered towards a **reference taken from the recording itself — the mean level of the speech windows**, not an absolute target, because its job is the *distance* between talkers; an absolute target would spend the boost budget lifting a quiet microphone and duplicate what normalize already does. Gain is capped at ±12 dB (boost is also a boost of that talker's room noise). A pause never computes a gain of its own — steering the room to the reference is how a naive AGC pumps up hiss — it holds the last talker's gain for 250 ms, then relaxes towards 0 dB over 300 ms, since a long pause after a far talker is exactly where whisper hallucinates. Being offline, it looks 300 ms ahead so a quiet talker's boost is in place before their first syllable; the gain falls faster (100 ms) than it rises (250 ms). It runs **unconditionally** — there is no Settings toggle yet ([roadmap.md](roadmap.md) backlog) — and every default above was validated only on synthetic signals.

- **The compressor.** One microphone on a meeting table records the nearest person far louder than the person at the far end — the working assumption is a gap on the order of 20 dB, from inverse-square reasoning rather than from a measurement of this project's own rooms. `normalize` applies one gain to the whole file, so it either leaves the far speaker under the noise floor or pushes the near one into the ceiling. A soft-knee feed-forward compressor narrows that distance *before* the single gain is applied. It defaults **on**: it is O(n), and lifting the far end of the table above the noise floor is the cheapest accuracy lever left in the front end. Automatic makeup gain is capped (12 dB, and never past −1 dBFS) so a recording that is mostly silence cannot have its room tone lifted to speech level.
- **Model-free noise reduction.** Spectral subtraction against a noise profile estimated from the recording itself: the quietest ~30% of frames, selected by *broadband frame energy* rather than a per-bin percentile, so "the room" means "what is there when nobody is talking". It exists as the fallback for when GTCRN is not there to run — deleted, corrupted, never installed because the user skipped setup, or failing at runtime — so such a machine still gets noise handling beyond the 80 Hz high-pass. A per-bin gain floor plus gain smoothing across neighbouring bins are what keep the residual from warbling into musical noise. Its limitation follows from the method rather than the implementation, and is written down rather than discovered: a perfectly stationary signal cannot be separated from the room by anything that defines the room as "what is there when nothing is happening", so a recording that is one continuous unbroken tone would have that tone subtracted.

Noise reduction now defaults **on**, like the compressor, but **GTCRN supersedes it rather than stacking with it** — the model-free reducer only ever runs when GTCRN does not, never as a second pass on top. It was previously off because it costs a full STFT over the whole recording against an open risk that processing time may already miss its target ([roadmap.md](roadmap.md), risk 4); turning it on as the fallback accepts that cost, and it is still unmeasured.

All three stages report what they actually did (`LevelerReport`, `CompressorReport` and `NoiseReductionReport` on `PreprocessReport`, printed by `pipeline_cli`), because "the compressor was enabled" and "the compressor did something" are different claims and only the second is evidence. The leveler's headline number is `speech_spread_before_db` → `speech_spread_after_db`: the distance between the loud and quiet ends of the speech, which narrowing is the whole point of the stage.

The GTCRN path is unchanged in shape: GTCRN is a 16 kHz model so it runs after the resample, and levelling, compressing or normalising before it would hand it a boosted or moving noise floor to fight — so the front end converts without gain, denoises, then applies only the level stages (`apply_gain`: leveler, compressor and normalize, no filtering) to the denoiser's output, so the leveler runs exactly once, after the denoiser. If GTCRN fails, the chain is re-run from `original.wav` *with* the model-free reducer; a second failure falls back to the un-denoised audio. Every failure there is a logged skip, never an error that costs a meeting.

`spectrum.rs` is the shared frequency-domain layer under all of this — one radix-2 FFT, averaged spectra with band queries, and a real analysis/synthesis STFT. It is hand-written for the same reason `preprocess::Biquad` is: a DSP crate is a dependency with its own API-stability risk in a project whose point is that it ships no runtime it did not choose deliberately.

### Audio profiling and noise classification

`audio_profile.rs` measures `original.wav` — peak and RMS level, clipping, a 90th-percentile speech level, crest factor, an SNR taken over the 300–3400 Hz speech band specifically, a timeline of at most 120 evenly spread buckets for the UI to draw, and a 0–100 quality score with a coarse grade. The score starts at 100 and deducts against the measurements it already has — sample rate, silence share, clipping, peak level, speech-band SNR, and the confidence of each noise finding below — rather than being an opinion, and the `warnings` list carries the reasons.

`noise_class.rs` answers the question a bare SNR number cannot. "There is a lot of background noise" is the same sentence whether the cause is a projector fan, a ground loop, someone typing, or a hard-walled room with a long tail — and those have four different fixes, only one of which is "move the microphone closer". It classifies the room tone into named causes — mains hum (distinguishing 50 Hz from 60 Hz by their harmonics), fan/HVAC, low-frequency rumble, hiss, impulsive noise, reverberation, babble, and clipping — each with the measurement behind it and advice for the user.

It is deliberately **a signal-processing classifier, not a model**. Every decision comes from measurable properties of the spectrum and the envelope, which keeps it deterministic and auditable, adds no download and no ONNX session, and means it works on a fresh install before any model has been fetched. A learned acoustic scene classifier would be more accurate on ambiguous rooms; that is not worth a model dependency for advice a person reads once and acts on.

One design detail worth stating: the spectral findings are gated behind a level test — a room 30 dB or more below the speech is not described at all. Without that gate they fire on every recording ever made, because every room tone has a shape. What the user needs to know is not the colour of the noise but whether it is loud enough to matter.

### MoM generation

```text
transcript → mask for grounding → speaker-aware chunks → per-chunk extraction →
merged facts → (condense if needed) → synthesis → validate/repair → grounding → mom.json
```

**Masking the input** (`mom::mask_for_grounding`) runs first, before anything else reads the text. Minutes are the artefact that gets circulated, and an evidence quote is transcript text the model lifted verbatim — so masking at render time only, the way the transcript view does, would be too late for anything the model chose to repeat back. Masking the *input* means every quote downstream is masked by construction, and no later addition to `mom.rs` can bypass it by forgetting to thread a mode through. It is a copy: `transcript.json` is untouched, and `ground` below deliberately still reads the original for labels, timestamps and durations. Only `mask` and `redact` change anything — `flag` is a highlight, and a prompt has nowhere to put one, so under `flag` the model sees exactly what the transcript view shows.

It **rescans with the current lexicon and unions the result with the flags already on the transcript**, the same call `export.rs`'s `Redactor` makes, for a stronger version of the same reason. Stored flags were written by whichever scan last ran, and the gap here is wider than an export's: `pipeline::run_redaction` returns early in `off` mode, so a meeting processed with redaction off carries *no* flags at all. A user who then switches to `mask` and hits "Regenerate minutes" — exactly the sequence that switch invites — would otherwise hand the model the unmasked transcript and get unmasked minutes back, with nothing showing that anything had failed. Rescanning also picks up the user's own lexicon edits and handles a `transcript.json` written before flags existed. `redact::apply` skips overlapping spans, so a span found by both passes is masked once.

**This is the one place in the pipeline that fails closed, and the asymmetry is deliberate.** Everywhere else the rule is never to lose a meeting, so a stage that cannot run degrades to doing less. Here "doing less" means printing the words into a document that gets circulated, which is unrecoverable in the way a missing transcript is not — the file has already left. So every fallback on this path errs towards masking more: a `data_dir` that cannot supply a lexicon falls back to the built-in list rather than to no masking, and a lexicon that loads empty (every file missing or unparseable) does the same rather than being taken at its word. Note the contrast with `Redactor::none()`, which renders verbatim for a caller with no settings — an export has an explicit human behind it, and this runs unattended at the end of a pipeline. Nothing here can fail hard either: `Lexicon::load` logs and skips an unreadable override and `redact::apply` clamps out-of-range offsets, so a failure to mask can only mask more than strictly needed, never cost the minutes.

**Chunking** never splits a speaker turn mid-sentence, and overlaps one turn at each boundary so a decision straddling a cut is visible on both sides. `chunk::max_chunk_chars` sizes the budget pessimistically (~2 chars/token) because Tamil tokenizes denser than English and a code-switched meeting is a mix of both.

**Extraction** runs one call per chunk (`prompts/extraction_v2.txt`) asking only for facts with timestamps — never prose. Results merge with dedup, keeping the earliest timestamp for anything appearing in overlapping chunks. **Synthesis** (`prompts/synthesis_v2.txt`) turns merged facts into the `Mom` structure; if they still overflow the context window, `mom::condense` synthesizes in batches and folds results back (bounded to 2 rounds), with `trim_to_budget` as a last resort rather than failing outright.

Both prompts moved to v2 to carry the redaction contract, in two halves: never reproduce offensive language in anything written, even where the transcript shows it unmasked ("Person 1 objected in strong terms"), and treat runs of `*` and `[expunged]` as deliberate masking to be carried through verbatim rather than guessed at, expanded or "corrected". That is the belt to `mask_for_grounding`'s braces — the model normally never sees the words at all, but a prompt costs nothing and covers a term the lexicon does not carry. v1 of both is still on disk and still loadable, because a superseded template is never deleted: `mom.json` files already written name the version that produced them, and minutes nobody can explain afterwards are worse than minutes nobody regenerated.

**Validate/repair**: malformed JSON goes through `prompts/repair_json_v1.txt` and is re-parsed once, rather than being accepted as-is or silently discarded. `MomMeta.repaired` records whether that happened, per meeting.

**Grounding** (`mom::ground`) is what makes the output trustworthy rather than merely well-formed. It runs *after* validation and overwrites everything the model is not allowed to invent:

- `participants` comes entirely from the transcript's real speaker labels — never from model output.
- Timestamps outside the recording's actual duration (5s slack for rounding) are **dropped, not clamped**: a wrong evidence link is worse than a missing one.
- Every `owner`/`speaker` is matched case-insensitively against real transcript labels; an invented name becomes `null`. This is where V1's "never identify real people" rule is enforced at the data layer rather than by prompt convention.
- An `action_items[].confidence` outside `high|medium|low` is downgraded to `low`.

Prompts are versioned plain text under `prompts/`, compiled in via `include_str!` so a packaged install has no loose files to lose. Their version strings land in every `mom.json`'s `meta` block alongside model/quantization/runtime, so any minutes can be traced to exactly what produced them.

`MomMeta.redaction_mode` completes that provenance: it records, as a `RedactionMode` name, the policy the transcript was rendered under *before the model saw it*. The same meeting summarised under `off` and under `mask` yields materially different minutes and the two files are otherwise indistinguishable, so without it the rest of the `meta` block cannot describe — let alone reproduce — the run. A `mom.json` written before the field existed reads back as the sentinel `"unknown"` rather than failing, and deliberately not as `off`: such a file recorded no mode, and naming a real one would invent the provenance the block exists to preserve, `off` most of all, since it would assert the single thing that cannot be checked afterwards — that the model was shown the meeting verbatim. `RedactionMode::parse` rejects `"unknown"`, so anything reading it back as a policy gets `None` and has to handle the gap.

`LlamaCppProvider` starts `llama-server.exe` when MoM generation begins and kills it via `Drop` when it ends, so no model stays resident between meetings — on a 16 GB machine that's the difference between an idle laptop and a swapping one. `MockLlm` is the scripted stand-in for tests and `--dry-run`.

## Unparliamentary-language annotation

Meeting minutes get circulated; a verbatim record of a heated discussion often should not be, the way an official record expunges unparliamentary words rather than reprinting them. `redact.rs` implements that.

**The design decision that matters most: it never destroys the text.** Detection produces *annotations*, not edits. Each `TranscriptSegment` gains a `flags` array — `{ start_char, end_char, term_id, category, severity, method }` — and `transcript.json` keeps exactly what was said. Masking is applied at render and export time instead. `original.wav`, `stt.json` and `diarization.json` are never touched by any of this; that rule is load-bearing here. The consequences are all in the same direction: the policy is adjustable without re-running STT, evidence timestamps stay valid, and a wrong flag is reversible.

Offsets are **Unicode scalar (`char`) offsets** into that turn's text — not bytes, which would be meaningless across the bridge, and not UTF-16 units. The frontend slices with `Array.from(text)` to match.

Two tiers are implemented:

- **Tier 1, a lexicon matcher.** Aho-Corasick over normalised turn text. Normalisation folds case, leetspeak (`@`→a, `0`→o, `1`/`!`→i, `$`→s), repeated characters (`fuuuck`) and separators inside a word (`f.u.c.k`, `f u c k`), keeping a map back to the original offsets so a match can actually be located. Word-boundary matching plus a per-lexicon allowlist (`!assessment`) is what keeps the Scunthorpe problem from producing embarrassing false positives. The automaton is hand-written for the same reason the FFT and the biquad are, and because the alphabet here is Unicode rather than bytes.
- **Tier 2, tolerance for STT error.** Whisper mishears profanity often and some builds sanitise it outright, so exact matching under-detects. Bounded edit distance (≤ 1, terms of 5+ characters) and a crude phonetic key close some of that gap. Both are **gated to `severe` terms only** and are off by default — applied broadly the false-positive rate climbs fast, and a false positive on innocent speech in a circulated record is the failure that actually costs something.

Tier 3, a learned classifier, is deliberately **not** built. See [roadmap.md](roadmap.md) for the reasoning.

**Terms are data, not code.** The built-in lists are `src-tauri/assets/lexicon/{en,ta,ta-latn}.txt`, compiled in via `include_str!`, in a `term | category | severity` format with `#` comments and `!word` allowlist entries. An organisation overrides them without a rebuild by dropping a file of the same format at:

```text
%APPDATA%\com.cadence.app\lexicon\{en,ta,ta-latn}.txt
```

Override files are merged *over* the built-ins at load time — later sources win, so an override entry for an existing term replaces its category and severity. This is the same pattern the versioned prompts already use. An unreadable override is skipped with a log line rather than being an error: a typo in an optional policy file must not stop a meeting from being processed.

Categories are `profanity | slur | sexual | violent | insult` and severities `mild | strong | severe`, because a slur and a mild expletive should not get identical handling. `Settings.redaction_mode` selects `off` (do not even scan) · `flag` (detect and highlight, original still visible) · `mask` (`f***`, first character and length kept) · `redact` (`[expunged]`, length hidden too). **`flag` is the default** — it is the only honest one for a lexicon that cannot tell a quotation from an insult, and it means a human sees what was found before anything is hidden. Every surface that shows transcript text has to respect the mode, and the two that can leak off the machine are held to a stricter rule than the rest: export, because the file leaves, and MoM generation, because minutes get circulated and the model would otherwise be free to quote the words back. Both re-scan rather than trusting stored flags; see [MoM generation](#mom-generation) and [Export](#export).

The shipped lexicon's coverage, and what it deliberately omits, is recorded in [roadmap.md](roadmap.md) rather than implied here.

## Meeting classification

`topic.rs` derives two things from the transcript alone: a **meeting type** (standup, planning, review, retrospective, decision, brainstorm, interview, one-on-one, training, troubleshooting, or the honest `general`) and a set of **topic tags** — the terms the meeting actually kept returning to. Both feed the history list, where twenty meetings called "Weekly sync" are otherwise indistinguishable.

It is lexical and deterministic, **not** an LLM call, and the reasoning is stronger here than for the noise classifier: the app already has a local LLM loaded during MoM generation, so asking it "what kind of meeting is this" would cost no extra download. That was rejected anyway. An LLM answer is non-deterministic, costs seconds of the processing budget, would change whenever the shipped model does (and did, back when the model depended on the machine's RAM tier), and cannot be reproduced when someone asks why a meeting was labelled the way it was. Counting words is reproducible.

Type scoring counts weighted cue phrases over the transcript, damped by the square root of the turn count so a two-hour meeting does not outscore a ten-minute one on every type simply by having more words in it. Structural signals — speaker count, mean turn length, question ratio, dominant-speaker share, duration — then apply as *multipliers* rather than additions, so structure can only support or undercut what the words already said, never invent a type on its own: a two-person meeting is not a one-on-one because it has two people in it, it is a one-on-one if it also sounds like one. Below a floor score the answer is `general`. The signals it used are serialized alongside the label so a classification can be argued with rather than merely believed.

Topic tags are term frequency crossed with *spread* — how many separate turns a term appears in — so one person saying "migration" nine times in one answer does not become what the meeting was about. Bigrams are scored alongside single words and suppress their own halves when they survive.

**Type classification is cued in English**, which is a stated limitation rather than a hidden one: a meeting conducted mostly in Tamil falls through to `general` with low confidence rather than being mislabelled. Topic extraction has no such problem — it is pure term frequency and works on whatever language the transcript is in.

## One model set, 16 GB minimum

Cadence used to bucket machines by RAM into `Low`/`Balanced`/`High` profiles and pick a smaller LLM (and on `Low`, a smaller Whisper) on smaller boxes. That is gone. The minimum supported machine is now 16 GB, and **every machine runs the same models**, sized to fit that minimum. One set means a meeting is summarised by the same model on a 16 GB laptop as on a 32 GB workstation, and there is exactly one configuration to benchmark and support.

The constants live in `src-tauri/src/system.rs`:

| Constant | Value | Notes |
|---|---|---|
| `MIN_TOTAL_RAM_MB` | 15,000 | Below 16,384 on purpose: Windows reports *usable* physical memory, and a 16 GB laptop with iGPU-reserved memory commonly shows ~15.7 GB |
| `STT_MODEL_ID` | `whisper-small` | Multilingual |
| `LLM_MODEL_ID` | `qwen3-8b-q4` | Qwen3-8B Q4_K_M, ~4.8 GB of weights |
| `LLM_CONTEXT_TOKENS` | 8192 | MoM generation is chunked so it never needs more |
| `LLM_RAM_BUDGET_MB` | 7,000 | Advisory ceiling on the LLM process, sized for 16 GB |

`system::HostInfo::detect()` still reads total/available RAM and core count via `GlobalMemoryStatusEx` (Windows API, no crate) — not to choose anything, but so `HostInfo::meets_minimum()` can tell a machine below the minimum so plainly. That check is **advisory only**: it produces a warning and never blocks recording, because a meeting captured to disk can always be processed later.

`settings::Settings` no longer has `profile`, `stt_model`, `llm_model` or `llm_context_tokens` fields (a legacy `settings.json` that still carries them loads fine; they are ignored). `effective_stt_model()`/`effective_llm_model()`/`effective_llm_context()` survive as accessors that return the `system` constants, so the pipeline still reads model choice in one place; `PipelineConfig::from_settings` and `paths::ModelPaths` build on them. Thread counts are still per-machine: `stt_threads`/`diarization_threads` are optional overrides, defaulting to the core count capped per stage — `MAX_STT_THREADS` 24, `MAX_DIARIZATION_THREADS` 8, `MAX_LLM_THREADS` 16. The reasoning is measured — see [benchmarks.md](benchmarks.md). The LLM default additionally reads the CPU's core layout: `HostInfo::topology()` calls `GetLogicalProcessorInformationEx(RelationProcessorCore)` (no crate) and treats the cores of the highest `EfficiencyClass` as the P-cores. On a hybrid Intel part llama.cpp's threads meet at a barrier every layer, so an E-core in the pool sets the pace for all of them; `settings::default_llm_threads` therefore uses the P-cores' logical count (floored at 4, capped at the logical count and 16) — 8 on an i7-1260P, 16 on the i9-13950HX, which is what the old `logical − 2` rule already gave it. A non-hybrid CPU, a failed query, or a hand-built `HostInfo` whose core count does not match the running machine gets the old rule unchanged. `RunMetrics` still records which models a run used, so a run from before the switch is never silently compared with one after it.

**llama-server runs with a quantized KV cache**: `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0` (spellings verified against the pinned b10717 build; a quantized V cache requires flash attention, so it is forced on rather than left to `auto`). That is what makes 8K context affordable on 16 GB — measured at ~540 MB less working set on Qwen3-4B at 8192 context ([benchmarks.md](benchmarks.md)). q8_0 was chosen over q4_0 as the smaller saving at the smaller precision cost; its effect on minutes quality is unmeasured.

Before starting `llama-server.exe`, `preflight_memory` checks the model's estimated RAM (weights + ~0.08 MB/token KV cache at q8_0 + ~600 MB overhead — 6,050 MB for the 8B at 8K) against actually-free RAM, reserving a 1,500 MB slice for Windows/webview/file cache, so it wants 7,550 MB free. A model that won't fit is refused *before* it OOMs mid-generation. There is no smaller model to fall back to, so the message states the threshold, what is free, and the shortfall — "…needs about 7550 MB of free RAM (6050 MB for the model at a 8192 token context, plus 1500 MB kept free for Windows and Cadence), but only 6079 MB is free … Close other applications (browsers especially) to free at least 1471 MB more, then regenerate the minutes" — and the transcript is already saved by then. That exact refusal is the one real data point so far: on a 16 GB laptop with ordinary apps open, the first in-app run was refused twice ([benchmarks.md](benchmarks.md#runtime-isolation-and-the-first-in-app-run-2026-11-19)). This is an estimate-based guard, not a hard allocator limit (llama.cpp has none), and it blocks only when *available RAM* is short; exceeding `LLM_RAM_BUDGET_MB` while the RAM is physically there is logged, not refused. Whether the 8B actually fits alongside Windows on a real 16 GB machine is the top unmeasured risk ([roadmap.md](roadmap.md)).

## Model Manager

`models_manager.rs` is the catalog-driven install/remove/verify system that replaced "run `scripts/setup-models.ps1` and hope" (the script remains a dev-time fallback provisioning the same layout).

`catalog()` is a static list of every installable item — three runtimes (whisper.cpp b4938, llama.cpp b10717, sherpa-onnx v1.13.6) plus weights: Whisper small, Qwen3-8B Q4_K_M, the two diarization ONNX models, the GTCRN speech denoiser and Silero VAD. There is exactly one STT and one LLM; the former alternatives (whisper-base, whisper-medium, Qwen3-4B, Gemma3-4B) were removed along with the memory tiers. Each entry carries its download URL, install path, and the file that must exist for it to count as installed. `required_ids()` is the whole set, in install order (runtimes first, the ~4.8 GB LLM last); `missing_required_ids()` and `required_download_mb()` feed the first-run screen.

GTCRN and Silero (about 1 MB each) are in the required set because they are the cheapest voice/noise gain there is, but `ModelPaths::missing()` does **not** gate processing on them: the pipeline skips each gracefully when absent, so a deleted or corrupt file never costs a meeting. Settings → Models lists every catalog item, grouped by kind — GTCRN as "Speech enhancement" and Silero as "Speech detection", which were required but invisible there before — with an "Other" group so a kind added later cannot silently disappear from the list. Sizes shown are what is on disk, rounded **up** to whole megabytes so a 1 MB model never reads "0 MB"; a runtime in its own folder is sized as that whole folder, and one still on the legacy layout (below) falls back to the catalog estimate, since its launcher alone would understate it by orders of magnitude.

`remove()` refuses to delete anything in the required set — which is now everything in the catalog — so a user can't accidentally leave the app unable to process a meeting; Verify and **Reinstall** (a button on every installed item, behind a confirmation that states the download size) are the recovery path for a damaged file. The exception is a runtime bundled with the app, which has no Reinstall button: reinstalling Cadence repairs it ([Runtime layout](#runtime-layout)).

### Runtime layout

**Bundled first.** A release installer ships all three runtimes in `runtimes\` beside the app executable: `runtimes\whisper\`, `runtimes\llama\`, `runtimes\sherpa\`, plus `manifest.json` and `THIRD_PARTY_NOTICES.txt`. Tauri installs bundle resources next to the executable on Windows. In a dev build, tauri-build copies `vendor/runtimes/` into `target/<profile>/runtimes/` in the same way, so `pipeline_cli` and `bench_cli`, built into that folder, resolve the same copy. `scripts/prepare-runtimes.ps1` fills `vendor/runtimes/` at build time (`build.beforeBundleCommand`). It takes each runtime from its pinned release archive, refusing one whose SHA-256 differs from the pin, keeps the DLLs and only the executables Cadence launches, signs them, and records every shipped file's SHA-256 in `manifest.json` ([release.md](release.md)). Only a placeholder README is committed there, so a fresh clone still builds and runs, with no bundled runtimes.

Resolution is decided per runtime, on its launcher, in this order: **the bundled copy → its own folder under app data (`<models>/bin/<runtime>/`) → the legacy flat `<models>/bin/`**. `models_manager::runtime_dir` is the one place that decides. The bundled root is `<exe dir>/runtimes`, read once per process (`bundled_runtimes_root`), and it is `None` under `cargo test`, so unit tests inject a temporary one instead of depending on what sits beside the test binary. A bundled runtime:

- counts as installed and never appears in `missing_required_ids`, so on a release build first-run setup lists **model data only**;
- is refused by `install_model`/`install()` before anything is fetched: "… is included with Cadence — reinstall Cadence to repair it". Settings shows it as **Included with Cadence**, with Verify but no Install or Reinstall (`ModelStatus.bundled`);
- is sized as its bundled folder;
- is verified against the bundled `manifest.json` rather than the app-data manifest (see [Integrity](#installs-one-at-a-time-never-during-a-meeting) below).

The download path for runtimes is unchanged and still works. It is what a development build without bundled runtimes uses.

**The downloaded layout.** Without a bundled copy, each runtime installs into **its own folder** — `<models>/bin/whisper/` (`whisper-cli.exe`, `whisper-server.exe`), `<models>/bin/llama/` (`llama-server.exe`) and `<models>/bin/sherpa/` (diarization, `sherpa-onnx-offline-denoiser.exe` for GTCRN, `sherpa-onnx-vad.exe`, and onnxruntime) — with that release's `.exe`/`.dll` files flattened into it. Everything that launches a native tool resolves it through `models_manager::runtime_dir` / `runtime_binary`; nothing builds a runtime path by hand.

**Why: the runtimes ship DLLs with the same names.** whisper.cpp and llama.cpp both ship `ggml.dll`, `ggml-base.dll` and `ggml-cpu-*.dll`, built from different ggml versions. Builds before 2026-11-19 flattened all three runtimes into one shared `<models>/bin/`, so whichever installed last won. On the test machine whisper.cpp b4938 had been running on llama.cpp b10717's ggml — working by ABI luck — and reinstalling whisper.cpp would have put the older ggml under `llama-server.exe`. Windows resolves an executable's DLLs from its own folder first, so a folder per runtime is what keeps each tool on the DLLs it was built against. **Never flatten them into one folder again.** The isolation has a measured price — whisper.cpp now runs on its own, older ggml, ~32% slower on STT than on the one it was borrowing ([benchmarks.md](benchmarks.md#runtime-isolation-and-the-first-in-app-run-2026-11-19)) — and the remedy for that is a newer whisper.cpp release, not shared DLLs ([roadmap.md](roadmap.md) backlog).

**Legacy fallback, decided per runtime.** A runtime counts as installed in the legacy layout when its launcher exe (`whisper-cli.exe`, `llama-server.exe`, `sherpa-onnx-offline-speaker-diarization.exe`) is in the flat `bin/` but not in its own folder; it then still resolves and runs from `bin/`, so an upgrade forces no reinstall. The choice is made on the launcher, never file by file, so a runtime's executables and DLLs always come from one folder together. An install or reinstall always writes the runtime's own folder, which then wins.

**A reinstall replaces the folder; it does not copy over it.** The new release is staged in `bin/<runtime>.new`, the current folder is renamed to `bin/<runtime>.old`, the staged one is renamed into place, and only then is `.old` deleted. A file the new release no longer ships therefore cannot linger beside it — and be vouched for by the new digests. If the swap fails (on Windows, usually a file still open), the old folder is put back and the install fails visibly, so the working runtime keeps working throughout. A `.new`/`.old` left by an interrupted attempt is cleared by the next install. This is what makes a runtime version change, such as the planned whisper.cpp upgrade, a clean swap rather than a mix of two releases' DLLs.

Verify on a legacy runtime says that it shares DLLs with the other runtimes and that reinstalling moves it into a folder of its own. `scripts/setup-models.ps1` provisions the same per-runtime layout.

**Clean-up of the old flat files.** Once all three runtimes are in their own folders, nothing can run from the flat `bin/` any more, and whatever files remain directly in it (~90 MB and ~100 files on the test machine) appear as **one** item, "old shared runtime files", in the same retired-models notice described below. Until every runtime has moved, that item is not offered at all, so a file a runtime still uses is never listed. Removing it deletes only plain files directly inside `bin/`: never a subfolder or anything in one, never a link, and nothing at all if `bin` is itself a link or junction.

**Clean-up once everything is bundled.** When all three runtimes resolve to the bundled copy, nothing under `<models>/bin` can run any more: not the per-runtime folders an earlier build downloaded, not any `.new`/`.old` left by an interrupted reinstall, and not the old flat files. All of it appears as **one** item instead, "downloaded runtimes no longer used" (key `bin/**`, which replaces `bin/*`; at most one of the two is listed). It is not offered while any runtime still runs from app data. Removing it deletes only the known runtime folders (`whisper`, `llama`, `sherpa` and their `.new`/`.old`) that are real directories directly inside a real `bin/`, plus the flat files above. It never follows a link or junction, never deletes a folder that is, contains or sits inside the bundled folder, and leaves anything else in `bin/` alone. The manifest records describing the deleted folders are dropped.

### Retired and unknown models

`find_retired()` lists anything directly inside a model folder (`llm/`, `stt/`, `diarization/`, …) that the current catalog does not own: a file an earlier catalog shipped (Qwen3-4B, Gemma 3 4B, Whisper base or medium), named as that model, or any other file, named by its file name. `check_models` returns the list and `RetiredModelsNotice` shows it on the setup screen and in Settings → Models, each with a Remove behind a confirmation. Nothing is removed on its own. `remove_retired()` only removes something `find_retired()` currently lists, rebuilds the path from the models folder rather than trusting a manifest record, refuses links and anything that resolves outside the models folder, never looks in `bench/` or `_tmp/`, and is refused while a download is running. It exists because a machine upgraded from the tiered builds kept the retired 4B silently ([roadmap.md](roadmap.md) risk 17). The manifest's informational `file` paths, which builds from before the project rename wrote under `%APPDATA%\com.echomind.app\…`, are rewritten to the current models folder whenever the manifest is next written; lookups go by id and SHA-256, so those paths never affected what was resolved.

### Installs: one at a time, never during a meeting

Every install — first-run setup, a single Install or Reinstall from Settings — goes through one guard in `configuration.rs` (`install_blocker`). **Downloads are strictly one at a time across the whole app**: the Model Manager and first-run setup share one progress map, and a new install is refused while any item in it is queued, downloading, extracting or verifying. (Before 2026-11-19, two different models could be downloading in parallel from Settings.) **No install starts while a meeting is being recorded**, where disk and CPU contention can drop audio, **or processed**, where a runtime reinstall would overwrite the executables and DLLs the pipeline has open. Each refusal is a plain sentence telling the user what to wait for. The guard is unit-tested. Each install downloads into `models/_tmp/<id>/`, which is removed on every exit path, success or failure, and the then-empty `_tmp` folder itself is removed after installs.

### First-run setup

When required models are missing, the app opens on `src/features/setup/SetupView.tsx`: the full list of what will be downloaded with sizes and a total, and **one explicit consent button**. Nothing starts until it is clicked. The click calls the `install_required_models` command, which queues every missing required id and installs them **strictly sequentially on one background thread** — one `curl.exe` at a time — with per-item progress polled by the UI, retry of just the failed items, and the same refusals as every other install (above): never beside another download, never during a recording or processing. "Record without setting up" always steps past it — recording never waits on a download — and is not persisted, so the next launch asks again.

This deliberately re-introduces a bulk download that commit `59ecece` removed: the earlier one-click "install recommended setup" banner fired several `curl.exe`/`tar.exe` downloads back to back with no per-item confirmation, which reads as dropper-like behaviour to endpoint security tooling. The mitigations are explicit consent over a visible list, and never more than one download on the wire. They are mitigations, not a fix. **Since the runtimes moved into the installer, a release build's setup downloads model data only** (`.bin`, `.gguf`, `.onnx`, and the one `.tar.bz2` holding the diarization segmentation model), never an executable. The remaining pattern is still "a desktop app spawns `curl.exe` and `tar.exe` to fetch files from GitHub and Hugging Face", and **whether EDR tooling flags it is untested** ([roadmap.md](roadmap.md)). The flow itself has been exercised once through the UI (2026-11-19): with GTCRN and Silero VAD moved aside, the screen listed exactly those two, installed them in turn on one click and handed over to Record, and the re-downloaded files were byte-identical to the originals. The per-model Install button in Settings → Models remains as the one-at-a-time alternative.

Downloads shell out to Windows' own `curl.exe` and `tar.exe` rather than linking an HTTP crate — see [Privacy and data handling](#privacy-and-data-handling) for why that's structural rather than convenient.

**Integrity is trust-on-first-install.** `install()` computes each file's SHA-256 right after installing and records it in `models/manifest.json`; `verify()` (Settings → Models → health check) recomputes and compares. For a weights file that is one digest. For a runtime the launcher alone says little — the DLLs beside it are what run — so a runtime install records a SHA-256 for **every file in its own folder** (`file_digests` on the manifest record; omitted for weights, whose records are unchanged), and Verify checks all of them. Untouched, it reports the folder: "OK — 38 files, 21 MB, all unchanged since install." It fails, naming the files and saying to reinstall, on a changed file ("1 of 38 changed (ggml.dll)"), a missing one, or a new `.exe`/`.dll` that was not there at install time — a planted DLL beside a runtime is exactly what Windows would load. A new non-binary file is reported but does not fail. A runtime installed before full-folder digests existed has only its launcher's digest on record, so Verify checks that alone and says how many other files it could not check and that reinstalling records a digest for each; a legacy flat-layout runtime is verified as before. `verify_model` runs off the UI thread (an async command on a blocking task), since hashing a folder of DLLs is not instant. That catches corruption, truncation, and post-install tampering — but the catalog ships **no pinned, independently-sourced hashes**, so it cannot detect a compromised upstream release or a download that was already bad when the hash was taken. TLS via `curl.exe` covers transit; nothing here independently attests to what a file *should* be. Inventing digests never checked against real upstream files would be worse than having none, so this gap is written down rather than papered over.

**Bundled runtimes are pinned, then verified against the build.** Runtimes are the exception to trust-on-first-install. `prepare-runtimes.ps1` refuses to bundle an archive whose SHA-256 differs from the pin in the script. Each pin was taken from a fresh download and cross-checked: every one of the 54 shipped files matched the digest the app had recorded when it downloaded the same releases a day earlier. That is two downloads agreeing, not upstream attestation; none of the three projects publishes checksums for these archives. After signing, the script writes `runtimes/manifest.json`. On a bundled runtime, Verify checks the installed folder against that manifest with the same rules as a downloaded runtime: it fails on a changed or missing file or on a new `.exe`/`.dll`, notes a new non-binary file without failing, says it is the copy included with Cadence, and tells the user to reinstall Cadence. If the manifest is absent (a dev build whose runtimes were placed by hand), Verify says the files cannot be checked rather than claiming integrity. If it is unreadable or does not list the runtime, the check fails. **Model data hashes are still unpinned**: weights remain trust-on-first-install.

## Storage and history

`storage.rs` owns the on-disk layout:

```text
<app_data>/
├── config/settings.json
├── lexicon/                   optional {en,ta,ta-latn}.txt overrides
├── models/                    catalog installs + manifest.json (SHA-256s)
│   ├── bin/whisper/, bin/llama/, bin/sherpa/   downloaded runtimes, one folder each, never shared
│   │                          (unused once the installer's bundled runtimes are present)
│   ├── stt/, llm/, diarization/, denoise/, vad/
│   └── bench/                 bench_cli candidates only, own manifest
├── database/meetings.db       SQLite index for history: metadata only
├── logs/
└── meetings/<id>/
    ├── metadata.json          authoritative record for this meeting
    ├── original.wav           never modified, never deleted automatically
    ├── processed.wav
    ├── diarization.json       raw stage output
    ├── stt.json               raw stage output
    ├── transcript.json        derived (turns carry `flags` annotations)
    ├── mom.json               derived
    ├── audio_profile.json     derived: measurements of original.wav
    ├── classification.json    derived: meeting type + topic tags
    ├── redaction.json         derived: what the lexicon scan found
    ├── metrics.json           per-stage wall time + peak RAM
    └── exports/
```

The three newest files are all **derived and rebuildable**: `audio_profile.json` can be recomputed from `original.wav`, and `classification.json` and `redaction.json` from `transcript.json`, which is why they carry commands to regenerate them (below) rather than only being written once during processing. Deleting any of them costs a re-scan, not a meeting.

`metadata.json` is the source of truth; the SQLite `meetings` table is a **rebuildable index** over it, holding only what the history list needs to sort and filter — never full transcript or MoM content. Every metadata write goes through `write_metadata`, which also upserts the index row; a failed upsert is logged, not propagated, because the meeting is already safe on disk. `list_meetings` reads the index, but falls back to `scan_meetings` (walking `meetings/*/metadata.json`) and rebuilds it if the index is empty or unusable — **no meeting can be lost to a corrupted database file.** `recover_interrupted` uses the same scan at startup to mark meetings left mid-recording or mid-processing as failed-but-recoverable, with a message explaining what's still safe.

`delete_meeting` is the only path that removes a recording, and only on explicit user action: the whole `meetings/<id>/` directory plus the SQLite row, in one call. Because the index is a rebuildable cache, there is no secondary copy left behind to forget about.

## Tauri command surface

`processing.rs` exposes the pipeline to the UI: `start_processing` (spawns on a worker thread, returns immediately), `get_processing_status` (polled for stage and progress), `regenerate_mom` (the retry path when only the LLM stage failed — re-uses the saved `transcript.json`), and `check_models` (so the UI warns about missing models *before* a meeting is recorded, not after). `library.rs` adds history/transcript/MoM/export/rename/delete; `configuration.rs` adds host info, settings, and Model Manager commands. Full registration list in `lib.rs`.

The derived-artifact commands follow the same read-or-recompute shape: `profile_meeting_audio` measures `original.wav`, `classify_meeting` / `get_meeting_classification` and `rescan_redactions` / `get_redaction_summary` write and read `classification.json` and `redaction.json`. The `get_*` variants return `null` for a meeting processed before these existed rather than failing, and the recompute variants are what a user reaches for after editing the lexicon or changing a setting — none of them touch `original.wav`, `stt.json` or `diarization.json`.

Processing status is in memory only (`Mutex<HashMap<meeting_id, MeetingStatus>>`). A restart loses progress but never artifacts — every stage's output is already on disk, and re-running is the recovery path.

## Export

`export.rs` renders Markdown and JSON (PDF/DOCX are out of V1 scope), written into the meeting's own `exports/` directory so deleting the meeting deletes them too. Both apply user speaker renames consistently: a rename is keyed by the stable id (`SPEAKER_00`) and applied to the presentation label wherever it appears, so it can never land on the transcript but not the MoM. Markdown carries a meeting-info header, participants, summary, topics/decisions/questions/risks/discussion each with their evidence, action items with owner/deadline/confidence, and the full transcript; JSON is the same content structured for reuse, tagged `"generatedBy": "Cadence (local processing, no cloud services)"`.

Export is also the surface where the redaction mode matters most, and it is held to a stricter rule than the rest of the app. Everywhere else a mask is cosmetic: the original text is one setting away, on the same machine, for the same user. An export is a file that leaves — mailed, pasted into a ticket, dropped in a shared folder — so at `mask` or `redact` the original word must not appear **anywhere** in the output: not in the transcript, not in a minutes evidence quote, and not in a JSON field a reader would not think to check. A single `Redactor` choke point enforces that, and it deliberately **re-scans with the lexicon rather than trusting the flags stored on the transcript**, because those flags were written by whichever scan last ran and the lexicon or the setting may have changed since. Stale flags on a cosmetic surface are a nuisance; stale flags on a file that leaves the machine are the leak. The exported flags themselves have their `term_id` masked too, since that field records the lexicon entry that matched.

Exports also carry the meeting's classification and audio-quality summary when those files exist.

## Privacy and data handling

### Security hardening (2026-10-05)

Meeting IDs and artifact names are validated as single Windows-safe path components before filesystem access. Absolute paths, traversal, separators, alternate data streams, and device names are rejected. Meeting roots, directories, and existing artifact links/junctions are rejected, and metadata IDs must match their directory. Export output paths are checked too. These checks protect the IPC read/write/delete surface; they do not defend against a same-user process racing filesystem changes between checking and opening a file.

Recording reserves `original.wav` atomically with `create_new` rather than truncating an existing destination. A duplicate meeting ID or pre-existing destination produces a recoverable error telling the user to start a new meeting, while preserving the earlier file. Synthetic WAV tests verify both successful new-file recording and preservation of existing audio and non-audio files.

The production webview has a content security policy limited to bundled scripts, Tauri IPC, and local asset audio playback. Inline styles remain allowed for React progress meters; inline scripts and remote resources are blocked. Development separately allows Vite's local hot-reload connection. Policy guidance: [Tauri CSP](https://v2.tauri.app/security/csp/).

Model downloads disable inherited curl configuration and require HTTPS for both the initial request and redirects, preventing curlrc settings or an HTTP downgrade from bypassing transport security. Release builds no longer use this path for executables: the runtimes ship inside the installer, pinned by archive SHA-256 at build time. Consent, sequential downloads, and trust-on-first-install hashing are unchanged. Upstream hashes for model data remain unpinned. Flag semantics: [curl manual](https://curl.se/docs/manpage.html).

The privacy model is architectural, not a policy statement: the app cannot send meeting content anywhere because the code paths to do so do not exist in the shipped binary.

**What is captured.** Everything listed in the `meetings/<id>/` layout above — raw audio, per-stage output, derived transcript and minutes, metadata, and per-run metrics (numbers only, no meeting content). None of it is created, read, or transmitted by anything other than the app running on the same machine.

**No telemetry.** No analytics SDK, no crash reporter, no update-check call, anywhere in the pipeline or UI.

**No cloud STT or LLM.** The crate links no HTTP client for AI processing, and `cargo tree | grep reqwest` returning nothing is the check — deliberately kept binary, so "no cloud path snuck in" is a one-line answer rather than something re-argued by reading every module. That check is now automated, along with several others, in `src-tauri/tests/offline_guarantee.rs` ([below](#the-offline-guarantee-and-what-tests-it)). The old networked-backend client and its UI backend-URL field were deleted in Phase 8.

Exactly two places in the crate contain anything network-shaped, and neither is a general-purpose HTTP client:

- **`models_manager.rs`** shells out to `curl.exe` and `tar.exe` to fetch model weights, and on a development build without bundled runtimes the runtimes too, from public release URLs. An installed release build downloads **model data only**: its executables arrive in the signed installer and a bundled runtime is refused for download. This runs *only* on an explicit click — the first-run setup screen's consent button, or a per-model Install or Reinstall in Settings → Models (or `setup-models.ps1` at dev time) — and is refused outright while a meeting is recording or processing. It downloads *into* the app; there is no code path from a meeting's audio, transcript, or MoM to this module at all. Shelling out to tools Windows already ships keeps the `cargo tree` check meaningful, which is the actual reason for it.
- **`pipeline/http.rs`** is a hand-rolled HTTP/1.1 client talking to the local `llama-server.exe` and `whisper-server.exe` over loopback — those servers' own IPC mechanism, the same role stdin/stdout plays for `whisper-cli.exe` and sherpa-onnx. `request()` takes a **port, not a host or URL**, and hard-codes `Ipv4Addr::LOCALHOST`, so no parameter or code path can point it off-machine even by mistake. It is hand-rolled rather than `reqwest` for the same reason as above: adding a real HTTP crate would defeat the check, for a need a small loopback client already meets. Its `Multipart` builder exists on the same logic — whisper-server takes audio as a `multipart/form-data` upload, and that format is a dozen lines of string building, not a crate.

**What's logged**: meeting id, processing stage, model used, timings, memory, errors. **Not** full transcript text, raw audio, or MoM content. A deeper debug mode exists (`Settings.debug_logging`, off by default) and turning it on is a deliberate, visible choice. Two named exceptions: `logs/llama-server.log` and `logs/whisper-server.log` are diagnostic tails, truncated every run, used only to surface the last few lines if one of those servers fails to start. Neither llama.cpp nor whisper.cpp logs prompt, response or transcript text at default verbosity, so they hold no meeting content in normal operation.

**Deletion** is user-initiated only — no retention limit, no background cleanup. See `delete_meeting` above.

**Masking the transcript does not mask the audio.** Redaction annotates text; `original.wav` still contains what was said and playback will play it. The app must say so rather than implying a meeting has been sanitized. Nothing in the redaction path is a deletion — that is the point of the annotation design — so `redact` mode is a presentation policy, not a data-destruction one.

### The offline guarantee, and what tests it

`src-tauri/tests/offline_guarantee.rs` is 15 integration tests that assert the paragraphs above rather than leaving them as prose. They are *static and structural*: they read the resolved dependency graph, the crate's own source, the frontend, and `tauri.conf.json`, and assert that no code path exists which could reach off the machine. One is behavioural — it stands up a listener, asks the client for that port and confirms the connection arrives from loopback with a loopback `Host` header — but a loopback socket is precisely the thing that keeps working with the NIC switched off. In six groups — no HTTP/networking/TLS/telemetry/cloud-SDK crate is linked; the only absolute URLs in crate source are the model catalog's, and every one is HTTPS at a sanctioned publisher; the only module opening a socket is the loopback client, which hard-codes `127.0.0.1` and takes no host; no Python or PyTorch runtime is pulled in; the only network-capable program shelled out to is the model downloader, the frontend issues no requests of its own, and the Tauri config grants no network capability; and a self-test feeding the scanners known-bad fixtures.

Two properties keep it honest. It needs no network itself — an offline-guarantee test that only passed on a connected machine would be self-defeating — so everything reads files inside the repo and the single subprocess, `cargo tree`, runs `--offline`. And every scan asserts it actually matched something before asserting what it matched is clean, with the self-test covering the detectors themselves, because a scanner that silently matches zero files passes forever while the guarantee rots.

**Why `cargo tree` and not `Cargo.lock`.** The lockfile was tried first and is wrong for the job: it is the union of every resolvable package across all targets and *all optional features*, enabled or not. This repo's `Cargo.lock` right now contains `reqwest`, `hyper` and `tower-http` — optional dependencies somewhere under Tauri that this feature set never turns on, and which `cargo tree` correctly shows as not linked. A lockfile-based check would fail today for no real reason, and the natural fix would be to allowlist `reqwest`, permanently defeating the one check this whole section rests on. `cargo tree` resolves features and the host target, so it answers the question actually being asked: what gets linked into the shipped binary?

**What a green run does not prove.** It does not prove the app *runs correctly* with the network disconnected — a different claim, covering whether Windows, WebView2, WASAPI, the SQLite VFS, whisper.cpp, llama.cpp and sherpa-onnx all behave when name resolution fails and the default route is gone, plus whether every model is present locally. Nothing in a test process can establish that; it needs a human running a full record → process → MoM cycle on a machine with the adapter disabled. That manual check is what [roadmap.md](roadmap.md) risk 6 has been narrowed to.

## Status

Fully local and single-machine, with no network path for AI processing. The whole route from recording through speaker-labelled transcript to generated minutes is **implemented and green under `cargo test`** — which is not the same as having been run against a real meeting in the app, and that distinction is the subject of [roadmap.md](roadmap.md) risk 1. "Complete" here means every stage exists and its unit tests pass. Since 2026-11-19 it also means one short real recording has gone through the app to a transcript; nobody has yet watched a real meeting go all the way through to minutes, because the RAM preflight refused the 8B on the 16 GB laptop it was tried on.

**Done:** audio capture (1), alignment (4), headless pipeline runner (5), Tauri integration (8), MoM generation (7), transcript/MoM UI (9/10), SQLite history with filesystem rebuild (11), Markdown/JSON export (12), Model Manager (13), first-run setup, native runtimes bundled into the signed per-user installer (built and unit-tested; no installer has yet been built and installed from it), and the single model set with a 16 GB minimum (cross-cutting, not a numbered phase — it replaced the earlier memory tiering).

**Partly executed:** `bench_cli`. Thread sweeps, diarization threshold sweeps, `compare` on one clip and the `baseline` test have run; `llm` and `full` have not. The STT model size, the diarization cluster threshold, and the choice of Qwen3-8B — which has not been run on any machine here, least of all a 16 GB one — are all provisional — see [benchmarks.md](benchmarks.md) for exactly what is and isn't measured, and [roadmap.md](roadmap.md) for full phase status.

**Implemented but unmeasured:** the leveler, the compressor, model-free noise reduction, noise classification, the audio quality score, unparliamentary-language annotation, and meeting classification are all built and unit-tested, and **none of them has been run against real meeting audio.** Every default they ship with — the leveler's range, window and time constants, compressor threshold and ratio, noise-reduction strength, the classifier's thresholds, the score's arithmetic, the meeting-type cue weights — is a reasoned starting point, not a measurement. [benchmarks.md](benchmarks.md) lists what would settle each. The two newest additions join that list rather than escaping it: the word-level alignment split has met real multi-speaker audio exactly once (below), and no meeting containing flagged language has ever been through the masked-grounding path.

**First contact with real audio (2026-10-05), one clip deep.** The shipped chain through the transcript has now run on real speech with a human-caption reference — `bench_cli compare --reference`, then the committed `baseline` test built from it ([benchmarks.md](benchmarks.md#baseline-regression-test-baseline)). On one multi-speaker broadcast clip the enhancement chain was neutral (WER 16.7% shipped vs 16.1% raw), the audio profile's findings were plausible, and the word-level split met real diarization intervals — which is how the island problem above was found. One English broadcast clip calibrates nothing: the defaults above are still reasoning, minutes were not generated from it, and none of this ran inside the app.

**Verified structurally, not end to end:** the offline guarantee. `offline_guarantee.rs` proves no outbound code path exists; a human still has to run a meeting through the app with the adapter disabled ([roadmap.md](roadmap.md) risk 6).
