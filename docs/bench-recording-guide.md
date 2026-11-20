# Recording a bench session

One 8–12 minute recording, made on purpose, that gives `bench_cli compare` something to measure: a noise floor, a word-for-word reference for a near and a far speaker, real Tanglish, overlap, a noisy stretch, and decisions with owners. It is the recording [benchmarks.md](benchmarks.md) keeps saying nobody has made yet.

You don't need audio-engineering knowledge for this. What matters is sticking to the plan and **writing the times down as you go**.

## Before you start

- **Laptop on AC power**, with a normal power plan. Battery throttling skews every timing.
- **Close heavy apps** (browsers with many tabs, IDE builds, games, sync clients). The LLM rows report peak RAM against a 7,000 MB budget, and background load distorts that.
- Record with **the app's normal Record tab**. Don't use a separate recorder, because the point is to test the audio path the app actually uses.
- Put the **mic flat on the table where it would sit in a real meeting**, usually near one end, not in the middle of everyone. Write down which mic it is (built-in laptop array, or the USB model name) and which device the Record tab had selected.
- Arrange the seats so that one reader sits **right at the mic** and one sits **at the far end of the table, about 2 m away**.
- Have the script below printed or on a phone for both readers. Keep a notes sheet (see [What to note down](#what-to-note-down)) and someone watching the app's elapsed-time counter.

## Session plan (8–12 min)

Tick these off as you go. The times are targets, not limits.

- [ ] **0:00–0:30: Room as-is, silence.** Nobody talks and nobody types. Leave the fan/AC in its normal state. This gives the noise floor.
- [ ] **~0:30–2:30: Read-aloud section.** Each reader goes in turn:
  - **Near reader** (at the mic) reads the whole script at a normal meeting pace. **Write down the elapsed time when they start and when they stop.**
  - Pause for **3 seconds of silence**.
  - **Far reader** (~2 m away) reads the same script at a normal meeting volume. Don't let them project as if on stage. **Write down start and stop.**
  - Stay silent for about 3 s before and after each reading. Those gaps keep other speech out of the reference range.
- [ ] **~2:30–9:00: Natural discussion, 3+ speakers.** Talk about anything real or made-up, but make sure all of the following happen, and write down the time for each:
  - [ ] **Mid-sentence Tanglish**: switch languages within a sentence the way you normally would, not whole sentences in one language and then the other.
  - [ ] **The far-end speaker contributes** at least two or three times, unprompted, at their normal volume.
  - [ ] **A fast back-and-forth** of about 30 s: interruptions, people talking over each other, short "yes / no / wait" turns.
  - [ ] **A noisy stretch** of about 1 min: turn the **fan/AC on** (or up) and have **someone type** on a keyboard while the talking continues. Note when it starts and stops.
  - [ ] **Explicit decisions and action items with named owners and dates.** State them plainly. Use your own or these:
    - "Priya will send the revised estimate by Friday."
    - "Arun owns the vendor call; he'll confirm the new date by the 12th."
    - "Decision: we're moving the launch to March 15."
    - "Karthik will update the test plan and share it before Monday's review."
- [ ] **Last ~20 s: closing silence.** Nobody talks. Then stop the recording.

## The read-aloud script

About 170 words. Lines marked **[EN]** are English, **[TANGLISH]** are romanized code-switched, and **[TAMIL SCRIPT]** are in Tamil script. Read them exactly as written, numbers included. Don't paraphrase, because the WER number assumes the reference is what was said.

> **[EN]** Good morning, everyone. Let's start with the client update for the Chennai project.
> **[TANGLISH]** Naan already Ramesh-kitta pesiten, avanga deadline-a extend panna ready-aa irukanga.
> **[EN]** The new deadline is March 15, and the budget stays at 42 lakh.
> **[TAMIL SCRIPT]** இந்த வாரம் நாம் புதிய வடிவமைப்பை இறுதி செய்ய வேண்டும்.
> **[TANGLISH]** Design review-ku munnadi, Priya neenga revised estimate-a Friday-kulla anuppa mudiyuma?
> **[EN]** Arun, please confirm the vendor call for Tuesday at 3 pm.
> **[TANGLISH]** Testing team-ku konjam time venum, so phase two-va next sprint-ku push pannalaam.
> **[TAMIL SCRIPT]** அடுத்த கூட்டம் வெள்ளிக்கிழமை மாலை நான்கு மணிக்கு நடக்கும்.
> **[EN]** Before we close, we have three open items: the login bug, the invoice template, and the training schedule for 20 new users.
> **[TANGLISH]** Idhu ellam clear-aa irundha, naan minutes-a today evening-kulla share panren.
> **[EN]** Thanks, everyone. Let's meet again on Friday.

### Save it as reference files

WER is computed per range, so both readers can use the same text. Save the file **outside the repo** so it can never be committed, for example in `%USERPROFILE%\cadence-bench\`:

1. Open Notepad and paste the script **without** the `[EN]`/`[TANGLISH]`/`[TAMIL SCRIPT]` labels and without the `>` markers, one line per sentence.
2. Save as `ref-near.txt`, **encoding UTF-8** (Notepad's default on Windows 11; check the dropdown), so the Tamil script survives.
3. Save the same text again as `ref-far.txt`.
4. If a reader stumbled, skipped or added a word, edit **that reader's file** to match what they actually said. The reference has to be the truth, not the script.

**Know what the WER number can and can't tell you here.** WER is exact word matching after lowercasing and stripping punctuation:
- If whisper writes a Tamil-script sentence in romanized form (or the reverse), every word counts as an error. That is a real finding about script fidelity, but it is not the same thing as mishearing.
- Romanized Tamil has no standard spelling, so `pesiten` against `pesinen` counts as an error even when both are fine. Read the Tanglish lines in the transcripts by eye. Don't trust WER alone for them.
- `42 lakh`, `3 pm` and `March 15` can come back as `forty-two lakh`, `3 p.m.` or `15th March`. Those count as errors too.

The best use of the number is **comparing variants against each other on the same range**, not as an absolute accuracy figure.

## What to note down

Fill this in during or right after the session. Claude needs it to interpret the report.

| Speaker | Seat / distance from mic | Languages used | Notes |
|---|---|---|---|
| e.g. Karthik | at the mic, ~30 cm | EN, Tanglish | near reader |
| e.g. Priya | far end, ~2 m | EN, Tamil | far reader |
| | | | |

| Elapsed time | Event |
|---|---|
| 00:00–00:30 | opening silence |
| __:__–__:__ | near reading |
| __:__–__:__ | far reading |
| __:__–__:__ | fast back-and-forth / overlap |
| __:__–__:__ | fan/AC on + typing |
| __:__ | decision / action item stated (who, what, when) |
| __:__–end | closing silence |

Also write down: the mic (model or "built-in"), the room (size, hard or soft walls), and any unplanned noise (door, phone, traffic) with its time.

## After recording

> The flags below describe the harness as designed. If a command errors out, run `cargo run --release --bin bench_cli -- --help` (or check the usage line it prints) and go by that.

**1. Find the meeting id.** The meeting appears in the **Meetings** tab. Its id is the name of its folder under `%APPDATA%\com.cadence.app\meetings\`. The newest folder is the one you just recorded:

```powershell
Get-ChildItem "$env:APPDATA\com.cadence.app\meetings" -Directory |
  Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name, LastWriteTime
```

`compare` only ever **reads** that meeting's `original.wav`. It never writes into the meeting folder.

**2. Install the bench candidates** (run everything from `src-tauri/`):

```powershell
cargo run --release --bin bench_cli -- setup --dry-run   # what would download, and the total size
cargo run --release --bin bench_cli -- setup             # download them
# optional: add Gemma 4 12B and Gemma 3 12B, quality-ceiling references that are over the 16 GB budget by design
cargo run --release --bin bench_cli -- setup --with-large
```

Everything goes into `%APPDATA%\com.cadence.app\models\bench\`, and the app never shows it. At the time of writing, the default set is roughly 30 GB and `--with-large` adds roughly 13 GB (about 43 GB in all). Trust the `--dry-run` total over these figures. This step is the only one that needs the Internet.

**3. Sanity check first** (shipped front end + whisper-small + Qwen3-8B Q4 only):

```powershell
cargo run --release --bin bench_cli -- compare <meeting-id> --quick
```

Open the `report.md` it writes and skim `transcript-shipped.txt`. If this looks broken, stop here and share it before running the full matrix.

**4. Full runs, once per reference range.** Use the times you wrote down, with about 1 s of slack each side:

```powershell
# near reader
cargo run --release --bin bench_cli -- compare <meeting-id> `
  --reference "$env:USERPROFILE\cadence-bench\ref-near.txt" --reference-range 00:33-01:28 --llm-cleanup

# far reader
cargo run --release --bin bench_cli -- compare <meeting-id> `
  --reference "$env:USERPROFILE\cadence-bench\ref-far.txt" --reference-range 01:31-02:30 --llm-cleanup
```

Each full run goes through every front-end and STT variant and every installed LLM, so it is slow. Nobody has timed it yet. Leave the laptop plugged in and otherwise idle. The far-reader run's WER, especially `shipped` against `raw` and `no-leveler`, is the number the leveler and compressor exist for.

**5. Hand it back.** Each run writes a folder `bench-runs/<timestamp>-<meeting-id>/` at the repo root. Tell Claude the folder names and paste your notes tables. Claude will read:
- `report.md`: front-end, STT and LLM tables (time, realtime factor, peak RAM against the 7,000 MB budget, words, speakers, hallucination signals, WER);
- `transcript-<variant>.txt`: to check the Tanglish and Tamil lines, speaker turns in the overlap stretch, and hallucinations in the silences;
- `mom-<model>.md`: whether each model caught the decisions and action items you stated;
- `cleanup-<model>.txt` and its diff: whether the experimental cleanup pass changed anything it shouldn't have.

## Privacy

The recording, the transcripts and the bench output all stay on this machine. `compare` makes no network calls. **Never commit `bench-runs/`.** It contains transcripts and processed audio of real people. Keep the reference files outside the repo too, as described above.
