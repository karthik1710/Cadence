//! Segment-wise transcription for code-switched meetings.
//!
//! ## Why `--language auto` is not enough
//!
//! whisper.cpp's `auto` detects the language **once**, from the opening
//! window, and then decodes the entire file in it. In an English/Tamil meeting
//! that means whisper locks to whichever language happened to open the
//! recording and degrades the other for the rest of it: Tamil comes back
//! romanized or dropped when it locked to `en`, English comes back
//! transliterated when it locked to `ta`. Auto-detect *looks* like it handles
//! code-switching. It does not.
//!
//! So this module hands whisper the recording in blocks, each detected on its
//! own, and stitches the results back onto the original timeline.
//!
//! ## Why blocks are contiguous cuts, not a `vad::Timeline`
//!
//! [`crate::pipeline::vad::Timeline`] maps timestamps across a file whose
//! silences were spliced out, and documents its one residual inexactness: a
//! whisper segment that straddles a splice gets a stretched interior, because
//! the silence it would have placed words in no longer exists.
//!
//! A block here is a single contiguous span of the recording — silence inside
//! it is kept — so it has no splices and no interior to stretch. Mapping back
//! is `block.start_ms + whatever whisper said`, exact by construction, which
//! is what the transcript, the diarization alignment, the evidence links
//! and click-to-play all need. VAD is still what decides *where* the cuts
//! fall, so a block boundary never lands in the middle of a word, and the
//! silence between blocks is never sent to whisper at all.

use crate::pipeline::http;
use crate::pipeline::models::SttSegment;
use crate::pipeline::paths::ModelPaths;
use crate::pipeline::preprocess;
use crate::pipeline::proc::StageMetrics;
use crate::pipeline::vad::SpeechSegment;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// whisper analyses audio in 30 s windows and detects the language once, from
/// the first one. A 30 s block is therefore the finest language granularity
/// whisper.cpp can be asked for, and it wastes nothing: a meaningfully shorter
/// block leaves most of that window padded with silence, which is both slower
/// per second of speech and worse for the detector, while a longer one starts
/// decoding later speech in a language read off its opening seconds — the
/// whole-file problem back again at a smaller scale.
pub const TARGET_BLOCK_MS: i64 = 30_000;

/// A block may run past the target to absorb a short tail, or because a single
/// unbroken stretch of speech has nowhere safe to cut, but never past this.
pub const MAX_BLOCK_MS: i64 = 45_000;

/// Below this a block gives the language detector almost nothing to read, so
/// it is merged into its neighbour whenever [`MAX_BLOCK_MS`] leaves room.
pub const MIN_BLOCK_MS: i64 = 6_000;

/// One block buys nothing that a single pass does not already do, and costs a
/// server start; two is the point where per-block detection starts to exist.
const MIN_BLOCKS_TO_BOTHER: usize = 2;

/// Loading whisper weights from a cold page cache can take a while on a laptop
/// SSD; `medium` is over a gigabyte.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// One 30 s block decodes in seconds, but a machine under load and a large
/// model together can stretch that a long way. Generous, because the cost of
/// being wrong is a lost block.
const BLOCK_TIMEOUT: Duration = Duration::from_secs(600);

/// A contiguous span of the recording handed to whisper as one file, on the
/// **original** recording's timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub start_ms: i64,
    pub end_ms: i64,
}

impl Block {
    pub fn duration_ms(&self) -> i64 {
        (self.end_ms - self.start_ms).max(0)
    }

    /// Moves a timestamp whisper reported *inside this block* back onto the
    /// original recording's timeline.
    ///
    /// Clamped to the block, for the same reason [`crate::pipeline::vad`]
    /// clamps: whisper occasionally reports an end time a few milliseconds
    /// past the audio it was handed, and that must not push a segment into the
    /// next block's territory.
    pub fn to_original_ms(&self, block_ms: i64) -> i64 {
        (self.start_ms + block_ms.max(0)).clamp(self.start_ms, self.end_ms)
    }
}

// ------------------------------------------------------------- block layout

/// Groups VAD speech into blocks, padded into the surrounding silence.
///
/// `total_ms` bounds the last block's padding; `pad_ms` is the same
/// `vad_speech_pad_ms` setting silence trimming uses, so a cut never clips a
/// word onset.
pub fn plan_blocks(speech: &[SpeechSegment], total_ms: i64, pad_ms: i64) -> Vec<Block> {
    let segments = split_overlong(normalize(speech));
    if segments.is_empty() {
        return Vec::new();
    }

    let mut blocks: Vec<Block> = Vec::new();
    for segment in segments {
        match blocks.last_mut() {
            Some(block) if segment.end_ms - block.start_ms <= TARGET_BLOCK_MS => {
                block.end_ms = segment.end_ms;
            }
            _ => blocks.push(Block {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
            }),
        }
    }

    absorb_short_blocks(&mut blocks);
    pad_into_silence(&mut blocks, pad_ms, total_ms);
    blocks
}

/// Sorts, drops the empty, and merges anything overlapping or touching, so the
/// grouping below can assume a clean ascending list.
fn normalize(speech: &[SpeechSegment]) -> Vec<SpeechSegment> {
    let mut sorted: Vec<SpeechSegment> = speech
        .iter()
        .copied()
        .filter(|s| s.end_ms > s.start_ms && s.start_ms >= 0)
        .collect();
    sorted.sort_unstable_by_key(|s| (s.start_ms, s.end_ms));

    let mut merged: Vec<SpeechSegment> = Vec::with_capacity(sorted.len());
    for segment in sorted {
        match merged.last_mut() {
            Some(last) if segment.start_ms <= last.end_ms => {
                last.end_ms = last.end_ms.max(segment.end_ms)
            }
            _ => merged.push(segment),
        }
    }
    merged
}

/// Cuts a stretch of unbroken speech longer than [`MAX_BLOCK_MS`] into equal
/// parts of at most [`TARGET_BLOCK_MS`].
///
/// This is the one place a cut can land mid-word, and it is the lesser evil:
/// the alternative is decoding a multi-minute monologue in a language read off
/// its first few seconds, which is exactly what a mixed-language meeting breaks.
fn split_overlong(segments: Vec<SpeechSegment>) -> Vec<SpeechSegment> {
    let mut out = Vec::with_capacity(segments.len());
    for segment in segments {
        let length = segment.duration_ms();
        if length <= MAX_BLOCK_MS {
            out.push(segment);
            continue;
        }
        let parts = (length + TARGET_BLOCK_MS - 1) / TARGET_BLOCK_MS;
        for part in 0..parts {
            let start = segment.start_ms + (length * part) / parts;
            let end = segment.start_ms + (length * (part + 1)) / parts;
            out.push(SpeechSegment {
                start_ms: start,
                end_ms: end,
            });
        }
    }
    out
}

/// Folds a too-short block into its neighbour when the result still fits.
/// A three-second block is a poor thing to ask a language detector about.
fn absorb_short_blocks(blocks: &mut Vec<Block>) {
    let mut index = 1;
    while index < blocks.len() {
        let short = blocks[index].duration_ms() < MIN_BLOCK_MS;
        let fits = blocks[index].end_ms - blocks[index - 1].start_ms <= MAX_BLOCK_MS;
        if short && fits {
            blocks[index - 1].end_ms = blocks[index].end_ms;
            blocks.remove(index);
        } else {
            index += 1;
        }
    }
    // The first block can be the short one, and has no earlier neighbour.
    if blocks.len() > 1
        && blocks[0].duration_ms() < MIN_BLOCK_MS
        && blocks[1].end_ms - blocks[0].start_ms <= MAX_BLOCK_MS
    {
        blocks[1].start_ms = blocks[0].start_ms;
        blocks.remove(0);
    }
}

/// Widens each block into the silence around it, never into its neighbour.
///
/// Two blocks that overlapped would transcribe the same words twice, so each
/// side may take at most half of the gap between them. Padding into silence is
/// free; padding into speech would duplicate it.
fn pad_into_silence(blocks: &mut [Block], pad_ms: i64, total_ms: i64) {
    if pad_ms <= 0 || blocks.is_empty() {
        return;
    }
    let original: Vec<Block> = blocks.to_vec();
    let last = original.len() - 1;
    for (index, block) in blocks.iter_mut().enumerate() {
        let room_before = if index == 0 {
            original[0].start_ms
        } else {
            let gap = original[index].start_ms - original[index - 1].end_ms;
            gap - gap / 2
        };
        let room_after = if index == last {
            (total_ms - original[last].end_ms).max(0)
        } else {
            (original[index + 1].start_ms - original[index].end_ms) / 2
        };
        block.start_ms -= pad_ms.min(room_before.max(0));
        block.end_ms += pad_ms.min(room_after.max(0));
    }
}

/// Whether blocking is worth doing at all. A recording that yields one block
/// is short enough that a single pass detects on the same audio anyway.
pub fn is_worth_blocking(blocks: &[Block]) -> bool {
    blocks.len() >= MIN_BLOCKS_TO_BOTHER
}

// --------------------------------------------------------- response parsing

#[derive(Debug, Deserialize)]
struct VerboseSegment {
    #[serde(default)]
    text: String,
    /// Seconds from the start of the block whisper was given.
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
}

#[derive(Debug, Deserialize)]
struct VerboseJson {
    /// The language whisper detected for this block. whisper-server spells it
    /// out in full ("english", "tamil"); whisper-cli's `--output-json` uses
    /// the ISO code. [`language_code`] normalises both.
    #[serde(default)]
    language: String,
    #[serde(default)]
    segments: Vec<VerboseSegment>,
}

/// Turns one block's `verbose_json` response into segments on the original
/// recording's timeline, tagged with the language detected for that block.
pub fn parse_block(raw: &str, block: &Block) -> Result<Vec<SttSegment>, String> {
    let parsed: VerboseJson = serde_json::from_str(raw)
        .map_err(|e| format!("could not parse whisper-server's response: {e}"))?;
    let language = language_code(&parsed.language);

    Ok(parsed
        .segments
        .into_iter()
        .filter_map(|segment| {
            let text = segment.text.trim().to_string();
            if text.is_empty() {
                return None;
            }
            let start_ms = block.to_original_ms(seconds_to_ms(segment.start));
            let end_ms = block.to_original_ms(seconds_to_ms(segment.end)).max(start_ms);
            Some(SttSegment {
                start_ms,
                end_ms,
                text,
                // Left empty on purpose: the single-pass path does not produce
                // word timestamps either, and nothing downstream reads them
                // yet — a mode that filled them would make `stt.json` mean two
                // different things depending on a setting.
                words: Vec::new(),
                language: language.clone(),
            })
        })
        .collect())
}

fn seconds_to_ms(seconds: f64) -> i64 {
    if !seconds.is_finite() {
        return 0;
    }
    (seconds * 1000.0).round() as i64
}

/// whisper's own name for a language → its ISO 639-1 code.
///
/// Only the languages plausibly heard in these meetings plus the widely spoken
/// ones are listed; anything else passes through as whisper spelled it, which
/// still tells a reader the pipeline noticed a switch.
const LANGUAGE_CODES: &[(&str, &str)] = &[
    ("english", "en"),
    ("tamil", "ta"),
    ("hindi", "hi"),
    ("telugu", "te"),
    ("kannada", "kn"),
    ("malayalam", "ml"),
    ("marathi", "mr"),
    ("bengali", "bn"),
    ("gujarati", "gu"),
    ("punjabi", "pa"),
    ("urdu", "ur"),
    ("nepali", "ne"),
    ("sinhala", "si"),
    ("sanskrit", "sa"),
    ("arabic", "ar"),
    ("chinese", "zh"),
    ("japanese", "ja"),
    ("korean", "ko"),
    ("french", "fr"),
    ("german", "de"),
    ("spanish", "es"),
    ("portuguese", "pt"),
    ("italian", "it"),
    ("russian", "ru"),
    ("dutch", "nl"),
    ("turkish", "tr"),
    ("indonesian", "id"),
    ("malay", "ms"),
    ("thai", "th"),
    ("vietnamese", "vi"),
];

/// Normalises whatever whisper reported into an ISO code. `None` when it
/// reported nothing usable — a missing tag is honest, an invented one is not.
pub fn language_code(reported: &str) -> Option<String> {
    let name = reported.trim().to_ascii_lowercase();
    if name.is_empty() || name == "auto" || name == "unknown" {
        return None;
    }
    if let Some((_, code)) = LANGUAGE_CODES.iter().find(|(n, _)| *n == name) {
        return Some((*code).to_string());
    }
    Some(name)
}

// ------------------------------------------------------------- the server

/// Lifecycle for one local `whisper-server.exe`, so the weights are loaded
/// once for a whole meeting instead of once per block.
///
/// The arithmetic that makes this necessary: a bare `whisper-cli.exe`
/// invocation costs ~2.5-3.1 s on this machine before it decodes anything
/// (process start plus a 487 MB `small` model load, page cache warm). A
/// 60-minute meeting is roughly 120 blocks, so per-block CLI invocations would
/// spend ~5-6 minutes reloading weights — most of the 10-minute budget,
/// before any transcription. The server pays that once.
///
/// Same shape as [`crate::pipeline::llm::server::LlamaServer`]: loopback only,
/// killed on drop, and never outliving the run that started it.
pub struct WhisperServer {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl WhisperServer {
    pub fn start(
        exe: &Path,
        model: &Path,
        threads: usize,
        log_path: &Path,
    ) -> Result<Self, String> {
        if !exe.exists() {
            return Err(format!("{} is not installed", exe.display()));
        }
        if !model.exists() {
            return Err(format!("{} is not installed", model.display()));
        }

        let port = http::free_local_port()?;
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        // Truncated per run: a diagnostic tail, not an audit trail. whisper.cpp
        // prints transcribed text to its log at higher verbosities, so this
        // must not accumulate meeting content.
        let log = std::fs::File::create(log_path)
            .map_err(|e| format!("could not open {}: {e}", log_path.display()))?;
        let log_err = log.try_clone().map_err(|e| e.to_string())?;

        let mut command = Command::new(exe);
        command
            .arg("-m")
            .arg(model)
            // Loopback only: never reachable from another machine.
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-t")
            .arg(threads.max(1).to_string())
            // whisper-server defaults to greedy decoding (`-bo 2 -bs -1`)
            // where whisper-cli defaults to beam search. Without these two,
            // switching language mode would quietly switch decoding quality
            // as well, and any comparison between the modes would be measuring
            // the wrong thing.
            .arg("-bo")
            .arg("5")
            .arg("-bs")
            .arg("5")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));
        crate::system::hide_console(&mut command);
        let child = command
            .spawn()
            .map_err(|e| format!("failed to launch whisper-server.exe: {e}"))?;

        let mut server = WhisperServer {
            child,
            port,
            log_path: log_path.to_path_buf(),
        };
        server.wait_until_healthy()?;
        Ok(server)
    }

    /// whisper.cpp loads the model before it binds the socket, so *any* HTTP
    /// answer means the weights are resident. Not just a 2xx: `/health` is a
    /// recent addition and an older build 404s it while being perfectly ready.
    fn wait_until_healthy(&mut self) -> Result<(), String> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!(
                    "whisper-server exited during startup ({status}). {}",
                    self.log_tail()
                ));
            }
            if http::request(self.port, "GET", "/health", None, Duration::from_secs(5)).is_ok() {
                return Ok(());
            }
            if started.elapsed() > STARTUP_TIMEOUT {
                return Err(format!(
                    "whisper-server did not become ready within {}s. {}",
                    STARTUP_TIMEOUT.as_secs(),
                    self.log_tail()
                ));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// Transcribes one block of 16 kHz mono WAV bytes and returns the raw
    /// `verbose_json` body.
    pub fn transcribe_block(&self, wav: &[u8]) -> Result<String, String> {
        let mut form = http::Multipart::new();
        form.file("file", "block.wav", "audio/wav", wav);
        // verbose_json is the only response format that carries both the
        // per-segment timestamps and the detected language — the two things
        // code-switching needs out of every block.
        form.field("response_format", "verbose_json");
        // Per block, not per file: this is the entire point of the mode.
        form.field("language", "auto");
        // Load-bearing, not tuning knobs (docs/benchmarks.md): without them
        // whisper.cpp has been observed to collapse a real ~13 s span of
        // speech into one short, wrong sentence.
        form.field("max_len", "80");
        form.field("split_on_word", "true");
        form.field("max_context", "0");
        form.field("temperature", "0.0");
        // NOTE: `translate` is never sent, and must never be added here. The transcript
        // must preserve the language actually spoken;
        // translation is the MoM stage's job, controlled by `mom_language`,
        // and doing it here would destroy the original words for good.
        let (content_type, body) = form.finish();

        let response = http::request_bytes(
            self.port,
            "POST",
            "/inference",
            Some((&content_type, &body)),
            BLOCK_TIMEOUT,
        )?;
        if !response.is_success() {
            return Err(format!(
                "whisper-server rejected a block (HTTP {}). {}",
                response.status,
                self.log_tail()
            ));
        }
        Ok(response.body)
    }

    pub fn peak_ram_mb(&self) -> Option<u64> {
        crate::system::peak_working_set_mb(&self.child)
    }

    pub fn log_tail(&self) -> String {
        match std::fs::read_to_string(&self.log_path) {
            Ok(text) => {
                let tail: Vec<&str> = text.lines().rev().take(6).collect();
                let tail: Vec<&str> = tail.into_iter().rev().collect();
                format!("Server log: {}", tail.join(" | "))
            }
            Err(_) => format!("See {}.", self.log_path.display()),
        }
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for WhisperServer {
    /// The process must not outlive the run under any exit path — a failed
    /// block, a pipeline error, or a panic mid-transcription.
    fn drop(&mut self) {
        self.stop();
    }
}

// ------------------------------------------------------------------- run

/// Everything the block run needs that is not already in `PipelineConfig`.
pub struct BlockRun<'a> {
    pub paths: &'a ModelPaths,
    pub wav_path: &'a Path,
    pub speech: &'a [SpeechSegment],
    pub threads: usize,
    pub pad_ms: i64,
    /// Where the server log goes. `None` puts it beside the audio.
    pub data_dir: Option<&'a Path>,
}

pub struct BlockOutcome {
    pub segments: Vec<SttSegment>,
    pub metrics: StageMetrics,
    pub blocks: usize,
}

/// Transcribes a recording block by block, each block detected on its own.
///
/// `Ok(None)` means blocking would buy nothing on this recording (it is short
/// enough to be one block) and the caller should make its single pass —
/// that is not a failure and needs no warning. An `Err` is a real problem
/// (no server binary, a block that would not decode) and the caller falls back
/// to a single pass *and* says so: never lose a meeting to a stage
/// that could not run, and forbids doing it silently.
pub fn transcribe(run: &BlockRun) -> Result<Option<BlockOutcome>, String> {
    let (samples, _, _) = preprocess::read_mono_16k(run.wav_path)?;
    let sample_rate = preprocess::TARGET_SAMPLE_RATE;
    let total_ms = samples_to_ms(samples.len(), sample_rate);

    let blocks = plan_blocks(run.speech, total_ms, run.pad_ms);
    if !is_worth_blocking(&blocks) {
        return Ok(None);
    }

    let started = Instant::now();
    let log_path = match run.data_dir {
        Some(dir) => dir.join("logs").join("whisper-server.log"),
        None => run.wav_path.with_extension("whisper-server.log"),
    };
    let server = WhisperServer::start(
        &run.paths.whisper_server(),
        &run.paths.stt_model,
        run.threads,
        &log_path,
    )?;

    let mut segments: Vec<SttSegment> = Vec::new();
    let mut peak_ram_mb: Option<u64> = None;
    for (index, block) in blocks.iter().enumerate() {
        let wav = encode_block(&samples, sample_rate, block)?;
        let raw = server.transcribe_block(&wav).map_err(|e| {
            format!(
                "block {} of {} ({}s-{}s) failed: {e}",
                index + 1,
                blocks.len(),
                block.start_ms / 1000,
                block.end_ms / 1000
            )
        })?;
        segments.extend(parse_block(&raw, block)?);
        peak_ram_mb = peak_ram_mb.max(server.peak_ram_mb());
    }

    Ok(Some(BlockOutcome {
        segments,
        metrics: StageMetrics {
            wall_ms: started.elapsed().as_millis() as u64,
            peak_ram_mb,
        },
        blocks: blocks.len(),
    }))
}

/// Cuts one block out of the recording as a 16 kHz mono WAV, in memory —
/// dozens of block files per meeting would be litter, and none of them is an
/// artifact anyone would ever want to keep.
fn encode_block(samples: &[f32], sample_rate: u32, block: &Block) -> Result<Vec<u8>, String> {
    let start = ms_to_samples(block.start_ms, sample_rate).min(samples.len());
    let end = ms_to_samples(block.end_ms, sample_rate).clamp(start, samples.len());
    preprocess::encode_mono_16k(&samples[start..end])
}

fn ms_to_samples(ms: i64, sample_rate: u32) -> usize {
    if ms <= 0 {
        return 0;
    }
    ((ms as i128 * sample_rate as i128) / 1000) as usize
}

fn samples_to_ms(samples: usize, sample_rate: u32) -> i64 {
    if sample_rate == 0 {
        return 0;
    }
    ((samples as i128 * 1000) / sample_rate as i128) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech(start_ms: i64, end_ms: i64) -> SpeechSegment {
        SpeechSegment { start_ms, end_ms }
    }

    fn block(start_ms: i64, end_ms: i64) -> Block {
        Block { start_ms, end_ms }
    }

    // ------------------------------------------------------- block layout

    #[test]
    fn groups_speech_up_to_the_target_block_length() {
        // Six 8 s utterances a second apart: 8+1+8+1+8 = 26 s fits one block,
        // adding a fourth would make it 35 s and does not.
        let speech: Vec<SpeechSegment> = (0..6).map(|i| speech(i * 9_000, i * 9_000 + 8_000)).collect();
        let blocks = plan_blocks(&speech, 60_000, 0);
        assert_eq!(blocks, vec![block(0, 26_000), block(27_000, 53_000)]);
        for b in &blocks {
            assert!(b.duration_ms() <= TARGET_BLOCK_MS, "{b:?}");
        }
    }

    #[test]
    fn a_recording_with_one_block_is_not_worth_splitting() {
        let blocks = plan_blocks(&[speech(1_000, 9_000)], 20_000, 0);
        assert_eq!(blocks, vec![block(1_000, 9_000)]);
        assert!(!is_worth_blocking(&blocks));
        assert!(!is_worth_blocking(&[]));
        assert!(is_worth_blocking(&[block(0, 1), block(2, 3)]));
    }

    #[test]
    fn silence_between_blocks_is_never_sent_to_whisper() {
        // Two utterances with ten minutes of silence between them.
        let blocks = plan_blocks(&[speech(0, 10_000), speech(610_000, 620_000)], 700_000, 0);
        assert_eq!(blocks, vec![block(0, 10_000), block(610_000, 620_000)]);
        let transcribed: i64 = blocks.iter().map(|b| b.duration_ms()).sum();
        assert_eq!(transcribed, 20_000);
    }

    #[test]
    fn an_unbroken_stretch_longer_than_the_maximum_is_split_evenly() {
        // Five minutes of continuous speech: no silence to cut at, so it is
        // divided rather than decoded as one language-locked span.
        let blocks = plan_blocks(&[speech(0, 300_000)], 300_000, 0);
        assert_eq!(blocks.len(), 10);
        for b in &blocks {
            assert_eq!(b.duration_ms(), 30_000);
        }
        // The split covers the stretch exactly, with no gap and no overlap.
        assert_eq!(blocks[0].start_ms, 0);
        assert_eq!(blocks.last().unwrap().end_ms, 300_000);
        for pair in blocks.windows(2) {
            assert_eq!(pair[0].end_ms, pair[1].start_ms);
        }
    }

    #[test]
    fn a_stretch_that_does_not_divide_evenly_still_covers_every_millisecond() {
        let blocks = plan_blocks(&[speech(500, 100_001)], 200_000, 0);
        assert_eq!(blocks[0].start_ms, 500);
        assert_eq!(blocks.last().unwrap().end_ms, 100_001);
        for pair in blocks.windows(2) {
            assert_eq!(pair[0].end_ms, pair[1].start_ms);
            assert!(pair[0].duration_ms() <= MAX_BLOCK_MS);
        }
    }

    #[test]
    fn a_short_block_is_folded_into_its_neighbour() {
        // A 2 s afterthought long after a 20 s turn: alone it would give the
        // language detector two seconds of audio to work with.
        let blocks = plan_blocks(&[speech(0, 20_000), speech(38_000, 40_000)], 60_000, 0);
        assert_eq!(blocks, vec![block(0, 40_000)]);
        // ...but not when folding it would blow past the maximum.
        let blocks = plan_blocks(&[speech(0, 28_000), speech(60_000, 62_000)], 90_000, 0);
        assert_eq!(blocks, vec![block(0, 28_000), block(60_000, 62_000)]);
    }

    #[test]
    fn a_short_first_block_is_folded_forwards() {
        let blocks = plan_blocks(&[speech(0, 2_000), speech(4_000, 24_000)], 60_000, 0);
        assert_eq!(blocks, vec![block(0, 24_000)]);
    }

    #[test]
    fn padding_widens_blocks_into_silence_but_never_into_each_other() {
        let blocks = plan_blocks(&[speech(5_000, 25_000), speech(40_000, 60_000)], 90_000, 200);
        assert_eq!(blocks, vec![block(4_800, 25_200), block(39_800, 60_200)]);
        for pair in blocks.windows(2) {
            assert!(pair[0].end_ms <= pair[1].start_ms, "{blocks:?}");
        }
    }

    #[test]
    fn padding_takes_at_most_half_a_narrow_gap_and_never_runs_off_the_ends() {
        // A 300 ms gap between blocks and only 100 ms of room at either end.
        let blocks = plan_blocks(&[speech(100, 20_000), speech(20_300, 40_000)], 40_100, 5_000);
        assert_eq!(blocks, vec![block(0, 20_150), block(20_150, 40_100)]);
        assert!(blocks[0].end_ms <= blocks[1].start_ms);
    }

    #[test]
    fn overlapping_and_unsorted_speech_is_normalised_first() {
        let blocks = plan_blocks(
            &[
                speech(40_000, 50_000),
                speech(0, 10_000),
                speech(5_000, 12_000),
                speech(9_000, 9_000),
                speech(-100, -50),
            ],
            60_000,
            0,
        );
        assert_eq!(blocks, vec![block(0, 12_000), block(40_000, 50_000)]);
    }

    #[test]
    fn no_speech_at_all_plans_no_blocks() {
        assert!(plan_blocks(&[], 60_000, 200).is_empty());
    }

    // --------------------------------------------------- timestamp mapping

    #[test]
    fn block_relative_timestamps_map_onto_the_original_timeline() {
        let b = block(120_000, 150_000);
        assert_eq!(b.to_original_ms(0), 120_000);
        assert_eq!(b.to_original_ms(1), 120_001);
        assert_eq!(b.to_original_ms(12_345), 132_345);
        assert_eq!(b.to_original_ms(30_000), 150_000);
    }

    #[test]
    fn a_timestamp_outside_the_block_clamps_to_it() {
        let b = block(120_000, 150_000);
        // whisper sometimes reports an end a few ms past the audio it got;
        // that must not leak into the next block's span.
        assert_eq!(b.to_original_ms(30_040), 150_000);
        assert_eq!(b.to_original_ms(-500), 120_000);
    }

    /// The case the whole feature turns on: a block that is *not* the first.
    #[test]
    fn a_later_blocks_segments_land_where_the_words_were_spoken() {
        let b = block(600_000, 630_000);
        let raw = r#"{
            "language": "tamil",
            "segments": [
                {"text": " இந்த feature ready ஆச்சா?", "start": 0.5, "end": 4.25},
                {"text": " Almost.", "start": 4.25, "end": 5.0}
            ]
        }"#;
        let segments = parse_block(raw, &b).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_ms, 600_500);
        assert_eq!(segments[0].end_ms, 604_250);
        assert_eq!(segments[1].start_ms, 604_250);
        assert_eq!(segments[1].end_ms, 605_000);
        assert_eq!(segments[0].language.as_deref(), Some("ta"));
        assert_eq!(segments[0].text, "இந்த feature ready ஆச்சா?");
    }

    /// A whole recording reassembled: three blocks with VAD gaps between them.
    /// Every segment must come back in order, inside the block it came from,
    /// and inside the recording.
    #[test]
    fn a_recording_with_gaps_reassembles_in_order_and_in_range() {
        let speech = [
            speech(2_000, 28_000),
            speech(90_000, 118_000),
            speech(300_000, 325_000),
        ];
        let total_ms = 400_000;
        let blocks = plan_blocks(&speech, total_ms, 200);
        assert_eq!(blocks.len(), 3);

        let mut all: Vec<SttSegment> = Vec::new();
        for b in &blocks {
            // Two segments per block, one of them touching the far edge.
            let span = b.duration_ms() as f64 / 1000.0;
            let raw = format!(
                r#"{{"language":"english","segments":[
                    {{"text":"a","start":0.0,"end":{half}}},
                    {{"text":"b","start":{half},"end":{span}}}
                ]}}"#,
                half = span / 2.0,
                span = span
            );
            all.extend(parse_block(&raw, b).unwrap());
        }

        assert_eq!(all.len(), 6);
        // Strictly ascending, which align.rs relies on.
        for pair in all.windows(2) {
            assert!(pair[0].start_ms <= pair[1].start_ms, "{all:?}");
            assert!(pair[0].end_ms <= pair[1].start_ms, "{all:?}");
        }
        // Nothing escapes the recording, and the first and last segments sit
        // on the outer edges of the first and last blocks.
        assert!(all.iter().all(|s| s.start_ms >= 0 && s.end_ms <= total_ms));
        assert_eq!(all[0].start_ms, blocks[0].start_ms);
        assert_eq!(all[5].end_ms, blocks[2].end_ms);
        // The silence between blocks holds no transcript at all.
        assert!(all
            .iter()
            .all(|s| s.start_ms >= blocks[1].end_ms || s.end_ms <= blocks[1].start_ms
                || (s.start_ms >= blocks[1].start_ms && s.end_ms <= blocks[1].end_ms)));
    }

    #[test]
    fn a_segment_on_a_block_boundary_stays_inside_its_block() {
        let blocks = plan_blocks(&[speech(0, 30_000), speech(31_000, 60_000)], 90_000, 0);
        assert_eq!(blocks.len(), 2);
        // whisper reports the very last instant of block one and the very
        // first of block two; they must meet, not overlap.
        let tail = parse_block(
            r#"{"language":"english","segments":[{"text":"end","start":29.5,"end":30.0}]}"#,
            &blocks[0],
        )
        .unwrap();
        let head = parse_block(
            r#"{"language":"tamil","segments":[{"text":"start","start":0.0,"end":0.5}]}"#,
            &blocks[1],
        )
        .unwrap();
        assert_eq!(tail[0].end_ms, 30_000);
        assert_eq!(head[0].start_ms, 31_000);
        assert!(tail[0].end_ms <= head[0].start_ms);
    }

    #[test]
    fn blank_segments_are_dropped_and_an_end_before_its_start_is_repaired() {
        let raw = r#"{"language":"english","segments":[
            {"text":"   ","start":0.0,"end":1.0},
            {"text":"real","start":5.0,"end":4.0}
        ]}"#;
        let segments = parse_block(raw, &block(1_000, 31_000)).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 6_000);
        assert_eq!(segments[0].end_ms, 6_000);
    }

    #[test]
    fn a_response_that_is_not_json_is_reported_not_ignored() {
        assert!(parse_block("<html>404</html>", &block(0, 1_000)).is_err());
    }

    #[test]
    fn a_response_with_no_language_leaves_the_tag_off() {
        let raw = r#"{"segments":[{"text":"hello","start":0.0,"end":1.0}]}"#;
        let segments = parse_block(raw, &block(0, 30_000)).unwrap();
        assert_eq!(segments[0].language, None);
    }

    // -------------------------------------------------------- language tags

    #[test]
    fn whisper_language_names_normalise_to_iso_codes() {
        assert_eq!(language_code("english").as_deref(), Some("en"));
        assert_eq!(language_code("Tamil").as_deref(), Some("ta"));
        assert_eq!(language_code(" TAMIL ").as_deref(), Some("ta"));
        // whisper-cli's `--output-json` already reports the code.
        assert_eq!(language_code("ta").as_deref(), Some("ta"));
        // An unlisted language is reported as whisper spelled it rather than
        // silently dropped.
        assert_eq!(language_code("yoruba").as_deref(), Some("yoruba"));
    }

    #[test]
    fn nothing_usable_produces_no_tag() {
        assert_eq!(language_code(""), None);
        assert_eq!(language_code("   "), None);
        assert_eq!(language_code("auto"), None);
        assert_eq!(language_code("unknown"), None);
    }

    // ------------------------------------------------------- cutting audio

    #[test]
    fn a_block_is_cut_at_exactly_the_samples_it_names() {
        let sr = preprocess::TARGET_SAMPLE_RATE;
        let samples: Vec<f32> = (0..(10 * sr as usize)).map(|i| (i % 100) as f32 / 1000.0).collect();
        let bytes = encode_block(&samples, sr, &block(2_000, 5_000)).unwrap();
        let dir = std::env::temp_dir().join("cadence-codeswitch-cut");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("block.wav");
        std::fs::write(&path, &bytes).unwrap();

        let (decoded, rate, channels) = preprocess::read_mono_16k(&path).unwrap();
        assert_eq!(rate, sr);
        assert_eq!(channels, 1);
        assert_eq!(decoded.len(), 3 * sr as usize);
        // The first decoded sample is the one at 2 s in the original.
        let expected = samples[2 * sr as usize];
        assert!((decoded[0] - expected).abs() < 1e-3, "{} vs {expected}", decoded[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_block_running_past_the_end_of_the_audio_is_clamped() {
        let sr = preprocess::TARGET_SAMPLE_RATE;
        let samples = vec![0.0f32; sr as usize];
        let bytes = encode_block(&samples, sr, &block(500, 9_000)).unwrap();
        // 44-byte canonical WAV header plus 0.5 s of 16-bit mono.
        assert_eq!(bytes.len(), 44 + (sr as usize / 2) * 2);
    }
}
