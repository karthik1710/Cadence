pub mod align;
pub mod audio_profile;
pub mod chunk;
pub mod codeswitch;
pub mod denoise;
pub mod diarization;
pub mod enhance;
pub mod http;
pub mod llm;
pub mod models;
pub mod noise_class;
pub mod mom;
pub mod paths;
pub mod preprocess;
pub mod proc;
pub mod prompts;
pub mod redact;
pub mod spectrum;
pub mod stt;
pub mod topic;
pub mod vad;

use models::{Mom, Transcript};
use preprocess::{PreprocessOptions, PreprocessReport};
use proc::StageMetrics;
use serde::{Deserialize, Serialize};
use paths::ModelPaths;
use std::path::Path;

use crate::settings::{self, Settings};
use crate::system::HostInfo;

/// Per-stage processing knobs, owned by the Settings screen ("Processing
/// threads") and overridable by the benchmark runner.
///
/// Both native CLIs default to far fewer threads than a modern laptop has —
/// whisper.cpp to 4, sherpa-onnx to 1 — so leaving them alone costs most of
/// the processing-time budget.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub stt_threads: usize,
    pub diarization_threads: usize,
    /// sherpa-onnx clustering distance. Smaller splits one voice across
    /// several speaker ids; larger collapses distinct voices into one.
    pub cluster_threshold: f32,
    /// The organiser's "how many people spoke?" hint for *this* meeting.
    /// `Some(n)` makes diarization cluster into exactly `n` speakers
    /// (`--clustering.num-clusters`), and sherpa-onnx then ignores
    /// `cluster_threshold` entirely; `None` is the automatic, threshold-based
    /// clustering. Per meeting, never from Settings — which is why
    /// [`PipelineConfig::from_settings`] always leaves it `None` and the caller
    /// fills it in. See [`validate_expected_speakers`].
    pub expected_speakers: Option<u32>,
    pub generate_mom: bool,
    /// "auto" | "code-switched" | an ISO code — see [`stt::SttLanguage`].
    pub stt_language: String,
    pub mom_language: String,
    /// The ASR front-end chain (audio preprocessing).
    pub preprocess: PreprocessOptions,
    /// Both of these depend on an *optional* model. Enabled here only means
    /// "run it if it is installed" — see `denoise::is_available`.
    pub denoise_enabled: bool,
    pub vad_enabled: bool,
    /// Cut silence out of the audio sent to STT. Off by default (vs the
    /// timestamp-mapping risk documented in `pipeline::vad`).
    pub vad_trim_silence: bool,
    pub vad_pad_ms: i64,
}

impl PipelineConfig {
    /// Derives the knobs from the user's settings and the machine underneath —
    /// which is how a 16 GB / 8-core laptop and the 32 GB / 32-thread
    /// reference machine both end up sensibly configured with no user input.
    pub fn from_settings(settings: &Settings, host: &HostInfo) -> Self {
        Self {
            stt_threads: settings.effective_stt_threads(host),
            diarization_threads: settings.effective_diarization_threads(host),
            cluster_threshold: settings.cluster_threshold,
            expected_speakers: None,
            generate_mom: settings.generate_mom,
            stt_language: settings.stt_language.clone(),
            mom_language: settings.mom_language.clone(),
            preprocess: PreprocessOptions::from_settings(settings),
            denoise_enabled: settings.denoise_enabled,
            vad_enabled: settings.vad_enabled,
            vad_trim_silence: settings.vad_trim_silence,
            vad_pad_ms: settings.vad_speech_pad_ms as i64,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::from_settings(&Settings::default(), &HostInfo::detect())
    }
}

pub use settings::{DEFAULT_CLUSTER_THRESHOLD, MAX_DIARIZATION_THREADS, MAX_STT_THREADS};

/// Bounds on the expected-speakers hint.
///
/// The hint is offered for small rooms only, because that is where a fixed
/// cluster count was measured to work. sherpa-onnx cuts its linkage tree at
/// `n` clusters, and stray embeddings consume some of those clusters while
/// emitting almost no speech — so a larger `n` does not yield `n` voices.
/// Measured 2026-10-06: on the council baseline clip (anchor, reporter and
/// several council voices), `n = 7` produced fewer substantial voices than the
/// 0.5 threshold did (4 ids holding ≥5% of the talk against 6 on the
/// pipeline's `processed.wav`; 2 against 5 on the raw audio). On the
/// one-speaker TED clip, `n = 1` was exact: one id, against 10 at 0.5.
///
/// Four is the cap because a mistake there is bounded: the fewer the voices,
/// the fewer stray clusters can take their place. Above it the UI's "5 or
/// more" means automatic clustering, the same as "Not sure". Raise this only
/// on labelled evidence that larger counts hold up.
pub const MIN_EXPECTED_SPEAKERS: u32 = 1;
pub const MAX_EXPECTED_SPEAKERS: u32 = 4;

/// Accepts `None` ("Not sure") or a count in
/// `MIN_EXPECTED_SPEAKERS..=MAX_EXPECTED_SPEAKERS`. Anything else is an error
/// the user sees, rather than a value silently clamped into one they did not
/// choose.
pub fn validate_expected_speakers(hint: Option<u32>) -> Result<Option<u32>, String> {
    match hint {
        None => Ok(None),
        Some(n) if (MIN_EXPECTED_SPEAKERS..=MAX_EXPECTED_SPEAKERS).contains(&n) => Ok(Some(n)),
        Some(n) => Err(format!(
            "The number of speakers must be between {MIN_EXPECTED_SPEAKERS} and \
             {MAX_EXPECTED_SPEAKERS} (got {n}). For a larger group, choose \"Not sure\" \
             and Cadence detects speakers automatically."
        )),
    }
}

/// Everything one pipeline run needs. Grouped into a struct because the run
/// spans four subsystems and a positional argument list would be unreadable.
pub struct RunRequest<'a> {
    pub paths: &'a ModelPaths,
    pub config: &'a PipelineConfig,
    pub settings: &'a Settings,
    pub host: &'a HostInfo,
    /// The untouched recording (never modified, never deleted).
    pub input_wav: &'a Path,
    /// The meeting's own directory, where every artifact is written.
    pub out_dir: &'a Path,
    /// App data dir, for prompt overrides and the server log.
    pub data_dir: &'a Path,
    pub meeting_id: &'a str,
    pub title: &'a str,
}

pub struct PipelineOutcome {
    pub transcript: Transcript,
    pub mom: Option<Mom>,
    /// Set when the transcript succeeded but MoM generation did not. The
    /// meeting is still usable; the UI offers a retry.
    pub mom_error: Option<String>,
    pub metrics: RunMetrics,
}

/// Per-stage cost of one run, written to `metrics.json` beside the
/// artifacts. This is what turns every real meeting into a data point for the
/// performance budget instead of only the benchmark runs.
/// `#[serde(default)]`: a `metrics.json` written before the audio-quality
/// stages existed must still parse, rather than costing the benchmark harness
/// its history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RunMetrics {
    pub audio_ms: i64,
    pub preprocess_ms: u64,
    pub diarization: StageMetrics,
    pub stt: StageMetrics,
    pub align_ms: u64,
    pub mom_ms: u64,
    pub total_ms: u64,
    pub stt_model: String,
    pub llm_model: String,
    pub stt_threads: usize,
    pub diarization_threads: usize,
    /// The expected-speakers hint diarization ran with; `None` means automatic
    /// (threshold) clustering. Recorded so a run's speaker count can be read
    /// against what it was told to produce.
    pub expected_speakers: Option<u32>,
    /// What the front-end chain actually did to the audio.
    pub preprocess_report: Option<PreprocessReport>,
    /// `None` when the optional denoiser did not run.
    pub denoise: Option<StageMetrics>,
    pub vad: Option<StageMetrics>,
    /// 0.0..1.0 — how much of the recording is speech, when VAD ran.
    pub speech_ratio: Option<f32>,
    /// Length of the speech-only audio actually sent to STT, when trimming ran.
    pub stt_input_ms: Option<i64>,
    /// Blocks whisper was handed in code-switched mode. `None` for a
    /// single pass — which is also what tells the benchmark harness which
    /// mode produced a row.
    pub stt_blocks: Option<usize>,
    /// Languages actually detected across those blocks. More than one is the
    /// pipeline having caught a code switch; exactly one on a meeting known to
    /// be mixed is the signal that this needs looking at.
    pub stt_languages: Vec<String>,
    /// Anything about this run a user should know: a skipped optional stage, a
    /// recording that is mostly silence. Never fatal.
    pub warnings: Vec<String>,
    /// Unparliamentary-language scan. `0` when the mode is `off` and the scan
    /// never ran, which is also how an old `metrics.json` reads back.
    pub redact_ms: u64,
    /// Meeting-type classification. Cheap enough that it is not worth a
    /// separate progress stage, but worth measuring in case that stops being
    /// true on a three-hour recording.
    pub classify_ms: u64,
    /// Spans the redaction scan flagged across the whole transcript.
    pub flags_found: usize,
}

impl RunMetrics {
    /// Processing time as a multiple of the meeting's length. the budget is
    /// stated in these terms: a 60-minute meeting in 10 minutes is 0.17x.
    pub fn realtime_factor(&self) -> f64 {
        if self.audio_ms <= 0 {
            return 0.0;
        }
        self.total_ms as f64 / self.audio_ms as f64
    }
}

/// Runs the full local pipeline: preprocess → diarize →
/// STT → align → extract → synthesize. Every stage's raw output is written to
/// `out_dir` before the next stage runs, so a later failure never costs the
/// work already done — `original.wav` is never touched, and
/// `diarization.json` / `stt.json` stay separate from the derived
/// `transcript.json` and `mom.json`.
///
/// `on_stage` receives the state-machine stage name and a rough percentage.
pub fn run(
    request: &RunRequest,
    on_stage: &dyn Fn(&str, i64),
) -> Result<PipelineOutcome, String> {
    let out_dir = request.out_dir;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("could not create {out_dir:?}: {e}"))?;

    let started = std::time::Instant::now();
    let mut metrics = RunMetrics {
        stt_model: request.settings.effective_stt_model(),
        llm_model: request.settings.effective_llm_model(),
        stt_threads: request.config.stt_threads,
        diarization_threads: request.config.diarization_threads,
        expected_speakers: request.config.expected_speakers,
        ..RunMetrics::default()
    };

    on_stage("PREPROCESSING", 5);
    let stage_started = std::time::Instant::now();
    let processed = out_dir.join("processed.wav");
    let report = run_front_end(request, &processed, &mut metrics)?;
    metrics.preprocess_report = Some(report);
    metrics.preprocess_ms = stage_started.elapsed().as_millis() as u64;

    // Profiling reads `original.wav`, not `processed.wav`: the point of the
    // report is to tell the user what their *room and microphone* did, and the
    // front end has by then already hidden some of it (normalised the level,
    // possibly subtracted the room tone). It runs after preprocessing rather
    // than before only so a profiling bug can never delay the stages that
    // matter.
    run_audio_profile(request, &mut metrics);

    // VAD runs on the final preprocessed audio, before the expensive stages,
    // so its speech ratio describes exactly what they will see — and, in
    // code-switched mode, so STT knows where it can cut without clipping a
    // word.
    let vad_outcome = run_vad(request, &processed, &mut metrics);

    // Diarization always sees the *whole* recording: it is what puts speakers
    // on the original timeline, and trimming it would mean remapping twice.
    on_stage("DIARIZING", 15);
    let (diarization, diarization_metrics) =
        diarization::diarize(request.paths, request.config, &processed)?;
    metrics.diarization = diarization_metrics;
    write_json(&out_dir.join("diarization.json"), &diarization)?;

    on_stage("TRANSCRIBING", 35);
    let outcome = stt::transcribe(&stt::SttRequest {
        paths: request.paths,
        config: request.config,
        wav_path: &vad_outcome.stt_input,
        speech: vad_outcome.speech.as_deref(),
        data_dir: Some(request.data_dir),
    })?;
    let mut stt_segments = outcome.segments;
    metrics.stt = outcome.metrics;
    metrics.stt_blocks = outcome.blocks;
    metrics.stt_languages = outcome.languages;
    metrics.warnings.extend(outcome.warnings);
    // Back onto the original recording's timeline before anything downstream —
    // alignment, MoM evidence links and click-to-play all assume it.
    // (Code-switched mode has already done its own, per-block, mapping.)
    if let Some(timeline) = vad_outcome.timeline.as_ref() {
        timeline.remap_stt(&mut stt_segments);
    }
    write_json(&out_dir.join("stt.json"), &stt_segments)?;

    on_stage("ALIGNING", 60);
    let stage_started = std::time::Instant::now();
    let mut transcript = Transcript {
        meeting_id: request.meeting_id.to_string(),
        segments: align::align(&stt_segments, &diarization),
    };
    metrics.align_ms = stage_started.elapsed().as_millis() as u64;
    metrics.audio_ms = transcript.duration_ms();

    // Deliberately inside the ALIGNING window rather than announced as a stage
    // of its own: `PIPELINE_STAGES` in the UI is a fixed list, and a stage name
    // it does not know renders as an unlabelled step. Both of these are
    // sub-second on a meeting-length transcript, so there is nothing for a
    // progress bar to show anyway.
    run_redaction(request, &mut transcript, &mut metrics);
    // Written only after the flags are attached: `transcript.json` is the file
    // every reader (the transcript view, export, `run_mom_only`) treats as
    // authoritative, so it must never exist in a half-annotated state.
    write_json(&out_dir.join("transcript.json"), &transcript)?;
    run_classification(request, &transcript, &mut metrics);

    if !request.config.generate_mom {
        metrics.total_ms = started.elapsed().as_millis() as u64;
        let _ = write_json(&out_dir.join("metrics.json"), &metrics);
        on_stage("COMPLETED", 100);
        return Ok(PipelineOutcome {
            transcript,
            mom: None,
            mom_error: None,
            metrics,
        });
    }

    // From here on, the transcript is already safely on disk. A failure in the
    // LLM stage is reported, not fatal: the meeting keeps its transcript and
    // the user can retry minutes generation on its own.
    let stage_started = std::time::Instant::now();
    let minutes = generate_minutes(request, &transcript, on_stage);
    metrics.mom_ms = stage_started.elapsed().as_millis() as u64;
    metrics.total_ms = started.elapsed().as_millis() as u64;
    let _ = write_json(&out_dir.join("metrics.json"), &metrics);

    match minutes {
        Ok(mom) => {
            write_json(&out_dir.join("mom.json"), &mom)?;
            on_stage("COMPLETED", 100);
            Ok(PipelineOutcome {
                transcript,
                mom: Some(mom),
                mom_error: None,
                metrics,
            })
        }
        Err(err) => {
            on_stage("COMPLETED", 100);
            Ok(PipelineOutcome {
                transcript,
                mom: None,
                mom_error: Some(err),
                metrics,
            })
        }
    }
}

/// Format conversion, optional denoising, then the gain stage — writing
/// `processed.wav`. `original.wav` is only ever read.
///
/// The order matters: GTCRN is a 16 kHz model so it must run after the
/// resample, and normalising before it would hand it a boosted noise floor to
/// fight, so the gain stage — compressor then normalize — is applied to the
/// denoiser's output instead.
///
/// The two noise stages are mutually exclusive. When GTCRN will run, the
/// model-free reducer is dropped; when it will not, the full chain runs and the
/// model-free reducer is the noise handling.
///
/// Every failure below is a *skip*, not an error: the un-denoised audio it
/// started from is perfectly usable, and a processing stage must
/// never cost a meeting.
fn run_front_end(
    request: &RunRequest,
    processed: &Path,
    metrics: &mut RunMetrics,
) -> Result<PreprocessReport, String> {
    let options = &request.config.preprocess;
    if !request.config.denoise_enabled {
        return preprocess::preprocess(request.input_wav, processed, options);
    }

    if let Some(reason) = denoise::unavailable_reason(request.data_dir) {
        eprintln!("[pipeline] skipping noise suppression: {reason}");
        return preprocess::preprocess(request.input_wav, processed, options);
    }

    let converted = request.out_dir.join("converted.wav");
    let denoised = request.out_dir.join("denoised.wav");
    // GTCRN supersedes the model-free reducer, so the spectral-subtraction
    // stage is dropped here rather than run twice over the same audio: the
    // second pass would be estimating a noise floor the neural model has
    // already removed, and its cost is a full STFT over the whole recording.
    let front_options = options.without_gain().without_noise_reduction();
    let front = preprocess::preprocess(request.input_wav, &converted, &front_options)?;

    match denoise::denoise(
        request.data_dir,
        request.config.diarization_threads,
        &converted,
        &denoised,
    ) {
        Ok(stage) => {
            metrics.denoise = Some(stage);
            let gain = preprocess::apply_gain(&denoised, processed, options)?;
            Ok(merge_reports(front, gain))
        }
        Err(error) => {
            eprintln!("[pipeline] noise suppression failed, continuing without it: {error}");
            metrics
                .warnings
                .push(format!("Noise suppression was skipped: {error}"));
            // The model-free reducer was skipped above on the assumption GTCRN
            // would run. It did not, so the chain is re-run from `original.wav`
            // *with* it — the user asked for noise handling and this is the
            // only one left. A second failure falls back to the audio already
            // on disk: slightly noisier is always better than no meeting.
            let fallback = request.out_dir.join("converted_nr.wav");
            match preprocess::preprocess(request.input_wav, &fallback, &options.without_gain()) {
                Ok(front) => {
                    let gain = preprocess::apply_gain(&fallback, processed, options)?;
                    Ok(merge_reports(front, gain))
                }
                Err(second) => {
                    eprintln!("[pipeline] noise reduction fallback also failed: {second}");
                    let gain = preprocess::apply_gain(&converted, processed, options)?;
                    Ok(merge_reports(front, gain))
                }
            }
        }
    }
}

/// Stitches the two halves of a split front end back into one report.
///
/// When denoising runs, the chain is cut in two — DC/high-pass/noise reduction
/// before the model, leveler/compressor/normalize after it — and each half returns its
/// own report describing only what it did. Reporting just the second half
/// would silently drop the input format and the noise-reduction result, which
/// is exactly the information someone reads `metrics.json` to find.
fn merge_reports(front: PreprocessReport, gain: PreprocessReport) -> PreprocessReport {
    PreprocessReport {
        // The true input, from the half that actually opened `original.wav`.
        input_sample_rate: front.input_sample_rate,
        input_channels: front.input_channels,
        removed_dc_offset: front.removed_dc_offset,
        input_rms_dbfs: front.input_rms_dbfs,
        noise_reduction: front.noise_reduction,
        // The output, and the gain stages, from the half that wrote it.
        ..gain
    }
}

/// Measures the untouched recording and writes `audio_profile.json`.
///
/// Best-effort by construction: a meeting whose audio cannot be profiled is
/// still a meeting, and the profile exists to explain a poor transcript, not to
/// gate one.
fn run_audio_profile(request: &RunRequest, metrics: &mut RunMetrics) {
    match audio_profile::profile_wav(request.input_wav) {
        Ok(profile) => {
            // The profile's warnings are the room's problems — a mic too far
            // away, a fan, clipping. They belong in the run's warnings so the
            // one place a user looks after a bad transcript has them.
            metrics.warnings.extend(profile.warnings.iter().cloned());
            let _ = write_json(&request.out_dir.join("audio_profile.json"), &profile);
        }
        Err(error) => {
            eprintln!("[pipeline] audio profiling skipped: {error}");
        }
    }
}

/// Annotates the transcript's turns with unparliamentary-language spans and
/// writes `redaction.json`.
///
/// Flags are annotations, never edits: `segment.text` stays exactly what was
/// said, which is what makes a wrong flag reversible and lets the mode be
/// changed after the fact without reprocessing. Masking happens at render and
/// export time.
fn run_redaction(request: &RunRequest, transcript: &mut Transcript, metrics: &mut RunMetrics) {
    let mode = request.settings.effective_redaction_mode();
    if mode == redact::RedactionMode::Off {
        return;
    }
    let started = std::time::Instant::now();

    // Loaded from the data dir rather than `builtin()`, so a user's own
    // additions and — more importantly — their removals of a false positive
    // apply to the run that is happening now.
    let lexicon = redact::Lexicon::load(request.data_dir);
    let turns: Vec<String> = transcript
        .segments
        .iter()
        .map(|segment| segment.text.clone())
        .collect();
    let options = redact::ScanOptions {
        fuzzy: request.settings.redaction_fuzzy,
    };
    let (per_turn, summary) = redact::scan_turns(&turns, &lexicon, &options);

    for (segment, flags) in transcript.segments.iter_mut().zip(per_turn) {
        segment.flags = flags;
    }
    metrics.flags_found = summary.total;
    metrics.redact_ms = started.elapsed().as_millis() as u64;
    let _ = write_json(&request.out_dir.join("redaction.json"), &summary);
}

/// Classifies the meeting and writes `classification.json`.
///
/// Reads the transcript that is already on disk, adds a file beside it, and
/// touches nothing else — so this is the one stage that could be deleted
/// wholesale without any other stage noticing.
fn run_classification(request: &RunRequest, transcript: &Transcript, metrics: &mut RunMetrics) {
    if !request.settings.classify_meeting {
        return;
    }
    let started = std::time::Instant::now();
    let classification = topic::classify(transcript);
    metrics.classify_ms = started.elapsed().as_millis() as u64;
    let _ = write_json(&request.out_dir.join("classification.json"), &classification);
}

/// What the VAD stage leaves for STT to work with.
struct VadOutcome {
    /// The WAV to transcribe: `processed.wav`, or the speech-only copy.
    stt_input: std::path::PathBuf,
    /// Set only when `stt_input` is a trimmed file — the mapping back to the
    /// original recording's timeline.
    timeline: Option<vad::Timeline>,
    /// Speech regions on `stt_input`'s timeline, for code-switched blocking.
    speech: Option<Vec<vad::SpeechSegment>>,
}

/// Measures how much of the recording is speech and, when the user has asked
/// for it, writes the speech-only audio STT will read.
fn run_vad(request: &RunRequest, processed: &Path, metrics: &mut RunMetrics) -> VadOutcome {
    let untrimmed = || VadOutcome {
        stt_input: processed.to_path_buf(),
        timeline: None,
        speech: None,
    };
    if !request.config.vad_enabled {
        return untrimmed();
    }
    if !vad::is_available(request.data_dir) {
        eprintln!("[pipeline] skipping voice activity detection: model not installed");
        return untrimmed();
    }

    let scratch = request.out_dir.join("vad_scratch.wav");
    let report = match vad::detect(
        request.data_dir,
        request.config.diarization_threads,
        processed,
        &scratch,
    ) {
        Ok((report, stage)) => {
            metrics.vad = Some(stage);
            report
        }
        Err(error) => {
            eprintln!("[pipeline] voice activity detection failed, continuing without it: {error}");
            metrics
                .warnings
                .push(format!("Voice activity detection was skipped: {error}"));
            return untrimmed();
        }
    };

    metrics.speech_ratio = Some(report.speech_ratio);
    if let Some(warning) = vad::speech_ratio_warning(report.speech_ratio) {
        metrics.warnings.push(warning);
    }
    let _ = write_json(&request.out_dir.join("vad.json"), &report);

    let untrimmed_with_speech = VadOutcome {
        stt_input: processed.to_path_buf(),
        timeline: None,
        speech: Some(report.segments.clone()),
    };

    // Code-switched STT already skips the silence between its blocks, and it
    // needs the speech segments to line up with the audio it cuts. Trimming
    // first would give it segments on one timeline and samples on another, so
    // the two features are mutually exclusive and blocking wins: it gets the
    // same speed benefit without `vad::Timeline`'s stretched-interior caveat.
    let code_switched = matches!(
        stt::SttLanguage::parse(&request.config.stt_language),
        stt::SttLanguage::CodeSwitched
    );
    if !request.config.vad_trim_silence || code_switched {
        return untrimmed_with_speech;
    }

    let speech_wav = request.out_dir.join("speech.wav");
    match vad::write_trimmed(
        processed,
        &speech_wav,
        &report.segments,
        request.config.vad_pad_ms,
    ) {
        Ok(timeline) => {
            metrics.stt_input_ms = Some(timeline.trimmed_duration_ms());
            VadOutcome {
                stt_input: speech_wav,
                timeline: Some(timeline),
                // Deliberately dropped: these are on the *original* timeline
                // and the audio is not.
                speech: None,
            }
        }
        Err(error) => {
            // Transcribing the whole file is slower, never wrong. Silently
            // dropping to it beats a transcript with misplaced timestamps.
            eprintln!("[pipeline] silence trimming skipped: {error}");
            metrics
                .warnings
                .push(format!("Silence trimming was skipped: {error}"));
            untrimmed_with_speech
        }
    }
}

fn generate_minutes(
    request: &RunRequest,
    transcript: &Transcript,
    on_stage: &dyn Fn(&str, i64),
) -> Result<Mom, String> {
    on_stage("EXTRACTING", 62);
    // The model server is started here and dropped at the end of this
    // function, so no weights stay resident once the meeting is processed.
    let provider = llm::LlamaCppProvider::start(
        request.paths,
        request.settings,
        request.host,
        request.data_dir,
    )?;

    mom::generate(
        &provider,
        &mom::MomRequest {
            transcript,
            title: request.title,
            language: &request.config.mom_language,
            data_dir: Some(request.data_dir),
            redaction: request.settings.effective_redaction_mode(),
        },
        on_stage,
    )
}

/// Regenerates only the minutes for a meeting whose transcript already exists —
/// the retry path for an LLM failure, and what the "Regenerate minutes" action
/// in the UI calls after a model or language change.
pub fn run_mom_only(request: &RunRequest, on_stage: &dyn Fn(&str, i64)) -> Result<Mom, String> {
    let transcript_path = request.out_dir.join("transcript.json");
    let raw = std::fs::read_to_string(&transcript_path).map_err(|e| {
        format!(
            "no transcript at {} to summarise: {e}",
            transcript_path.display()
        )
    })?;
    let transcript: Transcript =
        serde_json::from_str(&raw).map_err(|e| format!("could not parse transcript.json: {e}"))?;

    let mom = generate_minutes(request, &transcript, on_stage)?;
    write_json(&request.out_dir.join("mom.json"), &mom)?;
    on_stage("COMPLETED", 100);
    Ok(mom)
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| format!("could not write {path:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_follows_the_machine_by_default() {
        let host = HostInfo {
            total_ram_mb: 15_700,
            available_ram_mb: 9_000,
            logical_cores: 8,
        };
        let config = PipelineConfig::from_settings(&Settings::default(), &host);
        assert_eq!(config.stt_threads, 8);
        assert_eq!(config.diarization_threads, 8);
        assert_eq!(config.cluster_threshold, DEFAULT_CLUSTER_THRESHOLD);
        assert!(config.generate_mom);
        // A per-meeting hint never comes from Settings.
        assert_eq!(config.expected_speakers, None);
    }

    #[test]
    fn expected_speakers_accepts_not_sure_and_one_to_four_only() {
        assert_eq!(validate_expected_speakers(None), Ok(None));
        assert_eq!(validate_expected_speakers(Some(1)), Ok(Some(1)));
        assert_eq!(validate_expected_speakers(Some(4)), Ok(Some(4)));
        assert!(validate_expected_speakers(Some(0)).is_err());
        // Five or more is automatic clustering, never a fixed count.
        assert!(validate_expected_speakers(Some(5)).is_err());
        assert!(validate_expected_speakers(Some(7)).is_err());
        assert!(validate_expected_speakers(Some(u32::MAX)).is_err());
    }

    #[test]
    fn a_big_machine_caps_diarization_but_not_stt() {
        let host = HostInfo {
            total_ram_mb: 31_700,
            available_ram_mb: 20_000,
            logical_cores: 32,
        };
        let config = PipelineConfig::from_settings(&Settings::default(), &host);
        assert_eq!(config.stt_threads, MAX_STT_THREADS);
        assert_eq!(config.diarization_threads, MAX_DIARIZATION_THREADS);
    }

    #[test]
    fn realtime_factor_is_processing_time_over_meeting_length() {
        let metrics = RunMetrics {
            audio_ms: 3_600_000,
            total_ms: 600_000,
            ..RunMetrics::default()
        };
        // the target: a 60-minute meeting processed in 10 minutes.
        assert!((metrics.realtime_factor() - 0.166).abs() < 0.01);
        assert_eq!(RunMetrics::default().realtime_factor(), 0.0);
    }

    #[test]
    fn explicit_thread_settings_win() {
        let host = HostInfo {
            total_ram_mb: 31_700,
            available_ram_mb: 20_000,
            logical_cores: 32,
        };
        let settings = Settings {
            stt_threads: Some(6),
            diarization_threads: Some(2),
            ..Settings::default()
        };
        let config = PipelineConfig::from_settings(&settings, &host);
        assert_eq!(config.stt_threads, 6);
        assert_eq!(config.diarization_threads, 2);
    }
}
