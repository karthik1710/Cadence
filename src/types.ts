// Mirrors the Rust types in src-tauri/src. Field names are the serialized
// ones, so anything renamed with `#[serde(rename)]` on the Rust side (the MoM
// schema keys from the MoM schema) is spelled the same way here.

export interface AudioDeviceInfo {
  name: string;
  is_default: boolean;
}

export interface RecordingSummary {
  file_path: string;
  sample_rate: number;
  channels: number;
}

export type MeetingState =
  | "IDLE"
  | "RECORDING"
  | "RECORDED"
  | "PREPROCESSING"
  | "DIARIZING"
  | "TRANSCRIBING"
  | "ALIGNING"
  | "EXTRACTING"
  | "SYNTHESIZING"
  | "COMPLETED"
  | "FAILED";

export interface MeetingMetadata {
  id: string;
  title: string;
  notes: string;
  created_at_ms: number;
  duration_ms: number;
  state: MeetingState;
  error: string | null;
  device_name: string | null;
  sample_rate: number;
  channels: number;
  speaker_names: Record<string, string>;
  has_transcript: boolean;
  has_mom: boolean;
  mom_error: string | null;
}

export interface MeetingSummary extends MeetingMetadata {
  audio_path: string | null;
  /** "How many people spoke?" as processing was last started with it.
   * `null` is "Not sure" — automatic speaker clustering. */
  expected_speakers: number | null;
}

/** Bounds on the expected-speakers hint; mirrors `MIN/MAX_EXPECTED_SPEAKERS`
 * in `pipeline/mod.rs`, which is what actually enforces them. */
export const MIN_EXPECTED_SPEAKERS = 1;
export const MAX_EXPECTED_SPEAKERS = 4;

export interface MeetingStatus {
  status: "idle" | "processing" | "completed" | "failed";
  stage: string | null;
  progress: number;
  error: string | null;
  warning: string | null;
}

export interface TranscriptSegment {
  speaker: string;
  speaker_label: string;
  start_ms: number;
  end_ms: number;
  text: string;
  /** Language whisper detected for this turn, when transcribed in
   * code-switched mode. Null when a single language covered the recording. */
  language: string | null;
  /** Sensitive-language spans found in `text`. Absent when the turn has
   * none, and absent entirely on transcripts produced before the scan
   * existed. Offsets are Unicode code points — see `Flag`. */
  flags?: Flag[];
}

export interface Transcript {
  meeting_id: string;
  segments: TranscriptSegment[];
}

export interface Evidenced {
  text: string;
  timestamp: number | null;
  speaker: string | null;
}

export interface ActionItem {
  description: string;
  owner: string | null;
  deadline: string | null;
  timestamp: number | null;
  confidence: string;
}

export interface MomMeta {
  model: string;
  quantization: string;
  runtime: string;
  context_tokens: number;
  extraction_prompt: string;
  synthesis_prompt: string;
  language: string;
  /**
   * The redaction policy the transcript was rendered under before the model
   * saw it: "off" | "flag" | "mask" | "redact", or "unknown" for a mom.json
   * written before this was recorded — an old file genuinely does not say.
   */
  redaction_mode: string;
  chunk_count: number;
  generated_at_ms: number;
  repaired: boolean;
  duration_ms: number;
}

export interface Mom {
  summary: string;
  participants: string[];
  topics: Evidenced[];
  decisions: Evidenced[];
  actionItems: ActionItem[];
  openQuestions: Evidenced[];
  risks: Evidenced[];
  detailedDiscussion: Evidenced[];
  meta: MomMeta | null;
}

export interface Settings {
  /** "auto" | "code-switched" | an ISO code such as "ta"/"en". */
  stt_language: string;
  input_device: string | null;
  stt_threads: number | null;
  diarization_threads: number | null;
  llm_threads: number | null;
  cluster_threshold: number;
  mom_language: string;
  generate_mom: boolean;
  debug_logging: boolean;

  // --- audio front-end -------------------------------------------------
  // Each stage is a judgement call about someone else's room and microphone,
  // so each is toggleable on its own. These existed in Rust long before they
  // existed here; leaving them out of this interface meant the Settings
  // screen's read-modify-write had no way to show or preserve them.
  /** Subtract the microphone's constant electrical offset. */
  preprocess_remove_dc: boolean;
  /** High-pass corner in Hz. `0` disables it. */
  preprocess_high_pass_hz: number;
  /** Normalise loudness before transcription. */
  preprocess_normalize: boolean;
  preprocess_target_rms_dbfs: number;
  /** Run the GTCRN denoiser (installed at setup) over the audio. */
  denoise_enabled: boolean;
  /** Measure how much of the recording is speech. */
  vad_enabled: boolean;
  /** Cut silence out of the audio sent to STT. */
  vad_trim_silence: boolean;
  vad_speech_pad_ms: number;

  // --- dynamics and model-free noise reduction --------------------------
  /** On by default: O(n), and it is what lifts the far end of a meeting
   * table above the noise floor. */
  preprocess_compress: boolean;
  preprocess_compress_threshold_dbfs: number;
  preprocess_compress_ratio: number;
  preprocess_compress_knee_db: number;
  preprocess_compress_attack_ms: number;
  preprocess_compress_release_ms: number;
  /** On by default. A full STFT over the whole recording, and superseded by
   * the GTCRN denoiser whenever that runs. */
  preprocess_noise_reduction: boolean;
  preprocess_noise_reduction_strength: number;

  // --- sensitive language and classification ----------------------------
  /** "off" | "flag" | "mask" | "redact" — anything else is read as "flag".
   * Kept as a plain string because that is what Rust stores; parse it with
   * `parseRedactionMode` before using it. */
  redaction_mode: string;
  /** Also catch misspellings and spaced-out spellings. */
  redaction_fuzzy: boolean;
  /** Detect the meeting type and its topics after transcription. */
  classify_meeting: boolean;

  theme: string;
}

/** Every machine runs the same model set — there are no hardware tiers —
 * so this reports what is in use, not a recommendation. */
export interface HostReport {
  total_ram_mb: number;
  available_ram_mb: number;
  logical_cores: number;
  llm_model: string;
  stt_model: string;
  llm_context: number;
  minimum_ram_mb: number;
  meets_minimum_ram: boolean;
  llm_ram_budget_mb: number;
  stt_threads: number;
  diarization_threads: number;
  llm_threads: number;
  data_dir: string;
}

/** Mirrors the Rust `ModelKind` (serialized lowercase). */
export type ModelKind = "runtime" | "stt" | "llm" | "diarization" | "denoise" | "vad";

export interface ModelStatus {
  id: string;
  label: string;
  kind: ModelKind;
  installed: boolean;
  path: string;
  approx_size_mb: number;
  actual_size_mb: number;
  notes: string;
  required: boolean;
  recorded_sha256: string | null;
  /** A runtime shipped inside the installer: never downloaded or reinstalled
   * from the app (reinstalling Cadence repairs it), only verified. */
  bundled: boolean;
}

export interface SetupStatus {
  ready: boolean;
  missing: ModelStatus[];
  message: string | null;
  total_ram_mb: number;
  minimum_ram_mb: number;
  /** Advisory only: below the minimum, setup and recording still proceed. */
  meets_minimum_ram: boolean;
  /** Sum of every `missing` item's download size. */
  total_download_mb: number;
  /** Models the current set no longer uses. Never blocks setup. */
  retired: RetiredModel[];
}

/** Something in the models folder the current model set does not use — a
 * model an earlier build installed, or a file put there by hand. Removed only
 * on an explicit, confirmed click. */
export interface RetiredModel {
  /** Location under the models folder, `/`-separated; what removal takes.
   * Two keys are groups: `bin/*` (old flat runtime files, once every runtime
   * has its own folder) and `bin/**` (every downloaded runtime, once all three
   * run from the copies bundled with the app). At most one is listed. */
  key: string;
  id: string | null;
  label: string;
  path: string;
  size_mb: number;
  /** An earlier Cadence build installed it (vs. a file nobody recognises). */
  from_previous_version: boolean;
}

export interface InstallProgress {
  id: string;
  /** "queued" only occurs under `install_required_models`, which installs
   * one model at a time. */
  phase: "queued" | "downloading" | "extracting" | "verifying" | "done" | "failed";
  percent: number;
  downloaded_mb: number;
  total_mb: number;
  error: string | null;
}

export interface VerifyReport {
  id: string;
  ok: boolean;
  message: string;
  sha256: string | null;
}

/// The stages the UI shows as a checklist, in order.
export const PIPELINE_STAGES: { state: MeetingState; label: string }[] = [
  { state: "PREPROCESSING", label: "Audio processing" },
  { state: "DIARIZING", label: "Speaker detection" },
  { state: "TRANSCRIBING", label: "Transcription" },
  { state: "ALIGNING", label: "Alignment" },
  { state: "EXTRACTING", label: "Reading the discussion" },
  { state: "SYNTHESIZING", label: "Writing the minutes" },
];

/** Live recording-quality meter (input warnings). */
export interface MicMeter {
  peak: number;
  peak_dbfs: number;
  rms_dbfs: number;
  session_peak_dbfs: number;
  noise_floor_dbfs: number;
  estimated_snr_db: number;
  clipped_samples: number;
  clipped_percent: number;
  recording: boolean;
}

/** Audio-quality report for a recording, with plain-language advice. */
export interface AudioProfile {
  duration_ms: number;
  sample_rate: number;
  channels: number;
  peak_dbfs: number;
  rms_dbfs: number;
  noise_floor_dbfs: number;
  estimated_snr_db: number;
  clipped_samples: number;
  clipped_percent: number;
  silence_percent: number;
  dc_offset: number;
  /** Includes `noise.findings`' advice, folded in by the Rust side. */
  warnings: string[];

  /** 90th-percentile frame level: how loud the speech itself is, as opposed
   * to `rms_dbfs`, which the silences drag down. */
  speech_dbfs: number;
  /** `peak_dbfs - rms_dbfs`. Low means squashed or clipped, very high means
   * a quiet recording with occasional bangs in it. */
  crest_factor_db: number;
  /** Signal-to-noise measured only over 300–3400 Hz, the band speech
   * actually occupies — hum and hiss outside it do not flatter the number. */
  speech_band_snr_db: number;
  noise: NoiseReport;
  /** 0–100, computed from level, clipping, SNR and the noise findings. */
  quality_score: number;
  quality_grade: string;
  /** At most 120 buckets, evenly spread across the recording. */
  timeline: ProfileFrame[];
}

/** What kind of unwanted sound was identified in a recording. */
export type NoiseKind =
  | "mains_hum"
  | "fan"
  | "rumble"
  | "hiss"
  | "impulsive"
  | "reverberant"
  | "babble"
  | "clipping";

export interface NoiseFinding {
  kind: NoiseKind;
  label: string;
  /** 0–1. */
  confidence: number;
  /** What was measured. */
  detail: string;
  /** What to do about it next time. */
  advice: string;
}

export interface NoiseReport {
  findings: NoiseFinding[];
  quiet_frames: number;
  quiet_share: number;
  /** 50 or 60 Hz when mains hum was identified, else null. */
  mains_hz: number | null;
  /** Reverberation decay estimate, null when it could not be measured. */
  decay_ms: number | null;
  transients_per_minute: number;
  noise_flatness: number;
  noise_tilt_db: number;
  noise_floor_dbfs: number;
  speech_to_noise_db: number;
}

/** One bucket of the level timeline. */
export interface ProfileFrame {
  start_ms: number;
  rms_dbfs: number;
  peak_dbfs: number;
  clipped_percent: number;
}

// --- sensitive language ---------------------------------------------------

export type RedactionCategory =
  | "profanity"
  | "slur"
  | "sexual"
  | "violent"
  | "insult";

export type RedactionSeverity = "mild" | "strong" | "severe";

export type MatchMethod = "exact" | "spaced" | "fuzzy" | "phonetic";

/** How a flagged span is presented. Rust stores this as a plain string; use
 * `parseRedactionMode` to read one safely. */
export type RedactionMode = "off" | "flag" | "mask" | "redact";

/**
 * One sensitive-language hit inside a turn's text.
 *
 * `start_char`/`end_char` are Unicode **code point** offsets — not UTF-16
 * code units and not bytes. Slicing with `String.prototype.slice` is wrong
 * the moment a turn contains Tamil or an emoji, which in this app is the
 * normal case; go through `Array.from(text)` instead (see
 * `services/format.ts`).
 */
export interface Flag {
  start_char: number;
  end_char: number;
  term_id: string;
  category: RedactionCategory;
  severity: RedactionSeverity;
  method: MatchMethod;
}

export interface RedactionSummary {
  total: number;
  turns_affected: number;
  by_category: Record<string, number>;
  by_severity: Record<string, number>;
  /** `[term_id, count]`, most frequent first. */
  terms: [string, number][];
  fuzzy: boolean;
  lexicon_terms: number;
}

// --- meeting classification -----------------------------------------------

export interface TypeScore {
  type: string;
  label: string;
  score: number;
}

export interface TopicTag {
  term: string;
  count: number;
  /** How widely the term is spread across the meeting, 0–1. */
  spread: number;
  score: number;
}

export interface MeetingSignals {
  duration_ms: number;
  speaker_count: number;
  turn_count: number;
  word_count: number;
  mean_turn_words: number;
  question_ratio: number;
  dominant_speaker_share: number;
}

export interface MeetingClassification {
  /** The serialized key is `type`, not `meeting_type`. */
  type: string;
  label: string;
  /** 0–1. */
  confidence: number;
  alternatives: TypeScore[];
  topics: TopicTag[];
  signals: MeetingSignals;
}
