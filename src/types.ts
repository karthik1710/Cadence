// Mirrors the Rust types in src-tauri/src. Field names are the serialized
// ones, so anything renamed with `#[serde(rename)]` on the Rust side (the MoM
// schema keys from master prompt §15) is spelled the same way here.

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
}

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
  input_device: string | null;
  profile: string;
  stt_model: string | null;
  llm_model: string | null;
  stt_threads: number | null;
  diarization_threads: number | null;
  llm_threads: number | null;
  cluster_threshold: number;
  llm_context_tokens: number | null;
  mom_language: string;
  generate_mom: boolean;
  debug_logging: boolean;
  theme: string;
}

export interface HostReport {
  total_ram_mb: number;
  available_ram_mb: number;
  logical_cores: number;
  profile: string;
  detected_profile: string;
  llm_ram_budget_mb: number;
  recommended_llm: string;
  recommended_stt: string;
  recommended_context: number;
  stt_threads: number;
  diarization_threads: number;
  llm_threads: number;
  data_dir: string;
}

export type ModelKind = "runtime" | "stt" | "llm" | "diarization";

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
}

export interface SetupStatus {
  ready: boolean;
  missing: ModelStatus[];
  message: string | null;
  profile: string;
  total_ram_mb: number;
}

export interface InstallProgress {
  id: string;
  phase: "downloading" | "extracting" | "verifying" | "done" | "failed";
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

/// The §12 stages the UI shows as a checklist, in order.
export const PIPELINE_STAGES: { state: MeetingState; label: string }[] = [
  { state: "PREPROCESSING", label: "Audio processing" },
  { state: "DIARIZING", label: "Speaker detection" },
  { state: "TRANSCRIBING", label: "Transcription" },
  { state: "ALIGNING", label: "Alignment" },
  { state: "EXTRACTING", label: "Reading the discussion" },
  { state: "SYNTHESIZING", label: "Writing the minutes" },
];

/** Live recording-quality meter (§9 input warnings). */
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
  warnings: string[];
}
