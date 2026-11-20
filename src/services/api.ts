// The single place the UI talks to Rust. Components never call `invoke`
// directly — keeps business logic out of React, and this is
// the seam that makes that enforceable.

import { invoke } from "@tauri-apps/api/core";
import type {
  AudioProfile,
  MicMeter,
  AudioDeviceInfo,
  HostReport,
  InstallProgress,
  MeetingClassification,
  MeetingMetadata,
  MeetingStatus,
  MeetingSummary,
  ModelStatus,
  Mom,
  RecordingSummary,
  RedactionSummary,
  RetiredModel,
  SetupStatus,
  Settings,
  Transcript,
  VerifyReport,
} from "../types";

// --- audio -------------------------------------------------------

export const listInputDevices = () =>
  invoke<AudioDeviceInfo[]>("list_input_devices");

export const startRecording = (deviceName: string | null, meetingId: string) =>
  invoke<RecordingSummary>("start_recording", { deviceName, meetingId });

export const stopRecording = () => invoke<RecordingSummary>("stop_recording");

export const getMicLevel = () => invoke<number>("get_mic_level");

export const takeRecordingError = () =>
  invoke<string | null>("take_recording_error");

/** Richer than `getMicLevel`: dBFS, clipping count and a live noise-floor
 * estimate, so the UI can warn about a bad input while it can still be fixed. */
export const getMicMeter = () => invoke<MicMeter>("get_mic_meter");

/** Records a short sample and reports what the microphone actually sounds
 * like — a pre-flight check, so a bad mic is found before a real meeting. */
export const testMicrophone = (deviceName: string | null, seconds?: number) =>
  invoke<AudioProfile>("test_microphone", { deviceName, seconds: seconds ?? null });

export const profileMeetingAudio = (meetingId: string) =>
  invoke<AudioProfile>("profile_meeting_audio", { meetingId });

// --- meeting lifecycle -----------------------------------------

export const createMeeting = (title: string, notes: string) =>
  invoke<MeetingMetadata>("create_meeting", { title, notes });

export const finalizeRecording = (
  meetingId: string,
  deviceName: string | null,
  sampleRate: number,
  channels: number,
) =>
  invoke<MeetingMetadata>("finalize_recording", {
    meetingId,
    deviceName,
    sampleRate,
    channels,
  });

/** `expectedSpeakers` is required on purpose: `null` means "Not sure"
 * (automatic clustering) and is saved as such, so a caller that forgot to pass
 * the meeting's saved hint would silently clear it. */
export const startProcessing = (meetingId: string, expectedSpeakers: number | null) =>
  invoke<void>("start_processing", { meetingId, expectedSpeakers });

export const regenerateMom = (meetingId: string) =>
  invoke<void>("regenerate_mom", { meetingId });

export const getProcessingStatus = (meetingId: string) =>
  invoke<MeetingStatus>("get_processing_status", { meetingId });

// --- library -----------------------------------------------------

export const listMeetings = (
  query?: string,
  fromMs?: number | null,
  toMs?: number | null,
) =>
  invoke<MeetingMetadata[]>("list_meetings", {
    query: query || null,
    fromMs: fromMs ?? null,
    toMs: toMs ?? null,
  });

export const getMeeting = (meetingId: string) =>
  invoke<MeetingSummary>("get_meeting", { meetingId });

export const updateMeeting = (
  meetingId: string,
  title: string | null,
  notes: string | null,
) => invoke<MeetingMetadata>("update_meeting", { meetingId, title, notes });

export const renameSpeaker = (
  meetingId: string,
  speaker: string,
  name: string,
) => invoke<MeetingMetadata>("rename_speaker", { meetingId, speaker, name });

export const deleteMeeting = (meetingId: string) =>
  invoke<void>("delete_meeting", { meetingId });

export const getTranscript = (meetingId: string) =>
  invoke<Transcript>("get_transcript", { meetingId });

export const getMom = (meetingId: string) => invoke<Mom>("get_mom", { meetingId });

export const exportMeeting = (meetingId: string, format: "markdown" | "json") =>
  invoke<string>("export_meeting", { meetingId, format });

export const revealMeetingFolder = (meetingId: string) =>
  invoke<string>("reveal_meeting_folder", { meetingId });

// --- meeting insights ------------------------------------------------

/** Runs the classifier over a meeting's transcript and writes
 * `classification.json`. Cheap: it counts over the transcript, no model is
 * loaded and nothing leaves the machine. */
export const classifyMeeting = (meetingId: string) =>
  invoke<MeetingClassification>("classify_meeting", { meetingId });

/** Null for a meeting recorded before classification existed, or one that
 * has never been processed — the caller shows the panel only when it isn't. */
export const getMeetingClassification = (meetingId: string) =>
  invoke<MeetingClassification | null>("get_meeting_classification", {
    meetingId,
  });

/** Re-runs the sensitive-language scan over an existing transcript with the
 * current lexicon and settings, rewriting the per-turn flags in place. */
export const rescanRedactions = (meetingId: string) =>
  invoke<RedactionSummary>("rescan_redactions", { meetingId });

/** Null when this meeting has never been scanned. */
export const getRedactionSummary = (meetingId: string) =>
  invoke<RedactionSummary | null>("get_redaction_summary", { meetingId });

// --- settings and models --------------------------------------------

export const getHostInfo = () => invoke<HostReport>("get_host_info");

export const getSettings = () => invoke<Settings>("get_settings");

export const saveSettings = (newSettings: Settings) =>
  invoke<Settings>("save_settings", { newSettings });

export const checkModels = () => invoke<SetupStatus>("check_models");

export const listModels = () => invoke<ModelStatus[]>("list_models");

export const installModel = (modelId: string) =>
  invoke<void>("install_model", { modelId });

/** Queues every missing required model and installs them one after another
 * in the backend. Resolves with the queued ids in install order; progress is
 * read per id with `getInstallProgress`. Rejects if a setup run is already
 * going. Only ever called from an explicit click on the setup screen. */
export const installRequiredModels = () =>
  invoke<string[]>("install_required_models");

export const getInstallProgress = (modelId: string) =>
  invoke<InstallProgress | null>("get_install_progress", { modelId });

export const removeModel = (modelId: string) =>
  invoke<void>("remove_model", { modelId });

export const verifyModel = (modelId: string) =>
  invoke<VerifyReport>("verify_model", { modelId });

/** Models in the models folder that the current set no longer uses. */
export const listRetiredModels = () => invoke<RetiredModel[]>("list_retired_models");

/** Deletes one of them. Only ever called after a confirmed click; the backend
 * refuses while a download is running and never touches the current set. */
export const removeRetiredModel = (key: string) =>
  invoke<RetiredModel>("remove_retired_model", { key });

/// Errors from `invoke` arrive as strings; anything else is unexpected and
/// still needs to reach the user rather than being swallowed.
export function errorMessage(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  return String(err);
}
