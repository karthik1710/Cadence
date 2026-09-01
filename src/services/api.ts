// The single place the UI talks to Rust. Components never call `invoke`
// directly — master prompt §19 keeps business logic out of React, and this is
// the seam that makes that enforceable.

import { invoke } from "@tauri-apps/api/core";
import type {
  AudioProfile,
  MicMeter,
  AudioDeviceInfo,
  HostReport,
  InstallProgress,
  MeetingMetadata,
  MeetingStatus,
  MeetingSummary,
  ModelStatus,
  Mom,
  RecordingSummary,
  SetupStatus,
  Settings,
  Transcript,
  VerifyReport,
} from "../types";

// --- audio (§9, §10) -------------------------------------------------------

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

// --- meeting lifecycle (§11, §12) -----------------------------------------

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

export const startProcessing = (meetingId: string) =>
  invoke<void>("start_processing", { meetingId });

export const regenerateMom = (meetingId: string) =>
  invoke<void>("regenerate_mom", { meetingId });

export const getProcessingStatus = (meetingId: string) =>
  invoke<MeetingStatus>("get_processing_status", { meetingId });

// --- library (§23-§26) -----------------------------------------------------

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

// --- settings and models (§27) --------------------------------------------

export const getHostInfo = () => invoke<HostReport>("get_host_info");

export const getSettings = () => invoke<Settings>("get_settings");

export const saveSettings = (newSettings: Settings) =>
  invoke<Settings>("save_settings", { newSettings });

export const checkModels = () => invoke<SetupStatus>("check_models");

export const listModels = () => invoke<ModelStatus[]>("list_models");

export const installModel = (modelId: string) =>
  invoke<void>("install_model", { modelId });

export const getInstallProgress = (modelId: string) =>
  invoke<InstallProgress | null>("get_install_progress", { modelId });

export const removeModel = (modelId: string) =>
  invoke<void>("remove_model", { modelId });

export const verifyModel = (modelId: string) =>
  invoke<VerifyReport>("verify_model", { modelId });

const INSTALL_POLL_MS = 700;

export interface BatchInstallStep {
  modelId: string;
  index: number;
  total: number;
  progress: InstallProgress;
}

export interface BatchInstallFailure {
  modelId: string;
  error: string;
}

/**
 * Installs several catalog entries one after another — the "install the
 * recommended setup for this machine" action (§27), built entirely from the
 * per-model primitives above rather than a new backend command. `onProgress`
 * fires as each model's download/extract/verify advances; the returned list
 * holds only the ones that failed, so the caller can report success or
 * failures without re-deriving state.
 */
export async function installMissingModels(
  modelIds: string[],
  onProgress?: (step: BatchInstallStep) => void,
): Promise<BatchInstallFailure[]> {
  const failures: BatchInstallFailure[] = [];

  for (let index = 0; index < modelIds.length; index++) {
    const modelId = modelIds[index];
    try {
      await installModel(modelId);
    } catch (err) {
      failures.push({ modelId, error: errorMessage(err) });
      continue;
    }

    // eslint-disable-next-line no-await-in-loop -- installs are sequential by design, not accidental serialization.
    await new Promise<void>((resolve) => {
      const timer = window.setInterval(async () => {
        let current: InstallProgress | null;
        try {
          current = await getInstallProgress(modelId);
        } catch (err) {
          window.clearInterval(timer);
          failures.push({ modelId, error: errorMessage(err) });
          resolve();
          return;
        }
        if (!current) return;
        onProgress?.({ modelId, index, total: modelIds.length, progress: current });
        if (current.phase === "done" || current.phase === "failed") {
          window.clearInterval(timer);
          if (current.phase === "failed") {
            failures.push({ modelId, error: current.error ?? "The download failed." });
          }
          resolve();
        }
      }, INSTALL_POLL_MS);
    });
  }

  return failures;
}

/// Errors from `invoke` arrive as strings; anything else is unexpected and
/// still needs to reach the user rather than being swallowed.
export function errorMessage(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  return String(err);
}
