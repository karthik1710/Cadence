import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../services/api";
import type { AudioDeviceInfo } from "../types";

export type RecorderState = "idle" | "starting" | "recording" | "stopping";

const LEVEL_POLL_MS = 150;
/** Below this for a sustained stretch, the mic is probably not picking up. */
const LOW_LEVEL = 0.02;
/** At or above this, the input is clipping and the audio will be distorted. */
const CLIPPING_LEVEL = 0.98;
const WARNING_SAMPLES = Math.round(4000 / LEVEL_POLL_MS);

/**
 * Owns the recording half of the meeting state machine and the input
 * warnings this surfaces: low input, clipping, and device disconnection.
 */
export function useRecorder() {
  const [state, setState] = useState<RecorderState>("idle");
  const [devices, setDevices] = useState<AudioDeviceInfo[]>([]);
  const [devicesError, setDevicesError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [deviceError, setDeviceError] = useState<string | null>(null);
  const [inputWarning, setInputWarning] = useState<string | null>(null);

  const startedAt = useRef<number | null>(null);
  const quietSamples = useRef(0);
  const timer = useRef<number | null>(null);

  const refreshDevices = useCallback(async () => {
    try {
      const list = await api.listInputDevices();
      setDevices(list);
      setDevicesError(
        list.length === 0 ? "No microphone detected. Plug one in and refresh." : null,
      );
    } catch (err) {
      setDevices([]);
      setDevicesError(api.errorMessage(err));
    }
  }, []);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  useEffect(() => {
    if (state !== "recording") {
      if (timer.current !== null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
      return;
    }

    timer.current = window.setInterval(async () => {
      if (startedAt.current !== null) {
        setElapsedMs(Date.now() - startedAt.current);
      }
      try {
        const current = await api.getMicLevel();
        setLevel(current);
        if (current >= CLIPPING_LEVEL) {
          quietSamples.current = 0;
          setInputWarning("Input is clipping — lower the microphone gain.");
        } else if (current < LOW_LEVEL) {
          quietSamples.current += 1;
          if (quietSamples.current >= WARNING_SAMPLES) {
            setInputWarning(
              "Almost no sound is reaching the microphone. Check it is not muted.",
            );
          }
        } else {
          quietSamples.current = 0;
          setInputWarning(null);
        }
      } catch {
        // The recording may have just stopped; a transient poll failure is
        // not worth showing.
      }
      try {
        const err = await api.takeRecordingError();
        // A stream-level error means the device went away mid-meeting.
        if (err) setDeviceError(err);
      } catch {
        // ignore
      }
    }, LEVEL_POLL_MS);

    return () => {
      if (timer.current !== null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
    };
  }, [state]);

  const start = useCallback(
    async (deviceName: string | null, meetingId: string) => {
      setDeviceError(null);
      setInputWarning(null);
      quietSamples.current = 0;
      setState("starting");
      try {
        const summary = await api.startRecording(deviceName, meetingId);
        startedAt.current = Date.now();
        setElapsedMs(0);
        setState("recording");
        return summary;
      } catch (err) {
        setState("idle");
        throw err;
      }
    },
    [],
  );

  const stop = useCallback(async () => {
    setState("stopping");
    try {
      const summary = await api.stopRecording();
      return summary;
    } finally {
      startedAt.current = null;
      setLevel(0);
      setState("idle");
    }
  }, []);

  return {
    state,
    devices,
    devicesError,
    level,
    elapsedMs,
    deviceError,
    inputWarning,
    refreshDevices,
    start,
    stop,
  };
}
