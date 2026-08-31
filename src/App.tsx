import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

type MeetingState = "idle" | "starting" | "recording" | "stopping";
type PipelineState = "idle" | "uploading" | "processing" | "completed" | "failed";

interface AudioDeviceInfo {
  name: string;
  is_default: boolean;
}

interface RecordingSummary {
  file_path: string;
  sample_rate: number;
  channels: number;
}

interface MeetingStatus {
  status: string;
  stage: string | null;
  progress: number;
  error: string | null;
}

interface TranscriptSegment {
  start_ms: number;
  end_ms: number;
  text: string;
}

interface TranscriptResponse {
  id: string;
  segments: TranscriptSegment[];
}

const DEFAULT_DEVICE_VALUE = "__default__";
const DEFAULT_BACKEND_URL = "http://10.102.14.137:8090";

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function App() {
  const [state, setState] = useState<MeetingState>("idle");
  const [title, setTitle] = useState("");
  const [notes, setNotes] = useState("");
  const [startedAt, setStartedAt] = useState<Date | null>(null);

  const [devices, setDevices] = useState<AudioDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState(DEFAULT_DEVICE_VALUE);
  const [devicesError, setDevicesError] = useState<string | null>(null);

  const [level, setLevel] = useState(0);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [pipeline, setPipeline] = useState<PipelineState>("idle");
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<MeetingStatus | null>(null);
  const [transcript, setTranscript] = useState<TranscriptSegment[] | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);
  const statusPollRef = useRef<number | null>(null);

  async function refreshDevices() {
    try {
      const list = await invoke<AudioDeviceInfo[]>("list_input_devices");
      setDevices(list);
      setDevicesError(list.length === 0 ? "No microphone detected." : null);
    } catch (err) {
      setDevices([]);
      setDevicesError(String(err));
    }
  }

  useEffect(() => {
    refreshDevices();
  }, []);

  useEffect(() => {
    if (state !== "recording") {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    pollRef.current = window.setInterval(async () => {
      try {
        const l = await invoke<number>("get_mic_level");
        setLevel(l);
      } catch {
        // Recording may have just stopped; ignore transient poll errors.
      }
      try {
        const err = await invoke<string | null>("take_recording_error");
        if (err) setRecordingError(err);
      } catch {
        // ignore
      }
    }, 150);

    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [state]);

  useEffect(() => {
    if (pipeline !== "processing" || !meetingId) {
      if (statusPollRef.current !== null) {
        window.clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
      return;
    }

    statusPollRef.current = window.setInterval(async () => {
      try {
        const s = await invoke<MeetingStatus>("get_meeting_status", { backendUrl, meetingId });
        setPipelineStatus(s);
        if (s.status === "completed") {
          const t = await invoke<TranscriptResponse>("get_transcript", { backendUrl, meetingId });
          setTranscript(t.segments);
          setPipeline("completed");
        } else if (s.status === "failed") {
          setPipelineError(s.error ?? "Processing failed");
          setPipeline("failed");
        }
      } catch (err) {
        setPipelineError(String(err));
        setPipeline("failed");
      }
    }, 3000);

    return () => {
      if (statusPollRef.current !== null) {
        window.clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
    };
  }, [pipeline, meetingId, backendUrl]);

  async function startMeeting() {
    setActionError(null);
    setRecordingError(null);
    setPipeline("idle");
    setPipelineStatus(null);
    setPipelineError(null);
    setTranscript(null);
    setMeetingId(null);
    setState("starting");

    const localMeetingId = crypto.randomUUID();
    const deviceName = selectedDevice === DEFAULT_DEVICE_VALUE ? null : selectedDevice;

    try {
      await invoke("start_recording", { deviceName, meetingId: localMeetingId });
      setStartedAt(new Date());
      setState("recording");
    } catch (err) {
      setActionError(String(err));
      setState("idle");
    }
  }

  async function stopMeeting() {
    setActionError(null);
    setState("stopping");
    try {
      const summary = await invoke<RecordingSummary>("stop_recording");
      setState("idle");
      setLevel(0);
      await uploadAndProcess(summary);
    } catch (err) {
      setActionError(String(err));
      setState("idle");
    }
  }

  async function uploadAndProcess(summary: RecordingSummary) {
    setPipeline("uploading");
    setPipelineError(null);
    try {
      const id = await invoke<string>("upload_and_process", {
        backendUrl,
        filePath: summary.file_path,
        title: title || null,
      });
      setMeetingId(id);
      setPipeline("processing");
    } catch (err) {
      setPipelineError(String(err));
      setPipeline("failed");
    }
  }

  const isRecording = state === "recording";
  const busy = state === "starting" || state === "stopping";
  const pipelineBusy = pipeline === "uploading" || pipeline === "processing";

  return (
    <main className="container">
      <header className="app-header">
        <h1>EchoMind</h1>
        <p className="subtitle">Local-first meeting recorder</p>
      </header>

      <section className="card">
        <div className="field-row">
          <label className="field">
            <span>Microphone</span>
            <div className="inline">
              <select
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.currentTarget.value)}
                disabled={isRecording || busy}
              >
                <option value={DEFAULT_DEVICE_VALUE}>System default microphone</option>
                {devices.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                    {d.is_default ? " (default)" : ""}
                  </option>
                ))}
              </select>
              <button type="button" className="ghost" onClick={refreshDevices} disabled={isRecording || busy}>
                Refresh
              </button>
            </div>
          </label>
          <label className="field">
            <span>Backend</span>
            <input
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.currentTarget.value)}
              disabled={isRecording || busy}
            />
          </label>
        </div>
        {devicesError && <p className="notice error">{devicesError}</p>}
      </section>

      <section className="card">
        <form
          className="field-row"
          onSubmit={(e) => {
            e.preventDefault();
            if (busy) return;
            isRecording ? stopMeeting() : startMeeting();
          }}
        >
          <label className="field grow">
            <span>Meeting title</span>
            <input
              placeholder="Untitled meeting"
              value={title}
              onChange={(e) => setTitle(e.currentTarget.value)}
              disabled={isRecording || busy}
            />
          </label>
          <button type="submit" className={isRecording ? "danger" : "primary"} disabled={busy}>
            {isRecording ? "Stop Meeting" : "Start Meeting"}
          </button>
        </form>

        <label className="field">
          <span>Notes</span>
          <textarea
            placeholder="Optional notes"
            value={notes}
            onChange={(e) => setNotes(e.currentTarget.value)}
          />
        </label>

        {isRecording && (
          <div className="recording-status">
            <span className="rec-dot" />
            {startedAt && <span>Recording since {startedAt.toLocaleTimeString()}</span>}
            <div className="level-meter">
              <div className="level-meter-fill" style={{ width: `${Math.round(level * 100)}%` }} />
            </div>
          </div>
        )}

        {recordingError && <p className="notice error">Microphone error: {recordingError}</p>}
        {actionError && <p className="notice error">{actionError}</p>}
      </section>

      {pipeline !== "idle" && (
        <section className="card">
          <h2>Processing</h2>
          {pipelineBusy && (
            <p className="notice">
              <span className="spinner" />
              {pipeline === "uploading"
                ? "Uploading to backend…"
                : `Processing${pipelineStatus?.stage ? ` — ${pipelineStatus.stage}` : ""}${
                    pipelineStatus?.progress ? ` (${pipelineStatus.progress}%)` : ""
                  }`}
            </p>
          )}
          {pipeline === "failed" && <p className="notice error">{pipelineError}</p>}
          {pipeline === "completed" && transcript && (
            <div className="transcript">
              {transcript.length === 0 && <p className="notice">No speech detected.</p>}
              {transcript.map((seg, i) => (
                <div key={i} className="transcript-segment">
                  <span className="timestamp">{formatTime(seg.start_ms)}</span>
                  <span className="transcript-text">{seg.text}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

export default App;
