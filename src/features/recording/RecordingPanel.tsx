import { useState } from "react";
import * as api from "../../services/api";
import { timecode } from "../../services/format";
import { useRecorder } from "../../hooks/useRecorder";
import { useProcessingStatus } from "../../hooks/useProcessingStatus";
import { Card, LevelMeter, Notice, Spinner, StageChecklist } from "../../components/ui";
import { RecommendedSetupBanner } from "../../components/RecommendedSetup";
import type { ModelStatus } from "../../types";

interface Props {
  onOpenMeeting: (id: string) => void;
  setupReady: boolean;
  missingModels: ModelStatus[];
  onSetupChanged: () => void;
  onOpenSettings: () => void;
}

/** Sentinel for "whatever Windows considers default", which Rust spells `null`. */
const DEFAULT_DEVICE = "__default__";

/**
 * The §22 main screen: start a meeting, watch it record, watch it process.
 *
 * The meeting row is created *before* the first sample is captured, so the
 * audio always has a folder to land in and no failure after that point can
 * leave a recording without a meeting to recover it from (§33).
 */
export function RecordingPanel({
  onOpenMeeting,
  setupReady,
  missingModels,
  onSetupChanged,
  onOpenSettings,
}: Props) {
  const recorder = useRecorder();
  const [title, setTitle] = useState("");
  const [notes, setNotes] = useState("");
  const [selectedDevice, setSelectedDevice] = useState(DEFAULT_DEVICE);
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [runToken, setRunToken] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [processError, setProcessError] = useState<string | null>(null);

  const status = useProcessingStatus(activeId, runToken, (final) => {
    // Fires once per run. A failure here belongs to the pipeline, not the UI:
    // the recording is already on disk, so this is recoverable, not lost work.
    if (final.status === "failed") {
      setProcessError(final.error ?? "Processing failed.");
    }
  });

  const deviceName = selectedDevice === DEFAULT_DEVICE ? null : selectedDevice;
  const recording = recorder.state === "recording";
  const busy = recorder.state === "starting" || recorder.state === "stopping";
  const processing = status?.status === "processing";
  const completed = status?.status === "completed";
  const failed = status?.status === "failed" || processError !== null;

  const startMeeting = async () => {
    setError(null);
    setProcessError(null);
    setActiveId(null);
    try {
      const meeting = await api.createMeeting(title, notes);
      setRecordingId(meeting.id);
      await recorder.start(deviceName, meeting.id);
    } catch (err) {
      setError(api.errorMessage(err));
    }
  };

  const stopMeeting = async () => {
    setError(null);
    const id = recordingId;
    try {
      const summary = await recorder.stop();
      if (!id) {
        setError(
          "The recording stopped, but the meeting it belonged to could not be found. " +
            "The audio is still in the meetings folder — Settings shows where that is.",
        );
        return;
      }
      await api.finalizeRecording(id, deviceName, summary.sample_rate, summary.channels);
      await api.startProcessing(id);
      setRecordingId(null);
      setActiveId(id);
      setRunToken((n) => n + 1);
    } catch (err) {
      setError(api.errorMessage(err));
    }
  };

  const retry = async () => {
    if (!activeId) return;
    setProcessError(null);
    try {
      await api.startProcessing(activeId);
      setRunToken((n) => n + 1);
    } catch (err) {
      setProcessError(api.errorMessage(err));
    }
  };

  return (
    <div className="recording-panel">
      {!setupReady && (
        <RecommendedSetupBanner
          missing={missingModels}
          onChanged={onSetupChanged}
          onOpenSettings={onOpenSettings}
        />
      )}

      <Card title="New meeting">
        <div className="field-row">
          <label className="field grow">
            <span>Meeting title</span>
            <input
              placeholder="Untitled meeting"
              value={title}
              onChange={(e) => setTitle(e.currentTarget.value)}
              disabled={recording || busy}
            />
          </label>
          <label className="field">
            <span>Microphone</span>
            <div className="inline">
              <select
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.currentTarget.value)}
                disabled={recording || busy}
                aria-label="Microphone"
              >
                <option value={DEFAULT_DEVICE}>System default microphone</option>
                {recorder.devices.map((device) => (
                  <option key={device.name} value={device.name}>
                    {device.name}
                    {device.is_default ? " (default)" : ""}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="ghost"
                onClick={() => recorder.refreshDevices()}
                disabled={recording || busy}
              >
                Refresh
              </button>
            </div>
          </label>
        </div>

        <label className="field">
          <span>Notes</span>
          <textarea
            placeholder="Optional — agenda, attendees, anything worth remembering"
            value={notes}
            onChange={(e) => setNotes(e.currentTarget.value)}
            disabled={recording || busy}
          />
        </label>

        <div className="recording-status">
          {recording ? (
            <>
              <span className="rec-dot" />
              <span className="badge error">Recording</span>
            </>
          ) : (
            <span className="badge muted">{busy ? "Working" : "Ready"}</span>
          )}
          <span className="timer">{timecode(recorder.elapsedMs)}</span>
          {recording && <LevelMeter level={recorder.level} />}
        </div>

        <div className="button-row">
          <button
            type="button"
            className={recording ? "danger" : "primary"}
            disabled={busy || (!recording && !setupReady)}
            onClick={recording ? stopMeeting : startMeeting}
          >
            {recording ? "Stop Meeting" : "Start Meeting"}
          </button>
        </div>

        {recorder.devicesError && <Notice kind="error">{recorder.devicesError}</Notice>}
        {recorder.deviceError && (
          <Notice kind="error">
            The microphone was disconnected during the meeting: {recorder.deviceError}{" "}
            Everything captured before that point has been kept.
          </Notice>
        )}
        {recorder.inputWarning && <Notice kind="warning">{recorder.inputWarning}</Notice>}
        {error && <Notice kind="error">{error}</Notice>}
      </Card>

      {activeId && (
        <Card title="Processing on this machine">
          {processing && (
            <p className="notice">
              <Spinner />
              {status?.stage ?? "Starting"} — {status?.progress ?? 0}%
            </p>
          )}
          <StageChecklist stage={status?.stage ?? null} failed={failed} />

          {status?.warning && <Notice kind="warning">{status.warning}</Notice>}

          {completed && (
            <div className="button-row">
              <button
                type="button"
                className="primary"
                onClick={() => onOpenMeeting(activeId)}
              >
                View transcript and minutes
              </button>
            </div>
          )}

          {failed && (
            <>
              <Notice kind="error">
                {processError ?? status?.error ?? "Processing failed."} The recording
                itself is safe — retry once the cause is fixed.
              </Notice>
              <div className="button-row">
                <button type="button" className="ghost" onClick={retry}>
                  Retry processing
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => onOpenMeeting(activeId)}
                >
                  Open this meeting
                </button>
              </div>
            </>
          )}

          {processing && (
            <p className="muted">
              Nothing leaves this computer. You can switch tabs; processing continues.
            </p>
          )}
        </Card>
      )}

      <p className="muted footer-note">Local processing · No cloud</p>
    </div>
  );
}
