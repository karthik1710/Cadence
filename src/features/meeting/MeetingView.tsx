import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../../services/api";
import { dateLabel, duration } from "../../services/format";
import { useProcessingStatus } from "../../hooks/useProcessingStatus";
import { AudioPlayer, type AudioPlayerHandle } from "../../components/AudioPlayer";
import { Card, Notice, Spinner, StageChecklist, StateBadge } from "../../components/ui";
import { TranscriptView } from "../transcript/TranscriptView";
import { MomView } from "../mom/MomView";
import type { MeetingSummary, Mom, Transcript } from "../../types";

interface Props {
  meetingId: string;
  onBack: () => void;
  onDeleted: () => void;
}

type Tab = "transcript" | "minutes";

export function MeetingView({ meetingId, onBack, onDeleted }: Props) {
  const [meeting, setMeeting] = useState<MeetingSummary | null>(null);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [mom, setMom] = useState<Mom | null>(null);
  const [tab, setTab] = useState<Tab>("transcript");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [exported, setExported] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [runToken, setRunToken] = useState(0);
  const player = useRef<AudioPlayerHandle | null>(null);

  const load = useCallback(async () => {
    try {
      const summary = await api.getMeeting(meetingId);
      setMeeting(summary);
      setTranscript(
        summary.has_transcript ? await api.getTranscript(meetingId).catch(() => null) : null,
      );
      const minutes = summary.has_mom ? await api.getMom(meetingId).catch(() => null) : null;
      setMom(minutes);
      if (minutes) setTab("minutes");
    } catch (err) {
      setError(api.errorMessage(err));
    }
  }, [meetingId]);

  useEffect(() => {
    load();
  }, [load]);

  const status = useProcessingStatus(meetingId, runToken, () => {
    // Processing settled — pick up whatever it produced.
    load();
  });

  const processing = status?.status === "processing";

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setExported(null);
    try {
      await action();
    } catch (err) {
      setError(api.errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const seek = (ms: number) => player.current?.seekTo(ms);

  const renames: Record<string, string> = {};
  if (meeting && transcript) {
    for (const segment of transcript.segments) {
      const name = meeting.speaker_names[segment.speaker];
      if (name) renames[segment.speaker_label] = name;
    }
  }

  if (!meeting) {
    return (
      <Card>
        {error ? <Notice kind="error">{error}</Notice> : <Spinner />}
        <button type="button" className="ghost" onClick={onBack}>
          Back
        </button>
      </Card>
    );
  }

  return (
    <div className="meeting-view">
      <Card
        action={
          <button type="button" className="ghost" onClick={onBack}>
            ← All meetings
          </button>
        }
      >
        <div className="meeting-header">
          {editingTitle ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const value = new FormData(e.currentTarget).get("title");
                setEditingTitle(false);
                run(async () => {
                  const updated = await api.updateMeeting(
                    meetingId,
                    String(value ?? ""),
                    null,
                  );
                  setMeeting({ ...meeting, ...updated });
                });
              }}
            >
              <input name="title" defaultValue={meeting.title} autoFocus aria-label="Meeting title" />
              <button type="submit" className="ghost">
                Save
              </button>
            </form>
          ) : (
            <h1>
              {meeting.title || "Untitled meeting"}{" "}
              <button
                type="button"
                className="link"
                onClick={() => setEditingTitle(true)}
                title="Rename this meeting"
              >
                edit
              </button>
            </h1>
          )}
          <div className="meeting-meta">
            <StateBadge state={meeting.state} />
            <span>{dateLabel(meeting.created_at_ms)}</span>
            <span>{duration(meeting.duration_ms)}</span>
            {meeting.device_name && <span>{meeting.device_name}</span>}
          </div>
        </div>

        <AudioPlayer ref={player} path={meeting.audio_path} />

        <div className="button-row">
          <button
            type="button"
            className="ghost"
            disabled={busy || processing}
            onClick={() =>
              run(async () => {
                await api.startProcessing(meetingId);
                setRunToken((n) => n + 1);
              })
            }
          >
            {meeting.has_transcript ? "Re-run processing" : "Process this meeting"}
          </button>
          <button
            type="button"
            className="ghost"
            disabled={busy || processing || !meeting.has_transcript}
            onClick={() =>
              run(async () => {
                await api.regenerateMom(meetingId);
                setRunToken((n) => n + 1);
              })
            }
          >
            {meeting.has_mom ? "Regenerate minutes" : "Generate minutes"}
          </button>
          <button
            type="button"
            className="ghost"
            disabled={busy || (!meeting.has_transcript && !meeting.has_mom)}
            onClick={() =>
              run(async () => setExported(await api.exportMeeting(meetingId, "markdown")))
            }
          >
            Export Markdown
          </button>
          <button
            type="button"
            className="ghost"
            disabled={busy || (!meeting.has_transcript && !meeting.has_mom)}
            onClick={() =>
              run(async () => setExported(await api.exportMeeting(meetingId, "json")))
            }
          >
            Export JSON
          </button>
          <button
            type="button"
            className="danger ghost"
            disabled={busy || processing}
            onClick={() => {
              if (!window.confirm("Delete this meeting, its recording and everything derived from it? This cannot be undone.")) {
                return;
              }
              run(async () => {
                await api.deleteMeeting(meetingId);
                onDeleted();
              });
            }}
          >
            Delete
          </button>
        </div>

        {exported && (
          <Notice kind="success">
            Exported to <code>{exported}</code>
          </Notice>
        )}
        {error && <Notice kind="error">{error}</Notice>}
        {meeting.error && meeting.state === "FAILED" && (
          <Notice kind="error">{meeting.error}</Notice>
        )}
        {meeting.mom_error && (
          <Notice kind="warning">
            The transcript is complete, but the minutes could not be generated:{" "}
            {meeting.mom_error}
          </Notice>
        )}
      </Card>

      {processing && (
        <Card title="Processing on this machine">
          <p className="notice">
            <Spinner />
            {status?.stage ?? "Working"} — {status?.progress ?? 0}%
          </p>
          <StageChecklist stage={status?.stage ?? null} />
          <p className="muted">
            Nothing leaves this computer. You can close this view; processing continues.
          </p>
        </Card>
      )}

      {(transcript || mom) && (
        <Card>
          <div className="tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "minutes"}
              className={tab === "minutes" ? "active" : ""}
              onClick={() => setTab("minutes")}
              disabled={!mom}
            >
              Minutes
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "transcript"}
              className={tab === "transcript" ? "active" : ""}
              onClick={() => setTab("transcript")}
              disabled={!transcript}
            >
              Transcript
            </button>
          </div>

          {tab === "minutes" && mom && (
            <MomView mom={mom} renames={renames} onSeek={seek} />
          )}
          {tab === "transcript" && transcript && (
            <TranscriptView
              transcript={transcript}
              speakerNames={meeting.speaker_names}
              onSeek={seek}
              onRenameSpeaker={(speaker, name) =>
                run(async () => {
                  const updated = await api.renameSpeaker(meetingId, speaker, name);
                  setMeeting({ ...meeting, ...updated });
                })
              }
            />
          )}
        </Card>
      )}

      {!transcript && !mom && !processing && (
        <Card>
          <Notice>
            This meeting has a recording but has not been processed yet. Processing runs
            entirely on this machine and takes roughly a quarter of the meeting's length.
          </Notice>
        </Card>
      )}
    </div>
  );
}
