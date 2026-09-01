import { useEffect, useState } from "react";
import * as api from "../../services/api";
import { dateLabel, duration } from "../../services/format";
import { Card, Notice, Spinner, StateBadge } from "../../components/ui";
import type { MeetingMetadata } from "../../types";

interface Props {
  onOpenMeeting: (id: string) => void;
  /** Bumped by the shell after a delete, so the list stops showing a ghost. */
  reloadToken: number;
}

const DAY_MS = 86_400_000;

/** `<input type="date">` gives a local calendar day; the store wants epoch ms. */
function startOfDay(value: string): number | null {
  if (!value) return null;
  const ms = new Date(`${value}T00:00:00`).getTime();
  return Number.isNaN(ms) ? null : ms;
}

function endOfDay(value: string): number | null {
  const start = startOfDay(value);
  return start === null ? null : start + DAY_MS - 1;
}

/** The §25 library: list, title search, date filtering. */
export function HistoryView({ onOpenMeeting, reloadToken }: Props) {
  const [query, setQuery] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [meetings, setMeetings] = useState<MeetingMetadata[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    // Debounced: typing a title should not fire one query per keystroke.
    const handle = window.setTimeout(async () => {
      try {
        const rows = await api.listMeetings(query, startOfDay(from), endOfDay(to));
        if (cancelled) return;
        setMeetings(rows);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setMeetings([]);
        setError(api.errorMessage(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 250);

    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [query, from, to, reloadToken]);

  const filtered = query.trim() !== "" || from !== "" || to !== "";

  return (
    <Card title="Meetings">
      <div className="field-row">
        <label className="field grow">
          <span>Search titles and notes</span>
          <input
            type="search"
            placeholder="Search meetings"
            value={query}
            onChange={(e) => setQuery(e.currentTarget.value)}
          />
        </label>
        <label className="field">
          <span>From</span>
          <input type="date" value={from} onChange={(e) => setFrom(e.currentTarget.value)} />
        </label>
        <label className="field">
          <span>To</span>
          <input type="date" value={to} onChange={(e) => setTo(e.currentTarget.value)} />
        </label>
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      {loading && meetings.length === 0 && !error && (
        <p className="notice">
          <Spinner />
          Loading meetings
        </p>
      )}

      {!loading && !error && meetings.length === 0 && (
        <Notice>
          {filtered
            ? "No meetings match that search."
            : "No meetings yet. Record one from the Record tab."}
        </Notice>
      )}

      {meetings.length > 0 && (
        <ul className="history-list">
          {meetings.map((meeting) => (
            <li key={meeting.id}>
              <button
                type="button"
                className="history-row"
                onClick={() => onOpenMeeting(meeting.id)}
              >
                <span className="history-title">{meeting.title || "Untitled meeting"}</span>
                <span className="history-meta">
                  <StateBadge state={meeting.state} />
                  <span>{dateLabel(meeting.created_at_ms)}</span>
                  <span>{duration(meeting.duration_ms)}</span>
                  {meeting.has_transcript && <span className="marker">Transcript</span>}
                  {meeting.has_mom && <span className="marker">Minutes</span>}
                  {meeting.mom_error && <span className="marker warning">Minutes failed</span>}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
