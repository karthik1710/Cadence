import { useMemo, useState } from "react";
import { splitOnMatch, timecode } from "../../services/format";
import type { Transcript, TranscriptSegment } from "../../types";
import { Notice } from "../../components/ui";

interface Props {
  transcript: Transcript;
  /** Stable speaker id → user-assigned name (§23, presentation only). */
  speakerNames: Record<string, string>;
  onSeek: (ms: number) => void;
  onRenameSpeaker: (speaker: string, name: string) => void;
}

export function TranscriptView({
  transcript,
  speakerNames,
  onSeek,
  onRenameSpeaker,
}: Props) {
  const [query, setQuery] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState<string>("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [copied, setCopied] = useState(false);

  const speakers = useMemo(() => {
    const seen = new Map<string, string>();
    for (const segment of transcript.segments) {
      if (!seen.has(segment.speaker)) seen.set(segment.speaker, segment.speaker_label);
    }
    return [...seen.entries()].map(([speaker, label]) => ({ speaker, label }));
  }, [transcript]);

  const displayName = (segment: TranscriptSegment) =>
    speakerNames[segment.speaker] || segment.speaker_label;

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return transcript.segments.filter((segment) => {
      if (speakerFilter && segment.speaker !== speakerFilter) return false;
      if (!needle) return true;
      return segment.text.toLowerCase().includes(needle);
    });
  }, [transcript, query, speakerFilter]);

  const copyAll = async () => {
    const text = matches
      .map((s) => `[${timecode(s.start_ms)}] ${displayName(s)}\n${s.text}`)
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  if (transcript.segments.length === 0) {
    return <Notice>No speech was detected in this recording.</Notice>;
  }

  return (
    <div className="transcript-view">
      <div className="toolbar">
        <input
          type="search"
          placeholder="Search this transcript"
          value={query}
          onChange={(e) => setQuery(e.currentTarget.value)}
          aria-label="Search this transcript"
        />
        <select
          value={speakerFilter}
          onChange={(e) => setSpeakerFilter(e.currentTarget.value)}
          aria-label="Filter by speaker"
        >
          <option value="">All speakers</option>
          {speakers.map(({ speaker, label }) => (
            <option key={speaker} value={speaker}>
              {speakerNames[speaker] || label}
            </option>
          ))}
        </select>
        <button type="button" className="ghost" onClick={copyAll}>
          {copied ? "Copied" : "Copy"}
        </button>
        <span className="muted">
          {matches.length} of {transcript.segments.length} turns
        </span>
      </div>

      <div className="speaker-chips">
        {speakers.map(({ speaker, label }) => (
          <span key={speaker} className="chip">
            {renaming === speaker ? (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  onRenameSpeaker(speaker, draftName);
                  setRenaming(null);
                }}
              >
                <input
                  autoFocus
                  value={draftName}
                  placeholder={label}
                  onChange={(e) => setDraftName(e.currentTarget.value)}
                  onBlur={() => setRenaming(null)}
                  aria-label={`New name for ${label}`}
                />
              </form>
            ) : (
              <button
                type="button"
                className="link"
                title="Rename this speaker (display only — the diarization data is unchanged)"
                onClick={() => {
                  setRenaming(speaker);
                  setDraftName(speakerNames[speaker] || "");
                }}
              >
                {speakerNames[speaker] || label}
                {speakerNames[speaker] ? ` (${label})` : ""}
              </button>
            )}
          </span>
        ))}
      </div>

      {matches.length === 0 && <Notice>Nothing in this transcript matches that search.</Notice>}

      <div className="transcript">
        {matches.map((segment, index) => (
          <div key={`${segment.start_ms}-${index}`} className="transcript-segment">
            <button
              type="button"
              className="timestamp link"
              onClick={() => onSeek(segment.start_ms)}
              title="Play from here"
            >
              {timecode(segment.start_ms)}
            </button>
            <span className="speaker">{displayName(segment)}</span>
            <span className="transcript-text">
              {splitOnMatch(segment.text, query).map((part, i) =>
                part.match ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>,
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
