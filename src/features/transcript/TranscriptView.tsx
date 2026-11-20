import { useMemo, useState } from "react";
import {
  flagDescription,
  redactPieces,
  redactText,
  splitOnMatch,
  timecode,
} from "../../services/format";
import type {
  RedactionMode,
  RedactionSummary,
  Transcript,
  TranscriptSegment,
} from "../../types";
import { Notice } from "../../components/ui";

interface Props {
  transcript: Transcript;
  /** Stable speaker id → user-assigned name (presentation only). */
  speakerNames: Record<string, string>;
  onSeek: (ms: number) => void;
  onRenameSpeaker: (speaker: string, name: string) => void;
  /** How flagged spans are shown. Comes from Settings; `flag` is the
   * review mode, `mask`/`redact` change what is displayed and copied. */
  redactionMode?: RedactionMode;
  /** Counts from the last scan, when there has been one. */
  redactionSummary?: RedactionSummary | null;
  /** Offered only when the caller can run the scan again. */
  onRescanRedactions?: () => void;
  rescanning?: boolean;
}

export function TranscriptView({
  transcript,
  speakerNames,
  onSeek,
  onRenameSpeaker,
  redactionMode = "flag",
  redactionSummary = null,
  onRescanRedactions,
  rescanning = false,
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

  // Search runs over the TRUE text, never the displayed form: a flagged word
  // has to stay findable, or masking would quietly delete evidence from the
  // only tool anyone has for finding it. What comes back from the search is
  // still rendered through the redaction mode below, so a hit inside a
  // hidden span reveals the turn without revealing the word.
  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return transcript.segments.filter((segment) => {
      if (speakerFilter && segment.speaker !== speakerFilter) return false;
      if (!needle) return true;
      return segment.text.toLowerCase().includes(needle);
    });
  }, [transcript, query, speakerFilter]);

  const flaggedTurns = useMemo(
    () => transcript.segments.filter((s) => (s.flags?.length ?? 0) > 0).length,
    [transcript],
  );
  const flaggedSpans = useMemo(
    () => transcript.segments.reduce((n, s) => n + (s.flags?.length ?? 0), 0),
    [transcript],
  );

  const copyAll = async () => {
    // Copy what is on screen, not what is underneath it. Copying the true
    // text out of a masked transcript would defeat the mode entirely.
    const text = matches
      .map(
        (s) =>
          `[${timecode(s.start_ms)}] ${displayName(s)}\n${redactText(
            s.text,
            s.flags,
            redactionMode,
          )}`,
      )
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

      <RedactionBanner
        mode={redactionMode}
        flaggedTurns={flaggedTurns}
        flaggedSpans={flaggedSpans}
        totalTurns={transcript.segments.length}
        summary={redactionSummary}
        onRescan={onRescanRedactions}
        rescanning={rescanning}
      />

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
            {/* Only present when transcribed in code-switched mode — it
                shows the pipeline noticed the switch, without cluttering a
                transcript people read end to end. */}
            {segment.language && (
              <span className="lang-tag" title={`Detected language: ${segment.language}`}>
                {segment.language}
              </span>
            )}
            <SegmentText segment={segment} query={query} mode={redactionMode} />
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * One turn's words.
 *
 * The turn is cut into runs at its flag boundaries (in code points — see
 * `redactPieces`), and each run is rendered according to whether it is
 * hidden:
 *
 * - a visible run gets the ordinary yellow search `<mark>`;
 * - a hidden run (`mask`/`redact`) is rendered as-transformed and is never
 *   search-highlighted, because the offsets no longer line up with the true
 *   text and the match is the very thing being hidden;
 * - in `flag` mode nothing is hidden, so the true words show with a distinct
 *   underline treatment that a search `<mark>` can sit inside.
 */
function SegmentText({
  segment,
  query,
  mode,
}: {
  segment: TranscriptSegment;
  query: string;
  mode: RedactionMode;
}) {
  const pieces = redactPieces(segment.text, segment.flags, mode);
  const needle = query.trim().toLowerCase();
  // A search hit that landed entirely inside a hidden span: say so, rather
  // than showing a turn with no visible reason to be in the results.
  const hiddenOnlyMatch =
    needle.length > 0 &&
    segment.text.toLowerCase().includes(needle) &&
    !pieces.some((piece) => !piece.hidden && piece.text.toLowerCase().includes(needle));

  return (
    <span className="transcript-text">
      {pieces.map((piece, index) => {
        if (piece.hidden) {
          return (
            <span
              key={index}
              className={`redacted ${mode}`}
              title={piece.flag ? flagDescription(piece.flag) : undefined}
            >
              {piece.text}
            </span>
          );
        }
        const inner = splitOnMatch(piece.text, query).map((part, i) =>
          part.match ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>,
        );
        if (!piece.flag) return <span key={index}>{inner}</span>;
        return (
          <span
            key={index}
            className={`flagged sev-${piece.flag.severity}`}
            title={flagDescription(piece.flag)}
          >
            {inner}
          </span>
        );
      })}
      {hiddenOnlyMatch && (
        <span className="hidden-match" title="The words that matched are hidden by the current redaction mode.">
          match hidden
        </span>
      )}
    </span>
  );
}

/** What was flagged, what the current mode does about it, and the one thing
 * a masked transcript must not be allowed to imply. */
function RedactionBanner({
  mode,
  flaggedTurns,
  flaggedSpans,
  totalTurns,
  summary,
  onRescan,
  rescanning,
}: {
  mode: RedactionMode;
  flaggedTurns: number;
  flaggedSpans: number;
  totalTurns: number;
  summary: RedactionSummary | null;
  onRescan?: () => void;
  rescanning: boolean;
}) {
  if (flaggedSpans === 0) {
    // Nothing to hide. Worth one quiet line only if a scan actually ran, so
    // "nothing was found" is distinguishable from "nothing was looked for".
    if (!summary && !onRescan) return null;
    return (
      <div className="redaction-banner quiet">
        <div className="redaction-banner-head">
          <span className="muted">
            {summary
              ? `No sensitive language was flagged across ${totalTurns} turns.`
              : "This transcript has not been scanned for sensitive language."}
          </span>
          {onRescan && (
            <button type="button" className="ghost" onClick={onRescan} disabled={rescanning}>
              {rescanning ? "Scanning…" : "Scan now"}
            </button>
          )}
        </div>
      </div>
    );
  }

  const modeLine: Record<RedactionMode, string> = {
    off: "Redaction is off, so the words are shown exactly as transcribed.",
    flag: "Flagged for review — the words are shown, marked in place.",
    mask: "Masked here and in anything exported or copied from this screen.",
    redact: "Replaced with [expunged] here and in anything exported or copied from this screen.",
  };

  return (
    <div className="redaction-banner">
      <div className="redaction-banner-head">
        <span className="badge warn">
          {flaggedSpans} flagged {flaggedSpans === 1 ? "word" : "words"}
        </span>
        <span className="muted">
          in {flaggedTurns} of {totalTurns} turns
        </span>
        {summary?.fuzzy && (
          <span className="marker" title="Misspellings and spaced-out spellings were matched too.">
            fuzzy matching
          </span>
        )}
        {onRescan && (
          <button type="button" className="ghost" onClick={onRescan} disabled={rescanning}>
            {rescanning ? "Scanning…" : "Re-scan"}
          </button>
        )}
      </div>
      <p className="muted">{modeLine[mode]}</p>
      <p className="muted">
        This changes the text only. <code>original.wav</code> still contains
        everything that was said, and the player above will play it back in
        full. Nothing here has been removed from the recording.
      </p>
    </div>
  );
}
