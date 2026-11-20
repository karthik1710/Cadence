// Presentation-only helpers. No app logic lives here.

import type { Flag, RedactionMode } from "../types";

/** `hh:mm:ss`, matching the timecodes the exports and the Rust side produce. */
export function timecode(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}

/** Human duration: "30 s", "42 min", "1 h 12 min". */
export function duration(ms: number): string {
  // Decide seconds vs minutes on the seconds value, not on rounded minutes —
  // rounding minutes first labelled a 30 s meeting "1 min".
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds} s`;
  const totalMinutes = Math.round(ms / 60000);
  if (totalMinutes < 60) return `${totalMinutes} min`;
  return `${Math.floor(totalMinutes / 60)} h ${totalMinutes % 60} min`;
}

/** Local date and time — the meeting happened where the user is. */
export function dateLabel(ms: number): string {
  if (!ms) return "Unknown date";
  return new Date(ms).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function sizeLabel(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

/** Highlights every occurrence of `needle`, for transcript search. */
export function splitOnMatch(
  text: string,
  needle: string,
): { text: string; match: boolean }[] {
  const query = needle.trim().toLowerCase();
  if (!query) return [{ text, match: false }];

  const parts: { text: string; match: boolean }[] = [];
  const haystack = text.toLowerCase();
  let index = 0;
  for (;;) {
    const found = haystack.indexOf(query, index);
    if (found === -1) {
      if (index < text.length) parts.push({ text: text.slice(index), match: false });
      return parts;
    }
    if (found > index) parts.push({ text: text.slice(index, found), match: false });
    parts.push({ text: text.slice(found, found + query.length), match: true });
    index = found + query.length;
  }
}

// --- sensitive language ---------------------------------------------------
//
// The one rule that matters in here: `Flag.start_char` and `Flag.end_char`
// are Unicode CODE POINT offsets, because that is what the Rust scanner
// counts in. JavaScript strings are UTF-16, so `"…".slice(a, b)` disagrees
// with those offsets the moment the turn contains a character outside the
// BMP — and disagrees about *width* for anything the Rust side counted as
// one `char`. These meetings are English/Tamil code-switched and people
// paste emoji into meeting notes, so every slice below goes through
// `Array.from`, which iterates code points.

/** Reads the stored setting safely. Unknown values fall back to `flag`,
 * matching the Rust side's `RedactionMode::parse`. */
export function parseRedactionMode(value: string | null | undefined): RedactionMode {
  switch (value) {
    case "off":
    case "flag":
    case "mask":
    case "redact":
      return value;
    default:
      return "flag";
  }
}

/** One run of a turn's text, already transformed for display. */
export interface TextPiece {
  /** What to render. Never contains anything `mode` says to hide. */
  text: string;
  /** The true text this piece covers. Safe to *search*, not to show. */
  source: string;
  /** The flag this run belongs to; null for ordinary text. */
  flag: Flag | null;
  /** True when `text` differs from `source` — the caller must not highlight
   * a search hit inside it, because the offsets no longer line up and the
   * hit itself is the thing being hidden. */
  hidden: boolean;
}

/** Keeps the first character and replaces the rest with `*`, preserving the
 * span's length in code points so the line's shape survives. */
function maskSpan(span: string): string {
  const chars = Array.from(span);
  if (chars.length === 0) return span;
  return chars[0] + "*".repeat(chars.length - 1);
}

/** Applies a redaction mode to a single flagged span. */
export function applyRedactionToSpan(span: string, mode: RedactionMode): string {
  switch (mode) {
    case "mask":
      return maskSpan(span);
    case "redact":
      return "[expunged]";
    // `off` shows the text with no marking at all; `flag` shows the same
    // text and leaves the marking to the UI.
    default:
      return span;
  }
}

/**
 * Splits a turn's text into runs at its flag boundaries, applying `mode` to
 * the flagged ones.
 *
 * Flags are sorted and clamped, and any that overlaps one already emitted is
 * dropped rather than double-counted — a garbled flag must never be able to
 * drop text out of the transcript or duplicate it.
 */
export function redactPieces(
  text: string,
  flags: Flag[] | undefined,
  mode: RedactionMode,
): TextPiece[] {
  const plain = [{ text, source: text, flag: null, hidden: false }];
  if (mode === "off" || !flags || flags.length === 0) return plain;

  const chars = Array.from(text);
  const sorted = [...flags].sort(
    (a, b) => a.start_char - b.start_char || a.end_char - b.end_char,
  );

  const pieces: TextPiece[] = [];
  let cursor = 0;
  for (const flag of sorted) {
    const start = Math.max(cursor, Math.min(flag.start_char, chars.length));
    const end = Math.max(start, Math.min(flag.end_char, chars.length));
    if (end <= start) continue;
    if (start > cursor) {
      const before = chars.slice(cursor, start).join("");
      pieces.push({ text: before, source: before, flag: null, hidden: false });
    }
    const source = chars.slice(start, end).join("");
    const shown = applyRedactionToSpan(source, mode);
    pieces.push({ text: shown, source, flag, hidden: shown !== source });
    cursor = end;
  }
  if (cursor < chars.length) {
    const rest = chars.slice(cursor).join("");
    pieces.push({ text: rest, source: rest, flag: null, hidden: false });
  }
  return pieces.length > 0 ? pieces : plain;
}

/** The whole turn as it should be shown or copied. */
export function redactText(
  text: string,
  flags: Flag[] | undefined,
  mode: RedactionMode,
): string {
  return redactPieces(text, flags, mode)
    .map((piece) => piece.text)
    .join("");
}

/** Plain-language description of one flag, for a `title` tooltip. */
export function flagDescription(flag: Flag): string {
  const category: Record<string, string> = {
    profanity: "profanity",
    slur: "a slur",
    sexual: "sexual language",
    violent: "violent language",
    insult: "an insult",
  };
  const method: Record<string, string> = {
    exact: "matched exactly",
    spaced: "matched through spacing",
    fuzzy: "matched as a probable misspelling",
    phonetic: "matched by how it sounds",
  };
  const what = category[flag.category] ?? flag.category;
  const how = method[flag.method] ?? flag.method;
  return `Flagged as ${flag.severity} ${what} — ${how}.`;
}

/** `0.82` → `82%`. Confidences and shares arrive as 0–1 fractions. */
export function percent(fraction: number): string {
  if (!Number.isFinite(fraction)) return "—";
  return `${Math.round(fraction * 100)}%`;
}

/** Decibel values, always signed and to one place: `-20.4 dB`. */
export function decibels(value: number, unit = "dBFS"): string {
  if (!Number.isFinite(value)) return "—";
  return `${value.toFixed(1)} ${unit}`;
}
