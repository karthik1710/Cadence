// Presentation-only helpers. No app logic lives here.

/** `hh:mm:ss`, matching the timecodes the exports and the Rust side produce. */
export function timecode(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}

/** Human duration: "42 min", "1 h 12 min". */
export function duration(ms: number): string {
  const totalMinutes = Math.round(ms / 60000);
  if (totalMinutes < 1) return `${Math.max(0, Math.round(ms / 1000))} s`;
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

/** Highlights every occurrence of `needle`, for transcript search (§23). */
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
