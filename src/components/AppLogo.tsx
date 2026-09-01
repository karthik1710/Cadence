/**
 * The EchoMind mark.
 *
 * An analog waveform — a continuous swept curve, not the blocky bar-graph
 * every recorder app uses — drawn twice: the solid stroke, plus a faded copy
 * trailing behind it. That trailing copy is the "echo" half of the name, and
 * it's the only decoration here; at 22px in the header, anything more detailed
 * turns to mush.
 *
 * The gradient ids are suffixed per instance because two SVGs on one page with
 * the same `<defs>` id will silently share the first one's definition.
 */
export function AppLogo({
  size = 28,
  id = "app",
}: {
  size?: number;
  /** Unique per rendered instance — see the note on gradient ids above. */
  id?: string;
}) {
  const fill = `echomind-tile-${id}`;
  const wave = `echomind-wave-${id}`;

  return (
    <svg
      className="app-logo"
      width={size}
      height={size}
      viewBox="0 0 48 48"
      role="img"
      aria-label="EchoMind"
    >
      <defs>
        <linearGradient id={fill} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#7c8fff" />
          <stop offset="100%" stopColor="#58d8c9" />
        </linearGradient>
        <linearGradient id={wave} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.85" />
          <stop offset="50%" stopColor="#ffffff" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0.85" />
        </linearGradient>
      </defs>

      <rect x="0" y="0" width="48" height="48" rx="12" fill={`url(#${fill})`} />

      {/* The echo: the same curve, trailing and dimmed. */}
      <path
        d="M9 24 C11 24 12 20 14.5 20 C17 20 17.5 31 20.5 31 C23.5 31 23.5 15 26.5 15 C29.5 15 29.5 30 32.5 30 C35 30 35.5 22 37 22 C38.5 22 38.5 24 39 24"
        fill="none"
        stroke="#ffffff"
        strokeOpacity="0.32"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        transform="translate(2.5 3.5)"
      />

      {/* The signal itself. */}
      <path
        d="M9 24 C11 24 12 20 14.5 20 C17 20 17.5 31 20.5 31 C23.5 31 23.5 15 26.5 15 C29.5 15 29.5 30 32.5 30 C35 30 35.5 22 37 22 C38.5 22 38.5 24 39 24"
        fill="none"
        stroke={`url(#${wave})`}
        strokeWidth="3.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
