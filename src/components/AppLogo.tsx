/**
 * The Cadence mark.
 *
 * An analog waveform — a continuous swept curve, not the blocky bar-graph
 * every recorder app uses — drawn twice: the solid stroke, plus a faded copy
 * trailing behind it. That trailing copy is the "echo" half of the name.
 *
 * This renders `public/logo.png`, which is the single raster every surface
 * shares: this header mark, the browser/webview favicon (index.html), the
 * README, and — via `tauri icon` — every file in `src-tauri/icons/`. It used
 * to be a hand-inlined <svg> here, which meant the mark existed in two places
 * that could (and did) drift apart. One asset now, so they cannot.
 *
 * `public/logo.svg` remains the vector master. If you change it, regenerate
 * both the raster and the app icons so nothing falls out of step:
 *
 *   npx tauri icon public/logo.svg -o <tmp> --png 1024
 *   cp <tmp>/1024x1024.png public/logo.png
 *   npx tauri icon public/logo.png       # rewrites src-tauri/icons/
 */
export function AppLogo({ size = 28 }: { size?: number }) {
  return (
    <img
      className="app-logo"
      src="/logo.png"
      width={size}
      height={size}
      alt="Cadence"
      // The mark is decorative next to the "Cadence" wordmark it sits beside;
      // dragging it out of the header is never what anyone means to do.
      draggable={false}
    />
  );
}
