import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { timecode } from "../services/format";

export interface AudioPlayerHandle {
  /** Seeks to a transcript timestamp and plays from there. */
  seekTo: (ms: number) => void;
}

/**
 * Plays the meeting's own recording, straight off disk.
 *
 * The file is served through Tauri's asset protocol rather than read into a
 * blob: a 60-minute recording is hundreds of megabytes, and loading all of it
 * into the webview to play a 10-second span would be absurd. The asset scope
 * in `tauri.conf.json` limits that protocol to the app's own data directory.
 */
export const AudioPlayer = forwardRef<AudioPlayerHandle, { path: string | null }>(
  function AudioPlayer({ path }, ref) {
    const audio = useRef<HTMLAudioElement | null>(null);
    const [position, setPosition] = useState(0);
    const [error, setError] = useState<string | null>(null);

    useImperativeHandle(ref, () => ({
      seekTo(ms: number) {
        const element = audio.current;
        if (!element) return;
        element.currentTime = Math.max(0, ms / 1000);
        void element.play().catch(() => {
          // Autoplay refusal is not worth an error banner; the user can press
          // play themselves.
        });
      },
    }));

    if (!path) {
      return <p className="muted">The recording for this meeting is no longer on disk.</p>;
    }

    return (
      <div className="audio-player">
        <audio
          ref={audio}
          controls
          preload="metadata"
          src={convertFileSrc(path)}
          onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime * 1000)}
          onError={() =>
            setError("The recording could not be played. It may have been moved or deleted.")
          }
        />
        <span className="muted">{timecode(position)}</span>
        {error && <span className="notice error">{error}</span>}
      </div>
    );
  },
);
