import { useEffect, useRef, useState } from "react";
import * as api from "../services/api";
import type { MeetingStatus } from "../types";

const POLL_MS = 1000;

/**
 * Polls the local pipeline's status for one meeting (§12) and stops as soon as
 * it settles. `onSettled` fires once per completed or failed run, which is
 * where the caller reloads the transcript and minutes.
 */
export function useProcessingStatus(
  meetingId: string | null,
  /** Bump to restart polling after kicking off a new run on the same meeting. */
  restartToken: number,
  onSettled?: (status: MeetingStatus) => void,
) {
  const [status, setStatus] = useState<MeetingStatus | null>(null);
  const settled = useRef(false);
  const callback = useRef(onSettled);
  callback.current = onSettled;

  useEffect(() => {
    settled.current = false;
    setStatus(null);
    if (!meetingId) return;

    let cancelled = false;
    let handle: number | null = null;

    const poll = async () => {
      try {
        const next = await api.getProcessingStatus(meetingId);
        if (cancelled) return;
        setStatus(next);
        if (next.status === "completed" || next.status === "failed") {
          if (!settled.current) {
            settled.current = true;
            callback.current?.(next);
          }
          if (handle !== null) {
            window.clearInterval(handle);
            handle = null;
          }
        }
      } catch {
        // Keep polling: a transient IPC failure should not look like a
        // processing failure.
      }
    };

    poll();
    handle = window.setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      if (handle !== null) window.clearInterval(handle);
    };
  }, [meetingId, restartToken]);

  return status;
}
