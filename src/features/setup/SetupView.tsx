import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../../services/api";
import { sizeLabel } from "../../services/format";
import { Card, Notice, Spinner } from "../../components/ui";
import type { InstallProgress, ModelStatus, SetupStatus } from "../../types";
import { RetiredModelsNotice } from "./RetiredModelsNotice";

interface Props {
  status: SetupStatus;
  /** Re-runs `check_models`; the shell leaves this screen once it is ready. */
  onRecheck: () => Promise<void>;
  /** Into the app without the models. Recording never waits on setup. */
  onSkip: () => void;
}

const POLL_MS = 700;

const PHASE_LABEL: Record<InstallProgress["phase"], string> = {
  queued: "Waiting",
  downloading: "Downloading",
  extracting: "Unpacking",
  verifying: "Checking",
  done: "Ready",
  failed: "Failed",
};

const isTerminal = (p: InstallProgress | undefined) =>
  p !== undefined && (p.phase === "done" || p.phase === "failed");

/** How far one item is, 0–1. Unpacking and checking come after the bytes are
 * all here, so they count as downloaded for the overall bar. */
function fraction(p: InstallProgress | undefined): number {
  if (!p) return 0;
  switch (p.phase) {
    case "downloading":
      return Math.min(1, Math.max(0, p.percent / 100));
    case "extracting":
    case "verifying":
    case "done":
      return 1;
    default:
      return 0;
  }
}

/**
 * First-run setup: the one place models are downloaded in bulk.
 *
 * The full list is shown, and nothing starts until the user clicks. That is
 * deliberate: an earlier one-click batch installer was removed because
 * silent, back-to-back downloads of executables and model files looked like
 * dropper behaviour to endpoint-security tools. Explicit consent over a
 * visible list is the fix, so do not make this start on its own.
 */
export function SetupView({ status, onRecheck, onSkip }: Props) {
  // Frozen when a run starts, so rows don't vanish from under the user as
  // each model lands and `status.missing` shrinks.
  const [plan, setPlan] = useState<ModelStatus[]>(status.missing);
  const [queued, setQueued] = useState<string[]>([]);
  const [progress, setProgress] = useState<Record<string, InstallProgress>>({});
  const [starting, setStarting] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timer = useRef<number | null>(null);
  const mounted = useRef(true);
  // The poll decides "all finished?" from this, not from React state, which
  // may not have applied the latest tick yet.
  const latest = useRef<Record<string, InstallProgress>>({});

  const stopPolling = () => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  };

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stopPolling();
    };
  }, []);

  // Outside a run, the list follows whatever `check_models` last said.
  const running = queued.length > 0 && queued.some((id) => !isTerminal(progress[id]));
  useEffect(() => {
    if (!running && !starting) setPlan(status.missing);
  }, [status]);

  const poll = useCallback(
    (ids: string[]) => {
      stopPolling();
      const tick = async () => {
        let results: (InstallProgress | null)[];
        try {
          results = await Promise.all(ids.map((id) => api.getInstallProgress(id)));
        } catch (err) {
          stopPolling();
          if (mounted.current) setError(api.errorMessage(err));
          return;
        }
        if (!mounted.current) return;
        const snapshot = { ...latest.current };
        results.forEach((p, i) => {
          if (p) snapshot[ids[i]] = p;
        });
        latest.current = snapshot;
        setProgress(snapshot);
        if (ids.every((id) => isTerminal(snapshot[id]))) {
          stopPolling();
          if (ids.every((id) => snapshot[id]?.phase === "done")) {
            setFinishing(true);
            try {
              await onRecheck();
            } finally {
              if (mounted.current) setFinishing(false);
            }
          }
        }
      };
      timer.current = window.setInterval(tick, POLL_MS);
      void tick();
    },
    [onRecheck],
  );

  // Coming back to this screen mid-run (after "Record without setting up")
  // picks the run back up rather than offering to start a second one.
  useEffect(() => {
    let cancelled = false;
    const ids = status.missing.map((m) => m.id);
    (async () => {
      try {
        const results = await Promise.all(ids.map((id) => api.getInstallProgress(id)));
        if (cancelled) return;
        const known = ids.filter((_, i) => results[i] !== null);
        const anyActive = results.some((p) => p !== null && !isTerminal(p));
        if (anyActive) {
          setQueued(known);
          poll(known);
        }
      } catch {
        // Nothing to resume; the button still works.
      }
    })();
    return () => {
      cancelled = true;
    };
    // Only on mount.
  }, []);

  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      const ids = await api.installRequiredModels();
      if (!mounted.current) return;
      if (ids.length === 0) {
        // Nothing left to fetch — whatever was missing has since landed.
        await onRecheck();
        return;
      }
      setPlan((prev) => (prev.length > 0 ? prev : status.missing));
      setQueued(ids);
      // A retry re-queues only what failed; clear their old failure so the
      // row doesn't show a stale error while it waits its turn.
      const next = { ...latest.current };
      for (const id of ids) delete next[id];
      latest.current = next;
      setProgress(next);
      poll(ids);
    } catch (err) {
      if (mounted.current) setError(api.errorMessage(err));
    } finally {
      if (mounted.current) setStarting(false);
    }
  };

  const rows = plan.length > 0 ? plan : status.missing;
  const totalMb = rows.reduce((sum, m) => sum + m.approx_size_mb, 0) || 1;
  const doneMb = rows.reduce((sum, m) => sum + m.approx_size_mb * fraction(progress[m.id]), 0);
  const overall = Math.round((doneMb / totalMb) * 100);
  const started = queued.length > 0;
  const failedIds = queued.filter((id) => progress[id]?.phase === "failed");
  const downloadMb = status.total_download_mb || totalMb;
  // A release build ships the runtimes inside its installer, so setup then
  // lists model data only; a development build may still need them.
  const needsRuntimes = rows.some((m) => m.kind === "runtime");

  return (
    <div className="setup-view">
      <Card title="Set up Cadence">
        <p>
          Cadence runs entirely on this computer. Before it can transcribe a meeting,
          tell speakers apart and write the minutes, it needs to download the models
          {needsRuntimes ? " and runtimes" : ""} listed below, once — about{" "}
          <strong>{sizeLabel(downloadMb)}</strong>. After that it works fully offline.
        </p>
        <p className="muted">
          The files come from Hugging Face and GitHub, where the upstream projects
          publish their models and runtimes. Nothing about you or your meetings is
          sent — these are plain downloads. Make sure the drive holding your user
          profile has room for them, plus space for your recordings.
        </p>

        {!status.meets_minimum_ram && (
          <Notice kind="warning">
            Cadence needs {sizeLabel(status.minimum_ram_mb)} of RAM, and this computer
            has {sizeLabel(status.total_ram_mb)}. Recording and transcription will work,
            but writing the minutes may fail here.
          </Notice>
        )}

        {/* An upgrade from the old memory tiers can leave a multi-gigabyte
            model behind; naming it here lets the user free that space before
            the new download, rather than finding out from a full disk. */}
        <RetiredModelsNotice
          models={status.retired ?? []}
          busy={starting || running}
          onRemoved={onRecheck}
        />

        <h3>What will be downloaded</h3>
        <ul className="setup-list">
          {rows.map((model) => {
            const current = progress[model.id];
            const isQueued = queued.includes(model.id);
            const phase = current?.phase ?? (isQueued ? "queued" : null);
            return (
              <li key={model.id} className={`setup-item${phase ? ` ${phase}` : ""}`}>
                <div className="setup-item-row">
                  <span className="model-label">{model.label}</span>
                  <span className="setup-item-size">{sizeLabel(model.approx_size_mb)}</span>
                  {phase && (
                    <span
                      className={`badge ${
                        phase === "done"
                          ? "success"
                          : phase === "failed"
                            ? "error"
                            : phase === "queued"
                              ? "muted"
                              : "active"
                      }`}
                    >
                      {PHASE_LABEL[phase]}
                      {phase === "downloading" && current ? ` ${current.percent}%` : ""}
                    </span>
                  )}
                </div>
                {model.notes && <span className="model-notes">{model.notes}</span>}
                {current?.phase === "downloading" && (
                  <span className="progress-track" role="presentation">
                    <span
                      className="progress-fill"
                      style={{ width: `${Math.min(100, Math.max(0, current.percent))}%` }}
                    />
                  </span>
                )}
                {current?.phase === "failed" && (
                  <Notice kind="error">{current.error ?? "The download failed."}</Notice>
                )}
              </li>
            );
          })}
        </ul>

        {started && (
          <div className="setup-overall">
            <span className="install-progress">
              {overall}% — {sizeLabel(Math.round(doneMb))} of {sizeLabel(totalMb)}
            </span>
            <span
              className="progress-track"
              role="progressbar"
              aria-valuenow={overall}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Overall setup progress"
            >
              <span
                className={`progress-fill${failedIds.length > 0 && !running ? " failed" : ""}`}
                style={{ width: `${overall}%` }}
              />
            </span>
          </div>
        )}

        {error && <Notice kind="error">{error}</Notice>}
        {failedIds.length > 0 && !running && (
          <Notice kind="error">
            {failedIds.length} download{failedIds.length === 1 ? "" : "s"} failed. Retry
            fetches only what is still missing; what already finished is kept.
          </Notice>
        )}
        {finishing && (
          <p className="notice">
            <Spinner />
            Checking the installed models
          </p>
        )}

        <div className="button-row">
          {failedIds.length > 0 && !running ? (
            <button type="button" className="primary" disabled={starting} onClick={start}>
              {starting ? "Retrying…" : "Retry"}
            </button>
          ) : (
            <button
              type="button"
              className="primary"
              disabled={starting || running || finishing}
              onClick={start}
            >
              {running || finishing ? (
                <>
                  <Spinner />
                  Setting up…
                </>
              ) : starting ? (
                "Starting…"
              ) : (
                "Download and set up"
              )}
            </button>
          )}
        </div>

        <p className="setup-skip">
          <button type="button" className="link" onClick={onSkip}>
            Record without setting up
          </button>
          <span className="muted">
            {" "}
            — recordings are kept, and can be processed once the models are installed.
            {running && " Downloads carry on in the background."}
          </span>
        </p>
      </Card>
    </div>
  );
}
