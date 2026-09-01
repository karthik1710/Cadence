import type { ReactNode } from "react";
import type { MeetingState } from "../types";
import { PIPELINE_STAGES } from "../types";
import type { Theme } from "../hooks/useTheme";

export function Card({
  title,
  action,
  children,
}: {
  title?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      {(title || action) && (
        <header className="card-header">
          {title && <h2>{title}</h2>}
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Notice({
  kind = "info",
  children,
}: {
  kind?: "info" | "error" | "warning" | "success";
  children: ReactNode;
}) {
  return <p className={`notice ${kind}`}>{children}</p>;
}

export function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

/** Sun/moon toggle for light/dark theme — lives in the fixed app header so
 * it's reachable without scrolling. Purely presentational: `useTheme` owns
 * the actual state and persistence. */
export function ThemeToggle({
  theme,
  onToggle,
}: {
  theme: Theme;
  onToggle: () => void;
}) {
  const isLight = theme === "light";
  const label = isLight ? "Switch to dark theme" : "Switch to light theme";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      aria-label={label}
      title={label}
    >
      {isLight ? (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4.5" fill="currentColor" />
          <g stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <line x1="12" y1="1.5" x2="12" y2="4" />
            <line x1="12" y1="20" x2="12" y2="22.5" />
            <line x1="1.5" y1="12" x2="4" y2="12" />
            <line x1="20" y1="12" x2="22.5" y2="12" />
            <line x1="4.5" y1="4.5" x2="6.3" y2="6.3" />
            <line x1="17.7" y1="17.7" x2="19.5" y2="19.5" />
            <line x1="4.5" y1="19.5" x2="6.3" y2="17.7" />
            <line x1="17.7" y1="6.3" x2="19.5" y2="4.5" />
          </g>
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M20.5 14.7A8.5 8.5 0 0 1 9.3 3.5a8.5 8.5 0 1 0 11.2 11.2Z"
            fill="currentColor"
          />
        </svg>
      )}
    </button>
  );
}

export function LevelMeter({ level }: { level: number }) {
  const percent = Math.min(100, Math.round(level * 100));
  return (
    <div
      className="level-meter"
      role="meter"
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Microphone input level"
    >
      <div
        className={`level-meter-fill${percent >= 98 ? " clipping" : ""}`}
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

/**
 * The §22 "after recording" checklist. Stages before the current one are done,
 * the current one is running, later ones are pending.
 */
export function StageChecklist({
  stage,
  failed,
  skipMinutes,
}: {
  stage: string | null;
  failed?: boolean;
  skipMinutes?: boolean;
}) {
  const stages = skipMinutes
    ? PIPELINE_STAGES.filter(
        (s) => s.state !== "EXTRACTING" && s.state !== "SYNTHESIZING",
      )
    : PIPELINE_STAGES;
  const currentIndex = stages.findIndex((s) => s.state === stage);
  const done = stage === "COMPLETED";

  return (
    <ol className="stage-list">
      {stages.map((entry, index) => {
        const state = stageState(index, currentIndex, done, failed);
        return (
          <li key={entry.state} className={`stage ${state}`}>
            <span className="stage-icon" aria-hidden="true">
              {state === "done" ? "✓" : state === "active" ? "•" : state === "failed" ? "!" : "○"}
            </span>
            {entry.label}
          </li>
        );
      })}
    </ol>
  );
}

function stageState(
  index: number,
  currentIndex: number,
  done: boolean,
  failed?: boolean,
): "done" | "active" | "pending" | "failed" {
  if (done) return "done";
  if (currentIndex === -1) return "pending";
  if (index < currentIndex) return "done";
  if (index === currentIndex) return failed ? "failed" : "active";
  return "pending";
}

export function StateBadge({ state }: { state: MeetingState }) {
  const label: Record<MeetingState, string> = {
    IDLE: "Idle",
    RECORDING: "Recording",
    RECORDED: "Not processed",
    PREPROCESSING: "Processing",
    DIARIZING: "Processing",
    TRANSCRIBING: "Processing",
    ALIGNING: "Processing",
    EXTRACTING: "Processing",
    SYNTHESIZING: "Processing",
    COMPLETED: "Ready",
    FAILED: "Needs attention",
  };
  const tone =
    state === "COMPLETED"
      ? "success"
      : state === "FAILED"
        ? "error"
        : state === "RECORDED" || state === "IDLE"
          ? "muted"
          : "active";
  return <span className={`badge ${tone}`}>{label[state]}</span>;
}
