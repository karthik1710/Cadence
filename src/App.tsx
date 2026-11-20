import { useCallback, useEffect, useState } from "react";
import * as api from "./services/api";
import { Notice, Spinner, ThemeToggle } from "./components/ui";
import { AppLogo } from "./components/AppLogo";
import { RecordingPanel } from "./features/recording/RecordingPanel";
import { HistoryView } from "./features/history/HistoryView";
import { SettingsView } from "./features/settings/SettingsView";
import { MeetingView } from "./features/meeting/MeetingView";
import { SetupView } from "./features/setup/SetupView";
import { useTheme } from "./hooks/useTheme";
import type { SetupStatus } from "./types";
import "./App.css";

type Tab = "record" | "meetings" | "settings";

const TABS: { id: Tab; label: string }[] = [
  { id: "record", label: "Record" },
  { id: "meetings", label: "Meetings" },
  { id: "settings", label: "Settings" },
];

function App() {
  const [tab, setTab] = useState<Tab>("record");
  const [openMeetingId, setOpenMeetingId] = useState<string | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);
  // "Record without setting up": recording must never wait on a download, so
  // the setup screen can always be stepped past. Not persisted — the next
  // launch asks again while models are still missing.
  const [setupSkipped, setSetupSkipped] = useState(false);
  const [historyToken, setHistoryToken] = useState(0);
  const { theme, toggleTheme, persistError, dismissPersistError, syncTheme } = useTheme();

  // Checked up front, not after a meeting has been captured: a missing model
  // must never be discovered only once there is real audio to lose.
  const refreshSetup = useCallback(async () => {
    try {
      setSetupStatus(await api.checkModels());
      setSetupError(null);
    } catch (err) {
      setSetupStatus(null);
      setSetupError(api.errorMessage(err));
    }
  }, []);

  useEffect(() => {
    refreshSetup();
  }, [refreshSetup]);

  const openSetup = () => {
    setOpenMeetingId(null);
    setSetupSkipped(false);
  };

  // Until the first check lands there is nothing to decide on, and showing
  // the tabs would flash them before the setup screen replaced them. A failed
  // check falls through to the tabs, with its error shown: never a dead end.
  const checking = setupStatus === null && setupError === null;
  const showSetup = setupStatus !== null && !setupStatus.ready && !setupSkipped;

  return (
    <main className="container">
      <header className="app-header">
        <div className="app-header-row">
          <div className="app-header-titles">
            <h1>
              <AppLogo size={30} />
              <span>Cadence</span>
            </h1>
            <p className="subtitle">Local meeting intelligence · nothing leaves this computer</p>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
        {persistError && (
          <Notice kind="error">
            <span className="grow">Theme preference couldn't be saved: {persistError}</span>
            <button type="button" className="ghost" onClick={dismissPersistError}>
              Dismiss
            </button>
          </Notice>
        )}
      </header>

      {/* The tab bar is a fixed, non-scrolling shell element — only the
          content region below it (rendered into .content-scroll) scrolls. */}
      {!openMeetingId && !showSetup && !checking && (
        <nav className="tabs" role="tablist">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={tab === entry.id}
              className={tab === entry.id ? "active" : ""}
              onClick={() => setTab(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </nav>
      )}

      <div className="content-scroll">
        {checking ? (
          <p className="notice">
            <Spinner />
            Checking the local models
          </p>
        ) : showSetup && setupStatus ? (
          <SetupView
            status={setupStatus}
            onRecheck={refreshSetup}
            onSkip={() => setSetupSkipped(true)}
          />
        ) : (
          <>
            {setupError && (
              <Notice kind="error">
                The local model set could not be checked: {setupError}
              </Notice>
            )}

            {/* On the Meetings tab the fix lives on the setup screen, so this
                stays a short pointer rather than repeating the full banner. */}
            {setupStatus && !setupStatus.ready && tab === "meetings" && !openMeetingId && (
              <Notice kind="warning">
                <span className="grow">
                  {setupStatus.message ?? "Some models still need to be installed."}
                </span>
                <button type="button" className="ghost" onClick={openSetup}>
                  Finish setting up
                </button>
              </Notice>
            )}

            {openMeetingId ? (
              <MeetingView
                meetingId={openMeetingId}
                onBack={() => setOpenMeetingId(null)}
                onDeleted={() => {
                  setOpenMeetingId(null);
                  setHistoryToken((n) => n + 1);
                  setTab("meetings");
                }}
              />
            ) : (
              <>
                {tab === "record" && (
                  <RecordingPanel
                    onOpenMeeting={setOpenMeetingId}
                    setupReady={setupStatus?.ready ?? false}
                    missingModels={setupStatus?.missing ?? []}
                    onOpenSetup={openSetup}
                  />
                )}
                {tab === "meetings" && (
                  <HistoryView onOpenMeeting={setOpenMeetingId} reloadToken={historyToken} />
                )}
                {tab === "settings" && (
                  <SettingsView onSetupChanged={refreshSetup} onThemeSaved={syncTheme} />
                )}
              </>
            )}
          </>
        )}
      </div>
    </main>
  );
}

export default App;
