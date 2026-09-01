import { useCallback, useEffect, useState } from "react";
import * as api from "./services/api";
import { Notice, ThemeToggle } from "./components/ui";
import { AppLogo } from "./components/AppLogo";
import { RecordingPanel } from "./features/recording/RecordingPanel";
import { HistoryView } from "./features/history/HistoryView";
import { SettingsView } from "./features/settings/SettingsView";
import { MeetingView } from "./features/meeting/MeetingView";
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

  const openSettings = () => {
    setOpenMeetingId(null);
    setTab("settings");
  };

  return (
    <main className="container">
      <header className="app-header">
        <div className="app-header-row">
          <div className="app-header-titles">
            <h1>
              <AppLogo size={30} />
              <span>EchoMind</span>
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
      {!openMeetingId && (
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
        {setupError && (
          <Notice kind="error">
            The local model set could not be checked: {setupError}
          </Notice>
        )}

        {/* On the Meetings tab the fix (Record or Settings) lives elsewhere, so
            this stays a short pointer rather than repeating the full banner. */}
        {setupStatus && !setupStatus.ready && tab === "meetings" && !openMeetingId && (
          <Notice kind="warning">
            <span className="grow">
              {setupStatus.message ?? "Some models still need to be installed."}
            </span>
            <button type="button" className="ghost" onClick={openSettings}>
              Open Settings
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
                onSetupChanged={refreshSetup}
                onOpenSettings={openSettings}
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
      </div>
    </main>
  );
}

export default App;
