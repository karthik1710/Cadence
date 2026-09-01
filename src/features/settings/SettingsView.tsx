import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../../services/api";
import { sizeLabel } from "../../services/format";
import { Card, Notice, Spinner } from "../../components/ui";
import { RecommendedSetupBanner } from "../../components/RecommendedSetup";
import type {
  HostReport,
  InstallProgress,
  ModelKind,
  ModelStatus,
  Settings,
  VerifyReport,
} from "../../types";

interface Props {
  /** Lets the shell re-run `check_models` once the installed set changes. */
  onSetupChanged: () => void;
  /** Lets the shell apply a theme change from this screen's Save button
   * immediately, without a redundant second read-modify-write — this screen
   * already persisted the whole settings object itself. */
  onThemeSaved?: (theme: string) => void;
}

const INSTALL_POLL_MS = 700;

const KIND_SECTIONS: { kind: ModelKind; label: string }[] = [
  { kind: "runtime", label: "Runtimes" },
  { kind: "stt", label: "Speech recognition" },
  { kind: "llm", label: "Meeting intelligence" },
  { kind: "diarization", label: "Speaker diarization" },
];

const PROFILES = ["auto", "low", "balanced", "high"];

/** Empty means "let the machine decide", which the Rust settings spell `null`. */
function numberOrNull(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** The §27 settings screen: what this machine is, what is installed, how it runs. */
export function SettingsView({ onSetupChanged, onThemeSaved }: Props) {
  const [host, setHost] = useState<HostReport | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [progress, setProgress] = useState<Record<string, InstallProgress>>({});
  const [verified, setVerified] = useState<Record<string, VerifyReport>>({});
  // The threshold is a free-typed decimal, so it needs a draft the user can
  // clear mid-edit without the stored value snapping to 0.
  const [thresholdText, setThresholdText] = useState("");

  const timers = useRef<Record<string, number>>({});

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.listModels());
    } catch (err) {
      setError(api.errorMessage(err));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [hostReport, stored, list] = await Promise.all([
          api.getHostInfo(),
          api.getSettings(),
          api.listModels(),
        ]);
        if (cancelled) return;
        setHost(hostReport);
        setSettings(stored);
        setThresholdText(String(stored.cluster_threshold));
        setModels(list);
      } catch (err) {
        if (!cancelled) setError(api.errorMessage(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const handles = timers.current;
    return () => {
      for (const handle of Object.values(handles)) window.clearInterval(handle);
    };
  }, []);

  const update = (patch: Partial<Settings>) => {
    setSaved(false);
    setSettings((prev) => (prev ? { ...prev, ...patch } : prev));
  };

  const stopPolling = (id: string) => {
    const handle = timers.current[id];
    if (handle !== undefined) {
      window.clearInterval(handle);
      delete timers.current[id];
    }
  };

  const install = async (id: string) => {
    setError(null);
    setVerified((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    try {
      await api.installModel(id);
    } catch (err) {
      setError(api.errorMessage(err));
      return;
    }
    stopPolling(id);
    timers.current[id] = window.setInterval(async () => {
      try {
        const current = await api.getInstallProgress(id);
        if (!current) return;
        setProgress((prev) => ({ ...prev, [id]: current }));
        if (current.phase === "done" || current.phase === "failed") {
          stopPolling(id);
          if (current.phase === "done") {
            await refreshModels();
            onSetupChanged();
          }
        }
      } catch (err) {
        stopPolling(id);
        setError(api.errorMessage(err));
      }
    }, INSTALL_POLL_MS);
  };

  const verify = async (id: string) => {
    setError(null);
    try {
      const report = await api.verifyModel(id);
      setVerified((prev) => ({ ...prev, [id]: report }));
    } catch (err) {
      setError(api.errorMessage(err));
    }
  };

  const remove = async (model: ModelStatus) => {
    if (
      !window.confirm(
        `Remove ${model.label} from this machine? It has to be downloaded again before it can be used.`,
      )
    ) {
      return;
    }
    setError(null);
    try {
      await api.removeModel(model.id);
      setProgress((prev) => {
        const next = { ...prev };
        delete next[model.id];
        return next;
      });
      await refreshModels();
      onSetupChanged();
    } catch (err) {
      setError(api.errorMessage(err));
    }
  };

  const save = async () => {
    if (!settings) return;
    setError(null);
    setSaved(false);
    try {
      const stored = await api.saveSettings(settings);
      setSettings(stored);
      setThresholdText(String(stored.cluster_threshold));
      setSaved(true);
      // Profile and model choices decide which models count as required.
      onSetupChanged();
      // Apply the theme choice live — this save already wrote it to disk.
      onThemeSaved?.(stored.theme);
    } catch (err) {
      setError(api.errorMessage(err));
    }
  };

  const installedOf = (kind: ModelKind) =>
    models.filter((model) => model.kind === kind && model.installed);

  const missingRequired = models.filter((model) => model.required && !model.installed);

  return (
    <div className="settings-view">
      {error && <Notice kind="error">{error}</Notice>}

      {missingRequired.length > 0 && (
        <RecommendedSetupBanner
          missing={missingRequired}
          onChanged={() => {
            refreshModels();
            onSetupChanged();
          }}
        />
      )}

      <Card title="This machine">
        {!host ? (
          <p className="notice">
            <Spinner />
            Reading this machine
          </p>
        ) : (
          <>
            <dl className="host-grid">
              <div>
                <dt>Memory</dt>
                <dd>
                  {sizeLabel(host.available_ram_mb)} free of {sizeLabel(host.total_ram_mb)}
                </dd>
              </div>
              <div>
                <dt>Logical cores</dt>
                <dd>{host.logical_cores}</dd>
              </div>
              <div>
                <dt>Profile in force</dt>
                <dd>
                  {host.profile}
                  {host.profile !== host.detected_profile && ` (detected: ${host.detected_profile})`}
                </dd>
              </div>
              <div>
                <dt>Language model budget</dt>
                <dd>{sizeLabel(host.llm_ram_budget_mb)}</dd>
              </div>
              <div>
                <dt>Recommended language model</dt>
                <dd>{host.recommended_llm}</dd>
              </div>
              <div>
                <dt>Recommended speech model</dt>
                <dd>{host.recommended_stt}</dd>
              </div>
              <div>
                <dt>Recommended context</dt>
                <dd>{host.recommended_context} tokens</dd>
              </div>
              <div>
                <dt>Threads</dt>
                <dd>
                  {host.stt_threads} speech · {host.diarization_threads} diarization ·{" "}
                  {host.llm_threads} language model
                </dd>
              </div>
            </dl>
            <p className="muted">
              EchoMind picks a smaller language model and a shorter context on machines
              with less RAM: a 16 GB laptop gets the 4B model, a 32 GB one gets the 8B.
            </p>
          </>
        )}
      </Card>

      <Card title="Models">
        {models.length === 0 && !error && (
          <p className="notice">
            <Spinner />
            Reading the model list
          </p>
        )}

        {KIND_SECTIONS.map((section) => {
          const rows = models.filter((model) => model.kind === section.kind);
          if (rows.length === 0) return null;
          return (
            <section key={section.kind} className="model-group">
              <h3>{section.label}</h3>
              <table className="model-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Size</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((model) => {
                    const current = progress[model.id];
                    const report = verified[model.id];
                    const installing =
                      current !== undefined &&
                      current.phase !== "done" &&
                      current.phase !== "failed";
                    return (
                      <tr key={model.id}>
                        <td>
                          <span className="model-label">{model.label}</span>
                          {model.required && <span className="marker">Required</span>}
                          {model.notes && <span className="model-notes">{model.notes}</span>}
                          {current && (
                            <>
                              <span className="install-progress">
                                {current.phase} — {current.percent}% (
                                {sizeLabel(current.downloaded_mb)} of{" "}
                                {sizeLabel(current.total_mb)})
                              </span>
                              <span className="progress-track" role="presentation">
                                <span
                                  className={`progress-fill${
                                    current.phase === "failed" ? " failed" : ""
                                  }`}
                                  style={{
                                    width: `${Math.min(100, Math.max(0, current.percent))}%`,
                                  }}
                                />
                              </span>
                            </>
                          )}
                          {current?.phase === "failed" && (
                            <Notice kind="error">
                              {current.error ?? "The download failed."}
                            </Notice>
                          )}
                          {report && (
                            <Notice kind={report.ok ? "success" : "error"}>
                              {report.message}
                            </Notice>
                          )}
                        </td>
                        <td>
                          {model.installed
                            ? sizeLabel(model.actual_size_mb)
                            : sizeLabel(model.approx_size_mb)}
                        </td>
                        <td>
                          <span className={`badge ${model.installed ? "success" : "muted"}`}>
                            {model.installed ? "Installed" : "Not installed"}
                          </span>
                        </td>
                        <td>
                          <div className="model-actions">
                            {!model.installed && (
                              <button
                                type="button"
                                className="primary"
                                disabled={installing}
                                onClick={() => install(model.id)}
                              >
                                {installing ? "Installing…" : "Install"}
                              </button>
                            )}
                            <button
                              type="button"
                              className="ghost"
                              disabled={!model.installed || installing}
                              onClick={() => verify(model.id)}
                            >
                              Verify
                            </button>
                            <button
                              type="button"
                              className="ghost danger"
                              disabled={!model.installed || installing}
                              onClick={() => remove(model)}
                            >
                              Remove
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>
          );
        })}

        <p className="muted">
          Models are downloaded once. Everything after that — recording, transcription,
          diarization and the minutes — runs offline on this machine.
        </p>
      </Card>

      <Card title="Processing">
        {!settings ? (
          <p className="notice">
            <Spinner />
            Loading settings
          </p>
        ) : (
          <>
            <div className="field-row">
              <label className="field">
                <span>Performance profile</span>
                <select
                  value={settings.profile}
                  onChange={(e) => update({ profile: e.currentTarget.value })}
                >
                  {PROFILES.map((profile) => (
                    <option key={profile} value={profile}>
                      {profile}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Speech model</span>
                <select
                  value={settings.stt_model ?? ""}
                  onChange={(e) =>
                    update({ stt_model: e.currentTarget.value || null })
                  }
                >
                  <option value="">Automatic (recommended for this machine)</option>
                  {installedOf("stt").map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Language model</span>
                <select
                  value={settings.llm_model ?? ""}
                  onChange={(e) =>
                    update({ llm_model: e.currentTarget.value || null })
                  }
                >
                  <option value="">Automatic (recommended for this machine)</option>
                  {installedOf("llm").map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="field-row">
              <label className="field">
                <span>Speech threads</span>
                <input
                  type="number"
                  min={1}
                  placeholder="Automatic"
                  value={settings.stt_threads ?? ""}
                  onChange={(e) =>
                    update({ stt_threads: numberOrNull(e.currentTarget.value) })
                  }
                />
              </label>
              <label className="field">
                <span>Diarization threads</span>
                <input
                  type="number"
                  min={1}
                  placeholder="Automatic"
                  value={settings.diarization_threads ?? ""}
                  onChange={(e) =>
                    update({ diarization_threads: numberOrNull(e.currentTarget.value) })
                  }
                />
              </label>
              <label className="field">
                <span>Language model threads</span>
                <input
                  type="number"
                  min={1}
                  placeholder="Automatic"
                  value={settings.llm_threads ?? ""}
                  onChange={(e) =>
                    update({ llm_threads: numberOrNull(e.currentTarget.value) })
                  }
                />
              </label>
              <label className="field">
                <span>Context window (tokens)</span>
                <input
                  type="number"
                  min={512}
                  placeholder="Automatic"
                  value={settings.llm_context_tokens ?? ""}
                  onChange={(e) =>
                    update({ llm_context_tokens: numberOrNull(e.currentTarget.value) })
                  }
                />
              </label>
            </div>

            <div className="field-row">
              <label className="field">
                <span>Speaker cluster threshold</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={thresholdText}
                  onChange={(e) => {
                    const text = e.currentTarget.value;
                    setThresholdText(text);
                    const parsed = numberOrNull(text);
                    if (parsed !== null) update({ cluster_threshold: parsed });
                  }}
                />
              </label>
              <label className="field">
                <span>Minutes language</span>
                <input
                  value={settings.mom_language}
                  onChange={(e) => update({ mom_language: e.currentTarget.value })}
                />
              </label>
            </div>
            <p className="muted">
              0.8 is provisional — it rests on one unlabelled sample. Lower it if one
              person is split across several speakers, raise it if two people are merged.
            </p>

            <div className="field-row">
              <label className="field">
                <span>Theme</span>
                <select
                  value={settings.theme === "light" ? "light" : "dark"}
                  onChange={(e) => update({ theme: e.currentTarget.value })}
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                </select>
              </label>
            </div>
            <p className="muted">
              Also reachable at any time from the toggle in the top-right corner.
            </p>

            <label className="field checkbox">
              <input
                type="checkbox"
                checked={settings.generate_mom}
                onChange={(e) => update({ generate_mom: e.currentTarget.checked })}
              />
              <span>Generate minutes of meeting</span>
            </label>
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={settings.debug_logging}
                onChange={(e) => update({ debug_logging: e.currentTarget.checked })}
              />
              <span>Debug logging</span>
            </label>

            <div className="button-row">
              <button type="button" className="primary" onClick={save}>
                Save settings
              </button>
            </div>
            {saved && <Notice kind="success">Settings saved.</Notice>}
          </>
        )}
      </Card>

      <Card title="Where your data lives">
        {host ? (
          <>
            <code className="data-dir">{host.data_dir}</code>
            <p className="muted">
              Recordings, transcripts and minutes are written here and never leave this
              folder.
            </p>
          </>
        ) : (
          <p className="muted">Reading this machine…</p>
        )}
      </Card>
    </div>
  );
}
