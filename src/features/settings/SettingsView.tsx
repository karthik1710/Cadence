import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../../services/api";
import { parseRedactionMode, sizeLabel } from "../../services/format";
import { Card, Notice, Spinner } from "../../components/ui";
import type {
  HostReport,
  InstallProgress,
  ModelKind,
  ModelStatus,
  RedactionMode,
  RetiredModel,
  Settings,
  VerifyReport,
} from "../../types";
import { RetiredModelsNotice } from "../setup/RetiredModelsNotice";

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
  { kind: "denoise", label: "Speech enhancement" },
  { kind: "vad", label: "Speech detection" },
];

/** Catches any model whose kind has no section above, so a model the backend
 * lists can never silently drop out of this screen. */
const OTHER_SECTION = { key: "other", label: "Other" };

/** In the order they escalate: see nothing, see everything marked, then two
 * that change what the transcript says. */
const REDACTION_MODES: { mode: RedactionMode; label: string }[] = [
  { mode: "off", label: "Do nothing — don't even mark them" },
  { mode: "flag", label: "Mark them for review (recommended)" },
  { mode: "mask", label: "Mask them (d****)" },
  { mode: "redact", label: "Replace them with [expunged]" },
];

/** Empty means "let the machine decide", which the Rust settings spell `null`. */
function numberOrNull(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** The settings screen: what this machine is, what is installed, how it runs. */
export function SettingsView({ onSetupChanged, onThemeSaved }: Props) {
  const [host, setHost] = useState<HostReport | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [retired, setRetired] = useState<RetiredModel[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [progress, setProgress] = useState<Record<string, InstallProgress>>({});
  const [verified, setVerified] = useState<Record<string, VerifyReport>>({});
  // Downloads are one at a time across the app; the backend refuses a second,
  // and every install button waits rather than offering one it would refuse.
  const anyInstalling = Object.values(progress).some(
    (p) => p.phase !== "done" && p.phase !== "failed",
  );
  // The threshold is a free-typed decimal, so it needs a draft the user can
  // clear mid-edit without the stored value snapping to 0.
  const [thresholdText, setThresholdText] = useState("");

  const timers = useRef<Record<string, number>>({});

  const refreshModels = useCallback(async () => {
    try {
      const [list, leftovers] = await Promise.all([
        api.listModels(),
        api.listRetiredModels(),
      ]);
      setModels(list);
      setRetired(leftovers);
    } catch (err) {
      setError(api.errorMessage(err));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [hostReport, stored, list, leftovers] = await Promise.all([
          api.getHostInfo(),
          api.getSettings(),
          api.listModels(),
          api.listRetiredModels(),
        ]);
        if (cancelled) return;
        setHost(hostReport);
        setSettings(stored);
        setThresholdText(String(stored.cluster_threshold));
        setModels(list);
        setRetired(leftovers);
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

  // Reinstall and Remove confirm in the page itself, next to the row they act
  // on, the same way RetiredModelsNotice does — rather than in a native dialog
  // detached from the model it is asking about.
  const [confirming, setConfirming] = useState<{ id: string; action: "reinstall" | "remove" } | null>(null);

  const reinstall = (model: ModelStatus) => {
    setConfirming(null);
    void install(model.id);
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
    setConfirming(null);
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
      // Apply the theme choice live — this save already wrote it to disk.
      onThemeSaved?.(stored.theme);
    } catch (err) {
      setError(api.errorMessage(err));
    }
  };

  return (
    <div className="settings-view">
      {error && <Notice kind="error">{error}</Notice>}

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
                <dt>Threads</dt>
                <dd>
                  {host.stt_threads} speech · {host.diarization_threads} diarization ·{" "}
                  {host.llm_threads} language model
                </dd>
              </div>
            </dl>
          </>
        )}
      </Card>

      {/* Read-only on purpose: every machine runs the same model set, so
          there is nothing here to choose. */}
      <Card title="Models in use">
        {!host ? (
          <p className="notice">
            <Spinner />
            Reading this machine
          </p>
        ) : (
          <>
            <dl className="host-grid">
              <div>
                <dt>Language model</dt>
                <dd>{host.llm_model}</dd>
              </div>
              <div>
                <dt>Speech model</dt>
                <dd>{host.stt_model}</dd>
              </div>
              <div>
                <dt>Context</dt>
                <dd>{host.llm_context.toLocaleString()} tokens</dd>
              </div>
              <div>
                <dt>Memory</dt>
                <dd>
                  {sizeLabel(host.total_ram_mb)} (minimum {sizeLabel(host.minimum_ram_mb)})
                </dd>
              </div>
              <div>
                <dt>Language model budget</dt>
                <dd>{sizeLabel(host.llm_ram_budget_mb)}</dd>
              </div>
            </dl>
            {!host.meets_minimum_ram && (
              <Notice kind="warning">
                This computer has less than the {sizeLabel(host.minimum_ram_mb)} of RAM
                Cadence needs. Recording and transcription work, but writing the minutes
                may fail.
              </Notice>
            )}
            <p className="muted">
              Every computer runs the same models, so a meeting is transcribed and
              summarised the same way wherever it is processed.
            </p>
          </>
        )}
      </Card>

      <Card title="Installed models">
        {models.length === 0 && !error && (
          <p className="notice">
            <Spinner />
            Reading the model list
          </p>
        )}

        <RetiredModelsNotice
          models={retired}
          busy={anyInstalling}
          onRemoved={async () => {
            await refreshModels();
            onSetupChanged();
          }}
        />

        {[
          ...KIND_SECTIONS.map((section) => ({
            key: section.kind as string,
            label: section.label,
            rows: models.filter((model) => model.kind === section.kind),
          })),
          {
            ...OTHER_SECTION,
            rows: models.filter(
              (model) => !KIND_SECTIONS.some((section) => section.kind === model.kind),
            ),
          },
        ].map((section) => {
          const rows = section.rows;
          if (rows.length === 0) return null;
          return (
            <section key={section.key} className="model-group">
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
                          {confirming?.id === model.id && (
                            <Notice kind="warning">
                              <span className="grow">
                                {confirming.action === "reinstall"
                                  ? `Download ${model.label} again (about ${sizeLabel(model.approx_size_mb)}) and replace the installed copy? The current copy keeps working until the new one is in place.`
                                  : `Remove ${model.label} from this computer? It has to be downloaded again before it can be used.`}
                              </span>
                              <button
                                type="button"
                                className={confirming.action === "remove" ? "danger" : "primary"}
                                disabled={anyInstalling}
                                onClick={() =>
                                  confirming.action === "reinstall" ? reinstall(model) : remove(model)
                                }
                              >
                                {confirming.action === "reinstall" ? "Reinstall" : "Remove"}
                              </button>
                              <button
                                type="button"
                                className="ghost"
                                onClick={() => setConfirming(null)}
                              >
                                Cancel
                              </button>
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
                            {model.bundled
                              ? "Included with Cadence"
                              : model.installed
                                ? "Installed"
                                : "Not installed"}
                          </span>
                        </td>
                        <td>
                          <div className="model-actions">
                            {/* A runtime shipped inside the installer is never
                                downloaded from here: reinstalling Cadence is
                                what repairs it, and Verify says so. */}
                            {!model.installed && !model.bundled && (
                              <button
                                type="button"
                                className="primary"
                                disabled={anyInstalling}
                                onClick={() => install(model.id)}
                              >
                                {installing ? "Installing…" : "Install"}
                              </button>
                            )}
                            {/* Repairs a damaged install, records a digest
                                for one placed by hand, and moves a runtime
                                out of the shared folder older versions used. */}
                            {model.installed && !model.bundled && (
                              <button
                                type="button"
                                className="ghost"
                                disabled={anyInstalling}
                                onClick={() => {
                                  setError(null);
                                  setConfirming({ id: model.id, action: "reinstall" });
                                }}
                              >
                                {installing ? "Reinstalling…" : "Reinstall"}
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
                            {/* The backend refuses to remove a required model,
                                and every catalog entry is required. */}
                            {!model.required && (
                              <button
                                type="button"
                                className="ghost danger"
                                disabled={!model.installed || installing}
                                onClick={() => {
                                  setError(null);
                                  setConfirming({ id: model.id, action: "remove" });
                                }}
                              >
                                Remove
                              </button>
                            )}
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
          Models are downloaded once, at setup; runtimes marked “Included with
          Cadence” came with the installer. Everything after that — recording,
          transcription, diarization and the minutes — runs offline on this machine.
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
                <span>Spoken language</span>
                <select
                  value={settings.stt_language}
                  onChange={(e) => update({ stt_language: e.currentTarget.value })}
                >
                  <option value="code-switched">Mixed Tamil + English</option>
                  <option value="auto">Auto-detect (one language)</option>
                  <option value="ta">Tamil only</option>
                  <option value="en">English only</option>
                </select>
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
              0.5 is provisional — it rests on two small samples that disagreed. Raise it
              if one person is split across several speakers, lower it if two people are
              merged into one.
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
            <Toggle
              label="Detect the meeting type and its topics"
              hint="A count over the finished transcript. No model is loaded."
              checked={settings.classify_meeting}
              onChange={(checked) => update({ classify_meeting: checked })}
            />
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

      <Card title="Audio processing">
        {!settings ? (
          <p className="muted">Loading settings…</p>
        ) : (
          <>
            <p className="muted">
              What happens to the audio before it reaches the speech model, in order:
              DC removal, then the high-pass filter, then noise reduction, then the
              leveler, then the compressor, then normalisation. The leveler slowly
              evens out near and far talkers and always runs. Every other stage is a
              judgement about your room and your microphone, so each can be turned off
              on its own. The recording itself is never modified — every stage works on
              a copy.
            </p>

            <h3>Cleaning up the room</h3>
            <Toggle
              label="Remove the microphone's DC offset"
              hint="Subtracts the constant electrical bias some microphones add. Harmless when there is none."
              checked={settings.preprocess_remove_dc}
              onChange={(checked) => update({ preprocess_remove_dc: checked })}
            />
            <div className="field-row">
              <NumberField
                label="High-pass corner"
                unit="Hz"
                step={5}
                min={0}
                max={300}
                value={settings.preprocess_high_pass_hz}
                onChange={(value) => update({ preprocess_high_pass_hz: value })}
                hint="Cuts air conditioning, desk thumps and handling noise below the bottom of a voice. 0 turns it off; 80 is the default."
              />
            </div>

            <h3>Levels</h3>
            <Toggle
              label="Normalise loudness"
              hint="Whisper was trained on normalised audio. The single biggest accuracy win on a quiet laptop microphone."
              checked={settings.preprocess_normalize}
              onChange={(checked) => update({ preprocess_normalize: checked })}
            />
            <div className="field-row">
              <NumberField
                label="Target level"
                unit="dBFS RMS"
                step={0.5}
                min={-40}
                max={-6}
                value={settings.preprocess_target_rms_dbfs}
                onChange={(value) => update({ preprocess_target_rms_dbfs: value })}
                hint="-20 is where broadcast speech sits. Peak limiting keeps the gain from clipping."
                disabled={!settings.preprocess_normalize}
              />
            </div>

            <Toggle
              label="Even out loud and quiet speakers (compressor)"
              hint="Lifts the far end of the meeting table above the noise floor. Costs nothing measurable, so it is on by default."
              checked={settings.preprocess_compress}
              onChange={(checked) => update({ preprocess_compress: checked })}
            />
            <div className="field-row">
              <NumberField
                label="Threshold"
                unit="dBFS"
                step={1}
                min={-60}
                max={0}
                value={settings.preprocess_compress_threshold_dbfs}
                onChange={(value) => update({ preprocess_compress_threshold_dbfs: value })}
                hint="Above this level, gain starts coming down."
                disabled={!settings.preprocess_compress}
              />
              <NumberField
                label="Ratio"
                unit=":1"
                step={0.1}
                min={1}
                max={20}
                value={settings.preprocess_compress_ratio}
                onChange={(value) => update({ preprocess_compress_ratio: value })}
                hint="How hard. 2.5:1 is gentle; above 6:1 starts to sound processed."
                disabled={!settings.preprocess_compress}
              />
              <NumberField
                label="Knee"
                unit="dB"
                step={1}
                min={0}
                max={24}
                value={settings.preprocess_compress_knee_db}
                onChange={(value) => update({ preprocess_compress_knee_db: value })}
                hint="How gradually it engages around the threshold."
                disabled={!settings.preprocess_compress}
              />
            </div>
            <div className="field-row">
              <NumberField
                label="Attack"
                unit="ms"
                step={1}
                min={0}
                max={200}
                value={settings.preprocess_compress_attack_ms}
                onChange={(value) => update({ preprocess_compress_attack_ms: value })}
                hint="Fast enough to catch a raised voice, slow enough to leave consonants alone."
                disabled={!settings.preprocess_compress}
              />
              <NumberField
                label="Release"
                unit="ms"
                step={10}
                min={10}
                max={2000}
                value={settings.preprocess_compress_release_ms}
                onChange={(value) => update({ preprocess_compress_release_ms: value })}
                hint="Too short pumps the room tone up between words."
                disabled={!settings.preprocess_compress}
              />
            </div>

            <h3>Noise reduction</h3>
            <Toggle
              label="Model-free noise reduction (spectral)"
              hint="Estimates the room tone and subtracts it. Costs a full spectral pass over the whole recording, and is skipped whenever the GTCRN denoiser below runs."
              checked={settings.preprocess_noise_reduction}
              onChange={(checked) => update({ preprocess_noise_reduction: checked })}
            />
            <div className="field-row">
              <NumberField
                label="Strength"
                step={0.05}
                min={0}
                max={1}
                value={settings.preprocess_noise_reduction_strength}
                onChange={(value) =>
                  update({ preprocess_noise_reduction_strength: value })
                }
                hint="0 does nothing, 1 is aggressive and can hollow out quiet speech. 0.5 is the default."
                disabled={!settings.preprocess_noise_reduction}
              />
            </div>
            <Toggle
              label="Speech denoiser (GTCRN model)"
              hint="Better than the model-free pass and cheaper than it too. The GTCRN model is installed at setup; if it is ever missing this does nothing and logs a skip, so it is safe to leave on."
              checked={settings.denoise_enabled}
              onChange={(checked) => update({ denoise_enabled: checked })}
            />
            <Notice kind="info">
              The two are not additive. If GTCRN is installed it supersedes the
              model-free pass, and running both only spends time twice.
            </Notice>

            <h3>Speech detection</h3>
            <Toggle
              label="Measure how much of the recording is speech"
              hint="Uses the Silero VAD model installed at setup; without it this is a no-op."
              checked={settings.vad_enabled}
              onChange={(checked) => update({ vad_enabled: checked })}
            />
            <Toggle
              label="Cut the silence out before transcribing"
              hint="Faster, but the timestamp mapping back onto the original recording is not yet trusted on real meeting audio. Off by default for that reason."
              checked={settings.vad_trim_silence}
              onChange={(checked) => update({ vad_trim_silence: checked })}
              disabled={!settings.vad_enabled}
            />
            <div className="field-row">
              <NumberField
                label="Padding kept around speech"
                unit="ms"
                step={10}
                min={0}
                max={2000}
                value={settings.vad_speech_pad_ms}
                onChange={(value) => update({ vad_speech_pad_ms: Math.round(value) })}
                hint="So a cut never clips the start of a word."
                disabled={!settings.vad_enabled || !settings.vad_trim_silence}
              />
            </div>

            <div className="button-row">
              <button type="button" className="primary" onClick={save}>
                Save settings
              </button>
            </div>
            {saved && <Notice kind="success">Settings saved.</Notice>}
          </>
        )}
      </Card>

      <Card title="Sensitive language">
        {!settings ? (
          <p className="muted">Loading settings…</p>
        ) : (
          <>
            <p className="muted">
              After transcription, Cadence checks the transcript against a local word
              list and records where the hits are. The list ships with the app; nothing
              is sent anywhere to do this.
            </p>

            <div className="field-row">
              <label className="field">
                <span>When a word is flagged</span>
                <select
                  value={parseRedactionMode(settings.redaction_mode)}
                  onChange={(e) => update({ redaction_mode: e.currentTarget.value })}
                  aria-label="What to do with flagged words"
                >
                  {REDACTION_MODES.map(({ mode, label }) => (
                    <option key={mode} value={mode}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <ul className="mode-help">
              <li>
                <strong>Mark for review</strong> changes nothing about the text. It
                highlights the words in the transcript so you can decide, and exports
                keep them as they were said.
              </li>
              <li>
                <strong>Mask</strong> and <strong>replace</strong> change what you see,
                what you copy, and what an export contains.
              </li>
              <li>
                None of the four touch <code>original.wav</code>. The recording still
                contains every word, and playback will play it. Masking a transcript is
                not the same as sanitising a meeting.
              </li>
            </ul>

            <Toggle
              label="Catch misspellings and spaced-out spellings"
              hint="Also matches near-misses and words broken up with spaces or punctuation. Finds more, and produces more false positives on ordinary words — which is why the review mode exists."
              checked={settings.redaction_fuzzy}
              onChange={(checked) => update({ redaction_fuzzy: checked })}
            />

            <p className="muted">
              Changing any of this affects meetings processed from now on. Use “Re-scan”
              on a meeting's transcript to apply it to one already processed.
            </p>

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

/**
 * A numeric setting with its own text draft.
 *
 * The draft matters: these are decimals, and several are negative. Binding
 * the input straight to the stored number means a half-typed `-` or a
 * trailing `.` parses as nothing and the value snaps out from under the
 * cursor. The draft is only replaced when the stored value genuinely differs
 * from what is typed, so `-2` on the way to `-24` survives.
 */
function NumberField({
  label,
  value,
  onChange,
  step,
  min,
  max,
  unit,
  hint,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
  unit?: string;
  hint?: string;
  disabled?: boolean;
}) {
  const [text, setText] = useState(() => String(value));

  useEffect(() => {
    setText((previous) =>
      previous.trim() !== "" && Number(previous) === value ? previous : String(value),
    );
  }, [value]);

  return (
    <label className="field">
      <span>
        {label}
        {unit && <em className="field-unit">{unit}</em>}
      </span>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={text}
        disabled={disabled}
        aria-label={unit ? `${label} in ${unit}` : label}
        onChange={(e) => {
          const next = e.currentTarget.value;
          setText(next);
          const parsed = numberOrNull(next);
          if (parsed !== null) onChange(parsed);
        }}
      />
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

/** A checkbox with the sentence that explains what turning it on costs. */
function Toggle({
  label,
  hint,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`field checkbox toggle${disabled ? " disabled" : ""}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
      <span>
        {label}
        {hint && <em className="field-hint">{hint}</em>}
      </span>
    </label>
  );
}
