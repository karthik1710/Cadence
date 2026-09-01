import { useEffect, useState } from "react";
import * as api from "../services/api";
import { sizeLabel } from "../services/format";
import { Card, Notice, Spinner } from "./ui";
import type { HostReport, ModelStatus } from "../types";

interface Props {
  /// Catalog entries this machine still needs — usually
  /// `SetupStatus.missing` or `ModelStatus[]` filtered to `required && !installed`.
  missing: ModelStatus[];
  /// Re-checks setup readiness once the batch finishes, successfully or not.
  onChanged: () => void;
  /// Optional escape hatch to the full model table, for anyone who wants to
  /// pick something other than the recommendation by hand.
  onOpenSettings?: () => void;
}

/**
 * The one-click "install what this machine needs" action.
 *
 * Cadence already picks the right model for the machine's RAM tier —
 * `Settings.stt_model`/`llm_model` default to `null`, which resolves to
 * `MemoryProfile::default_*` — so there is nothing to *choose* here, only
 * something to *fetch*. This exists so a first-time user does not have to
 * find the Models table and click Install on four separate rows to reach
 * that already-correct default.
 */
export function RecommendedSetupBanner({ missing, onChanged, onOpenSettings }: Props) {
  const [host, setHost] = useState<HostReport | null>(null);
  const [installing, setInstalling] = useState(false);
  const [current, setCurrent] = useState<{ label: string; index: number; total: number; percent: number } | null>(
    null,
  );
  const [failures, setFailures] = useState<api.BatchInstallFailure[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .getHostInfo()
      .then((report) => {
        if (!cancelled) setHost(report);
      })
      .catch(() => {
        // The banner still works without host copy — it just loses the
        // "for your Nâ€¯GB machine" framing.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (missing.length === 0) return null;

  const totalMb = missing.reduce((sum, model) => sum + model.approx_size_mb, 0);
  const idsToInstall = missing.map((model) => model.id);

  const install = async () => {
    setInstalling(true);
    setFailures([]);
    setCurrent(null);
    const result = await api.installMissingModels(idsToInstall, (step) => {
      const label = missing.find((m) => m.id === step.modelId)?.label ?? step.modelId;
      setCurrent({
        label,
        index: step.index + 1,
        total: step.total,
        percent: step.progress.percent,
      });
    });
    setInstalling(false);
    setCurrent(null);
    setFailures(result);
    onChanged();
  };

  return (
    <Card>
      <div className="recommended-setup">
        <div className="recommended-setup-copy">
          <h2>Recommended setup for this machine{host ? ` (${sizeLabel(host.total_ram_mb)} RAM)` : ""}</h2>
          <p className="muted">
            Cadence picks the speech and language models that fit this machine's memory
            automatically — {host ? `${host.recommended_stt} for speech and ${host.recommended_llm} for minutes` : "a smaller or larger model depending on available RAM"}
            . {missing.length} component{missing.length === 1 ? "" : "s"} still need
            downloading (about {sizeLabel(totalMb)} total), one time, after which everything
            runs offline.
          </p>
        </div>
        <div className="recommended-setup-action">
          <button type="button" className="primary" disabled={installing} onClick={install}>
            {installing ? "Installing…" : "Install recommended setup"}
          </button>
          {onOpenSettings && !installing && (
            <button type="button" className="link" onClick={onOpenSettings}>
              Choose different models
            </button>
          )}
        </div>
      </div>

      {installing && (
        <div className="recommended-setup-progress">
          <p className="notice">
            <Spinner />
            {current
              ? `${current.label} — item ${current.index} of ${current.total} (${current.percent}%)`
              : "Starting…"}
          </p>
          <span className="progress-track">
            <span
              className="progress-fill"
              style={{
                width: current ? `${((current.index - 1) / current.total) * 100 + current.percent / current.total}%` : "0%",
              }}
            />
          </span>
        </div>
      )}

      {failures.length > 0 && (
        <Notice kind="error">
          {failures.length} component{failures.length === 1 ? "" : "s"} could not be
          installed: {failures.map((f) => f.error).join("; ")}
        </Notice>
      )}
    </Card>
  );
}
