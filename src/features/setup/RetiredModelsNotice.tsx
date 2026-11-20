import { useState } from "react";
import * as api from "../../services/api";
import { sizeLabel } from "../../services/format";
import { Notice } from "../../components/ui";
import type { RetiredModel } from "../../types";

interface Props {
  models: RetiredModel[];
  /** A download is running; the backend refuses removal meanwhile. */
  busy?: boolean;
  /** Re-reads the model state once something has been removed. */
  onRemoved: () => void | Promise<void>;
}

/**
 * Names models the current set no longer uses — typically a model from the
 * old memory tiers left behind by an upgrade — and offers to delete them.
 *
 * Nothing is deleted without two clicks: Remove opens a confirm step in the
 * page itself, beside the item it concerns, and only Delete there calls the
 * backend. (WebView2 on Windows does show a native `window.confirm` — checked
 * 2026-11-19 — so this is a design choice, not a workaround.)
 */
export function RetiredModelsNotice({ models, busy = false, onRemoved }: Props) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (models.length === 0 && !error) return null;

  const remove = async (model: RetiredModel) => {
    setError(null);
    setRemoving(model.key);
    try {
      await api.removeRetiredModel(model.key);
      setConfirming(null);
      await onRemoved();
    } catch (err) {
      setError(api.errorMessage(err));
    } finally {
      setRemoving(null);
    }
  };

  return (
    <>
      {models.map((model) => {
        const size = sizeLabel(model.size_mb);
        if (confirming === model.key) {
          return (
            <Notice key={model.key} kind="warning">
              <span className="grow">
                Delete {model.label} from this computer? This frees {size} and can't be
                undone.
              </span>
              <button
                type="button"
                className="danger"
                disabled={busy || removing !== null}
                onClick={() => remove(model)}
              >
                {removing === model.key ? "Deleting…" : "Delete"}
              </button>
              <button
                type="button"
                className="ghost"
                disabled={removing === model.key}
                onClick={() => setConfirming(null)}
              >
                Cancel
              </button>
            </Notice>
          );
        }
        return (
          <Notice key={model.key} kind="info">
            <span className="grow">
              {model.key === "bin/**"
                ? `Runtimes downloaded by an earlier version of Cadence are still in its models folder (${size}). This version includes its own copies, so these are no longer used.`
                : model.key === "bin/*"
                ? `Runtime files from an older version of Cadence are still in its models folder (${size}). Each runtime now has a folder of its own, so these are no longer used.`
                : model.from_previous_version
                ? `An older model from a previous version is still installed (${model.label}, ${size}). Cadence no longer uses it.`
                : `A file Cadence doesn't use is in its models folder (${model.label}, ${size}).`}
            </span>
            <button
              type="button"
              className="ghost"
              disabled={busy || removing !== null}
              onClick={() => {
                setError(null);
                setConfirming(model.key);
              }}
            >
              Remove
            </button>
          </Notice>
        );
      })}
      {error && <Notice kind="error">{error}</Notice>}
    </>
  );
}
