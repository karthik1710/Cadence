import { useCallback, useEffect, useState } from "react";
import * as api from "../services/api";

export type Theme = "light" | "dark";

const STORAGE_KEY = "echomind-theme";

/** Anything other than exactly "light" is dark — the fail-safe default. */
function normalize(value: string | null | undefined): Theme {
  return value === "light" ? "light" : "dark";
}

function cachedTheme(): Theme {
  try {
    return normalize(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return "dark";
  }
}

/** Applies the theme to the DOM immediately and updates the paint cache that
 * `index.html`'s inline bootstrap script reads on the next cold start. This
 * never talks to the backend — see `persist` for that. */
function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage can throw (private mode, disabled site data, ...). The theme
    // still applies to the DOM this session; it just won't be cached for
    // next launch, which only costs a possible one-time flash — not fatal.
  }
}

/**
 * Owns the current theme end to end:
 *  - instant DOM + localStorage-cache application (no IPC wait), and
 *  - the backend `Settings.theme` round trip, which is the source of truth.
 *
 * The persistence step always reads fresh settings and patches only `theme`
 * before saving — never a settings object cached elsewhere in the app —
 * so it can never clobber edits in-flight on the Settings screen (see
 * SettingsView, which keeps its own independently-loaded Settings state).
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const domTheme = document.documentElement.dataset.theme;
    return domTheme ? normalize(domTheme) : cachedTheme();
  });
  const [persistError, setPersistError] = useState<string | null>(null);

  // Reconcile against the backend once it's known. The bootstrap script and
  // the localStorage cache only ever paint a guess; if the backend disagrees
  // (first run ever, or the cache was cleared / a different machine), the
  // backend wins.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const settings = await api.getSettings();
        if (cancelled) return;
        const backendTheme = normalize(settings.theme);
        if (backendTheme !== normalize(document.documentElement.dataset.theme)) {
          applyTheme(backendTheme);
          setTheme(backendTheme);
        }
      } catch {
        // Could not read settings — keep whatever is already painted rather
        // than surfacing an error for a purely cosmetic reconciliation.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // A failed save is low-stakes (the toggle already applied visually) —
  // auto-dismiss so it doesn't linger.
  useEffect(() => {
    if (!persistError) return;
    const id = window.setTimeout(() => setPersistError(null), 6000);
    return () => window.clearTimeout(id);
  }, [persistError]);

  const setThemeAndPersist = useCallback(async (next: Theme) => {
    applyTheme(next);
    setTheme(next);
    setPersistError(null);
    try {
      // Fetch-modify-save, every time — never trust a copy of Settings held
      // elsewhere, which may be stale relative to unsaved edits in progress
      // on the Settings screen.
      const current = await api.getSettings();
      await api.saveSettings({ ...current, theme: next });
    } catch (err) {
      setPersistError(api.errorMessage(err));
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeAndPersist(theme === "light" ? "dark" : "light");
  }, [theme, setThemeAndPersist]);

  const dismissPersistError = useCallback(() => setPersistError(null), []);

  // For flows that already persisted a theme value themselves (the Settings
  // screen's own "Save settings" button, which read-modify-writes its own
  // full local copy) — apply it live without a redundant second save.
  const syncTheme = useCallback((value: string) => {
    const next = normalize(value);
    applyTheme(next);
    setTheme(next);
  }, []);

  return { theme, toggleTheme, persistError, dismissPersistError, syncTheme };
}
