import { useEffect, useState } from "react";
import { MAX_EXPECTED_SPEAKERS } from "../../types";

const NOT_SURE = "";
const MANY = "many";

interface Props {
  /** `null` is automatic speaker clustering — "Not sure" or "5 or more". */
  value: number | null;
  onChange: (value: number | null) => void;
  disabled?: boolean;
}

/**
 * The optional "How many people spoke?" hint for one meeting.
 *
 * Automatic clustering has no threshold that is right for every room — one
 * talker gets split into several speakers, two similar voices get merged — so
 * an organiser who knows the headcount of a *small* room can say so, and
 * diarization then produces exactly that many speakers.
 *
 * Only 1–4 are offered because only small counts were measured to work: a
 * fixed count of 7 on a real council recording found fewer distinct voices
 * than automatic clustering did (`MAX_EXPECTED_SPEAKERS` in
 * `pipeline/mod.rs` has the numbers). "5 or more" therefore sends `null`,
 * exactly like "Not sure" — it is a clearer answer for a big room, not a
 * different setting.
 */
export function SpeakerCountField({ value, onChange, disabled }: Props) {
  // Both "Not sure" and "5 or more" are `null`, so which one is showing is
  // this field's own state. Resets whenever a real count is chosen, here or by
  // the caller.
  const [many, setMany] = useState(false);
  useEffect(() => {
    if (value !== null) setMany(false);
  }, [value]);

  const selected = value !== null ? String(value) : many ? MANY : NOT_SURE;
  const counts = Array.from({ length: MAX_EXPECTED_SPEAKERS }, (_, i) => i + 1);

  return (
    <label className="field">
      <span>How many people spoke? (optional)</span>
      <select
        value={selected}
        disabled={disabled}
        aria-label="How many people spoke?"
        onChange={(e) => {
          const next = e.currentTarget.value;
          setMany(next === MANY);
          onChange(next === NOT_SURE || next === MANY ? null : Number(next));
        }}
      >
        <option value={NOT_SURE}>Not sure</option>
        {counts.map((n) => (
          <option key={n} value={String(n)}>
            {n}
          </option>
        ))}
        <option value={MANY}>{MAX_EXPECTED_SPEAKERS + 1} or more</option>
      </select>
      <span className="field-hint">
        Leave as Not sure unless the speaker labels come out wrong.
      </span>
      {selected === MANY && (
        <span className="field-hint">
          Cadence detects speakers automatically for larger groups.
        </span>
      )}
    </label>
  );
}
