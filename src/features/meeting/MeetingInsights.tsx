// The read-out side of the audio profiler and the meeting classifier.
// Presentational only: everything here is handed a finished artifact (or
// null, for a meeting recorded before the artifact existed) and draws it.

import { decibels, duration, percent, timecode } from "../../services/format";
import { Card, Notice, Spinner } from "../../components/ui";
import type { AudioProfile, MeetingClassification, ProfileFrame } from "../../types";

/** dBFS floor for the timeline and the level bars. Below this a meeting
 * recording is silence for practical purposes. */
const FLOOR_DBFS = -60;

function levelFraction(dbfs: number): number {
  if (!Number.isFinite(dbfs)) return 0;
  const fraction = (dbfs - FLOOR_DBFS) / -FLOOR_DBFS;
  return Math.min(1, Math.max(0, fraction));
}

function gradeTone(grade: string): string {
  switch (grade) {
    case "excellent":
    case "good":
      return "success";
    case "fair":
      return "warn";
    default:
      return "error";
  }
}

/**
 * What the microphone actually captured, in the terms someone can act on:
 * a score, the numbers behind it, a picture of the level over time, and the
 * named noises with what to do about each.
 */
export function AudioQualityPanel({
  profile,
  loading,
  error,
}: {
  profile: AudioProfile | null;
  loading?: boolean;
  error?: string | null;
}) {
  if (loading) {
    return (
      <Card title="Audio quality">
        <p className="notice">
          <Spinner />
          Measuring the recording
        </p>
      </Card>
    );
  }
  if (!profile) {
    return null;
  }

  const score = Math.max(0, Math.min(100, Math.round(profile.quality_score)));
  const tone = gradeTone(profile.quality_grade);
  const noise = profile.noise;

  return (
    <Card title="Audio quality">
      {error && <Notice kind="warning">{error}</Notice>}

      <div className="quality-head">
        <div className={`quality-score ${tone}`}>
          <span className="quality-score-value">{score}</span>
          <span className="quality-score-max">/100</span>
        </div>
        <div className="quality-head-text">
          <span className={`badge ${tone}`}>{profile.quality_grade}</span>
          <p className="muted">
            {duration(profile.duration_ms)} · {profile.sample_rate.toLocaleString()} Hz ·{" "}
            {profile.channels === 1 ? "mono" : `${profile.channels} channels`}
          </p>
          <div className="quality-bar" role="presentation">
            <span className={`quality-bar-fill ${tone}`} style={{ width: `${score}%` }} />
          </div>
        </div>
      </div>

      <dl className="host-grid quality-grid">
        <Metric
          label="Speech level"
          value={decibels(profile.speech_dbfs)}
          note="90th-percentile frame — how loud the talking is"
        />
        <Metric label="Peak" value={decibels(profile.peak_dbfs)} note="Loudest single sample" />
        <Metric
          label="Average level"
          value={decibels(profile.rms_dbfs)}
          note="Whole file, silences included"
        />
        <Metric
          label="Noise floor"
          value={decibels(profile.noise_floor_dbfs)}
          note="The room with nobody speaking"
        />
        <Metric
          label="Speech-band SNR"
          value={decibels(profile.speech_band_snr_db, "dB")}
          note="Measured over 300–3400 Hz only"
        />
        <Metric
          label="Crest factor"
          value={decibels(profile.crest_factor_db, "dB")}
          note="Peak above average — low means squashed"
        />
        <Metric
          label="Clipping"
          value={`${profile.clipped_percent.toFixed(2)}%`}
          note={`${profile.clipped_samples.toLocaleString()} samples at the ceiling`}
        />
        <Metric
          label="Silence"
          value={`${profile.silence_percent.toFixed(1)}%`}
          note="Of the recording with nobody talking"
        />
      </dl>

      <LevelTimeline frames={profile.timeline} durationMs={profile.duration_ms} />

      {noise && noise.findings.length > 0 && (
        <section className="noise-findings">
          <h3>What is in the background</h3>
          <ul>
            {noise.findings.map((finding) => (
              <li key={finding.kind} className={`noise-finding ${finding.kind}`}>
                <div className="noise-finding-head">
                  <span className="noise-label">{finding.label}</span>
                  <span
                    className="marker"
                    title="How confident the detector is that this is really present"
                  >
                    {percent(finding.confidence)} confident
                  </span>
                </div>
                <p className="noise-detail">{finding.detail}</p>
                <p className="noise-advice">{finding.advice}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      {noise && (
        <p className="muted">
          Noise floor {decibels(noise.noise_floor_dbfs)}, speech {decibels(
            noise.speech_to_noise_db,
            "dB",
          )}{" "}
          above it.
          {noise.mains_hz !== null && ` Mains hum at ${noise.mains_hz} Hz.`}
          {noise.decay_ms !== null && ` Reverberation decays over ~${Math.round(noise.decay_ms)} ms.`}
          {noise.transients_per_minute > 0 &&
            ` ${noise.transients_per_minute.toFixed(1)} knocks or bumps per minute.`}
        </p>
      )}

      {profile.warnings.length > 0 && (
        <ul className="quality-warnings">
          {profile.warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      )}

      <p className="muted">
        Measured from <code>original.wav</code>. Nothing here changes the recording — the
        processing pipeline works on a copy.
      </p>
    </Card>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
      <dd className="metric-note">{note}</dd>
    </div>
  );
}

/**
 * The level over time, as one bar per bucket: a faint bar for the peak and a
 * solid one for the RMS, with clipped buckets tinted. Hand-drawn with divs
 * on purpose — a chart library would be a dependency, a network font and a
 * theme of its own, for something that is fundamentally 120 rectangles.
 */
function LevelTimeline({
  frames,
  durationMs,
}: {
  frames: ProfileFrame[];
  durationMs: number;
}) {
  if (!frames || frames.length === 0) return null;

  const loudest = frames.reduce((best, frame) => Math.max(best, frame.peak_dbfs), FLOOR_DBFS);

  return (
    <section className="level-timeline">
      <h3>Level over the meeting</h3>
      <div
        className="level-strip"
        role="img"
        aria-label={`Recording level over ${duration(durationMs)}, in ${frames.length} steps. Loudest peak ${decibels(loudest)}.`}
      >
        {frames.map((frame, index) => {
          const clipped = frame.clipped_percent > 0;
          return (
            <span
              key={index}
              className={`level-bar${clipped ? " clipped" : ""}`}
              title={`${timecode(frame.start_ms)} — ${decibels(frame.rms_dbfs)} average, ${decibels(
                frame.peak_dbfs,
              )} peak${clipped ? `, ${frame.clipped_percent.toFixed(1)}% clipped` : ""}`}
            >
              <span
                className="level-bar-peak"
                style={{ height: `${levelFraction(frame.peak_dbfs) * 100}%` }}
              />
              <span
                className="level-bar-rms"
                style={{ height: `${levelFraction(frame.rms_dbfs) * 100}%` }}
              />
            </span>
          );
        })}
      </div>
      <div className="level-axis">
        <span>{timecode(0)}</span>
        <span className="muted">
          full height is 0 dBFS, the bottom is {FLOOR_DBFS} dBFS
        </span>
        <span>{timecode(durationMs)}</span>
      </div>
    </section>
  );
}

/** The detected meeting type, how sure the classifier is, and what the
 * meeting kept coming back to. */
export function ClassificationPanel({
  classification,
  loading,
  onClassify,
  busy,
  canClassify,
}: {
  classification: MeetingClassification | null;
  loading?: boolean;
  onClassify?: () => void;
  busy?: boolean;
  canClassify?: boolean;
}) {
  if (loading) return null;

  if (!classification) {
    // Older meetings were processed before the classifier existed; offer to
    // run it rather than showing an empty panel with no way forward.
    if (!canClassify || !onClassify) return null;
    return (
      <Card title="Meeting type">
        <p className="muted">
          This meeting has not been classified. It is a count over the transcript —
          no model is loaded and it takes a moment.
        </p>
        <div className="button-row">
          <button type="button" className="ghost" onClick={onClassify} disabled={busy}>
            {busy ? "Working…" : "Detect meeting type"}
          </button>
        </div>
      </Card>
    );
  }

  const signals = classification.signals;
  const confidence = Math.round(classification.confidence * 100);

  return (
    <Card
      title="Meeting type"
      action={
        onClassify && (
          <button type="button" className="ghost" onClick={onClassify} disabled={busy}>
            {busy ? "Working…" : "Re-detect"}
          </button>
        )
      }
    >
      <div className="classification-head">
        <span className="classification-label">{classification.label}</span>
        <span
          className="confidence-pill"
          title="How far ahead this type scored over the next best one"
        >
          {confidence}% confident
        </span>
      </div>

      {classification.alternatives.length > 0 && (
        <p className="muted">
          Also considered:{" "}
          {classification.alternatives
            .slice(0, 3)
            .map((alt) => alt.label)
            .join(", ")}
          .
        </p>
      )}

      {classification.topics.length > 0 && (
        <>
          <h3>What it kept coming back to</h3>
          <div className="topic-chips">
            {classification.topics.map((topic) => (
              <span
                key={topic.term}
                className="chip topic-chip"
                title={`Said ${topic.count} ${
                  topic.count === 1 ? "time" : "times"
                }, across ${percent(topic.spread)} of the meeting`}
              >
                {topic.term}
                <span className="topic-count">{topic.count}</span>
              </span>
            ))}
          </div>
        </>
      )}

      <dl className="host-grid signals-grid">
        <div>
          <dt>Speakers</dt>
          <dd>{signals.speaker_count}</dd>
        </div>
        <div>
          <dt>Turns</dt>
          <dd>{signals.turn_count}</dd>
        </div>
        <div>
          <dt>Words</dt>
          <dd>{signals.word_count.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Words per turn</dt>
          <dd>{signals.mean_turn_words.toFixed(1)}</dd>
        </div>
        <div>
          <dt>Questions</dt>
          <dd>{percent(signals.question_ratio)} of turns</dd>
        </div>
        <div>
          <dt>Busiest speaker</dt>
          <dd>{percent(signals.dominant_speaker_share)} of the talking</dd>
        </div>
      </dl>

      <p className="muted">
        Worked out from the shape of the conversation — its length, who spoke and how
        often, and which words recur. It is a hint for filing the meeting, not a
        judgement about it.
      </p>
    </Card>
  );
}
