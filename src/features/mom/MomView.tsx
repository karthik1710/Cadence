import { timecode } from "../../services/format";
import type { Evidenced, Mom } from "../../types";
import { Notice } from "../../components/ui";

interface Props {
  mom: Mom;
  /** Presentation label → user-assigned name, for renamed speakers. */
  renames: Record<string, string>;
  onSeek: (ms: number) => void;
}

/**
 * The §24 minutes view. Every generated item shows the transcript evidence
 * behind it where there is any, and clicking that evidence plays the moment it
 * came from — which is what makes the minutes checkable rather than something
 * the user has to take on trust.
 */
export function MomView({ mom, renames, onSeek }: Props) {
  const name = (label: string | null | undefined) =>
    label ? renames[label] || label : null;

  return (
    <div className="mom-view">
      <section>
        <h3>Executive summary</h3>
        <p>{mom.summary}</p>
      </section>

      <section>
        <h3>Participants</h3>
        {mom.participants.length === 0 ? (
          <p className="muted">No speakers were detected.</p>
        ) : (
          <ul className="inline-list">
            {mom.participants.map((participant) => (
              <li key={participant}>{name(participant)}</li>
            ))}
          </ul>
        )}
      </section>

      <EvidencedSection
        title="Topics discussed"
        items={mom.topics}
        onSeek={onSeek}
        name={name}
      />
      <EvidencedSection
        title="Key decisions"
        items={mom.decisions}
        onSeek={onSeek}
        name={name}
      />

      <section>
        <h3>Action items</h3>
        {mom.actionItems.length === 0 ? (
          <p className="muted">None recorded.</p>
        ) : (
          <ul className="action-items">
            {mom.actionItems.map((item, index) => (
              <li key={index}>
                <span className="action-description">{item.description}</span>
                <span className="action-meta">
                  <span>Owner: {name(item.owner) ?? "unassigned"}</span>
                  <span>Deadline: {item.deadline ?? "none"}</span>
                  <span className={`confidence ${item.confidence || "low"}`}>
                    {item.confidence || "low"} confidence
                  </span>
                  {item.timestamp !== null && (
                    <button
                      type="button"
                      className="link"
                      onClick={() => onSeek(item.timestamp as number)}
                    >
                      {timecode(item.timestamp)}
                    </button>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <EvidencedSection
        title="Open questions"
        items={mom.openQuestions}
        onSeek={onSeek}
        name={name}
      />
      <EvidencedSection
        title="Risks and concerns"
        items={mom.risks}
        onSeek={onSeek}
        name={name}
      />
      <EvidencedSection
        title="Detailed discussion"
        items={mom.detailedDiscussion}
        onSeek={onSeek}
        name={name}
      />

      {mom.meta && (
        <section className="mom-provenance">
          <h3>How these minutes were made</h3>
          <p className="muted">
            Generated on this machine by {mom.meta.model} ({mom.meta.quantization}) via{" "}
            {mom.meta.runtime}, {mom.meta.context_tokens} token context, from{" "}
            {mom.meta.chunk_count} transcript chunk(s) in {mom.meta.language}. Prompts{" "}
            {mom.meta.extraction_prompt} / {mom.meta.synthesis_prompt}.
          </p>
          {mom.meta.repaired && (
            <Notice kind="warning">
              The model's first answer was malformed and had to be repaired before it
              validated. Worth a closer read than usual.
            </Notice>
          )}
        </section>
      )}
    </div>
  );
}

function EvidencedSection({
  title,
  items,
  onSeek,
  name,
}: {
  title: string;
  items: Evidenced[];
  onSeek: (ms: number) => void;
  name: (label: string | null | undefined) => string | null;
}) {
  return (
    <section>
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p className="muted">None recorded.</p>
      ) : (
        <ul className="evidenced-list">
          {items.map((item, index) => (
            <li key={index}>
              <span>{item.text}</span>
              <span className="evidence">
                {item.speaker && <span className="muted">{name(item.speaker)}</span>}
                {item.timestamp !== null && (
                  <button
                    type="button"
                    className="link"
                    onClick={() => onSeek(item.timestamp as number)}
                    title="Play from here"
                  >
                    {timecode(item.timestamp)}
                  </button>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
