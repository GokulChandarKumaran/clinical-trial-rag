import type { Groundedness } from "../api";

/**
 * Surfaces the guardrail result to the user.
 *
 * A fabricated citation is shown as a distinct, louder state than merely-low
 * overlap, because they mean different things: low overlap is "this answer
 * paraphrased a lot", a fabricated NCT id is "this answer invented a trial".
 */
export function GroundednessBadge({ g }: { g: Groundedness }) {
  if (g.hallucinated_ids.length > 0) {
    return (
      <span className="badge bad" title={`Not in retrieved set: ${g.hallucinated_ids.join(", ")}`}>
        ⚠ Fabricated citation ({g.hallucinated_ids.length})
      </span>
    );
  }
  const pct = Math.round(g.overlap * 100);
  return (
    <span
      className={`badge ${g.grounded ? "good" : "warn"}`}
      title={`${pct}% of the answer's content words appear in the retrieved trials`}
    >
      {g.grounded ? "✓ Grounded" : "◐ Weakly grounded"} · {pct}%
    </span>
  );
}