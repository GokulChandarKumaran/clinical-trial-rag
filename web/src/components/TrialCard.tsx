import type { TrialHit } from "../api";

export function TrialCard({ trial, cited }: { trial: TrialHit; cited: boolean }) {
  return (
    <article className={`card${cited ? " cited" : ""}`}>
      <div className="card-top">
        <a href={trial.url} target="_blank" rel="noreferrer" className="nct">
          {trial.nct_id}
        </a>
        <span className={`status ${trial.status.toLowerCase()}`}>
          {trial.status.replace(/_/g, " ").toLowerCase()}
        </span>
      </div>
      <h3>{trial.title}</h3>
      {trial.conditions && <p className="conditions">{trial.conditions}</p>}
      <div className="card-foot">
        <span className="score" title="cosine similarity to the query">
          {trial.score.toFixed(3)}
        </span>
        {cited && <span className="cited-tag">cited in answer</span>}
      </div>
    </article>
  );
}