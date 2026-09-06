import { useRef, useState } from "react";
import { streamAnswer, type Groundedness, type TrialHit } from "./api";
import { GroundednessBadge } from "./components/GroundednessBadge";
import { TrialCard } from "./components/TrialCard";

const EXAMPLES = [
  "breast cancer, stage II, HER2 positive",
  "non-small cell lung cancer with EGFR mutation",
  "metastatic melanoma, BRAF V600E",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState<TrialHit[]>([]);
  const [answer, setAnswer] = useState("");
  const [groundedness, setGroundedness] = useState<Groundedness | null>(null);
  const [tookMs, setTookMs] = useState<number | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function ask(q: string) {
    if (!q.trim() || streaming) return;

    // Clear everything first: leaving the previous answer visible while a new
    // one streams in is how people end up reading a stale result.
    setSources([]);
    setAnswer("");
    setGroundedness(null);
    setTookMs(null);
    setError(null);
    setStreaming(true);

    abortRef.current = new AbortController();

    try {
      await streamAnswer(
        q,
        {
          onSources: setSources,
          onToken: (t) => setAnswer((prev) => prev + t),
          onFinal: (g, ms) => {
            setGroundedness(g);
            setTookMs(ms);
          },
          onError: setError,
        },
        abortRef.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String(e));
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>Clinical Trial Search</h1>
        <p className="sub">
          Semantic retrieval over 114,900 ClinicalTrials.gov oncology trials,
          with every answer checked against what was actually retrieved.
        </p>
      </header>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Describe the patient: condition, stage, biomarkers…"
          aria-label="Clinical query"
          disabled={streaming}
        />
        <button type="submit" disabled={streaming || !question.trim()}>
          {streaming ? "Searching…" : "Search"}
        </button>
        {streaming && (
          <button
            type="button"
            className="secondary"
            onClick={() => abortRef.current?.abort()}
          >
            Stop
          </button>
        )}
      </form>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="chip"
            disabled={streaming}
            onClick={() => {
              setQuestion(ex);
              ask(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}

      {(answer || streaming) && (
        <section className="answer">
          <div className="answer-head">
            <h2>Answer</h2>
            {groundedness && <GroundednessBadge g={groundedness} />}
          </div>
          <p className="answer-body">
            {answer}
            {streaming && <span className="cursor" aria-hidden />}
          </p>
          {tookMs !== null && (
            <p className="timing">generated in {(tookMs / 1000).toFixed(1)}s</p>
          )}
        </section>
      )}

      {sources.length > 0 && (
        <section>
          <h2>
            Retrieved trials <span className="count">{sources.length}</span>
          </h2>
          <div className="cards">
            {sources.map((t) => (
              <TrialCard
                key={t.nct_id}
                trial={t}
                cited={groundedness?.cited_ids.includes(t.nct_id) ?? false}
              />
            ))}
          </div>
        </section>
      )}

      <footer>
        Decision support for search only. Not a medical device, and not a
        substitute for clinical eligibility review.
      </footer>
    </div>
  );
}
