# Clinical Trial RAG — grounded answering over 114,900 trials

A running FastAPI service that retrieves oncology clinical trials by meaning and
answers questions about them **without being allowed to make things up**.

Patients and coordinators search ClinicalTrials.gov by keyword. Keyword search
fails the way clinical language fails: a trial titled *NSCLC* does not match a
search for *lung cancer*. This does semantic retrieval instead, then generates a
grounded answer — and checks the answer against what was actually retrieved
before serving it.

**Stack:** FastAPI · FAISS · sentence-transformers · PyTorch · SSE streaming · Docker · GitHub Actions

---

## Measured results

Six oncology queries, k = 5, over 114,900 indexed trials:

| Metric | Result |
|---|---|
| Mean Precision@5 | **0.867** |
| Mean retrieval latency | **13.2 ms** |
| Mean generation latency | 2,253 ms |
| **Answers citing a fabricated NCT id** | **0 of 6** |
| Grounded rate | 1.00 |
| Mean lexical overlap with context | 0.972 |

Reproduce with `python eval/run_eval.py`; raw output in
[`eval/results.json`](eval/results.json).

**Read the groundedness numbers carefully.** Six queries is a small sample, and
the default 0.5B model writes short answers — the mean answer is under 100
characters. Short answers are *easy* to keep grounded, so 1.00 is a floor
result, not proof the guardrail is strong. It demonstrates the mechanism works;
it does not establish a rate.

### Live request

```bash
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"breast cancer stage II HER2 positive","top_k":3,"open_only":true}'
```

```json
{
  "query": "breast cancer stage II HER2 positive",
  "took_ms": 25.1,
  "results": [
    {"nct_id": "NCT06603597", "title": "HER2-positive Breast Cancer Registry",
     "status": "RECRUITING", "score": 0.8558},
    {"nct_id": "NCT06762977", "title": "A Composite Assay for HER2-positive Early-stage Breast Cancer Management",
     "status": "RECRUITING", "score": 0.8423},
    {"nct_id": "NCT06533670", "title": "TME Alteration in HER2 Positive Breast Cancer",
     "status": "RECRUITING", "score": 0.7843}
  ]
}
```

---

## The guardrail is the point

A RAG service that retrieves well and then invents an NCT number is worse than
no service, because the invented answer is confident and checkable-looking. Two
deterministic checks run on every generated answer, neither using a second model
— *asking the model whether the model was right* is not a control.

**Citation validity.** Every `NCT\d{8}` in the answer must appear in the
retrieved set. One that does not is a fabrication, and fails hard regardless of
anything else.

**Lexical grounding.** The share of the answer's content words present in the
retrieved context, stopwords removed. This is a blunt instrument: it will flag a
correct answer that paraphrases heavily and pass a wrong answer that reuses
context vocabulary. It is reported as a score, not a verdict, for exactly that
reason.

```python
{"grounded": true, "overlap": 0.972, "cited_ids": ["NCT06603597"],
 "hallucinated_ids": [], "unsupported_terms": []}
```

Sent as the final SSE event, after the full answer exists — groundedness cannot
be judged from a partial answer.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Status, document count, active backend, uptime |
| `/search` | POST | Retrieval only. No model call, tens of milliseconds |
| `/answer` | POST | Retrieval + grounded generation, streamed over SSE |
| `/docs` | GET | OpenAPI UI, generated from the pydantic schemas |

`/answer` emits three event types: `sources` first so the client can render
citations before any text arrives, then `token` events as generation proceeds,
then `final` carrying the groundedness report.

Streaming is not decoration. Generation takes seconds, and a user watching a
blank page assumes it broke.

---

## Design decisions

**Pluggable LLM backend.** `local` (in-process transformers) is the default
because a service nobody can run without buying an API key is not a
demonstrable service. `ollama` and `openai` are drop-in via one environment
variable, because production would not want a 0.5B model doing clinical
summarisation.

```bash
LLM_BACKEND=local   # default, no key
LLM_BACKEND=ollama  OLLAMA_MODEL=llama3.2
LLM_BACKEND=openai  OPENAI_API_KEY=sk-...
```

**Exact search, not approximate.** `IndexFlatIP` over normalised vectors is
exact cosine similarity. The corpus fits in memory and queries take 13 ms, so an
ANN index would trade recall for speed that is not needed.

**Structured filters run after retrieval, not before.** `open_only` over-fetches
then filters, rather than maintaining a second index. Trial statuses change
weekly; index rebuilds should not have to.

**Index is a mounted artefact, not baked into the image.** 168 MB of binary has
no business in a container layer, and it can be rebuilt without rebuilding the
image.

**Non-root container.** The service reads a corpus and serves HTTP. It never
needs root.

---

## Limitations

- **The corpus has no eligibility criteria.** ClinicalTrials.gov exposes them,
  this export does not. Without inclusion/exclusion text the service matches on
  *topic*, not eligibility — it cannot tell you whether a patient qualifies, only
  which trials are about their condition. That is the single biggest gap between
  this and something clinically useful.
- **Precision@k uses a weak label** — does the trial's Conditions field mention
  the queried condition. It misses trials listed as "Solid Tumor" for a lung
  query, and accepts trials a patient is ineligible for.
- **The default model is small.** Qwen2.5-0.5B keeps the service runnable
  anywhere; answer quality is correspondingly modest. Point `LLM_BACKEND` at
  something larger for real use.
- **Not a medical device.** Decision support for search, nothing more.

---

## Running it

```bash
pip install -r requirements.txt

python scripts/build_index.py        # ~10 min, 114,900 embeddings
uvicorn app.main:app --reload

python eval/run_eval.py              # reproduce the table above
pytest tests/ -q                     # 15 tests
```

Or with Docker:

```bash
docker compose up --build
docker compose --profile ollama up   # with a larger local model
```

## Layout

```
app/
  main.py        FastAPI app, SSE streaming, structured logging
  index.py       FAISS build/load/search, corpus parsing
  generate.py    local | ollama | openai backends behind one interface
  guardrail.py   citation validity + lexical grounding
  schemas.py     pydantic request/response models
  config.py      environment-driven settings
eval/run_eval.py retrieval + groundedness evaluation
tests/           15 tests, retrieval stubbed
```

## Data

[ClinicalTrials.gov](https://clinicaltrials.gov/) oncology export, 114,900
trials. Public; not redistributed here.

## License

MIT — see [LICENSE](LICENSE).
