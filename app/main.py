"""FastAPI service: retrieval, grounded generation, streaming, health.

Endpoint design follows what the pieces actually cost. /search is fast and
synchronous. /answer runs a language model and takes seconds, so it streams
over SSE — and the guardrail result is sent as a final event once the full
answer exists, because groundedness cannot be judged from a partial answer.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import guardrail
from .config import settings
from .generate import build_prompt, get_backend
from .index import TrialIndex
from .schemas import (
    AnswerRequest,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    TrialHit,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("app")

state: dict = {"index": None, "started": time.time()}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the index once at startup, not per request. A cold load is seconds;
    # per-request would make the service unusable.
    try:
        state["index"] = TrialIndex.load()
        log.info("index ready: %d vectors", state["index"].index.ntotal)
    except Exception as exc:                       # pragma: no cover
        log.error("index unavailable: %s — run scripts/build_index.py", exc)
    yield


app = FastAPI(
    title="Clinical Trial RAG",
    description="Semantic retrieval and grounded answering over ClinicalTrials.gov.",
    version="1.0.0",
    lifespan=lifespan,
)


def _index() -> TrialIndex:
    if state["index"] is None:
        raise HTTPException(503, "Index not built. Run scripts/build_index.py first.")
    return state["index"]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    idx = state["index"]
    return HealthResponse(
        status="ok" if idx else "degraded",
        documents=idx.index.ntotal if idx else 0,
        embed_model=settings.embed_model,
        llm_backend=settings.llm_backend,
        uptime_seconds=round(time.time() - state["started"], 1),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Retrieval only. No model call, so this stays in the tens of milliseconds."""
    t0 = time.perf_counter()
    hits = _index().search(req.query, k=req.top_k, open_only=req.open_only)
    took = (time.perf_counter() - t0) * 1000

    log.info("search q=%r k=%d hits=%d ms=%.1f", req.query, req.top_k, len(hits), took)
    return SearchResponse(
        query=req.query,
        took_ms=round(took, 1),
        results=[
            TrialHit(nct_id=d.nct_id, title=d.title, conditions=d.conditions,
                     status=d.status, url=d.url, score=round(s, 4))
            for d, s in hits
        ],
    )


@app.post("/answer")
def answer(req: AnswerRequest) -> StreamingResponse:
    """Retrieve, then generate an answer grounded in what was retrieved.

    Server-sent events. Three event types:
      sources  — the retrieved trials, sent first so the client can render
                 citations before any text arrives
      token    — generated text, incrementally
      final    — the groundedness report, once the whole answer exists
    """
    index = _index()
    hits = index.search(req.question, k=req.top_k, open_only=req.open_only)

    if not hits:
        raise HTTPException(404, "No trials matched the question.")

    docs = [d for d, _ in hits]
    prompt = build_prompt(req.question, docs)

    def events():
        sources = [
            {"nct_id": d.nct_id, "title": d.title, "url": d.url,
             "status": d.status, "score": round(s, 4)}
            for d, s in hits
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        t0 = time.perf_counter()
        chunks: list[str] = []
        try:
            for chunk in get_backend(req.backend).stream(prompt):
                chunks.append(chunk)
                yield f"event: token\ndata: {json.dumps({'t': chunk})}\n\n"
        except Exception as exc:                   # pragma: no cover
            log.error("generation failed: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        full = "".join(chunks)
        report = guardrail.check(full, docs)
        took = (time.perf_counter() - t0) * 1000

        log.info("answer q=%r grounded=%s overlap=%.2f hallucinated=%d ms=%.0f",
                 req.question, report.grounded, report.overlap,
                 len(report.hallucinated_ids), took)

        yield ("event: final\ndata: "
               + json.dumps({"groundedness": report.as_dict(),
                             "took_ms": round(took, 1)})
               + "\n\n")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Static frontend
#
# Mounted last so it cannot shadow the API routes above. Only mounted if the
# build output exists, so running the API alone (in dev, or in CI) does not
# require a frontend build to have happened.
# ---------------------------------------------------------------------------
_WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
    log.info("serving frontend from %s", _WEB_DIST)
else:
    log.info("no frontend build at %s — API only", _WEB_DIST)
