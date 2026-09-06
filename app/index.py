"""FAISS index over the trial corpus: build once, memory-map thereafter.

Why a real vector index rather than a similarity matrix: cosine similarity over
a NumPy array is O(n) per query and needs the whole embedding matrix resident.
At 114,900 documents that is workable; at ten times that it is not. FAISS gives
sub-linear search and a persisted artefact the service can load in a second
instead of re-embedding on every boot.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class Doc:
    nct_id: str
    title: str
    conditions: str
    interventions: str
    status: str
    url: str

    def to_text(self) -> str:
        """The string that gets embedded.

        Title carries the specific intervention, conditions carry the disease
        vocabulary. Interventions are included because a query naming a drug
        class should match trials of that drug even when the title does not
        repeat the condition.
        """
        return (f"{self.title}. Conditions: {self.conditions}. "
                f"Interventions: {self.interventions}")


def load_corpus(path: Path | None = None, limit: int = 0) -> list[Doc]:
    """Read the ClinicalTrials.gov export.

    Default csv quoting is deliberate: trial titles contain commas inside
    quotes, and QUOTE_NONE silently shreds ~39% of rows into misaligned fields.
    """
    path = path or settings.corpus_path
    docs: list[Doc] = []

    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            nct = (row.get("NCT Number") or "").strip()
            if not nct:
                continue
            docs.append(Doc(
                nct_id=nct,
                title=(row.get("Study Title") or "").strip(),
                conditions=(row.get("Conditions") or "").strip(),
                interventions=(row.get("Interventions") or "").strip(),
                status=(row.get("Study Status") or "").strip(),
                url=(row.get("Study URL") or "").strip(),
            ))
            if limit and len(docs) >= limit:
                break

    log.info("loaded %d documents from %s", len(docs), path)
    return docs


class TrialIndex:
    """Embedding index with persistence."""

    def __init__(self, index, docs: list[Doc], model):
        self.index = index
        self.docs = docs
        self.model = model

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, docs: list[Doc], model_name: str | None = None,
              batch_size: int = 256) -> "TrialIndex":
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name or settings.embed_model)
        log.info("embedding %d documents", len(docs))

        emb = model.encode(
            [d.to_text() for d in docs],
            batch_size=batch_size,
            normalize_embeddings=True,       # so inner product == cosine
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        # IndexFlatIP on normalised vectors is exact cosine similarity. Exact
        # beats approximate here: the corpus fits in memory, and an ANN index
        # would trade recall for speed we do not need.
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(index, docs, model)

    # ------------------------------------------------------------ persist

    def save(self, directory: Path | None = None) -> Path:
        import faiss

        directory = directory or settings.index_dir
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "trials.faiss"))
        with (directory / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d.__dict__) + "\n")

        log.info("index saved to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Path | None = None,
             model_name: str | None = None) -> "TrialIndex":
        import faiss
        from sentence_transformers import SentenceTransformer

        directory = directory or settings.index_dir
        index = faiss.read_index(str(directory / "trials.faiss"))

        docs = []
        with (directory / "docs.jsonl").open(encoding="utf-8") as f:
            for line in f:
                docs.append(Doc(**json.loads(line)))

        model = SentenceTransformer(model_name or settings.embed_model)
        log.info("index loaded: %d vectors", index.ntotal)
        return cls(index, docs, model)

    # ------------------------------------------------------------- search

    def search(self, query: str, k: int | None = None,
               open_only: bool = False) -> list[tuple[Doc, float]]:
        """Return (doc, score) ranked by cosine similarity.

        open_only over-fetches then filters, rather than filtering the corpus
        up front: keeping one index avoids rebuilding whenever trial statuses
        change, at the cost of retrieving a few extra candidates.
        """
        k = k or settings.top_k
        fetch = k * 6 if open_only else k

        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(q, min(fetch, self.index.ntotal))

        out: list[tuple[Doc, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            doc = self.docs[idx]
            if open_only and doc.status.upper() not in {
                "RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"
            }:
                continue
            out.append((doc, float(score)))
            if len(out) >= k:
                break
        return out
