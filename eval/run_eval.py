"""Evaluate retrieval quality and answer groundedness.

Two things are measured, because they fail independently: retrieval can surface
the wrong trials, and generation can invent things about the right ones.

RETRIEVAL — Precision@k against a weak relevance label (does the trial's
Conditions field mention the queried condition). A proxy, and stated as one: it
misses trials listed as "Solid Tumor" for a lung-cancer query, and accepts
trials the patient is ineligible for. It is used because it is derivable from
the data, unlike true clinical relevance.

GROUNDEDNESS — the share of generated answers that cite only real NCT ids from
the retrieved set and stay lexically anchored to it. This one is not a proxy:
a fabricated NCT number is unambiguously wrong.

    python eval/run_eval.py --limit-queries 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import guardrail
from app.generate import build_prompt, get_backend
from app.index import TrialIndex

QUERIES = [
    ("breast cancer, stage II, HER2 positive", "breast"),
    ("non-small cell lung cancer with EGFR mutation", "lung"),
    ("metastatic melanoma, BRAF V600E", "melanoma"),
    ("colorectal cancer, stage III", "colorectal"),
    ("prostate cancer, BRCA2 mutation", "prostate"),
    ("pancreatic cancer, stage IV", "pancrea"),
]


def precision_at_k(hits, term: str) -> float:
    if not hits:
        return 0.0
    relevant = sum(1 for d, _ in hits
                   if term in (d.conditions + " " + d.title).lower())
    return relevant / len(hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit-queries", type=int, default=len(QUERIES))
    ap.add_argument("--skip-generation", action="store_true",
                    help="retrieval metrics only; no model load")
    ap.add_argument("--out", type=Path, default=Path("eval/results.json"))
    args = ap.parse_args()

    index = TrialIndex.load()
    print(f"index: {index.index.ntotal:,} documents\n")

    backend = None if args.skip_generation else get_backend()
    rows = []

    for query, term in QUERIES[:args.limit_queries]:
        t0 = time.perf_counter()
        hits = index.search(query, k=args.k, open_only=True)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        row = {
            "query": query,
            "precision_at_k": round(precision_at_k(hits, term), 3),
            "top_score": round(hits[0][1], 4) if hits else 0.0,
            "retrieval_ms": round(retrieval_ms, 1),
        }

        if backend is not None:
            docs = [d for d, _ in hits]
            t0 = time.perf_counter()
            answer = "".join(backend.stream(build_prompt(query, docs)))
            gen_ms = (time.perf_counter() - t0) * 1000

            report = guardrail.check(answer, docs)
            row.update({
                "generation_ms": round(gen_ms, 0),
                "grounded": report.grounded,
                "overlap": round(report.overlap, 3),
                "cited": len(report.cited_ids),
                "hallucinated_ids": report.hallucinated_ids,
                "answer_chars": len(answer),
            })

        rows.append(row)
        print(json.dumps(row, indent=2))

    n = len(rows)
    summary = {
        "queries": n,
        "k": args.k,
        "mean_precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 3),
        "mean_retrieval_ms": round(sum(r["retrieval_ms"] for r in rows) / n, 1),
    }
    if backend is not None:
        summary.update({
            "grounded_rate": round(sum(r["grounded"] for r in rows) / n, 3),
            "mean_overlap": round(sum(r["overlap"] for r in rows) / n, 3),
            "answers_with_fabricated_ids":
                sum(1 for r in rows if r["hallucinated_ids"]),
            "mean_generation_ms": round(sum(r["generation_ms"] for r in rows) / n, 0),
        })

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_query": rows}, indent=2),
                        encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
