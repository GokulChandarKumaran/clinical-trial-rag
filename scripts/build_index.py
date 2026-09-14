"""Build and persist the FAISS index."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.index import TrialIndex, load_corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--limit", type=int, default=settings.max_docs)
ap.add_argument("--corpus", type=Path, default=settings.corpus_path)
ap.add_argument("--out", type=Path, default=settings.index_dir)
a = ap.parse_args()

docs = load_corpus(a.corpus, limit=a.limit)
idx = TrialIndex.build(docs)
idx.save(a.out)
print(f"indexed {idx.index.ntotal:,} documents -> {a.out}")
