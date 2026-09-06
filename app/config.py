"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Data
    corpus_path: Path = Path(os.getenv("CORPUS_PATH", "data/oncology_trials.csv"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "data/index"))
    max_docs: int = int(os.getenv("MAX_DOCS", "0"))          # 0 = all

    # Retrieval
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Generation. 'local' needs no key and no external service, which is why it
    # is the default: the service must be runnable by anyone who clones it.
    llm_backend: str = os.getenv("LLM_BACKEND", "local")      # local | ollama | openai
    local_model: str = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "220"))

    # Guardrail: minimum share of the answer's content words that must appear in
    # the retrieved context before the answer is served without a warning.
    groundedness_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.55"))


settings = Settings()
