"""Pluggable LLM backends.

Three implementations behind one interface. The default is `local` because a
service nobody can run without buying an API key is not a demonstrable service;
`ollama` and `openai` exist because production deployments will not want a
0.5B model doing clinical summarisation.

Every backend streams. Streaming is not decoration: a RAG answer takes seconds
to generate, and a user watching a blank page assumes it broke.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Protocol

from .config import settings
from .index import Doc

log = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a clinical trial search assistant. Answer ONLY from the numbered "
    "trials provided. Cite trials by their NCT number in square brackets, like "
    "[NCT01234567]. If the trials do not contain the answer, say so plainly "
    "rather than guessing. Never invent an NCT number, an eligibility "
    "criterion, or a trial phase. Be concise."
)


def build_prompt(question: str, docs: list[Doc]) -> str:
    """Assemble the grounded prompt.

    Numbering the context and demanding bracketed NCT citations is what makes
    the guardrail in guardrail.py checkable: an answer that cites nothing, or
    cites an id absent from the context, is detectable without a second model.
    """
    context = "\n\n".join(
        f"[{i}] {d.nct_id} — {d.title}\n"
        f"    Conditions: {d.conditions}\n"
        f"    Interventions: {d.interventions}\n"
        f"    Status: {d.status}"
        for i, d in enumerate(docs, 1)
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"TRIALS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


class Backend(Protocol):
    name: str
    def stream(self, prompt: str) -> Iterator[str]: ...


# ------------------------------------------------------------------- local

class LocalBackend:
    """transformers running on CPU. No key, no network, no cost."""

    name = "local"

    def __init__(self, model_name: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name or settings.local_model
        log.info("loading local model %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def stream(self, prompt: str) -> Iterator[str]:
        import threading

        from transformers import TextIteratorStreamer

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        # Generation blocks, so it runs on its own thread and the tokens are
        # consumed from the streamer as they appear.
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(**inputs, streamer=streamer,
                        max_new_tokens=settings.max_new_tokens,
                        do_sample=False),          # deterministic: eval repeatability
        )
        thread.start()
        yield from streamer
        thread.join()


# ------------------------------------------------------------------ ollama

class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or settings.ollama_model
        self.host = (host or settings.ollama_host).rstrip("/")

    def stream(self, prompt: str) -> Iterator[str]:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps({"model": self.model, "prompt": prompt,
                             "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("response"):
                    yield chunk["response"]
                if chunk.get("done"):
                    break


# ------------------------------------------------------------------ openai

class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.model = model or settings.openai_model

    def stream(self, prompt: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.max_new_tokens,
            temperature=0,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_BACKENDS = {"local": LocalBackend, "ollama": OllamaBackend, "openai": OpenAIBackend}
_cached: Backend | None = None


def get_backend(name: str | None = None) -> Backend:
    """Lazily construct and cache the backend — loading a model is expensive."""
    global _cached
    name = name or settings.llm_backend
    if _cached is None or _cached.name != name:
        if name not in _BACKENDS:
            raise ValueError(f"unknown backend {name!r}; choose from {list(_BACKENDS)}")
        _cached = _BACKENDS[name]()
    return _cached
