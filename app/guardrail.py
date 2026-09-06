"""Groundedness checks on the generated answer.

The failure mode this exists to catch is not the model being wrong — it is the
model being *confidently* wrong about something a clinician might act on: an
NCT number that does not exist, or an eligibility claim absent from the context.

Two checks, both deterministic and both cheap. Neither uses a second LLM,
because "ask the model whether the model was right" is not a control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Doc

NCT_RE = re.compile(r"NCT\d{8}")

# Words that carry no evidential weight, so counting them would inflate the
# overlap score without saying anything about grounding.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "at", "by", "from", "has", "have", "had", "not",
    "no", "if", "then", "than", "there", "their", "they", "you", "your", "we",
    "can", "may", "will", "would", "should", "could", "also", "more", "most",
    "some", "any", "all", "such", "which", "who", "what", "when", "where",
    "trial", "trials", "study", "studies", "patient", "patients",
}


@dataclass
class GroundednessReport:
    grounded: bool
    overlap: float
    cited_ids: list[str] = field(default_factory=list)
    hallucinated_ids: list[str] = field(default_factory=list)
    unsupported_terms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "overlap": round(self.overlap, 3),
            "cited_ids": self.cited_ids,
            "hallucinated_ids": self.hallucinated_ids,
            "unsupported_terms": self.unsupported_terms[:12],
        }


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def check(answer: str, docs: list[Doc], threshold: float | None = None
          ) -> GroundednessReport:
    """Score how much of the answer is traceable to the retrieved context.

    Check 1 — citation validity. Any NCT id in the answer that is not in the
    retrieved set is a fabrication, full stop. This is the hard failure.

    Check 2 — lexical overlap. The share of the answer's content words that
    appear in the context. A blunt instrument: it will flag a correct answer
    that paraphrases heavily, and will pass a wrong answer that reuses context
    vocabulary. It is a smoke alarm, not a proof, and is reported as a score
    rather than a verdict for that reason.
    """
    from .config import settings
    threshold = settings.groundedness_threshold if threshold is None else threshold

    context = " ".join(d.to_text() + " " + d.nct_id for d in docs)
    context_words = set(_content_words(context))
    valid_ids = {d.nct_id for d in docs}

    cited = NCT_RE.findall(answer)
    hallucinated = sorted({c for c in cited if c not in valid_ids})

    answer_words = _content_words(answer)
    if answer_words:
        supported = [w for w in answer_words if w in context_words]
        overlap = len(supported) / len(answer_words)
        unsupported = sorted({w for w in answer_words if w not in context_words})
    else:
        overlap, unsupported = 0.0, []

    return GroundednessReport(
        grounded=(not hallucinated) and overlap >= threshold,
        overlap=overlap,
        cited_ids=sorted(set(cited)),
        hallucinated_ids=hallucinated,
        unsupported_terms=unsupported,
    )
