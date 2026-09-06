"""Request and response models. Validation lives here, not in the handlers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       description="Patient profile or free-text clinical query",
                       examples=["breast cancer, stage II, HER2 positive"])
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    open_only: bool = Field(default=False,
                            description="Restrict to trials still enrolling")


class TrialHit(BaseModel):
    nct_id: str
    title: str
    conditions: str
    status: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    results: list[TrialHit]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500,
                          examples=["Which trials are open for HER2+ breast cancer?"])
    top_k: int = Field(default=settings.top_k, ge=1, le=20)
    open_only: bool = Field(default=True)
    backend: str | None = Field(default=None,
                                description="Override the LLM backend for this request")


class HealthResponse(BaseModel):
    status: str
    documents: int
    embed_model: str
    llm_backend: str
    uptime_seconds: float
