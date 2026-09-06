"""API contract tests.

Retrieval is stubbed so these run in CI without the 168 MB index: what is being
tested is the HTTP contract and validation, not FAISS.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.index import Doc


@pytest.fixture
def client(monkeypatch):
    class StubIndex:
        class _Inner:
            ntotal = 3
        index = _Inner()

        def search(self, query, k=5, open_only=False):
            docs = [
                Doc("NCT00000001", "Trastuzumab in HER2-positive Breast Cancer",
                    "HER2-positive Breast Cancer", "DRUG: Trastuzumab",
                    "RECRUITING", "https://example.org/1"),
                Doc("NCT00000002", "Pembrolizumab for Advanced Melanoma",
                    "Melanoma", "DRUG: Pembrolizumab",
                    "COMPLETED", "https://example.org/2"),
            ]
            if open_only:
                docs = [d for d in docs if d.status == "RECRUITING"]
            return [(d, 0.9 - i * 0.1) for i, d in enumerate(docs[:k])]

    monkeypatch.setitem(main.state, "index", StubIndex())
    return TestClient(main.app)


def test_health_reports_document_count(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents"] == 3


def test_search_returns_ranked_hits(client):
    r = client.post("/search", json={"query": "HER2 positive breast cancer"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    scores = [h["score"] for h in body["results"]]
    assert scores == sorted(scores, reverse=True), "results must be ranked"


def test_open_only_filters_closed_trials(client):
    r = client.post("/search", json={"query": "cancer", "open_only": True})
    statuses = {h["status"] for h in r.json()["results"]}
    assert statuses == {"RECRUITING"}


@pytest.mark.parametrize("payload", [
    {"query": "ab"},                       # below min_length
    {"query": "valid query", "top_k": 0},  # below ge=1
    {"query": "valid query", "top_k": 99},  # above le=50
    {},                                     # missing required field
])
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_service_reports_503_without_an_index(monkeypatch):
    monkeypatch.setitem(main.state, "index", None)
    r = TestClient(main.app).post("/search", json={"query": "breast cancer"})
    assert r.status_code == 503
