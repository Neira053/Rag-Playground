from fastapi.testclient import TestClient

from app.main import app


def test_relevant_query_returns_sources_and_answer():
    with TestClient(app) as client:
        r = client.post(
            "/ask",
            headers={"X-API-Key": "demo-key-123"},
            json={"query": "How long does an on-call engineer have to acknowledge a page?"},
        )
    body = r.json()
    assert r.status_code == 200
    assert not body["blocked"]
    assert len(body["sources"]) > 0
    assert body["sources"][0]["doc_id"].startswith("incident-response")


def test_out_of_scope_query_finds_nothing_instead_of_guessing():
    """Regression test for the retrieval-threshold fix: a query with no real
    match in the knowledge base should come back with zero sources and an
    honest 'don't know' style answer, not a low-relevance chunk dressed up
    as a real answer."""
    with TestClient(app) as client:
        r = client.post(
            "/ask",
            headers={"X-API-Key": "demo-key-123"},
            json={"query": "What's the best recipe for a chocolate souffle?"},
        )
    body = r.json()
    assert r.status_code == 200
    assert not body["blocked"]
    assert body["sources"] == []
    assert "don't have relevant context" in body["answer"]


def test_repeated_query_is_served_from_cache():
    query = "How does the blue-green deployment strategy shift traffic?"
    with TestClient(app) as client:
        first = client.post("/ask", headers={"X-API-Key": "demo-key-123"}, json={"query": query}).json()
        second = client.post("/ask", headers={"X-API-Key": "demo-key-123"}, json={"query": query}).json()

    assert not first["cache_hit"]
    assert second["cache_hit"]
    assert second["answer"] == first["answer"]


def test_missing_api_key_is_rejected():
    with TestClient(app) as client:
        r = client.post("/ask", headers={"X-API-Key": "wrong-key"}, json={"query": "hello"})
    assert r.status_code == 401
