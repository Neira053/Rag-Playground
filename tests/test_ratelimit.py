import time

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
import app.ratelimit as ratelimit_module


def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 3)
    ratelimit_module._hits.clear()

    with TestClient(app) as client:
        codes = []
        for i in range(6):
            r = client.post(
                "/ask",
                headers={"X-API-Key": "demo-key-123"},
                json={"query": f"q{i}"},
            )
            codes.append(r.status_code)

    assert codes.count(200) == 3
    assert codes.count(429) == 3


def test_stale_clients_are_evicted(monkeypatch):
    """Regression test for the unbounded-growth bug: clients with no recent
    activity should eventually be pruned from the in-memory dict rather than
    accumulating forever in a long-running process."""
    from collections import deque

    ratelimit_module._hits.clear()
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 1000)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 1)

    # simulate 1500 long-idle clients sitting in the dict
    old_timestamp = time.time() - 999
    for i in range(1500):
        ratelimit_module._hits[f"stale-client-{i}"] = deque([old_timestamp])

    with TestClient(app) as client:
        r = client.post("/ask", headers={"X-API-Key": "demo-key-123"}, json={"query": "trigger cleanup"})
        assert r.status_code == 200

    # cleanup only runs once the dict exceeds 1000 entries; after one fresh
    # request, all 1500 stale entries should have been pruned
    assert len(ratelimit_module._hits) < 1500
