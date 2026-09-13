import pytest

from app.cache import Cache
from app.config import settings


@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    """Tests must be deterministic and offline. Without this, adding a real
    GEMINI_API_KEY to .env for manual testing silently makes the test suite
    depend on a live network call — brittle and not what these tests check."""
    monkeypatch.setattr(settings, "MOCK_MODE", True)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """Route the pipeline through a fresh in-memory cache for every test, so
    a real Redis instance on the developer's machine (populated by manually
    poking the dashboard) can never leak a cached answer into the suite."""
    fresh = Cache()
    fresh._client = None
    fresh.backend = "memory"
    fresh._memory_store = {}
    monkeypatch.setattr("app.pipeline.cache", fresh)