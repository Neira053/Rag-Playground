from app.cache import Cache


def test_key_for_is_stable_and_case_insensitive():
    k1 = Cache.key_for("What is the on-call policy?")
    k2 = Cache.key_for("  what IS the on-call POLICY?  ")
    assert k1 == k2


def test_key_for_differs_for_different_queries():
    assert Cache.key_for("query one") != Cache.key_for("query two")


def test_set_and_get_roundtrip():
    c = Cache()
    key = Cache.key_for("test query")
    c.set(key, {"answer": "hello"}, ttl=60)
    assert c.get(key) == {"answer": "hello"}


def test_get_missing_key_returns_none():
    c = Cache()
    assert c.get(Cache.key_for("never cached")) is None


def test_expired_entry_is_not_returned():
    c = Cache()
    key = Cache.key_for("expiring query")
    c.set(key, {"answer": "will expire"}, ttl=-1)  # already expired
    assert c.get(key) is None
