from llm_semantic_cache.cache import SemanticCache
from llm_semantic_cache.embedder import HashingEmbedder
from llm_semantic_cache.store import Entry, VectorCache


def _entry(emb, text, now):
    return Entry(vector=emb.embed(text), prompt=text, response=text, created_at=now, last_used=now)


def test_lru_eviction_keeps_size_bounded():
    emb = HashingEmbedder()
    store = VectorCache(max_size=2)
    store.add(_entry(emb, "alpha", now=1))
    store.add(_entry(emb, "bravo", now=2))
    store.add(_entry(emb, "charlie", now=3))  # evicts the least recently used
    assert len(store) == 2
    prompts = {e.prompt for e in store._entries}
    assert "alpha" not in prompts


def test_ttl_expires_entries():
    clock = {"t": 1000.0}
    cache = SemanticCache(threshold=0.8, ttl=10, clock=lambda: clock["t"])
    model = lambda p: "cached-answer"

    cache.get_or_call("hello world", model)  # stored at t=1000
    clock["t"] = 1005
    assert cache.get_or_call("hello world", model).cached is True  # still fresh
    clock["t"] = 1020
    assert cache.get_or_call("hello world", model).cached is False  # expired
