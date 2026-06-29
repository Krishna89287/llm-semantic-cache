import pytest

from llm_semantic_cache.cache import SemanticCache


def counting_model():
    calls = {"n": 0}

    def model(prompt: str) -> str:
        calls["n"] += 1
        return f"answer-{calls['n']}"

    return model, calls


def test_exact_repeat_is_a_hit_and_calls_model_once():
    cache = SemanticCache(threshold=0.8)
    model, calls = counting_model()
    first = cache.get_or_call("how do I reset my password", model)
    second = cache.get_or_call("how do I reset my password", model)
    assert first.cached is False
    assert second.cached is True
    assert second.response == first.response
    assert calls["n"] == 1


def test_unrelated_prompt_misses():
    cache = SemanticCache(threshold=0.8)
    model, calls = counting_model()
    cache.get_or_call("how do I reset my password", model)
    result = cache.get_or_call("what is the weather in munich today", model)
    assert result.cached is False
    assert calls["n"] == 2


def test_threshold_blocks_weak_matches():
    cache = SemanticCache(threshold=0.99)
    model, calls = counting_model()
    cache.get_or_call("how do I reset my password", model)
    # A reword is similar but below 0.99, so it should miss.
    result = cache.get_or_call("how can I reset my password", model)
    assert result.cached is False
    assert calls["n"] == 2


def test_metrics_track_hit_rate():
    cache = SemanticCache(threshold=0.8)
    model, _ = counting_model()
    cache.get_or_call("ping", model)
    cache.get_or_call("ping", model)
    snap = cache.metrics.snapshot()
    assert snap["lookups"] == 2
    assert snap["hits"] == 1
    assert snap["hit_rate"] == pytest.approx(0.5)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        SemanticCache(threshold=0)
