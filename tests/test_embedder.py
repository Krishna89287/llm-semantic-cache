import pytest

from llm_semantic_cache.embedder import HashingEmbedder, cosine


def test_identical_text_is_self_similar():
    e = HashingEmbedder()
    v = e.embed("how do I reset my password")
    assert cosine(v, v) == pytest.approx(1.0)


def test_reword_scores_higher_than_unrelated():
    e = HashingEmbedder()
    base = e.embed("how do I reset my password")
    reword = e.embed("how can I reset my password")
    unrelated = e.embed("what is the weather in munich today")
    assert cosine(base, reword) > cosine(base, unrelated)


def test_empty_text_does_not_crash():
    e = HashingEmbedder()
    v = e.embed("")
    assert len(v) == e.dim
    assert cosine(v, v) == 0.0  # zero vector, defined as no similarity


def test_dim_validation():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)
