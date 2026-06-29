from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _stable_hash(s: str) -> int:
    # md5 keeps the hash stable across processes, so cache behavior and the demo
    # numbers do not change run to run the way Python's salted hash() would.
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "little")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    squished = re.sub(r"\s+", " ", text.lower()).strip()
    if len(squished) < n:
        return [squished] if squished else []
    return [squished[i : i + n] for i in range(len(squished) - n + 1)]


class HashingEmbedder:
    """Local feature-hashing embedder. No API key, deterministic, fast.

    It hashes word tokens and character trigrams into a fixed vector with signed
    buckets to reduce collisions. That captures lexical overlap and small
    rewordings well. It does not capture deep meaning. For true semantic
    matching, implement Embedder against sentence-transformers or a hosted
    embedding endpoint and pass it to SemanticCache instead.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            self._add(vec, token, 1.0)
        for gram in _char_ngrams(text, 3):
            self._add(vec, gram, 0.5)
        return _l2_normalize(vec)

    def _add(self, vec: list[float], feature: str, weight: float) -> None:
        h = _stable_hash(feature)
        bucket = h % self.dim
        sign = 1.0 if (h >> 33) & 1 else -1.0
        vec[bucket] += sign * weight


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    # Inputs are L2 normalized, so the dot product is the cosine similarity.
    return sum(x * y for x, y in zip(a, b))
