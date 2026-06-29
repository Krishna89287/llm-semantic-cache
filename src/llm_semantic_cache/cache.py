from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .embedder import Embedder, HashingEmbedder
from .metrics import CacheMetrics
from .store import Entry, VectorCache

LLMFn = Callable[[str], str]


@dataclass
class CacheResult:
    response: str
    cached: bool
    similarity: float
    latency_ms: float


def _estimate_tokens(text: str) -> int:
    # Rough words-to-tokens estimate. Good enough for savings accounting; swap
    # in a real tokenizer if you need exact numbers.
    return max(1, int(len(text.split()) * 1.3))


class SemanticCache:
    """Cache LLM responses by meaning, not exact string match.

    A new prompt is embedded and compared to what is already cached. If the
    closest entry is similar enough, its response is returned without calling the
    model. Otherwise the model runs and the answer is stored for next time.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.85,
        max_size: int = 1000,
        ttl: float | None = None,
        cost_per_1k: float = 0.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.store = VectorCache(max_size=max_size, ttl=ttl)
        self.metrics = CacheMetrics(cost_per_1k=cost_per_1k)
        self.clock = clock

    def get_or_call(self, prompt: str, llm_fn: LLMFn) -> CacheResult:
        start = time.perf_counter()
        now = self.clock()
        vector = self.embedder.embed(prompt)
        entry, similarity = self.store.search(vector, now)

        if entry is not None and similarity >= self.threshold:
            self.store.touch(entry, now)
            self.metrics.record_hit(entry.est_tokens)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return CacheResult(entry.response, True, similarity, latency_ms)

        response = llm_fn(prompt)
        self.store.add(
            Entry(
                vector=vector,
                prompt=prompt,
                response=response,
                created_at=now,
                last_used=now,
                est_tokens=_estimate_tokens(response),
            )
        )
        self.metrics.record_miss()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return CacheResult(response, False, similarity, latency_ms)
