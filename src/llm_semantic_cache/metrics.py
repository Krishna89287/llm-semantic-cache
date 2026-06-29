from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheMetrics:
    """Counters for what the cache saved.

    cost_per_1k is the blended price you pay the provider per 1000 tokens. It is
    only used to turn saved tokens into a saved-dollars estimate, so set it to
    whatever your model actually costs.
    """

    cost_per_1k: float = 0.0
    lookups: int = 0
    hits: int = 0
    tokens_saved: int = 0

    def record_hit(self, est_tokens: int) -> None:
        self.lookups += 1
        self.hits += 1
        self.tokens_saved += est_tokens

    def record_miss(self) -> None:
        self.lookups += 1

    @property
    def misses(self) -> int:
        return self.lookups - self.hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def cost_saved(self) -> float:
        return (self.tokens_saved / 1000.0) * self.cost_per_1k

    def snapshot(self) -> dict:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "tokens_saved": self.tokens_saved,
            "cost_saved": round(self.cost_saved, 4),
        }
