from __future__ import annotations

from dataclasses import dataclass, field

from .embedder import cosine


@dataclass
class Entry:
    vector: list[float]
    prompt: str
    response: str
    created_at: float
    last_used: float
    hits: int = 0
    est_tokens: int = 0


@dataclass
class VectorCache:
    """A small in-memory vector store with nearest-neighbor lookup, TTL, and
    LRU eviction.

    Lookup is a linear cosine scan. That is the right call for cache sizes in
    the thousands and keeps the dependency list empty. Swap in a real ANN index
    (faiss, hnswlib) behind the same two methods if you outgrow it.
    """

    max_size: int = 1000
    ttl: float | None = None
    _entries: list[Entry] = field(default_factory=list)

    def search(self, vector: list[float], now: float) -> tuple[Entry | None, float]:
        self._expire(now)
        best: Entry | None = None
        best_sim = -1.0
        for entry in self._entries:
            sim = cosine(vector, entry.vector)
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is None:
            return None, 0.0
        return best, best_sim

    def add(self, entry: Entry) -> None:
        self._entries.append(entry)
        self._evict()

    def touch(self, entry: Entry, now: float) -> None:
        entry.hits += 1
        entry.last_used = now

    def __len__(self) -> int:
        return len(self._entries)

    def _expire(self, now: float) -> None:
        if self.ttl is None:
            return
        self._entries = [e for e in self._entries if now - e.created_at <= self.ttl]

    def _evict(self) -> None:
        while len(self._entries) > self.max_size:
            oldest = min(self._entries, key=lambda e: e.last_used)
            self._entries.remove(oldest)
