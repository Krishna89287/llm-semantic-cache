"""Send a stream of prompts through the cache and report what it saved.

The model here is a stand-in that sleeps 50ms per call, so the latency gap
between a hit and a miss is obvious. Some prompts are rewordings of earlier ones.
"""
from __future__ import annotations

import time

from llm_semantic_cache.cache import SemanticCache

PROMPTS = [
    "How do I reset my password?",
    "How can I reset my password?",
    "What is your refund policy?",
    "Tell me about your refund policy",
    "How do I reset my password?",
    "What are your business hours?",
    "When are you open?",
    "How can I reset my password?",
]


def slow_model(prompt: str) -> str:
    time.sleep(0.05)
    return f"Generated answer for: {prompt}"


def main() -> None:
    cache = SemanticCache(threshold=0.8, cost_per_1k=0.5)
    for prompt in PROMPTS:
        result = cache.get_or_call(prompt, slow_model)
        tag = "HIT " if result.cached else "MISS"
        print(f"  {tag}  sim={result.similarity:0.3f}  {result.latency_ms:6.1f}ms  {prompt}")

    snap = cache.metrics.snapshot()
    print()
    print(f"lookups={snap['lookups']}  hits={snap['hits']}  hit_rate={snap['hit_rate']:.0%}")
    print(f"tokens saved={snap['tokens_saved']}  est cost saved=${snap['cost_saved']}")


if __name__ == "__main__":
    main()
