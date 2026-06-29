from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

from .cache import SemanticCache

app = FastAPI(title="llm-semantic-cache", version="0.1.0")
cache = SemanticCache(threshold=0.8, cost_per_1k=0.5)


def _slow_model(prompt: str) -> str:
    # Stand-in for a real provider call. The sleep makes the cache speedup
    # visible in latency. Replace with your client (Anthropic, OpenAI, vLLM).
    time.sleep(0.05)
    return f"Generated answer for: {prompt}"


class CompleteRequest(BaseModel):
    prompt: str


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/complete")
def complete(req: CompleteRequest) -> dict:
    result = cache.get_or_call(req.prompt, _slow_model)
    return {
        "response": result.response,
        "cached": result.cached,
        "similarity": round(result.similarity, 4),
        "latency_ms": round(result.latency_ms, 3),
    }


@app.get("/metrics")
def metrics() -> dict:
    return cache.metrics.snapshot()
