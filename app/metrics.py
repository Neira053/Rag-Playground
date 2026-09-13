"""
In-memory metrics + trace log.

A real system ships these to Prometheus/Grafana or an observability
platform (Langfuse, Arize, Datadog). For a POC, an in-process rolling
window is enough to prove you understand *what* to measure and *why* —
the export target is a config change, not a redesign.
"""
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from app.config import settings


@dataclass
class Trace:
    trace_id: str
    query: str
    cache_hit: bool
    blocked: bool
    block_reason: Optional[str]
    api_overhead_ms: float
    retrieval_ms: float
    reranking_ms: float
    llm_ms: float
    guardrail_ms: float
    total_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retrieved_doc_ids: List[str] = field(default_factory=list)
    error: bool = False
    timestamp: float = field(default_factory=time.time)


class MetricsStore:
    def __init__(self, max_history: int = settings.TRACE_HISTORY_SIZE):
        self._lock = threading.Lock()
        self._traces: Deque[Trace] = deque(maxlen=max_history)

    def record(self, trace: Trace):
        with self._lock:
            self._traces.append(trace)

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def latest(self) -> Optional[Trace]:
        with self._lock:
            return self._traces[-1] if self._traces else None

    def get(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            for t in reversed(self._traces):
                if t.trace_id == trace_id:
                    return t
        return None

    def summary(self) -> dict:
        with self._lock:
            traces = list(self._traces)

        n = len(traces)
        if n == 0:
            return {
                "requests": 0, "avg_latency_ms": 0, "cache_hit_rate": 0,
                "avg_input_tokens": 0, "avg_output_tokens": 0, "avg_total_tokens": 0,
                "estimated_cost_usd": 0, "error_rate": 0,
            }

        cache_hits = sum(1 for t in traces if t.cache_hit)
        errors = sum(1 for t in traces if t.error or t.blocked)
        total_cost = sum(t.cost_usd for t in traces)
        avg_latency = sum(t.total_ms for t in traces) / n
        avg_in = sum(t.input_tokens for t in traces) / n
        avg_out = sum(t.output_tokens for t in traces) / n

        return {
            "requests": n,
            "avg_latency_ms": round(avg_latency, 1),
            "cache_hit_rate": round(cache_hits / n, 3),
            "avg_input_tokens": round(avg_in, 1),
            "avg_output_tokens": round(avg_out, 1),
            "avg_total_tokens": round(avg_in + avg_out, 1),
            "estimated_cost_usd": round(total_cost, 4),
            "error_rate": round(errors / n, 3),
        }


metrics_store = MetricsStore()
