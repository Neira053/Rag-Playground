from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class RetrievedChunk(BaseModel):
    doc_id: str
    text: str
    score: float


class StageTimings(BaseModel):
    api_overhead_ms: float
    retrieval_ms: float
    reranking_ms: float
    llm_ms: float
    guardrail_ms: float
    total_ms: float


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    sources: List[RetrievedChunk]
    cache_hit: bool
    blocked: bool = False
    block_reason: Optional[str] = None
    timings: StageTimings
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
