import time

from app.cache import cache
from app.config import settings
from app.guardrails import check_input, check_output
from app.llm import estimate_cost, generate
from app.metrics import Trace, metrics_store
from app.reranker import rerank
from app.schemas import AskResponse, RetrievedChunk, StageTimings
from app.vectorstore import vector_store


def run_pipeline(query: str) -> AskResponse:
    request_start = time.perf_counter()
    trace_id = metrics_store.new_trace_id()

    # ---- API overhead: whatever happens before real work starts ----
    t0 = time.perf_counter()
    input_check = check_input(query)
    api_overhead_ms = (time.perf_counter() - t0) * 1000

    if not input_check.passed:
        timings = StageTimings(
            api_overhead_ms=api_overhead_ms, retrieval_ms=0, reranking_ms=0,
            llm_ms=0, guardrail_ms=0, total_ms=(time.perf_counter() - request_start) * 1000,
        )
        _record(trace_id, query, False, True, input_check.reason, timings, 0, 0, 0.0, [])
        return AskResponse(
            trace_id=trace_id, answer="", sources=[], cache_hit=False, blocked=True,
            block_reason=input_check.reason, timings=timings, input_tokens=0,
            output_tokens=0, estimated_cost_usd=0.0,
        )

    clean_query = input_check.redacted_text

    # ---- cache lookup ----
    cache_key = cache.key_for(clean_query)
    cached = cache.get(cache_key)
    if cached:
        total_ms = (time.perf_counter() - request_start) * 1000
        timings = StageTimings(
            api_overhead_ms=api_overhead_ms, retrieval_ms=0, reranking_ms=0,
            llm_ms=0, guardrail_ms=0, total_ms=total_ms,
        )
        _record(trace_id, query, True, False, None, timings,
                cached["input_tokens"], cached["output_tokens"], cached["cost_usd"],
                [s["doc_id"] for s in cached["sources"]])
        return AskResponse(
            trace_id=trace_id, answer=cached["answer"],
            sources=[RetrievedChunk(**s) for s in cached["sources"]],
            cache_hit=True, timings=timings,
            input_tokens=cached["input_tokens"], output_tokens=cached["output_tokens"],
            estimated_cost_usd=cached["cost_usd"],
        )

    # ---- retrieval ----
    t0 = time.perf_counter()
    candidates = vector_store.query(clean_query, top_k=settings.TOP_K_RETRIEVE)
    candidates = [(c, s) for c, s in candidates if s >= settings.MIN_RETRIEVAL_SCORE]
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # ---- reranking ----
    t0 = time.perf_counter()
    top = rerank(clean_query, candidates, top_k=settings.TOP_K_RERANK)
    reranking_ms = (time.perf_counter() - t0) * 1000

    sources = [RetrievedChunk(doc_id=c.doc_id, text=c.text, score=round(score, 4)) for c, score in top]
    context_texts = [c.text for c, _ in top]

    # ---- LLM ----
    llm_result, llm_elapsed = generate(clean_query, context_texts)
    llm_ms = llm_elapsed * 1000

    # ---- output guardrail ----
    t0 = time.perf_counter()
    output_check = check_output(llm_result.text, context_texts)
    guardrail_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - request_start) * 1000
    timings = StageTimings(
        api_overhead_ms=api_overhead_ms, retrieval_ms=retrieval_ms, reranking_ms=reranking_ms,
        llm_ms=llm_ms, guardrail_ms=guardrail_ms, total_ms=total_ms,
    )

    if not output_check.passed:
        _record(trace_id, query, False, True, output_check.reason, timings,
                llm_result.input_tokens, llm_result.output_tokens, 0.0,
                [c.doc_id for c, _ in top])
        return AskResponse(
            trace_id=trace_id, answer="I can't provide a confident answer to that.", sources=sources,
            cache_hit=False, blocked=True, block_reason=output_check.reason, timings=timings,
            input_tokens=llm_result.input_tokens, output_tokens=llm_result.output_tokens,
            estimated_cost_usd=0.0,
        )

    final_answer = output_check.redacted_text
    cost = estimate_cost(llm_result.input_tokens, llm_result.output_tokens)

    if llm_result.success:
        cache.set(cache_key, {
            "answer": final_answer,
            "sources": [s.model_dump() for s in sources],
            "input_tokens": llm_result.input_tokens,
            "output_tokens": llm_result.output_tokens,
            "cost_usd": cost,
        })

    _record(trace_id, query, False, False, None, timings,
            llm_result.input_tokens, llm_result.output_tokens, cost,
            [c.doc_id for c, _ in top])

    return AskResponse(
        trace_id=trace_id, answer=final_answer, sources=sources, cache_hit=False,
        timings=timings, input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens, estimated_cost_usd=cost,
    )


def _record(trace_id, query, cache_hit, blocked, block_reason, timings,
            input_tokens, output_tokens, cost, doc_ids):
    metrics_store.record(Trace(
        trace_id=trace_id, query=query, cache_hit=cache_hit, blocked=blocked,
        block_reason=block_reason, api_overhead_ms=round(timings.api_overhead_ms, 2),
        retrieval_ms=round(timings.retrieval_ms, 2), reranking_ms=round(timings.reranking_ms, 2),
        llm_ms=round(timings.llm_ms, 2), guardrail_ms=round(timings.guardrail_ms, 2),
        total_ms=round(timings.total_ms, 2), input_tokens=input_tokens,
        output_tokens=output_tokens, cost_usd=cost, retrieved_doc_ids=doc_ids,
    ))