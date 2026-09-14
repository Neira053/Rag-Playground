"""
Gemini call wrapper.

Runs in MOCK_MODE automatically when no GEMINI_API_KEY is set, so the
whole pipeline (retrieval → rerank → "LLM" → guardrails → metrics) is
runnable and demo-able with zero API key, zero cost. Set a real key in
.env to get real answers (Gemini's free tier covers this easily).
"""
import time
from dataclasses import dataclass
from typing import List, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

_client = None
if not settings.MOCK_MODE and genai is not None:
    _client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options={"timeout": int(settings.LLM_TIMEOUT_SECONDS * 1000)},  # ms
    )


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    success: bool = True


class LLMTransientError(Exception):
    pass


def _build_prompt(query: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context retrieved)"
    return (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )


def _mock_answer(query: str, context_chunks: List[str]) -> LLMResult:
    if context_chunks:
        snippet = context_chunks[0][:180].strip()
        text = (
            f"[MOCK MODE — no GEMINI_API_KEY set] Based on the retrieved context, "
            f"here's a stand-in answer for \"{query}\": {snippet}..."
        )
    else:
        text = f"[MOCK MODE] I don't have relevant context to answer \"{query}\"."
    approx_in = sum(len(c.split()) for c in context_chunks) + len(query.split())
    approx_out = len(text.split())
    return LLMResult(text=text, input_tokens=int(approx_in * 1.3), output_tokens=int(approx_out * 1.3))


@retry(
    stop=stop_after_attempt(settings.LLM_MAX_RETRIES + 1),
    wait=wait_exponential(multiplier=0.4, min=0.4, max=3),
    retry=retry_if_exception_type(LLMTransientError),
    reraise=True,
)
def _call_gemini(prompt: str) -> LLMResult:
    try:
        resp = _client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "max_output_tokens": settings.LLM_MAX_TOKENS,
                "thinking_config": {"thinking_level": "MINIMAL"},  # Gemini 3 models default to
                # deep "thinking" which eats into max_output_tokens and can truncate the visible
                # answer. This is a direct doc-grounded QA task, not a reasoning task, so we
                # keep thinking minimal. (Gemini 3 uses thinking_level, not the older
                # thinking_budget field from Gemini 2.5 — sending the wrong one 400s.)
            },
        )
    except Exception as e:  # noqa: BLE001 — google-genai raises several transient error types
        raise LLMTransientError(str(e)) from e

    text = resp.text or ""
    usage = getattr(resp, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    return LLMResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def generate(query: str, context_chunks: List[str]) -> Tuple[LLMResult, float]:
    """Returns (result, elapsed_seconds)."""
    start = time.perf_counter()
    if settings.MOCK_MODE or _client is None:
        result = _mock_answer(query, context_chunks)
    else:
        prompt = _build_prompt(query, context_chunks)
        try:
            result = _call_gemini(prompt)
        except LLMTransientError:
            result = LLMResult(
                text="The model is temporarily unavailable after retrying — please try again shortly.",
                input_tokens=0,
                output_tokens=0,
                success=False,
            )
    elapsed = time.perf_counter() - start
    return result, elapsed


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * settings.PRICE_INPUT_PER_1M + (
        output_tokens / 1_000_000
    ) * settings.PRICE_OUTPUT_PER_1M