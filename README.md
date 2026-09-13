# Production RAG Playground

A small, fully runnable RAG system built to demonstrate the engineering
*around* the LLM call — the part that turns "a working AI feature" into
"a production-ready AI system":

- caching
- retries & timeouts
- rate limiting
- an evaluation harness (retrieval recall@k)
- cost tracking
- per-stage tracing & metrics
- guardrails (input + output)
- basic auth

It answers questions over a small internal "engineering handbook" (deployment
process, incident response, coding standards, on-call policy, security
policy) — swap in your own `.txt` docs and it works the same way.

```
                User
                  ↓
              FastAPI  ── auth, rate limiting
                  ↓
          ┌───────┴───────┐
          ↓               ↓
       Redis          RAG Pipeline
      Cache               ↓
     (+ in-mem       Retrieval (TF-IDF vector store)
      fallback)           ↓
          │           Rerank (lexical + vector blend)
          │               ↓
          │           LLM (Gemini, retries + timeout,
          │               mock mode with no API key)
          │               ↓
          │           Guardrails (PII, injection, groundedness)
          └───────→   Metrics & Trace store
                          ↓
                       Answer + trace_id
```

Dashboard (served at `/`) is a terminal-style monitor showing live request
volume, latency, cache hit rate, token/cost estimates, error rate, retrieval
recall, and a waterfall of the latest request's per-stage timings — plus a
box to actually ask it questions.

## Design choices worth knowing (and mentioning if asked)

- **No external model downloads.** Retrieval uses TF-IDF + cosine similarity
  instead of a neural embedding model, so the whole thing runs offline with
  a one-line `pip install`. The vector-store *interface* (`build_from_docs`,
  `query`) is what you'd swap out for Chroma/Pinecone + a real embedding
  model — that's a config change, not a redesign.
- **Mock mode by default.** With no `GEMINI_API_KEY` set, the LLM stage
  returns a clearly-labeled mock answer built from the retrieved context, so
  every other stage (retrieval, rerank, guardrails, caching, tracing) still
  runs for real. Add a key to `.env` to get real answers.
- **Graceful cache degradation.** If Redis isn't running, the cache silently
  falls back to an in-memory dict rather than crashing the demo.
- **Cheap, explainable reranker.** Rather than a second heavy model, the
  rerank stage blends vector similarity with keyword overlap — enough to
  show *why* retrieve-then-rerank is a two-stage design (recall, then
  precision) without adding a second dependency.
- **A real eval, not a made-up number.** `POST /eval` runs a small labeled
  set (`data/eval_set.json`) of query → expected-document pairs through
  retrieval and reports recall@k. The set includes paraphrases that don't
  share vocabulary with the source doc, and negative cases (out-of-scope
  queries where the correct answer is "find nothing"). Recall@5 on this
  harder set is **~79%**, not the inflated ~100% you'd get from an eval
  set that just echoes the source text — see `/eval`'s `cases` output for
  exactly which queries miss and why (mostly: TF-IDF can't bridge synonyms
  like "text-message-verify" → "SMS-based MFA"; a real embedding model
  would catch that).
- **A minimum similarity threshold before an answer is attempted.**
  Candidates below `MIN_RETRIEVAL_SCORE` are dropped before reranking, so
  an out-of-scope query gets an honest "I don't have relevant context"
  instead of a low-confidence chunk dressed up as a real answer.
- **A pytest suite** (`tests/`) covering guardrails, caching, the rate
  limiter (including a regression test for a stale-client memory leak
  that was fixed after writing the tests), the eval harness, and the
  full pipeline end to end.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optionally add GEMINI_API_KEY for real answers (free at aistudio.google.com)
python scripts/ingest.py      # builds the vector index from data/docs/*.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the dashboard.

Run the test suite with:

```bash
pytest tests/ -v
```

Optional: run Redis locally (`redis-server` or `docker run -p 6379:6379 redis`)
to see `cache_backend: "redis"` on `/health` instead of the in-memory fallback.

## API

| Endpoint | Method | Notes |
|---|---|---|
| `/ask` | POST | Body `{"query": "..."}`, header `X-API-Key: demo-key-123`. Runs the full pipeline. |
| `/metrics` | GET | Aggregate stats across the trace history. |
| `/trace/latest` | GET | Full per-stage timing breakdown of the most recent request. |
| `/trace/{trace_id}` | GET | Look up any past request by its trace ID. |
| `/eval` | POST | Runs `data/eval_set.json` through retrieval, reports recall@k. |
| `/reindex` | POST | Rebuilds the vector index from `data/docs/`. Auth required. |
| `/health` | GET | Mode (mock/live), cache backend, index size. |

Change `PLAYGROUND_API_KEY` in `.env` (and the matching constant at the top
of `static/dashboard.html`'s `<script>`) if you want a different demo key.

## Project layout

```
app/
  main.py        FastAPI app, routes, auth, rate-limit middleware
  pipeline.py     orchestrates cache → retrieval → rerank → LLM → guardrails → metrics
  vectorstore.py  TF-IDF vector store (build / persist / query)
  reranker.py     lexical + vector blended rerank
  llm.py          Gemini call wrapper: retries, timeout, mock mode, cost calc
  guardrails.py   input (injection/PII/blocklist) + output (groundedness/PII) checks
  cache.py        Redis client with in-memory fallback
  metrics.py      in-memory trace log + aggregate summary
  eval.py         retrieval recall@k harness
  ratelimit.py    sliding-window rate limit middleware
data/
  docs/           sample knowledge base (5 .txt files)
  eval_set.json   labeled query -> expected-doc pairs
static/
  dashboard.html  the monitor UI
scripts/ingest.py  CLI reindex
```

## Known limitations (honest, not hidden)

- **No streaming.** Real chat UIs stream tokens; `/ask` returns the whole
  answer at once.
- **No multi-turn/session memory.** Every `/ask` call is independent —
  there's no conversation history.
- **The reranker's benefit is asserted, not measured.** There's no
  before/after comparison showing the lexical rerank actually improves
  ordering versus raw vector similarity alone.
- **The retrieval threshold is a global constant.** The "parental leave
  policy" negative case in the eval set still returns weak false-positive
  matches above `MIN_RETRIEVAL_SCORE` — a fixed cutoff doesn't generalize
  perfectly across query types, which a real system would handle with a
  learned or per-domain threshold.

## Natural next steps

- Swap TF-IDF for a real embedding model + Chroma/Pinecone — the eval
  harness above already shows exactly the synonym-matching failures this
  would fix
- Cross-encoder reranker instead of the lexical blend, evaluated with a
  before/after recall or MRR comparison
- Ship metrics to Prometheus/Grafana or Langfuse instead of in-process memory
- Move rate-limit and cache state into Redis so it works across multiple
  app instances (cache.py already shows that fallback pattern)
- Structured output validation (e.g. Pydantic-typed answers) as a guardrail stage
- CI job that runs `pytest` and `/eval` on every PR and fails on test or recall regression
