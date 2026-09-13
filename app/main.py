from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import eval as eval_module
from app.config import settings
from app.metrics import metrics_store
from app.pipeline import run_pipeline
from app.ratelimit import RateLimitMiddleware
from app.schemas import AskRequest, AskResponse
from app.vectorstore import vector_store

app = FastAPI(title="Production RAG Playground", version="0.1.0")
app.add_middleware(RateLimitMiddleware)


@app.on_event("startup")
def load_index():
    if not vector_store.load():
        vector_store.build_from_docs()
        vector_store.save()


def _check_auth(x_api_key: str | None):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_mode": settings.MOCK_MODE,
        "cache_backend": __import__("app.cache", fromlist=["cache"]).cache.backend,
        "indexed_chunks": len(vector_store.chunks),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, x_api_key: str | None = Header(default=None)):
    _check_auth(x_api_key)
    return run_pipeline(payload.query)


@app.get("/metrics")
def metrics():
    return metrics_store.summary()


@app.get("/trace/latest")
def trace_latest():
    t = metrics_store.latest()
    if not t:
        raise HTTPException(status_code=404, detail="No requests recorded yet")
    return t


@app.get("/trace/{trace_id}")
def trace_by_id(trace_id: str):
    t = metrics_store.get(trace_id)
    if not t:
        raise HTTPException(status_code=404, detail="Trace not found")
    return t


@app.post("/eval")
def run_evaluation():
    return eval_module.run_eval(top_k=settings.TOP_K_RETRIEVE)


@app.post("/reindex")
def reindex(x_api_key: str | None = Header(default=None)):
    _check_auth(x_api_key)
    n = vector_store.build_from_docs()
    vector_store.save()
    return {"status": "reindexed", "chunks": n}


@app.get("/")
def dashboard():
    return FileResponse("static/dashboard.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
