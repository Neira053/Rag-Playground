import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    MOCK_MODE: bool = GEMINI_API_KEY == ""  # graceful degradation, no key needed for a demo run

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", "5"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "3"))
    # Candidates below this raw cosine-similarity score are dropped before
    # reranking. Without this, an out-of-scope query still hands the LLM
    # some barely-related chunk instead of honestly finding nothing.
    MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.05"))

    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "8.0"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    API_KEY: str = os.getenv("PLAYGROUND_API_KEY", "demo-key-123")  # simple auth for this POC

    # Approx per-1M-token pricing used only for the cost estimate on the dashboard.
    # Defaults match Gemini 2.5 Flash's paid-tier rate; if you're on the free
    # tier (no billing enabled) your actual cost is $0 regardless of what the
    # dashboard estimates — set both to 0 if you want the dashboard to reflect that.
    PRICE_INPUT_PER_1M: float = float(os.getenv("PRICE_INPUT_PER_1M", "0.30"))
    PRICE_OUTPUT_PER_1M: float = float(os.getenv("PRICE_OUTPUT_PER_1M", "2.50"))

    DOCS_DIR: str = os.getenv("DOCS_DIR", "data/docs")
    EVAL_SET_PATH: str = os.getenv("EVAL_SET_PATH", "data/eval_set.json")
    INDEX_PATH: str = os.getenv("INDEX_PATH", "data/index.pkl")

    TRACE_HISTORY_SIZE: int = int(os.getenv("TRACE_HISTORY_SIZE", "200"))


settings = Settings()
