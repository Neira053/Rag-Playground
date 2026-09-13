"""
Sliding-window rate limiter, in-memory. Good enough for a single-process
POC; a real deployment would move this state into Redis so it works across
multiple app instances (the cache module already shows that pattern).
"""
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

_hits: dict[str, deque] = defaultdict(deque)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/ask",):
            client_id = request.client.host if request.client else "unknown"
            now = time.time()
            window = _hits[client_id]

            while window and now - window[0] > settings.RATE_LIMIT_WINDOW_SECONDS:
                window.popleft()

            if len(window) >= settings.RATE_LIMIT_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Slow down and try again shortly."},
                )
            window.append(now)

            # Evict clients with no activity in the current window so the dict
            # doesn't grow unbounded over the life of a long-running process.
            if len(_hits) > 1000:
                stale = [cid for cid, w in _hits.items() if not w or now - w[-1] > settings.RATE_LIMIT_WINDOW_SECONDS]
                for cid in stale:
                    del _hits[cid]

        return await call_next(request)
