import time
import asyncio
from typing import Dict, List, Optional
from fastapi import HTTPException, Request
from app.core.config import settings

class RateLimiter:
    """
    Thread-safe, multi-worker ready rate and concurrency limiter.
    Supports per-user identifiers (primary) and client-IP fallback.
    Prevents concurrent request abuse per user.
    """
    def __init__(self, requests_per_minute: int = 10, max_concurrent: int = 1):
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        # In-memory sliding window history: {key: [timestamp1, timestamp2, ...]}
        self._history: Dict[str, List[float]] = {}
        # In-flight active requests per key: {key: count}
        self._active: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    @property
    def raw_history(self) -> Dict[str, List[float]]:
        return self._history

    def clear(self):
        """Clears all tracking history (primarily for test resets)."""
        self._history.clear()
        self._active.clear()

    @staticmethod
    def extract_client_ip(request: Optional[Request]) -> str:
        if not request:
            return "unknown"
        # Check standard reverse proxy headers (Render, Cloudflare, etc.)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # First IP in X-Forwarded-For is client
            return forwarded.split(",")[0].strip()
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
        return request.client.host if request.client else "unknown"

    def get_key(self, user_id: Optional[str], client_ip: Optional[str]) -> str:
        if user_id and str(user_id).strip():
            return f"user:{str(user_id).strip()}"
        clean_ip = (client_ip or "unknown").strip()
        return f"ip:{clean_ip}"

    def check(self, user_id: Optional[str] = None, client_ip: Optional[str] = None, request: Optional[Request] = None):
        """
        Enforces requests-per-minute rate limit.
        Raises HTTP 429 if limit is exceeded.
        """
        resolved_ip = client_ip or self.extract_client_ip(request)
        key = self.get_key(user_id, resolved_ip)
        now = time.time()
        minute_ago = now - 60.0

        # Prune expired timestamps and drop empty entries to keep memory bounded
        history = self._history.get(key, [])
        valid_history = [t for t in history if t > minute_ago]
        if valid_history:
            self._history[key] = valid_history
        else:
            self._history.pop(key, None)

        if len(self._history) > 500:
            stale_keys = [
                k for k, ts_list in self._history.items()
                if not ts_list or ts_list[-1] <= minute_ago
            ]
            for stale_key in stale_keys:
                self._history.pop(stale_key, None)

        limit = getattr(settings, "AI_RATE_LIMIT_PER_MINUTE", self.requests_per_minute)
        if len(valid_history) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Allowed: {limit} requests per minute. Please try again shortly."
            )

        self._history.setdefault(key, []).append(now)

    async def acquire_concurrency(self, user_id: Optional[str] = None, client_ip: Optional[str] = None, request: Optional[Request] = None):
        """
        Ensures a user cannot launch multiple heavy AI generations concurrently.
        Raises HTTP 429 if another generation is already in progress.
        """
        resolved_ip = client_ip or self.extract_client_ip(request)
        key = self.get_key(user_id, resolved_ip)

        async with self._lock:
            current_active = self._active.get(key, 0)
            if current_active >= self.max_concurrent:
                raise HTTPException(
                    status_code=429,
                    detail="An AI generation is already running for your account. Please wait for it to complete."
                )
            self._active[key] = current_active + 1

    async def release_concurrency(self, user_id: Optional[str] = None, client_ip: Optional[str] = None, request: Optional[Request] = None):
        """Releases the concurrent generation lock."""
        resolved_ip = client_ip or self.extract_client_ip(request)
        key = self.get_key(user_id, resolved_ip)

        async with self._lock:
            current_active = self._active.get(key, 0)
            if current_active > 1:
                self._active[key] = current_active - 1
            else:
                self._active.pop(key, None)

# Global rate limiter instance
ai_rate_limiter = RateLimiter(requests_per_minute=10, max_concurrent=1)
