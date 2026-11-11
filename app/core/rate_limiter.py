import asyncio
import math
import threading
import time
from typing import Dict, List, Optional, Tuple
from fastapi import HTTPException, Request
from app.core.config import settings


def _safe_int(value, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return fallback
    return fallback


class RateLimiter:
    """
    Thread-safe sliding-window rate and concurrency limiter.
    Supports per-user identifiers (primary), client-IP fallback, scoped rate buckets
    (e.g., separating chat from ATS analysis), and per-conversation active stream locking.
    """

    DEFAULT_LEASE_TTL_SECONDS = 135.0

    def __init__(self, requests_per_minute: int = 10, max_concurrent: int = 1):
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        # In-memory sliding window history: {scoped_key: [timestamp1, timestamp2, ...]}
        self._history: Dict[str, List[float]] = {}
        # In-flight active requests per user key: {key: count}
        self._active: Dict[str, int] = {}
        # Lease timestamps to prevent permanently stuck concurrency locks if a worker drops
        self._active_leases: Dict[str, List[float]] = {}
        # Per-conversation active stream lock: {f"{key}:conv:{conv_id}": acquired_monotonic}
        self._conversation_streams: Dict[str, float] = {}
        self._history_lock = threading.Lock()
        self._lock = asyncio.Lock()

    @property
    def raw_history(self) -> Dict[str, List[float]]:
        return self._history

    def clear(self):
        """Clears all tracking history and active locks (primarily for test resets)."""
        with self._history_lock:
            self._history.clear()
        self._active.clear()
        self._active_leases.clear()
        self._conversation_streams.clear()

    @staticmethod
    def extract_client_ip(request: Optional[Request]) -> str:
        if not request:
            return "unknown"
        headers = getattr(request, "headers", None)
        if headers:
            forwarded = headers.get("x-forwarded-for")
            if isinstance(forwarded, str) and forwarded.strip():
                return forwarded.split(",")[0].strip()
            cf_ip = headers.get("cf-connecting-ip")
            if isinstance(cf_ip, str) and cf_ip.strip():
                return cf_ip.strip()
        client = getattr(request, "client", None)
        host = getattr(client, "host", None) if client else None
        return host if isinstance(host, str) and host.strip() else "unknown"

    def get_key(
        self,
        user_id: Optional[str],
        client_ip: Optional[str],
        scope: Optional[str] = None,
    ) -> str:
        if user_id and str(user_id).strip():
            base = f"user:{str(user_id).strip()}"
        else:
            clean_ip = (client_ip or "unknown").strip()
            base = f"ip:{clean_ip}"
        if scope and scope != "default":
            return f"{scope}:{base}"
        return base

    def _effective_limit(self, scope: Optional[str]) -> int:
        if scope == "chat":
            raw_chat = getattr(settings, "AI_CHAT_RATE_LIMIT_PER_MINUTE", None)
            if isinstance(raw_chat, (int, float, str)) and not isinstance(raw_chat, bool):
                return max(0, _safe_int(raw_chat, 20))
        raw_default = getattr(settings, "AI_RATE_LIMIT_PER_MINUTE", self.requests_per_minute)
        return max(0, _safe_int(raw_default, self.requests_per_minute))

    def _effective_max_concurrent(self) -> int:
        raw_val = getattr(settings, "AI_MAX_CONCURRENT_REQUESTS", self.max_concurrent)
        return max(1, _safe_int(raw_val, self.max_concurrent))

    @staticmethod
    def _conv_stream_key(user_key: str, conversation_id: Optional[str]) -> Optional[str]:
        if conversation_id and str(conversation_id).strip():
            return f"{user_key}:conv:{str(conversation_id).strip()}"
        return None

    def _prune_stale_leases(self, now_mono: float):
        ttl = self.DEFAULT_LEASE_TTL_SECONDS
        stale_keys: List[str] = []
        for k, timestamps in self._active_leases.items():
            fresh = [ts for ts in timestamps if (now_mono - ts) < ttl]
            if fresh:
                self._active_leases[k] = fresh
                self._active[k] = len(fresh)
            else:
                stale_keys.append(k)
        for k in stale_keys:
            self._active_leases.pop(k, None)
            self._active.pop(k, None)

        stale_convs = [
            ck for ck, ts in self._conversation_streams.items()
            if (now_mono - ts) >= ttl
        ]
        for ck in stale_convs:
            self._conversation_streams.pop(ck, None)

    def check(
        self,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request: Optional[Request] = None,
        scope: Optional[str] = None,
    ):
        """
        Enforces sliding-window requests-per-minute rate limit.
        Raises HTTP 429 with Retry-After header when exceeded.
        """
        resolved_ip = client_ip or self.extract_client_ip(request)
        key = self.get_key(user_id, resolved_ip, scope=scope)
        now = time.time()
        minute_ago = now - 60.0
        limit = self._effective_limit(scope)

        with self._history_lock:
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

            if len(valid_history) >= limit:
                oldest = valid_history[0] if valid_history else now
                retry_after = max(1, math.ceil(60.0 - (now - oldest)))
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Allowed: {limit} requests per minute. Please wait {retry_after} seconds and try again.",
                    headers={"Retry-After": str(retry_after)},
                )

            self._history.setdefault(key, []).append(now)

    async def acquire_concurrency(
        self,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request: Optional[Request] = None,
        conversation_id: Optional[str] = None,
    ):
        """
        Ensures only one active stream per conversation and enforces per-user concurrent request caps.
        Raises HTTP 429 with Retry-After header if another request is active.
        """
        resolved_ip = client_ip or self.extract_client_ip(request)
        key = self.get_key(user_id, resolved_ip)
        conv_key = self._conv_stream_key(key, conversation_id)
        now_mono = time.monotonic()

        async with self._lock:
            self._prune_stale_leases(now_mono)

            if conv_key and conv_key in self._conversation_streams:
                raise HTTPException(
                    status_code=429,
                    detail="A response is already streaming in this conversation. Please wait for it to finish.",
                    headers={"Retry-After": "3"},
                )

            current_active = self._active.get(key, 0)
            limit = self._effective_max_concurrent()
            if current_active >= limit:
                raise HTTPException(
                    status_code=429,
                    detail="An AI generation is already running for your account. Please wait for it to complete.",
                    headers={"Retry-After": "3"},
                )

            self._active[key] = current_active + 1
            self._active_leases.setdefault(key, []).append(now_mono)
            if conv_key:
                self._conversation_streams[conv_key] = now_mono

    async def release_concurrency(
        self,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request: Optional[Request] = None,
        conversation_id: Optional[str] = None,
    ):
        """Releases the concurrent generation lock and optional conversation stream lock."""
        resolved_ip = client_ip or self.extract_client_ip(request)
        key = self.get_key(user_id, resolved_ip)
        conv_key = self._conv_stream_key(key, conversation_id)

        async with self._lock:
            if conv_key:
                self._conversation_streams.pop(conv_key, None)

            leases = self._active_leases.get(key, [])
            if leases:
                leases.pop(0)
                if leases:
                    self._active[key] = len(leases)
                else:
                    self._active_leases.pop(key, None)
                    self._active.pop(key, None)
            else:
                current_active = self._active.get(key, 0)
                if current_active > 1:
                    self._active[key] = current_active - 1
                else:
                    self._active.pop(key, None)


# Global rate limiter instance
ai_rate_limiter = RateLimiter(requests_per_minute=10, max_concurrent=1)
