"""
Rate Limiting for Web API

Provides in-memory rate limiting with optional Redis backend.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Optional, Tuple, Callable
import time
from collections import defaultdict
from dataclasses import dataclass, field
import asyncio


@dataclass
class RateLimitEntry:
    """Rate limit tracking entry"""
    count: int = 0
    window_start: float = field(default_factory=time.time)


class InMemoryRateLimiter:
    """Thread-safe in-memory rate limiter using sliding window"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.rpd = requests_per_day

        self.minute_windows: Dict[str, RateLimitEntry] = {}
        self.hour_windows: Dict[str, RateLimitEntry] = {}
        self.day_windows: Dict[str, RateLimitEntry] = {}

        self._lock = asyncio.Lock()

    async def check(self, key: str) -> Tuple[bool, dict]:
        """
        Check if request is allowed.

        Args:
            key: Client identifier (IP, API key, etc.)

        Returns:
            (allowed: bool, headers: dict with rate limit info)
        """
        async with self._lock:
            now = time.time()

            # Check minute limit
            minute_entry = self.minute_windows.get(key)
            if minute_entry is None or now - minute_entry.window_start >= 60:
                minute_entry = RateLimitEntry(count=0, window_start=now)
                self.minute_windows[key] = minute_entry

            minute_remaining = self.rpm - minute_entry.count
            if minute_remaining <= 0:
                retry_after = int(60 - (now - minute_entry.window_start)) + 1
                return False, {
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(minute_entry.window_start + 60)),
                    "Retry-After": str(retry_after),
                }

            # Check hour limit
            hour_entry = self.hour_windows.get(key)
            if hour_entry is None or now - hour_entry.window_start >= 3600:
                hour_entry = RateLimitEntry(count=0, window_start=now)
                self.hour_windows[key] = hour_entry

            hour_remaining = self.rph - hour_entry.count
            if hour_remaining <= 0:
                retry_after = int(3600 - (now - hour_entry.window_start)) + 1
                return False, {
                    "X-RateLimit-Limit": str(self.rph),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(hour_entry.window_start + 3600)),
                    "Retry-After": str(retry_after),
                }

            # All limits passed, increment counters
            minute_entry.count += 1
            hour_entry.count += 1

            return True, {
                "X-RateLimit-Limit": str(self.rpm),
                "X-RateLimit-Remaining": str(max(0, minute_remaining - 1)),
                "X-RateLimit-Reset": str(int(minute_entry.window_start + 60)),
            }

    def get_client_key(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try X-Forwarded-For header first (for reverse proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to client host
        if request.client:
            return request.client.host

        # Fallback for edge cases
        return "unknown"

    def cleanup(self, max_age: float = 86400):
        """Remove stale entries older than max_age seconds"""
        now = time.time()

        # Clean minute windows
        stale_keys = [
            k for k, v in self.minute_windows.items()
            if now - v.window_start > 60
        ]
        for k in stale_keys:
            del self.minute_windows[k]

        # Clean hour windows
        stale_keys = [
            k for k, v in self.hour_windows.items()
            if now - v.window_start > 3600
        ]
        for k in stale_keys:
            del self.hour_windows[k]

        # Clean day windows
        stale_keys = [
            k for k, v in self.day_windows.items()
            if now - v.window_start > max_age
        ]
        for k in stale_keys:
            del self.day_windows[k]


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""

    def __init__(
        self,
        rate_limiter: InMemoryRateLimiter = None,
        exclude_paths: list = None,
        exclude_methods: list = None
    ):
        self.limiter = rate_limiter or InMemoryRateLimiter()
        self.exclude_paths = set(exclude_paths or ['/api/status', '/', '/health', '/docs', '/openapi.json'])
        self.exclude_methods = set(exclude_methods or ['OPTIONS', 'HEAD'])

    async def __call__(self, request: Request, call_next: Callable):
        """Process request through rate limiter"""

        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Skip excluded methods
        if request.method in self.exclude_methods:
            return await call_next(request)

        # Get client key
        client_key = self.limiter.get_client_key(request)

        # Check rate limit
        allowed, headers = await self.limiter.check(client_key)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": "Too many requests. Please try again later."
                },
                headers=headers
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response


class RedisRateLimiter:
    """Redis-backed rate limiter for distributed deployments"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.redis_url = redis_url
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._redis = None

    async def _get_redis(self):
        """Lazy Redis connection"""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = await redis.from_url(self.redis_url)
            except ImportError:
                raise RuntimeError("Redis package not installed. Run: pip install redis")
        return self._redis

    async def check(self, key: str) -> Tuple[bool, dict]:
        """Check rate limit using Redis"""
        redis = await self._get_redis()
        now = time.time()

        # Use Redis pipeline for atomic operations
        async with redis.pipeline() as pipe:
            minute_key = f"rl:{key}:minute"
            hour_key = f"rl:{key}:hour"

            # Increment counters
            await pipe.incr(minute_key)
            await pipe.expire(minute_key, 60)
            await pipe.incr(hour_key)
            await pipe.expire(hour_key, 3600)

            results = await pipe.execute()

        minute_count = results[0]
        hour_count = results[2]

        if minute_count > self.rpm:
            return False, {
                "X-RateLimit-Limit": str(self.rpm),
                "X-RateLimit-Remaining": "0",
            }

        if hour_count > self.rph:
            return False, {
                "X-RateLimit-Limit": str(self.rph),
                "X-RateLimit-Remaining": "0",
            }

        return True, {
            "X-RateLimit-Limit": str(self.rpm),
            "X-RateLimit-Remaining": str(max(0, self.rpm - minute_count)),
        }
