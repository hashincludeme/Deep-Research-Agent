"""
Token bucket rate limiter, async-safe, configurable per API endpoint.

A token bucket allows short bursts (when tokens have accumulated during
idle periods) while enforcing a long-term average throughput. Tokens
refill continuously at `rate` per second up to `capacity` — there is no
window boundary where a burst of 2x the intended rate becomes valid.

The asyncio.Lock ensures atomic check-and-consume: without it, two
coroutines could both observe "1 token remaining" and both consume it,
overdrafting the bucket. acquire() releases the lock before sleeping so
other coroutines aren't blocked during the wait, then re-acquires after
waking up to do the final consume.

Pre-configured instances at the bottom of this module match the actual
rate limits of the APIs Thena uses. Import and call `await limiter.acquire()`
before any outbound API request.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucketRateLimiter:
    """
    Async-safe token bucket rate limiter for a single API endpoint.

    Parameters
    ----------
    rate:
        Tokens added per second. This is the sustained throughput ceiling.
        Example: rate=0.9 allows ~54 requests/minute long-term.
    capacity:
        Maximum tokens the bucket can hold. This is the burst allowance.
        An idle limiter accumulates tokens up to this cap, then allows a
        burst of `capacity` requests before throttling to `rate` per second.
    name:
        Human-readable label used in log output and repr.
    """

    rate: float
    capacity: float
    name: str = "default"

    _tokens: float = field(init=False, repr=False)
    _last_refill: float = field(init=False, repr=False)
    _lock: Optional[asyncio.Lock] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._tokens = self.capacity  # start full — first burst doesn't wait
        self._last_refill = time.monotonic()

    def _get_lock(self) -> asyncio.Lock:
        """
        Lazy lock creation. Deferring until first use avoids attaching to
        the wrong event loop if the limiter is instantiated at import time
        (before asyncio.run() is called).
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self) -> None:
        """
        Add tokens based on time elapsed since last refill.
        Must be called inside the lock — no concurrent modification.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Consume `tokens` from the bucket, waiting if necessary.

        The critical pattern:
        1. Acquire lock, refill, check.
        2. If sufficient: consume and return.
        3. If insufficient: calculate deficit wait, release lock, sleep.
        4. After sleep, loop back to step 1.

        Releasing the lock before sleeping means other coroutines can make
        progress while this one waits. After sleeping, we refill again to
        account for actual elapsed time (which may differ from calculated
        wait if the sleep ran long), then consume.

        The loop handles the case where another coroutine consumed tokens
        during our sleep — we just recalculate and wait again.
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds bucket capacity {self.capacity} "
                f"for limiter '{self.name}'. This request would wait forever."
            )

        while True:
            wait_seconds: float = 0.0

            async with self._get_lock():
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Compute how long until the bucket has enough tokens
                deficit = tokens - self._tokens
                wait_seconds = deficit / self.rate
                # Do NOT consume here — we release the lock and sleep first,
                # then re-check on the next loop iteration.

            # Sleep outside the lock so other coroutines aren't blocked
            await asyncio.sleep(wait_seconds)

    @property
    def available_tokens(self) -> float:
        """Estimated current token count, accounting for elapsed refill time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        return min(self.capacity, self._tokens + elapsed * self.rate)

    def __repr__(self) -> str:
        return (
            f"TokenBucketRateLimiter(name={self.name!r}, rate={self.rate}/s, "
            f"capacity={self.capacity}, available≈{self.available_tokens:.1f})"
        )


# ── Per-endpoint pre-configured limiters ──────────────────────────────────────
#
# Module-level singletons. Usage:
#
#   from thena.core.rate_limiter import ANTHROPIC_LIMITER
#   await ANTHROPIC_LIMITER.acquire()
#   response = await client.messages.create(...)
#
# Rates are set conservatively below documented API limits to leave
# headroom and avoid hard 429 bans.
#
# Anthropic Tier 1: ~50 req/min for claude-3 models.
#   Using rate=0.75/s (~45 req/min) with burst capacity=5.
#
# Tavily free tier: ~100 searches/month. Paid: ~60 req/min.
#   Conservative default: rate=0.5/s, capacity=5.
#
# Override for higher tiers:
#   import thena.core.rate_limiter as rl
#   rl.ANTHROPIC_LIMITER = TokenBucketRateLimiter(rate=5.0, capacity=20, name="anthropic")

ANTHROPIC_LIMITER: TokenBucketRateLimiter = TokenBucketRateLimiter(
    rate=0.75,
    capacity=5,
    name="anthropic",
)

TAVILY_LIMITER: TokenBucketRateLimiter = TokenBucketRateLimiter(
    rate=0.5,
    capacity=5,
    name="tavily",
)
