"""
Async retry decorator with exponential backoff and full jitter.

Only retries AriadneError subclasses where retryable=True. Non-retryable
errors (BudgetError, PlanError, SourceNotFoundError on 404) propagate
immediately — no delay, no retry attempt.

Jitter formula (full jitter, AWS recommendation):
    delay = uniform(0, min(max_delay, base_delay * 2^attempt))

Why full jitter over "equal jitter" (delay = cap/2 + uniform(0, cap/2)):
  Full jitter produces better load distribution at scale. Equal jitter
  guarantees at least half the max delay, which helps individual callers
  but concentrates retries in a narrower time band. For agents where
  10+ concurrent sub-tasks might hit the same rate limit simultaneously,
  full jitter spreads the retry cloud more evenly.
"""

from __future__ import annotations

import asyncio
import functools
import random
from typing import Callable, TypeVar

from .errors import AriadneError, RateLimitError

F = TypeVar("F", bound=Callable)


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[AriadneError], ...] = (AriadneError,),
) -> Callable[[F], F]:
    """
    Async retry decorator with full jitter exponential backoff.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts after the initial failure.
        Total calls = max_retries + 1. On the final attempt, the
        exception is re-raised rather than retried.
    base_delay:
        Base delay in seconds for the first retry. The cap for attempt n
        is min(max_delay, base_delay * 2^n). The actual delay is a
        uniform sample from [0, cap].
    max_delay:
        Upper bound on the jitter cap regardless of attempt number.
        Without this, delays would grow unboundedly for large max_retries.
    exceptions:
        Tuple of AriadneError subclasses to catch and retry.
        Defaults to (AriadneError,) — any retryable AriadneError.
        Restrict to specific types when you only want to retry certain
        failure modes: @retry(exceptions=(RateLimitError, ToolError))

    Non-retryable errors:
        Any AriadneError with retryable=False is re-raised immediately,
        even on the first attempt. Any non-AriadneError exception
        propagates without being caught at all.

    RateLimitError special case:
        If the caught error is a RateLimitError with retry_after set,
        the delay is max(jitter_delay, retry_after) to respect the
        API's own backoff instruction.

    Usage:

        @retry(max_retries=3, base_delay=1.0)
        async def call_tavily(query: str) -> list[dict]:
            try:
                return await tavily_client.search(query)
            except TavilyRateLimitError as e:
                raise RateLimitError(str(e), endpoint="tavily", retry_after=30.0)
            except requests.RequestException as e:
                raise ToolError(str(e), tool_name="search.tavily", retryable=True)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: AriadneError | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as exc:
                    if not exc.retryable:
                        raise

                    last_exc = exc

                    if attempt == max_retries:
                        # Exhausted all retries — re-raise the last exception
                        raise

                    # Full jitter: uniform(0, min(max_delay, base_delay * 2^attempt))
                    cap = min(max_delay, base_delay * (2 ** attempt))
                    delay = random.uniform(0, cap)

                    # Respect Retry-After headers for rate limit responses
                    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                        delay = max(delay, exc.retry_after)

                    await asyncio.sleep(delay)

            # Unreachable — the loop always raises on the final attempt —
            # but satisfies type checkers that need an explicit raise path.
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
