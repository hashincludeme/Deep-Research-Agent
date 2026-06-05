"""
Structured error hierarchy for Thena.

Every exception carries typed context — tool name, step ID, endpoint, etc. —
and a `retryable` flag the @retry decorator uses to decide whether to wait
and try again or escalate immediately.

Design rule: raise the most specific subclass you can, always with context.
Catching `ThenaError` at the orchestrator boundary gives you structured
context for logging and routing without needing to parse error strings.
"""

from __future__ import annotations

from typing import Optional


class ThenaError(Exception):
    """
    Base class for all Thena errors.

    The `retryable` flag is the contract between an error and the @retry
    decorator. True means the failure is transient (network glitch, rate limit)
    and the operation should be attempted again after backoff. False means
    the failure is permanent or structural and retrying would be wasteful.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.message!r}, retryable={self.retryable})"


class ToolError(ThenaError):
    """
    A registered tool failed during execution.

    Retryable by default because most tool failures are transient:
    network timeouts, DNS failures, empty responses from flaky APIs.
    Set retryable=False for deterministic failures like malformed input
    that will fail identically on every retry.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        step_id: str = "",
        retryable: bool = True,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.tool_name = tool_name
        self.step_id = step_id

    def __repr__(self) -> str:
        return (
            f"ToolError(tool={self.tool_name!r}, step={self.step_id!r}, "
            f"retryable={self.retryable}, message={self.message!r})"
        )


class SubagentError(ThenaError):
    """
    A sub-agent invocation failed.

    Retryable by default — sub-agent failures are typically transient
    (API timeout, context overflow on a single call). The orchestrator
    can reassign the step to a fresh agent instance.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: str = "",
        step_id: str = "",
        retryable: bool = True,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.agent_id = agent_id
        self.step_id = step_id

    def __repr__(self) -> str:
        return (
            f"SubagentError(agent={self.agent_id!r}, step={self.step_id!r}, "
            f"retryable={self.retryable}, message={self.message!r})"
        )


class PlanError(ThenaError):
    """
    The research plan is invalid, corrupt, or cannot be advanced.

    Not retryable by default. Plan errors are structural — a step_id
    that doesn't exist or a plan with no pending steps won't be fixed
    by waiting and trying again. The orchestrator should surface this
    to the user or abort the session.
    """

    def __init__(
        self,
        message: str,
        *,
        step_id: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.step_id = step_id

    def __repr__(self) -> str:
        return (
            f"PlanError(step={self.step_id!r}, retryable={self.retryable}, "
            f"message={self.message!r})"
        )


class BudgetError(ThenaError):
    """
    The session has exceeded its token or cost budget.

    Never retryable. Retrying a budgeted operation after it exceeds budget
    would only deepen the overage. The orchestrator should synthesize from
    whatever has been gathered so far and terminate.
    """

    def __init__(
        self,
        message: str,
        *,
        tokens_used: int = 0,
        budget: int = 0,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.tokens_used = tokens_used
        self.budget = budget

    def __repr__(self) -> str:
        return (
            f"BudgetError(used={self.tokens_used}, budget={self.budget}, "
            f"message={self.message!r})"
        )


class RateLimitError(ThenaError):
    """
    An API endpoint returned a rate limit response (HTTP 429 or equivalent).

    Always retryable. The @retry decorator uses `retry_after` as a
    minimum delay floor when provided — the bucket refill time is
    authoritative over our own backoff calculation.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str = "",
        retry_after: Optional[float] = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.endpoint = endpoint
        self.retry_after = retry_after

    def __repr__(self) -> str:
        return (
            f"RateLimitError(endpoint={self.endpoint!r}, "
            f"retry_after={self.retry_after}, message={self.message!r})"
        )


class SourceNotFoundError(ThenaError):
    """
    A URL or document source could not be retrieved.

    Not retryable by default. A 404 or permanently redirected URL
    won't be fixed by a second attempt. The orchestrator should mark
    the source as unavailable and continue with other sources.
    Set retryable=True for transient 503/504 responses where the
    server might recover.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.url = url
        self.status_code = status_code

    def __repr__(self) -> str:
        return (
            f"SourceNotFoundError(url={self.url!r}, status={self.status_code}, "
            f"retryable={self.retryable}, message={self.message!r})"
        )
