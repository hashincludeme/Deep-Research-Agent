"""
Structured logging for Ariadne using structlog.

Every log line is a JSON object. session_id and step_id are injected
automatically into every log call via contextvars — the orchestrator
binds them once at session start, and all downstream code picks them
up without passing them explicitly.

JSON output is the default. In development, call configure_logging()
with json_output=False for human-readable colored terminal output.

Usage:

    # In main.py, once at startup:
    from ariadne.core.logger import configure_logging
    configure_logging(json_output=True, log_level="INFO")

    # In the orchestrator, at session start:
    from ariadne.core.logger import bind_session
    bind_session(session_id=session.plan.session_id)

    # Before each step:
    from ariadne.core.logger import bind_step
    bind_step(step_id=step.id)

    # In any module:
    from ariadne.core.logger import get_logger
    log = get_logger(__name__)
    log.info("tool_called", tool_name="search.web", query="enterprise churn")

    # Output (JSON):
    # {"event": "tool_called", "tool_name": "search.web", "query": "enterprise churn",
    #  "session_id": "abc123", "step_id": "def456", "level": "info",
    #  "logger": "ariadne.tools.web_search", "timestamp": "2025-01-15T14:23:01Z"}
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


# ── Internal processors ───────────────────────────────────────────────────────

def _add_logger_name(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Add 'logger' key to the event dict.

    structlog.stdlib.add_logger_name requires a stdlib Logger with a .name
    attribute, but PrintLoggerFactory produces PrintLogger objects that don't
    have one. This processor handles both gracefully.
    """
    name = getattr(logger, "name", None)
    if name:
        event_dict["logger"] = name
    return event_dict


# ── Configuration ─────────────────────────────────────────────────────────────

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_state: dict[str, bool] = {"configured": False}


def configure_logging(json_output: bool = True, log_level: str = "INFO") -> None:
    """
    Configure structlog for the entire process.

    Call once at application startup (in main.py). Safe to call multiple
    times — subsequent calls reconfigure structlog, which is useful in tests.

    Parameters
    ----------
    json_output:
        True  → JSONRenderer, one JSON object per line, machine-parseable.
                Use in production and for any output piped to log aggregators.
        False → ConsoleRenderer with colors, human-readable dev output.
                Use during local development.
    log_level:
        Minimum level to emit. "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        INFO is the production default — DEBUG adds substantial volume.
    """
    level_int = _LEVEL_MAP.get(log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        # Inject session_id and step_id from context vars (set by bind_session/bind_step)
        structlog.contextvars.merge_contextvars,
        # Add "level" key (info, debug, error, etc.)
        structlog.stdlib.add_log_level,
        # Add "logger" key (the name passed to get_logger)
        _add_logger_name,
        # Add "timestamp" in ISO 8601 format
        structlog.processors.TimeStamper(fmt="iso"),
        # Render any attached stack info
        structlog.processors.StackInfoRenderer(),
        # Convert exc_info tuples to formatted tracebacks in the event dict
        structlog.processors.format_exc_info,
    ]

    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        # make_filtering_bound_logger creates a logger that drops calls
        # below the configured level before any processing happens.
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _state["configured"] = True


# ── Logger factory ────────────────────────────────────────────────────────────

def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to `name`.

    Call at module level: `log = get_logger(__name__)`
    The logger inherits whatever session_id/step_id are in the current context.
    All bound keys (tool_name, url, etc.) added via log.bind() are local to
    that logger instance — they don't bleed into the global context.
    """
    if not _state["configured"]:
        # Apply a safe default so log calls before configure_logging()
        # work in tests and scripts without explicit setup.
        configure_logging(json_output=False, log_level="INFO")

    return structlog.get_logger(name)


# ── Context binding ───────────────────────────────────────────────────────────

def bind_session(session_id: str) -> None:
    """
    Bind session_id to all log calls on this async task/thread.

    Uses structlog's contextvars integration — the binding is scoped
    to the current asyncio task context, not the entire process.
    Concurrent research sessions in different tasks have independent contexts.
    """
    structlog.contextvars.bind_contextvars(session_id=session_id)


def bind_step(step_id: str) -> None:
    """
    Bind step_id to all subsequent log calls on this async task/thread.

    Call at the start of each research step. The step_id overwrites
    the previous value — it's a running pointer to the current step,
    not a stack.
    """
    structlog.contextvars.bind_contextvars(step_id=step_id)


def unbind_step() -> None:
    """
    Remove step_id from the context (e.g., during synthesis, between steps).
    """
    structlog.contextvars.unbind_contextvars("step_id")


def clear_context() -> None:
    """
    Remove all bound context vars (session_id, step_id, and any others).
    Call at the start of each test to prevent context leakage between runs.
    """
    structlog.contextvars.clear_contextvars()
