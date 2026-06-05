"""
Tool registry for Ariadne.

Tools register themselves on import via the @tool decorator. The orchestrator
never hardcodes which tools exist — it calls to_anthropic_tools() to get the
current set and lets the LLM decide which to invoke based on each tool's
description. The dispatcher never grows. The if/else never gets written.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel


# ── Exceptions ────────────────────────────────────────────────────────────────


class ToolNotFoundError(KeyError):
    """Raised when get() or dispatch() is called with an unregistered name."""


class DuplicateToolError(ValueError):
    """Raised when register() is called with a name that is already taken."""


# ── ToolDefinition ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolDefinition:
    """
    Complete specification of one tool: its name, what it does, what it accepts,
    what it returns, and the function that executes it.

    Frozen so that registered tools cannot be mutated after registration.
    The ToolDefinition IS the contract — name, schema, and description in one place.
    """

    name: str
    """
    Namespaced tool name, e.g. 'search.pubmed_query' or 'scrape.full_page'.
    The part before the dot is the namespace, used by list_by_namespace().
    Dots are converted to double-underscores for the Anthropic API (see api_name).
    """

    description: str
    """
    Natural language description of WHAT this tool does and WHEN TO USE IT.

    This is the primary mechanism by which the model selects tools.
    Write it as a routing instruction, not a technical docstring.

      GOOD: "Search PubMed for peer-reviewed medical literature. Use when the
             question requires clinical evidence or systematic reviews.
             NOT for general web searches — use search.web instead."
      BAD:  "Calls the PubMed API and returns a list of articles."

    The model reads this string and decides whether this tool applies to the
    current research step. Every word is a signal for or against selection.
    """

    input_schema: Type[BaseModel]
    """
    Pydantic model class for input parameters. Serves two purposes:
    1. Runtime validation — input is validated before the handler is called.
    2. API schema generation — model_json_schema() produces the JSON Schema
       that Anthropic's API sends to the model so it knows what args to provide.
    """

    output_schema: Type[BaseModel]
    """
    Pydantic model class for the handler's return value.
    Used for runtime validation of output before returning to the orchestrator.
    Not sent to the Anthropic API — models don't need to know the output shape.
    """

    handler: Callable
    """
    The function (sync or async) that executes this tool.
    Called by dispatch() after input validation, with validated_input.model_dump()
    unpacked as keyword arguments.
    Must return either a dict or an instance of output_schema.
    """

    namespace: str
    """
    The prefix part of the name (e.g. 'search' from 'search.pubmed_query').
    Derived automatically from name in the @tool decorator if not provided.
    Used by list_by_namespace() to group related tools.
    """

    @property
    def api_name(self) -> str:
        """
        Sanitized name for Anthropic's API.
        Anthropic validates tool names against ^[a-zA-Z0-9_-]+$ — dots not allowed.
        We convert 'search.pubmed_query' → 'search__pubmed_query'.
        The registry reverses this in get_by_api_name() and dispatch_by_api_name().
        """
        return self.name.replace(".", "__")

    @property
    def short_name(self) -> str:
        """The part after the namespace prefix: 'pubmed_query' from 'search.pubmed_query'."""
        parts = self.name.split(".", 1)
        return parts[1] if len(parts) > 1 else self.name

    def input_json_schema(self) -> dict:
        """
        JSON Schema dict for the input Pydantic model, ready for Anthropic's API.
        Strips the 'title' key Pydantic adds at the root — not required by Anthropic
        and adds noise to the prompt the model reads.
        """
        schema = self.input_schema.model_json_schema()
        schema.pop("title", None)
        return schema


# ── ToolRegistry ──────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Central registry of all tools available to the agent.

    Tools self-register on import via the @tool decorator. The orchestrator
    never lists tool names in code — it calls to_anthropic_tools() and lets
    the LLM do the dispatch via description-based selection.

    To add a new tool:
      1. Create a file in the tools/ directory.
      2. Define input/output Pydantic models.
      3. Decorate the handler with @tool(...).
      4. Import the module anywhere before the agent runs.
         Registration is a side-effect of the import.

    For testing, instantiate a fresh ToolRegistry() rather than using the
    global registry — this keeps test state isolated from production state.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ── Mutation ──────────────────────────────────────────────────────────────

    def register(self, tool_def: ToolDefinition) -> None:
        """
        Add a ToolDefinition to the registry.
        Raises DuplicateToolError if the name is already taken — prevents silent
        overwrites when a module is accidentally imported twice.
        """
        if tool_def.name in self._tools:
            raise DuplicateToolError(
                f"Tool '{tool_def.name}' is already registered. "
                "Use a unique name or call unregister() first."
            )
        self._tools[tool_def.name] = tool_def

    def unregister(self, name: str) -> None:
        """Remove a tool by name. Primarily for test teardown."""
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Cannot unregister '{name}': not found. "
                f"Available: {sorted(self._tools.keys())}"
            )
        del self._tools[name]

    def clear(self) -> None:
        """Remove all registered tools. Primarily for test isolation."""
        self._tools.clear()

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> ToolDefinition:
        """
        Retrieve a ToolDefinition by its internal namespaced name.
        Raises ToolNotFoundError (with the list of available names) if not found.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"No tool registered as '{name}'. "
                f"Available: {sorted(self._tools.keys())}"
            )
        return self._tools[name]

    def get_by_api_name(self, api_name: str) -> ToolDefinition:
        """
        Retrieve a tool by its Anthropic API name (double-underscore separator).
        Called when processing the model's tool_use response blocks:
          'search__pubmed_query' → looks up 'search.pubmed_query'
        """
        return self.get(api_name.replace("__", "."))

    def list_by_namespace(self, namespace: str) -> list[ToolDefinition]:
        """
        Return all tools in the given namespace, sorted by name.
        Lets the orchestrator pass a focused subset to the model for each phase
        (e.g., only 'search' tools during evidence gathering, only 'analyze'
        tools during synthesis). Fewer choices → better model decisions.
        """
        return sorted(
            [t for t in self._tools.values() if t.namespace == namespace],
            key=lambda t: t.name,
        )

    def all_namespaces(self) -> list[str]:
        """Return sorted list of all unique namespaces currently registered."""
        return sorted({t.namespace for t in self._tools.values()})

    def all_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ── API format ────────────────────────────────────────────────────────────

    def to_anthropic_tools(
        self,
        namespace: Optional[str] = None,
    ) -> list[dict]:
        """
        Convert registered tools to the format Anthropic's messages API expects.

        Optionally filter to a single namespace — useful when the orchestrator
        wants to give the model a focused subset for a specific agent phase.

        Output format per tool:
          {
            "name": "search__pubmed_query",   # dots replaced with __
            "description": "...",              # the routing instruction
            "input_schema": {                  # JSON Schema from Pydantic model
              "type": "object",
              "properties": { ... },
              "required": [ ... ]
            }
          }

        The model receives this list and uses each tool's description to decide
        which one to call for the current step. input_schema tells it what
        arguments to construct. The orchestrator dispatches via dispatch_by_api_name().

        Usage:
          client.messages.create(
              model="...",
              tools=registry.to_anthropic_tools(),
              ...
          )
          # or for just search tools:
          tools=registry.to_anthropic_tools(namespace="search")
        """
        if namespace is not None:
            tools = self.list_by_namespace(namespace)
        else:
            tools = sorted(self._tools.values(), key=lambda t: t.name)

        return [
            {
                "name": t.api_name,
                "description": t.description,
                "input_schema": t.input_json_schema(),
            }
            for t in tools
        ]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def dispatch(self, tool_name: str, raw_args: dict[str, Any]) -> BaseModel:
        """
        The single execution path for all tool calls in the agent loop.

        Steps:
        1. Look up tool by internal name (raises ToolNotFoundError if missing)
        2. Validate input against input_schema (raises ValidationError if invalid)
        3. Call the handler (supports both sync and async handlers)
        4. Validate output against output_schema (raises ValidationError if invalid)
        5. Return the validated output model instance

        The orchestrator calls this with the tool name and args extracted from
        the model's tool_use response block. Validation at both ends means:
        - The model can't pass malformed arguments (caught by input_schema)
        - The handler can't return a malformed result (caught by output_schema)
        """
        tool_def = self.get(tool_name)

        validated_input = tool_def.input_schema(**raw_args)

        if asyncio.iscoroutinefunction(tool_def.handler):
            raw_result = await tool_def.handler(**validated_input.model_dump())
        else:
            raw_result = tool_def.handler(**validated_input.model_dump())

        if isinstance(raw_result, tool_def.output_schema):
            return raw_result
        if isinstance(raw_result, dict):
            return tool_def.output_schema(**raw_result)
        raise TypeError(
            f"Handler for '{tool_name}' returned {type(raw_result).__name__}, "
            f"expected dict or {tool_def.output_schema.__name__}."
        )

    async def dispatch_by_api_name(
        self, api_name: str, raw_args: dict[str, Any]
    ) -> BaseModel:
        """
        Dispatch using the Anthropic API name (double-underscore separator).
        Convenience wrapper for processing tool_use blocks from the model response:

          for block in response.content:
              if block.type == "tool_use":
                  result = await registry.dispatch_by_api_name(block.name, block.input)
        """
        return await self.dispatch(api_name.replace("__", "."), raw_args)


# ── Global registry and @tool decorator ───────────────────────────────────────


_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """
    Returns the module-level global ToolRegistry.
    Populated automatically as tool modules are imported.
    The orchestrator calls this once and holds the reference for the session.
    """
    return _global_registry


def tool(
    name: str,
    description: str,
    input_schema: Type[BaseModel],
    output_schema: Type[BaseModel],
    namespace: str = "",
) -> Callable:
    """
    Decorator that registers a function as a tool in the global registry.

    Registration is a side-effect of the import — just importing a module
    that uses @tool makes the tool available to any code that calls get_registry().
    No explicit registration call required anywhere.

    The namespace is derived from the name prefix if not provided:
      'search.web' → namespace='search'
      'analyze.credibility' → namespace='analyze'

    Usage:

        class WebSearchInput(BaseModel):
            query: str = Field(description="The search query")
            max_results: int = Field(default=5, ge=1, le=20)

        class WebSearchOutput(BaseModel):
            results: list[dict]
            total: int

        @tool(
            name="search.web",
            description=(
                "Search the web for recent information on any topic. "
                "Use for general research, current events, or company information. "
                "NOT for medical literature — use search.pubmed_query instead."
            ),
            input_schema=WebSearchInput,
            output_schema=WebSearchOutput,
        )
        async def web_search(query: str, max_results: int = 5) -> dict:
            ...

    After import, `get_registry().get("search.web")` returns the ToolDefinition.
    `get_registry().to_anthropic_tools()` includes this tool automatically.
    """
    derived_namespace = namespace or (name.split(".")[0] if "." in name else name)

    def decorator(func: Callable) -> Callable:
        tool_def = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            handler=func,
            namespace=derived_namespace,
        )
        _global_registry.register(tool_def)

        # Attach the ToolDefinition for introspection without going through the registry
        func._tool_definition = tool_def  # type: ignore[attr-defined]
        return func

    return decorator
