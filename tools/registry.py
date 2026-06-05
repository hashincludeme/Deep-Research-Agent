from typing import Callable, Dict

from tools import web_search, web_scraper, document_reader

# Maps tool names to their callables.
# The orchestrator dispatches by name — add new tools here to make them available.
TOOL_REGISTRY: Dict[str, Callable] = {
    "web_search": web_search.search,
    "web_scraper": web_scraper.scrape,
    "read_document": document_reader.read_document,
}


def get_tool(name: str) -> Callable:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool '{name}'. Available: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name]
