import requests
from typing import Any, Dict, List, Optional

import config

_TAVILY_URL = "https://api.tavily.com/search"


def search(query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Searches the web via Tavily API.
    Returns list of {url, title, content, score} dicts.
    Returns empty list (with a warning) if API key is missing — keeps the
    rest of the pipeline running so the agent gracefully degrades.
    """
    max_results = max_results or config.MAX_SEARCH_RESULTS

    if not config.TAVILY_API_KEY:
        print(f"  [search] TAVILY_API_KEY not set — skipping: {query[:60]}")
        return []

    try:
        response = requests.post(
            _TAVILY_URL,
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]
    except requests.RequestException as e:
        print(f"  [search] Request failed: {e}")
        return []
