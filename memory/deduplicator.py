from typing import Set


class Deduplicator:
    """
    Tracks URLs and search queries already used in this session.
    Prevents re-fetching the same page or re-running the same search query.
    Persisted as part of ResearchSession checkpointing.
    """

    def __init__(self):
        self._visited_urls: Set[str] = set()
        self._executed_queries: Set[str] = set()

    def is_url_visited(self, url: str) -> bool:
        return url in self._visited_urls

    def mark_url_visited(self, url: str) -> None:
        self._visited_urls.add(url)

    def is_query_executed(self, query: str) -> bool:
        return query.lower().strip() in self._executed_queries

    def mark_query_executed(self, query: str) -> None:
        self._executed_queries.add(query.lower().strip())

    def to_dict(self) -> dict:
        return {
            "visited_urls": list(self._visited_urls),
            "executed_queries": list(self._executed_queries),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Deduplicator":
        d = cls()
        d._visited_urls = set(data.get("visited_urls", []))
        d._executed_queries = set(data.get("executed_queries", []))
        return d
