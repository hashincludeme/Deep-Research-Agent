from typing import Dict, List
from urllib.parse import urlparse


def build_citation_list(sources: List[str]) -> List[Dict[str, str]]:
    """
    Converts an ordered list of source URLs into a numbered citation list.
    Deduplicates while preserving first-seen order.
    Returns [{index, url, domain}]
    """
    seen: Dict[str, int] = {}
    citations = []
    for url in sources:
        if not url or url in seen:
            continue
        idx = len(citations) + 1
        seen[url] = idx
        citations.append({
            "index": str(idx),
            "url": url,
            "domain": _extract_domain(url),
        })
    return citations


def format_citation_list(citations: List[Dict[str, str]]) -> str:
    """Formats citations as a markdown references section."""
    if not citations:
        return ""
    lines = ["## References", ""]
    for c in citations:
        lines.append(f"[{c['index']}] {c['url']}")
    return "\n".join(lines)


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url
