import requests
from bs4 import BeautifulSoup
from typing import Optional

import config

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Tags that add noise but no research value
_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "button"]

# Minimum line length to keep — filters out menu items, breadcrumbs, etc.
_MIN_LINE_LEN = 40


def scrape(url: str, max_chars: Optional[int] = None) -> Optional[str]:
    """
    Fetches a URL and returns clean text content.
    Prefers <article> or <main> over full body to reduce noise.
    Returns None if the page cannot be fetched or parsed.
    """
    max_chars = max_chars or config.CONTENT_MAX_CHARS

    try:
        response = requests.get(url, headers=_HEADERS, timeout=12)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [scraper] Fetch failed {url[:60]}: {e}")
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        content_tag = soup.find("article") or soup.find("main") or soup.body
        if not content_tag:
            return None

        text = content_tag.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= _MIN_LINE_LEN]
        cleaned = "\n".join(lines)

        return cleaned[:max_chars] if cleaned else None

    except Exception as e:
        print(f"  [scraper] Parse error {url[:60]}: {e}")
        return None
