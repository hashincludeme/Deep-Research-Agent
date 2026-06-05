import pytest
from unittest.mock import MagicMock, patch

from tools import web_search, web_scraper
from tools.registry import get_tool


# ── web_search ────────────────────────────────────────────────────────────────

def test_search_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr("config.TAVILY_API_KEY", "")
    results = web_search.search("test query")
    assert results == []


def test_search_returns_empty_on_network_error(monkeypatch):
    monkeypatch.setattr("config.TAVILY_API_KEY", "fake-key")
    with patch("requests.post") as mock_post:
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("Network error")
        results = web_search.search("test query")
    assert results == []


def test_search_normalizes_result_keys(monkeypatch):
    monkeypatch.setattr("config.TAVILY_API_KEY", "fake-key")
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com", "title": "Example", "content": "text", "score": 0.9}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        results = web_search.search("test")
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert results[0]["score"] == 0.9


# ── web_scraper ───────────────────────────────────────────────────────────────

def test_scraper_returns_none_on_network_error():
    with patch("requests.get") as mock_get:
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("Connection refused")
        result = web_scraper.scrape("https://example.com")
    assert result is None


def test_scraper_strips_script_tags():
    html = """
    <html><body>
    <script>alert('noise')</script>
    <article>
    <p>This is the meaningful research content that belongs in the output and is long enough to keep.</p>
    </article>
    </body></html>
    """
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = web_scraper.scrape("https://example.com")
        assert result is not None
        assert "alert" not in result
        assert "meaningful research content" in result


def test_scraper_respects_max_chars():
    html = "<html><body><article>" + ("word " * 10000) + "</article></body></html>"
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = web_scraper.scrape("https://example.com", max_chars=100)
        assert result is None or len(result) <= 100


# ── registry ──────────────────────────────────────────────────────────────────

def test_registry_returns_callable():
    tool = get_tool("web_search")
    assert callable(tool)


def test_registry_raises_on_unknown_tool():
    with pytest.raises(KeyError, match="Unknown tool"):
        get_tool("nonexistent_tool")
