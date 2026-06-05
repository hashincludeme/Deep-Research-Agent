"""
ariadne/tools/search.py

Ten search-namespace tools for Thena. All tools share the same execution contract:
  1. Acquire a rate-limiter token before the HTTP call.
  2. Make the request via asyncio.to_thread — sync requests.get runs in a thread
     pool so it never blocks the event loop.
  3. Convert HTTP/network failures to ThenaError subclasses so @retry can decide
     whether to wait and retry or surface the error immediately.
  4. Return a plain dict; registry.dispatch() validates it against output_schema.

Rate limits (conservative defaults below documented limits):
  NCBI/PubMed       3/s   (10/s with NCBI_API_KEY)
  Semantic Scholar  0.3/s (1/s with S2_API_KEY)
  ArXiv             1/s   (documented 3/s — be courteous)
  CrossRef          2/s   (polite pool)
  bioRxiv/medRxiv   1/s   (no documented limit)
  Web               2/s   (general scraping)
  Google Scholar    0.05/s (Scholar aggressively blocks — very conservative)
"""

from __future__ import annotations

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from core.errors import ThenaError, ToolError, RateLimitError, SourceNotFoundError
from core.retry import retry
from core.rate_limiter import TokenBucketRateLimiter
from registry.registry import tool


# ── Per-API rate limiters ──────────────────────────────────────────────────────

_PUBMED_LIMITER   = TokenBucketRateLimiter(rate=3.0,  capacity=5, name="ncbi")
_S2_LIMITER       = TokenBucketRateLimiter(rate=0.3,  capacity=3, name="semantic_scholar")
_ARXIV_LIMITER    = TokenBucketRateLimiter(rate=1.0,  capacity=3, name="arxiv")
_CROSSREF_LIMITER = TokenBucketRateLimiter(rate=2.0,  capacity=5, name="crossref")
_BIORXIV_LIMITER  = TokenBucketRateLimiter(rate=1.0,  capacity=3, name="biorxiv")
_WEB_LIMITER      = TokenBucketRateLimiter(rate=2.0,  capacity=5, name="web")
_SCHOLAR_LIMITER  = TokenBucketRateLimiter(rate=0.05, capacity=1, name="google_scholar")


# ── Configuration from environment ────────────────────────────────────────────

_NCBI_EMAIL   = os.getenv("NCBI_EMAIL", "research@thena.ai")
_NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
_S2_API_KEY   = os.getenv("S2_API_KEY", "")

_NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_S2_BASE      = "https://api.semanticscholar.org/graph/v1"
_ARXIV_BASE   = "http://export.arxiv.org/api/query"
_CROSSREF_BASE = "https://api.crossref.org/works"
_BIORXIV_BASE = "https://api.biorxiv.org"

_HEADERS = {"User-Agent": "Thena Research Agent/1.0 (contact: research@thena.ai)"}


# ── HTTP helper ────────────────────────────────────────────────────────────────

async def _get(
    url: str,
    tool_name: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 30,
) -> requests.Response:
    """
    Async GET via thread pool with ThenaError conversion.

    429  → RateLimitError (retryable, with Retry-After if present)
    404  → SourceNotFoundError (not retryable)
    5xx  → ToolError retryable=True
    4xx  → ToolError retryable=False
    network → ToolError retryable=True
    """
    merged = {**_HEADERS, **(headers or {})}
    try:
        resp: requests.Response = await asyncio.to_thread(
            requests.get, url, params=params, headers=merged, timeout=timeout
        )
    except requests.Timeout:
        raise ToolError(f"Timeout ({timeout}s) fetching {url}", tool_name=tool_name, retryable=True)
    except requests.RequestException as exc:
        raise ToolError(f"Network error: {exc}", tool_name=tool_name, retryable=True)

    if resp.status_code == 429:
        retry_after = float(resp.headers.get("Retry-After", 30.0))
        raise RateLimitError(f"Rate limited by {url}", endpoint=url, retry_after=retry_after)
    if resp.status_code == 404:
        raise SourceNotFoundError(f"Not found: {url}", url=url, status_code=404, retryable=False)
    if resp.status_code >= 500:
        raise ToolError(f"Server error {resp.status_code} from {url}", tool_name=tool_name, retryable=True)
    if resp.status_code >= 400:
        raise ToolError(f"Client error {resp.status_code} from {url}", tool_name=tool_name, retryable=False)
    return resp


# ── Pydantic models ────────────────────────────────────────────────────────────

class PubmedQueryInput(BaseModel):
    query: str = Field(description=(
        "PubMed search query. Supports MeSH terms and field tags: "
        "[ti] title, [ab] abstract, [au] author, [mh] MeSH heading. "
        "Example: 'metformin[ti] AND diabetes mellitus, type 2[mh]'"
    ))
    max_results: int = Field(default=10, ge=1, le=50)

class SearchResult(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int] = None
    journal: str

class PubmedQueryOutput(BaseModel):
    results: list[SearchResult]
    total_found: int
    query_translation: str = Field(
        default="",
        description="How PubMed expanded the query via MeSH — shows synonym expansion and controlled vocabulary mapping",
    )


class SemanticScholarInput(BaseModel):
    query: str = Field(description=(
        "Full-text semantic search query. Dense-embedding search — "
        "conceptual relevance matters more than exact keywords."
    ))
    max_results: int = Field(default=10, ge=1, le=50)

class SemanticResult(BaseModel):
    paper_id: str
    title: str
    abstract: str
    citation_count: int
    influential_citation_count: int
    year: Optional[int] = None
    authors: list[str]
    venue: str = ""

class SemanticScholarOutput(BaseModel):
    results: list[SemanticResult]
    total: int


class ArxivQueryInput(BaseModel):
    query: str = Field(description=(
        "ArXiv search query. Field qualifiers: ti: title, au: author, abs: abstract. "
        "Example: 'ti:attention AND au:vaswani'"
    ))
    max_results: int = Field(default=10, ge=1, le=50)
    category: str = Field(
        default="",
        description="ArXiv category filter e.g. 'cs.AI', 'q-bio.GN', 'stat.ML'. Empty = all categories.",
    )

class ArxivResult(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    categories: list[str]
    pdf_url: str
    doi: str = ""

class ArxivQueryOutput(BaseModel):
    results: list[ArxivResult]
    total_results: int


class CitationLookupInput(BaseModel):
    paper_id: str = Field(description=(
        "Semantic Scholar paper ID, or prefixed: "
        "'DOI:10.1038/nature12373', 'ARXIV:1706.03762', 'PMID:12345678'"
    ))
    limit: int = Field(default=20, ge=1, le=100)

class CitationEntry(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    citation_count: int
    is_influential: bool
    authors: list[str]
    venue: str = ""

class CitationLookupOutput(BaseModel):
    citations: list[CitationEntry]
    total_citations: int
    paper_title: str = ""


class DOIResolveInput(BaseModel):
    doi: str = Field(description=(
        "DOI to resolve. Accepts bare DOI ('10.1038/nature12373') "
        "or full URL ('https://doi.org/10.1038/nature12373')."
    ))

class DOIRecord(BaseModel):
    doi: str
    title: str
    authors: list[str]
    year: Optional[int] = None
    journal: str = ""
    publisher: str = ""
    url: str
    abstract: str = ""
    is_open_access: bool = False

class DOIResolveOutput(BaseModel):
    record: Optional[DOIRecord] = None
    found: bool


class PreprintCheckInput(BaseModel):
    query: str = Field(description=(
        "Title, keywords, or DOI to search on bioRxiv/medRxiv. "
        "DOIs (10.xxxx/...) attempt a direct bioRxiv lookup; "
        "text queries use Semantic Scholar filtered to preprint servers."
    ))
    server: str = Field(
        default="both",
        description="'biorxiv', 'medrxiv', or 'both'",
    )
    max_results: int = Field(default=10, ge=1, le=50)

class PreprintEntry(BaseModel):
    doi: str
    title: str
    abstract: str
    authors: list[str]
    date: str
    server: str
    category: str
    published_journal_doi: str = Field(
        default="",
        description="Set if this preprint was subsequently published in a journal",
    )
    is_published: bool = False

class PreprintCheckOutput(BaseModel):
    preprints: list[PreprintEntry]
    total: int


class HIndexLookupInput(BaseModel):
    author_name: str = Field(description=(
        "Author full name. More specific is better: 'Geoffrey Hinton' not 'G Hinton'."
    ))
    affiliation: str = Field(
        default="",
        description="Optional institution filter to disambiguate common names.",
    )

class AuthorMetrics(BaseModel):
    author_id: str
    name: str
    h_index: int
    citation_count: int
    paper_count: int
    affiliations: list[str]
    homepage: str = ""
    aliases: list[str] = Field(default_factory=list)

class HIndexLookupOutput(BaseModel):
    authors: list[AuthorMetrics]
    best_match: Optional[AuthorMetrics] = None


class ReferenceExpandInput(BaseModel):
    paper_id: str = Field(description=(
        "Semantic Scholar paper ID or prefixed ID (DOI:, ARXIV:, PMID:)"
    ))
    limit: int = Field(default=20, ge=1, le=100)
    min_citation_count: int = Field(
        default=0,
        ge=0,
        description="Filter out references with fewer than this many citations. "
                    "Set to 10+ to focus on high-impact foundational works.",
    )

class ReferenceEntry(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    citation_count: int
    authors: list[str]
    venue: str = ""
    is_open_access: bool = False

class ReferenceExpandOutput(BaseModel):
    references: list[ReferenceEntry]
    total_references: int
    paper_title: str = ""


class WebFetchInput(BaseModel):
    url: str = Field(description=(
        "URL to fetch. Extracts main article text, stripping nav/ads/footers. "
        "For topic searches, prefer search.semantic_scholar or search.pubmed_query."
    ))
    max_chars: int = Field(default=5000, ge=100, le=50000)

class WebPage(BaseModel):
    url: str
    title: str
    content: str
    char_count: int
    fetch_succeeded: bool
    status_code: int = 0

class WebFetchOutput(BaseModel):
    page: WebPage


class GoogleScholarInput(BaseModel):
    query: str = Field(description="Google Scholar search query. Supports site: and filetype: operators.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Kept small to reduce CAPTCHA risk.",
    )

class ScholarResult(BaseModel):
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int] = None
    venue: str = ""
    citation_count: int
    url: str = ""

class GoogleScholarOutput(BaseModel):
    results: list[ScholarResult]
    total: int


# ── Tool 1: search.pubmed_query ────────────────────────────────────────────────

@tool(
    name="search.pubmed_query",
    description=(
        "Search PubMed for peer-reviewed biomedical literature via the NCBI Entrez API. "
        "Use for clinical evidence, epidemiology, pharmacology, genomics, and any question "
        "requiring MeSH controlled-vocabulary precision. "
        "NOT for CS, ML, physics, or general web content — use search.semantic_scholar or search.arxiv_query."
    ),
    input_schema=PubmedQueryInput,
    output_schema=PubmedQueryOutput,
)
@retry(max_retries=3, base_delay=1.0, exceptions=(ThenaError,))
async def pubmed_query(query: str, max_results: int = 10) -> dict:
    # ── esearch: get matching PMIDs ────────────────────────────────────────────
    await _PUBMED_LIMITER.acquire()
    esearch_params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": _NCBI_EMAIL,
    }
    if _NCBI_API_KEY:
        esearch_params["api_key"] = _NCBI_API_KEY

    resp = await _get(f"{_NCBI_BASE}/esearch.fcgi", "search.pubmed_query", params=esearch_params)
    esearch = resp.json().get("esearchresult", {})
    pmids = esearch.get("idlist", [])
    total_found = int(esearch.get("count", 0))
    query_translation = esearch.get("querytranslation", "")

    if not pmids:
        return {"results": [], "total_found": 0, "query_translation": query_translation}

    # ── efetch: retrieve full article XML for those PMIDs ─────────────────────
    await _PUBMED_LIMITER.acquire()
    efetch_params: dict = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": _NCBI_EMAIL,
    }
    if _NCBI_API_KEY:
        efetch_params["api_key"] = _NCBI_API_KEY

    resp2 = await _get(f"{_NCBI_BASE}/efetch.fcgi", "search.pubmed_query", params=efetch_params)
    root = ET.fromstring(resp2.text)

    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = (article.findtext(".//ArticleTitle", default="") or "").strip()

        # Structured abstracts have labeled sections: join them with section labels
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            f"{p.get('Label')}: {p.text or ''}" if p.get("Label") else (p.text or "")
            for p in abstract_parts
        ).strip()

        authors = []
        for a in article.findall(".//Author"):
            last = a.findtext("LastName", "")
            fore = a.findtext("ForeName", "") or a.findtext("Initials", "")
            if last:
                authors.append(f"{last} {fore}".strip())

        # Year lives in <Year> or free-text <MedlineDate> like "2023 Jan-Feb"
        year_text = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate", "")
        )
        year = int(year_text[:4]) if year_text and year_text[:4].isdigit() else None

        journal = article.findtext(".//Journal/Title", default="")

        results.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "journal": journal,
        })

    return {"results": results, "total_found": total_found, "query_translation": query_translation}


# ── Tool 2: search.semantic_scholar ───────────────────────────────────────────

@tool(
    name="search.semantic_scholar",
    description=(
        "Search Semantic Scholar across all scientific disciplines using dense-embedding semantic search. "
        "Use when conceptual relevance matters more than exact keyword precision, "
        "when you need citation and influential-citation counts, "
        "or for CS/ML/physics papers outside PubMed's scope. "
        "Use search.citation_lookup to explore the citation graph of a specific paper."
    ),
    input_schema=SemanticScholarInput,
    output_schema=SemanticScholarOutput,
)
@retry(max_retries=3, base_delay=2.0, exceptions=(ThenaError,))
async def semantic_scholar(query: str, max_results: int = 10) -> dict:
    await _S2_LIMITER.acquire()

    params = {
        "query": query,
        "limit": max_results,
        "fields": "paperId,title,abstract,year,citationCount,influentialCitationCount,authors,venue",
    }
    headers = {"x-api-key": _S2_API_KEY} if _S2_API_KEY else {}

    resp = await _get(f"{_S2_BASE}/paper/search", "search.semantic_scholar", params=params, headers=headers)
    data = resp.json()

    results = []
    for paper in data.get("data", []):
        results.append({
            "paper_id":                   paper.get("paperId", ""),
            "title":                      paper.get("title", "") or "",
            "abstract":                   paper.get("abstract", "") or "",
            "citation_count":             paper.get("citationCount", 0) or 0,
            "influential_citation_count": paper.get("influentialCitationCount", 0) or 0,
            "year":                       paper.get("year"),
            "authors":                    [a.get("name", "") for a in paper.get("authors", [])],
            "venue":                      paper.get("venue", "") or "",
        })

    return {"results": results, "total": data.get("total", len(results))}


# ── Tool 3: search.arxiv_query ─────────────────────────────────────────────────

@tool(
    name="search.arxiv_query",
    description=(
        "Search ArXiv for preprints and papers in CS, ML, physics, math, "
        "quantitative biology, quantitative finance, and statistics. "
        "Use for cutting-edge AI/ML research and theoretical work not yet peer-reviewed. "
        "Results may be preprints — verify with search.doi_resolve if publication status matters."
    ),
    input_schema=ArxivQueryInput,
    output_schema=ArxivQueryOutput,
)
@retry(max_retries=3, base_delay=2.0, exceptions=(ThenaError,))
async def arxiv_query(query: str, max_results: int = 10, category: str = "") -> dict:
    await _ARXIV_LIMITER.acquire()

    search_query = f"cat:{category} AND all:{query}" if category else f"all:{query}"
    params = {
        "search_query": search_query,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    resp = await _get(_ARXIV_BASE, "search.arxiv_query", params=params)

    _NS = {
        "atom":       "http://www.w3.org/2005/Atom",
        "arxiv":      "http://arxiv.org/schemas/atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    }
    root = ET.fromstring(resp.text)

    total_el = root.find("opensearch:totalResults", _NS)
    total_results = int(total_el.text) if total_el is not None and total_el.text else 0

    results = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = entry.findtext("atom:id", "", _NS)
        m = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", raw_id)
        arxiv_id = m.group(1) if m else raw_id

        title    = (entry.findtext("atom:title",   "", _NS) or "").replace("\n", " ").strip()
        abstract = (entry.findtext("atom:summary", "", _NS) or "").replace("\n", " ").strip()
        published = (entry.findtext("atom:published", "", _NS) or "")[:10]

        authors    = [a.findtext("atom:name", "", _NS) for a in entry.findall("atom:author", _NS)]
        categories = [c.get("term", "") for c in entry.findall("atom:category", _NS)]

        pdf_url = ""
        for link in entry.findall("atom:link", _NS):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        doi_el = entry.find("arxiv:doi", _NS)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else ""

        results.append({
            "arxiv_id":  arxiv_id,
            "title":     title,
            "abstract":  abstract,
            "authors":   authors,
            "published": published,
            "categories": categories,
            "pdf_url":   pdf_url,
            "doi":       doi,
        })

    return {"results": results, "total_results": total_results}


# ── Tool 4: search.citation_lookup ────────────────────────────────────────────

@tool(
    name="search.citation_lookup",
    description=(
        "Retrieve papers that cite a specific paper using Semantic Scholar's citation graph. "
        "Use to trace how an idea propagated through literature, find the most impactful "
        "follow-up works, or see who built on a seminal paper. "
        "is_influential=True means the citing paper substantially built on the cited work, "
        "not just mentioned it in passing."
    ),
    input_schema=CitationLookupInput,
    output_schema=CitationLookupOutput,
)
@retry(max_retries=3, base_delay=2.0, exceptions=(ThenaError,))
async def citation_lookup(paper_id: str, limit: int = 20) -> dict:
    await _S2_LIMITER.acquire()

    fields = (
        "citingPaper.paperId,citingPaper.title,citingPaper.year,"
        "citingPaper.citationCount,citingPaper.authors,citingPaper.venue,isInfluential"
    )
    params = {"fields": fields, "limit": limit}
    headers = {"x-api-key": _S2_API_KEY} if _S2_API_KEY else {}

    resp = await _get(
        f"{_S2_BASE}/paper/{paper_id}/citations",
        "search.citation_lookup",
        params=params,
        headers=headers,
    )
    data = resp.json()

    # Fetch paper title for context — non-critical, swallow failures
    paper_title = ""
    try:
        await _S2_LIMITER.acquire()
        tr = await _get(
            f"{_S2_BASE}/paper/{paper_id}",
            "search.citation_lookup",
            params={"fields": "title"},
            headers=headers,
        )
        paper_title = tr.json().get("title", "")
    except ThenaError:
        pass

    citations = []
    for item in data.get("data", []):
        citing = item.get("citingPaper", {})
        citations.append({
            "paper_id":       citing.get("paperId", "") or "",
            "title":          citing.get("title", "") or "",
            "year":           citing.get("year"),
            "citation_count": citing.get("citationCount", 0) or 0,
            "is_influential": item.get("isInfluential", False),
            "authors":        [a.get("name", "") for a in citing.get("authors", [])],
            "venue":          citing.get("venue", "") or "",
        })

    return {
        "citations":       citations,
        "total_citations": data.get("total", len(citations)),
        "paper_title":     paper_title,
    }


# ── Tool 5: search.doi_resolve ────────────────────────────────────────────────

@tool(
    name="search.doi_resolve",
    description=(
        "Resolve a DOI to full bibliographic metadata using CrossRef. "
        "Use to confirm journal publication details, get canonical citation data, "
        "or check if a preprint was formally published. "
        "CrossRef covers most journal publishers but not books or theses."
    ),
    input_schema=DOIResolveInput,
    output_schema=DOIResolveOutput,
)
@retry(max_retries=3, base_delay=1.0, exceptions=(ThenaError,))
async def doi_resolve(doi: str) -> dict:
    await _CROSSREF_LIMITER.acquire()

    # Strip URL prefix and trailing slashes
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip()).strip("/")

    try:
        resp = await _get(f"{_CROSSREF_BASE}/{doi}", "search.doi_resolve")
    except SourceNotFoundError:
        return {"record": None, "found": False}

    message = resp.json().get("message", {})

    titles = message.get("title", [])
    title = titles[0] if titles else ""

    authors = []
    for a in message.get("author", []):
        given, family = a.get("given", ""), a.get("family", "")
        if family:
            authors.append(f"{family}, {given}".strip(", "))

    # CrossRef stores dates in several fields; try in order of preference
    year: Optional[int] = None
    for date_field in ("published-print", "published-online", "created"):
        parts = message.get(date_field, {}).get("date-parts", [[]])
        if parts and parts[0]:
            year = parts[0][0]
            break

    journals = message.get("container-title", [])
    journal   = journals[0] if journals else ""
    publisher = message.get("publisher", "")
    url       = message.get("URL", f"https://doi.org/{doi}")

    # CrossRef abstracts may be wrapped in JATS XML — strip tags
    abstract = re.sub(r"<[^>]+>", "", message.get("abstract", "") or "").strip()

    # Open access if any CC license URL is present
    licenses     = message.get("license", [])
    is_open_access = any("creativecommons" in (lic.get("URL", "") or "").lower() for lic in licenses)

    record = {
        "doi":            doi,
        "title":          title,
        "authors":        authors,
        "year":           year,
        "journal":        journal,
        "publisher":      publisher,
        "url":            url,
        "abstract":       abstract,
        "is_open_access": is_open_access,
    }
    return {"record": record, "found": True}


# ── Tool 6: search.preprint_check ────────────────────────────────────────────

@tool(
    name="search.preprint_check",
    description=(
        "Find preprints on bioRxiv and medRxiv for a given query or DOI. "
        "Use when you need the latest biology or medicine results before peer-review, "
        "or to check if a study exists as a preprint. "
        "is_published=True means the preprint was formally published — always prefer "
        "the published version for clinical citations. "
        "Note: bioRxiv has no text search API; results come via Semantic Scholar filtering."
    ),
    input_schema=PreprintCheckInput,
    output_schema=PreprintCheckOutput,
)
@retry(max_retries=3, base_delay=2.0, exceptions=(ThenaError,))
async def preprint_check(query: str, server: str = "both", max_results: int = 10) -> dict:
    servers = ["biorxiv", "medrxiv"] if server == "both" else [server.lower()]

    # Direct DOI lookup via bioRxiv details API
    doi_pattern = re.compile(r"^10\.\d{4,}/")
    if doi_pattern.match(query.strip()):
        return await _preprint_doi_lookup(query.strip(), servers, max_results)

    # Text search: use Semantic Scholar and filter for bioRxiv/medRxiv papers
    await _S2_LIMITER.acquire()
    params = {
        "query": query,
        "limit": max_results * 3,  # fetch extra; most won't match a server
        "fields": "paperId,title,abstract,year,authors,externalIds,publicationDate,venue,journal",
    }
    headers = {"x-api-key": _S2_API_KEY} if _S2_API_KEY else {}

    resp = await _get(f"{_S2_BASE}/paper/search", "search.preprint_check", params=params, headers=headers)
    data = resp.json()

    preprints = []
    for paper in data.get("data", []):
        venue     = (paper.get("venue", "") or "").lower()
        journal_d = paper.get("journal") or {}
        j_name    = (journal_d.get("name", "") or "").lower()

        matched_server = None
        for s in servers:
            if s in venue or s in j_name or s in str(paper.get("externalIds", {})).lower():
                matched_server = s
                break
        if not matched_server:
            continue

        ext_ids     = paper.get("externalIds", {}) or {}
        doi         = ext_ids.get("DOI", "")
        is_published = bool(j_name and j_name not in ("biorxiv", "medrxiv"))

        preprints.append({
            "doi":                  doi,
            "title":                paper.get("title", "") or "",
            "abstract":             paper.get("abstract", "") or "",
            "authors":              [a.get("name", "") for a in paper.get("authors", [])],
            "date":                 paper.get("publicationDate", "") or "",
            "server":               matched_server,
            "category":             venue,
            "published_journal_doi": "",
            "is_published":         is_published,
        })
        if len(preprints) >= max_results:
            break

    return {"preprints": preprints, "total": len(preprints)}


async def _preprint_doi_lookup(doi: str, servers: list[str], max_results: int) -> dict:
    """Direct bioRxiv/medRxiv lookup by DOI."""
    preprints = []
    for s in servers:
        await _BIORXIV_LIMITER.acquire()
        try:
            resp = await _get(
                f"{_BIORXIV_BASE}/details/{s}/{doi}/json",
                "search.preprint_check",
            )
            collection = resp.json().get("collection", [])
            for item in collection[:max_results]:
                preprints.append({
                    "doi":                  item.get("doi", ""),
                    "title":                item.get("title", ""),
                    "abstract":             item.get("abstract", ""),
                    "authors":              [a.strip() for a in item.get("authors", "").split(";")],
                    "date":                 item.get("date", ""),
                    "server":               s,
                    "category":             item.get("category", ""),
                    "published_journal_doi": item.get("published", ""),
                    "is_published":         bool(item.get("published", "")),
                })
        except (SourceNotFoundError, ToolError):
            continue
    return {"preprints": preprints, "total": len(preprints)}


# ── Tool 7: search.h_index_lookup ────────────────────────────────────────────

@tool(
    name="search.h_index_lookup",
    description=(
        "Look up an academic author's h-index, citation count, and publication count "
        "from Semantic Scholar. "
        "Use to assess research impact, disambiguate authors, or find high-impact researchers "
        "in a field. Returns up to 5 candidate matches; best_match is the one with highest h-index."
    ),
    input_schema=HIndexLookupInput,
    output_schema=HIndexLookupOutput,
)
@retry(max_retries=3, base_delay=2.0, exceptions=(ThenaError,))
async def h_index_lookup(author_name: str, affiliation: str = "") -> dict:
    await _S2_LIMITER.acquire()

    fields = "authorId,name,hIndex,citationCount,paperCount,affiliations,homepage,aliases"
    params = {"query": author_name, "fields": fields, "limit": 5}
    headers = {"x-api-key": _S2_API_KEY} if _S2_API_KEY else {}

    resp = await _get(f"{_S2_BASE}/author/search", "search.h_index_lookup", params=params, headers=headers)
    data = resp.json()

    authors = []
    for a in data.get("data", []):
        raw_affs   = a.get("affiliations", []) or []
        affs       = [x if isinstance(x, str) else x.get("name", "") for x in raw_affs]

        if affiliation and not any(affiliation.lower() in aff.lower() for aff in affs):
            continue

        authors.append({
            "author_id":      a.get("authorId", ""),
            "name":           a.get("name", ""),
            "h_index":        a.get("hIndex", 0) or 0,
            "citation_count": a.get("citationCount", 0) or 0,
            "paper_count":    a.get("paperCount", 0) or 0,
            "affiliations":   affs,
            "homepage":       a.get("homepage", "") or "",
            "aliases":        a.get("aliases", []) or [],
        })

    best = max(authors, key=lambda x: x["h_index"], default=None)
    return {"authors": authors, "best_match": best}


# ── Tool 8: search.reference_expand ──────────────────────────────────────────

@tool(
    name="search.reference_expand",
    description=(
        "Retrieve the full reference list of a paper — the works it cites. "
        "Use to map the intellectual lineage of a paper, find the foundational works "
        "in a field, or discover what a high-impact paper is actually built on. "
        "Set min_citation_count > 0 to filter out peripheral citations and focus on "
        "high-impact foundational works."
    ),
    input_schema=ReferenceExpandInput,
    output_schema=ReferenceExpandOutput,
)
@retry(max_retries=3, base_delay=2.0, exceptions=(ThenaError,))
async def reference_expand(paper_id: str, limit: int = 20, min_citation_count: int = 0) -> dict:
    await _S2_LIMITER.acquire()

    fields = (
        "citedPaper.paperId,citedPaper.title,citedPaper.year,"
        "citedPaper.citationCount,citedPaper.authors,citedPaper.venue,citedPaper.isOpenAccess"
    )
    params  = {"fields": fields, "limit": min(limit, 100)}
    headers = {"x-api-key": _S2_API_KEY} if _S2_API_KEY else {}

    resp = await _get(
        f"{_S2_BASE}/paper/{paper_id}/references",
        "search.reference_expand",
        params=params,
        headers=headers,
    )
    data = resp.json()

    # Fetch paper title for context — non-critical
    paper_title = ""
    try:
        await _S2_LIMITER.acquire()
        tr = await _get(
            f"{_S2_BASE}/paper/{paper_id}",
            "search.reference_expand",
            params={"fields": "title"},
            headers=headers,
        )
        paper_title = tr.json().get("title", "")
    except ThenaError:
        pass

    references = []
    for item in data.get("data", []):
        cited         = item.get("citedPaper", {})
        citation_count = cited.get("citationCount", 0) or 0
        if citation_count < min_citation_count:
            continue
        references.append({
            "paper_id":       cited.get("paperId", "") or "",
            "title":          cited.get("title", "") or "",
            "year":           cited.get("year"),
            "citation_count": citation_count,
            "authors":        [a.get("name", "") for a in cited.get("authors", [])],
            "venue":          cited.get("venue", "") or "",
            "is_open_access": cited.get("isOpenAccess", False) or False,
        })

    return {
        "references":       references,
        "total_references": data.get("total", len(references)),
        "paper_title":      paper_title,
    }


# ── Tool 9: search.web_fetch ──────────────────────────────────────────────────

@tool(
    name="search.web_fetch",
    description=(
        "Fetch and extract main text content from any URL. "
        "Use when you have a specific URL from search results or citations and need "
        "to read its full content rather than a snippet. "
        "Prefers <article> and <main> semantic tags; strips nav, ads, and boilerplate. "
        "Returns fetch_succeeded=False (no exception) on HTTP errors so the orchestrator "
        "can continue without crashing when a URL is unavailable."
    ),
    input_schema=WebFetchInput,
    output_schema=WebFetchOutput,
)
@retry(max_retries=2, base_delay=2.0, exceptions=(ThenaError,))
async def web_fetch(url: str, max_chars: int = 5000) -> dict:
    await _WEB_LIMITER.acquire()

    try:
        resp: requests.Response = await asyncio.to_thread(
            requests.get, url, headers=_HEADERS, timeout=20
        )
    except requests.Timeout:
        # Timeout is not a hard error — return degraded result so the orchestrator continues
        return {"page": {"url": url, "title": "", "content": "", "char_count": 0,
                         "fetch_succeeded": False, "status_code": 0}}
    except requests.RequestException as exc:
        raise ToolError(f"Network error fetching {url}: {exc}", tool_name="search.web_fetch", retryable=True)

    if resp.status_code != 200:
        return {"page": {"url": url, "title": "", "content": "", "char_count": 0,
                         "fetch_succeeded": False, "status_code": resp.status_code}}

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    main  = soup.find("article") or soup.find("main") or soup.body

    if main:
        lines   = [l.strip() for l in main.get_text(separator="\n").splitlines() if len(l.strip()) >= 40]
        content = "\n".join(lines)[:max_chars]
    else:
        content = ""

    return {
        "page": {
            "url":             url,
            "title":           title,
            "content":         content,
            "char_count":      len(content),
            "fetch_succeeded": True,
            "status_code":     resp.status_code,
        }
    }


# ── Tool 10: search.google_scholar ────────────────────────────────────────────

@tool(
    name="search.google_scholar",
    description=(
        "Search Google Scholar for academic papers, citation counts, and grey literature. "
        "Use for papers not indexed on PubMed or Semantic Scholar, or when Google Scholar "
        "citation counts specifically are needed. "
        "Requires: pip install scholarly. "
        "Very rate-limited — Scholar aggressively CAPTCHAs automated access. "
        "Prefer search.semantic_scholar for most research tasks (more reliable, richer data)."
    ),
    input_schema=GoogleScholarInput,
    output_schema=GoogleScholarOutput,
)
@retry(max_retries=2, base_delay=10.0, exceptions=(ThenaError,))
async def google_scholar(query: str, max_results: int = 5) -> dict:
    try:
        from scholarly import scholarly as _scholarly
    except ImportError:
        raise ToolError(
            "scholarly package not installed. Run: pip install scholarly",
            tool_name="search.google_scholar",
            retryable=False,
        )

    await _SCHOLAR_LIMITER.acquire()

    def _run() -> list[dict]:
        results = []
        try:
            for i, pub in enumerate(_scholarly.search_pubs(query)):
                if i >= max_results:
                    break
                bib       = pub.get("bib", {})
                author_raw = bib.get("author", "")
                authors    = (
                    [a.strip() for a in author_raw.split(" and ")]
                    if isinstance(author_raw, str) and author_raw
                    else []
                )
                year_str = bib.get("pub_year", "")
                results.append({
                    "title":          bib.get("title", ""),
                    "abstract":       bib.get("abstract", ""),
                    "authors":        authors,
                    "year":           int(year_str) if year_str and year_str.isdigit() else None,
                    "venue":          bib.get("venue", ""),
                    "citation_count": pub.get("num_citations", 0),
                    "url":            pub.get("pub_url", ""),
                })
        except StopIteration:
            pass
        return results

    try:
        results = await asyncio.to_thread(_run)
    except Exception as exc:
        msg = str(exc).lower()
        if any(kw in msg for kw in ("captcha", "blocked", "forbidden", "too many")):
            raise RateLimitError(
                "Google Scholar CAPTCHA triggered. Back off and retry.",
                endpoint="google_scholar",
                retry_after=120.0,
            )
        raise ToolError(
            f"Google Scholar search failed: {exc}",
            tool_name="search.google_scholar",
            retryable=True,
        )

    return {"results": results, "total": len(results)}
