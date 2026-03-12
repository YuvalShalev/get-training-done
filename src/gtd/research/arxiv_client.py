"""arXiv API client for searching academic papers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
REQUEST_TIMEOUT = 10

ATOM_NS = "{http://www.w3.org/2005/Atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def search_arxiv(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search arXiv for papers matching the query.

    Args:
        query: Search query string (supports arXiv query syntax).
        max_results: Maximum number of results to return (default 10).

    Returns:
        Dict with 'results' list, 'total_results' count, and 'query' echo.
        On failure, returns an error dict with 'error' and 'query' keys.
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }

    try:
        response = requests.get(
            ARXIV_API_BASE,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {
            "error": "Request to arXiv API timed out",
            "query": query,
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "Failed to connect to arXiv API",
            "query": query,
        }
    except requests.exceptions.HTTPError as exc:
        return {
            "error": f"arXiv API returned HTTP {exc.response.status_code}",
            "query": query,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "error": f"arXiv API request failed: {exc}",
            "query": query,
        }

    return _parse_arxiv_response(response.text, query)


def _parse_arxiv_response(xml_text: str, query: str) -> dict[str, Any]:
    """Parse the Atom XML response from arXiv.

    Args:
        xml_text: Raw XML response body.
        query: Original query string for inclusion in output.

    Returns:
        Parsed results dict, or error dict on parse failure.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {
            "error": f"Failed to parse arXiv XML response: {exc}",
            "query": query,
        }

    total_element = root.find(f"{OPENSEARCH_NS}totalResults")
    has_total = total_element is not None and total_element.text
    total_results = int(total_element.text) if has_total else 0

    results = [
        _parse_entry(entry)
        for entry in root.findall(f"{ATOM_NS}entry")
    ]

    return {
        "results": results,
        "total_results": total_results,
        "query": query,
    }


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    """Parse a single Atom entry element into a result dict.

    Args:
        entry: An <entry> XML element from the arXiv feed.

    Returns:
        Dict with title, abstract, authors, url, published, and categories.
    """
    title = _get_text(entry, f"{ATOM_NS}title").replace("\n", " ").strip()
    abstract = _get_text(entry, f"{ATOM_NS}summary").strip()
    published = _get_text(entry, f"{ATOM_NS}published")

    authors = [
        name_el.text.strip()
        for author_el in entry.findall(f"{ATOM_NS}author")
        if (name_el := author_el.find(f"{ATOM_NS}name")) is not None
        and name_el.text
    ]

    url = ""
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.get("type") == "text/html" or link.get("rel") == "alternate":
            url = link.get("href", "")
            break
    if not url:
        id_el = entry.find(f"{ATOM_NS}id")
        url = id_el.text.strip() if id_el is not None and id_el.text else ""

    categories = [
        cat.get("term", "")
        for cat in entry.findall(f"{ARXIV_NS}primary_category")
    ] + [
        cat.get("term", "")
        for cat in entry.findall(f"{ATOM_NS}category")
    ]
    categories = list(dict.fromkeys(c for c in categories if c))

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "url": url,
        "published": published,
        "categories": categories,
    }


def _get_text(element: ET.Element, tag: str) -> str:
    """Safely extract text from a child element.

    Args:
        element: Parent XML element.
        tag: Fully-qualified tag name to find.

    Returns:
        Text content of the child element, or empty string if missing.
    """
    child = element.find(tag)
    if child is not None and child.text:
        return child.text
    return ""
