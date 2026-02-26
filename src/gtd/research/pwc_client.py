"""Papers with Code API client for searching papers and their implementations."""

from __future__ import annotations

from typing import Any

import requests

PWC_API_BASE = "https://paperswithcode.com/api/v1"
REQUEST_TIMEOUT = 10


def search_papers_with_code(
    query: str,
    task_type: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search Papers with Code for papers matching the query.

    Args:
        query: Search query string.
        task_type: Optional task filter (e.g. 'image-classification',
                   'object-detection'). Passed to the API as a task filter.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Dict with 'results' list and 'total_results' count.
        On failure, returns an error dict with 'error' and 'query' keys.
    """
    params: dict[str, Any] = {
        "q": query,
        "items_per_page": max_results,
        "page": 1,
    }

    if task_type:
        params["task"] = task_type

    try:
        response = requests.get(
            f"{PWC_API_BASE}/papers/",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {
            "error": "Request to Papers with Code API timed out",
            "query": query,
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "Failed to connect to Papers with Code API",
            "query": query,
        }
    except requests.exceptions.HTTPError as exc:
        return {
            "error": f"Papers with Code API returned HTTP {exc.response.status_code}",
            "query": query,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "error": f"Papers with Code API request failed: {exc}",
            "query": query,
        }

    return _parse_response(response.json(), query)


def _parse_response(data: dict[str, Any], query: str) -> dict[str, Any]:
    """Parse the Papers with Code API response.

    Args:
        data: JSON response body from the API.
        query: Original query string for inclusion in output.

    Returns:
        Formatted results dict.
    """
    total_results = data.get("count", 0)
    raw_results = data.get("results", [])

    results = [_parse_paper(paper) for paper in raw_results]

    return {
        "results": results,
        "total_results": total_results,
        "query": query,
    }


def _parse_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Parse a single paper entry from the API response.

    Args:
        paper: A paper object from the API results list.

    Returns:
        Dict with title, abstract, url, url_pdf, proceeding, and tasks.
    """
    paper_id = paper.get("id", "")
    url = paper.get("url_abs", "")
    if not url and paper_id:
        url = f"https://paperswithcode.com/paper/{paper_id}"

    tasks = paper.get("tasks", [])
    if isinstance(tasks, list):
        task_names = [
            t.get("name", t) if isinstance(t, dict) else str(t)
            for t in tasks
        ]
    else:
        task_names = []

    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "url": url,
        "url_pdf": paper.get("url_pdf") or None,
        "proceeding": paper.get("proceeding") or None,
        "tasks": task_names,
    }
