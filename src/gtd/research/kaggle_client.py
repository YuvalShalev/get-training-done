"""Kaggle API client for searching datasets and notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"
REQUEST_TIMEOUT = 10


def search_kaggle_datasets(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search Kaggle for datasets matching the query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Dict with 'results' list and 'total_results' count.
        On failure or missing credentials, returns an error dict.
    """
    auth = _get_kaggle_auth()
    if auth is None:
        return _credentials_error()

    params = {
        "search": query,
        "pageSize": max_results,
        "page": 1,
    }

    try:
        response = requests.get(
            f"{KAGGLE_API_BASE}/datasets/list",
            params=params,
            auth=auth,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": "Request to Kaggle API timed out", "query": query}
    except requests.exceptions.ConnectionError:
        return {"error": "Failed to connect to Kaggle API", "query": query}
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        if status == 401:
            return {"error": "Kaggle API authentication failed. Check your credentials.", "query": query}
        return {"error": f"Kaggle API returned HTTP {status}", "query": query}
    except requests.exceptions.RequestException as exc:
        return {"error": f"Kaggle API request failed: {exc}", "query": query}

    return _parse_datasets_response(response.json(), query)


def search_kaggle_notebooks(
    query: str,
    sort_by: str = "relevance",
    max_results: int = 10,
) -> dict[str, Any]:
    """Search Kaggle for notebooks (kernels) matching the query.

    Args:
        query: Search query string.
        sort_by: Sort order - one of 'relevance', 'hotness', 'dateCreated',
                 'dateRun', 'scoreAscending', 'scoreDescending', 'voteCount'.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Dict with 'results' list.
        On failure or missing credentials, returns an error dict.
    """
    auth = _get_kaggle_auth()
    if auth is None:
        return _credentials_error()

    sort_map = {
        "relevance": "relevance",
        "hotness": "hotness",
        "dateCreated": "dateCreated",
        "dateRun": "dateRun",
        "scoreAscending": "scoreAscending",
        "scoreDescending": "scoreDescending",
        "voteCount": "voteCount",
    }
    kaggle_sort = sort_map.get(sort_by, "relevance")

    params = {
        "search": query,
        "pageSize": max_results,
        "page": 1,
        "sortBy": kaggle_sort,
    }

    try:
        response = requests.get(
            f"{KAGGLE_API_BASE}/kernels/list",
            params=params,
            auth=auth,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": "Request to Kaggle API timed out", "query": query}
    except requests.exceptions.ConnectionError:
        return {"error": "Failed to connect to Kaggle API", "query": query}
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        if status == 401:
            return {"error": "Kaggle API authentication failed. Check your credentials.", "query": query}
        return {"error": f"Kaggle API returned HTTP {status}", "query": query}
    except requests.exceptions.RequestException as exc:
        return {"error": f"Kaggle API request failed: {exc}", "query": query}

    return _parse_notebooks_response(response.json(), query)


def _get_kaggle_auth() -> tuple[str, str] | None:
    """Retrieve Kaggle credentials from environment or kaggle.json.

    Checks in order:
    1. KAGGLE_USERNAME and KAGGLE_KEY environment variables
    2. ~/.kaggle/kaggle.json file

    Returns:
        Tuple of (username, key) for HTTP basic auth, or None if not found.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if username and key:
        return (username, key)

    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            with open(kaggle_json_path) as f:
                creds = json.load(f)
            stored_username = creds.get("username")
            stored_key = creds.get("key")
            if stored_username and stored_key:
                return (stored_username, stored_key)
        except (json.JSONDecodeError, OSError):
            return None

    return None


def diagnose_kaggle_credentials() -> str | None:
    """Check Kaggle credential setup and return a diagnostic message if issues found.

    Returns:
        None if credentials are valid, or a diagnostic string explaining the problem.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return None

    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.exists():
        return (
            "~/.kaggle/kaggle.json not found and KAGGLE_USERNAME/KAGGLE_KEY env vars not set."
        )

    try:
        raw = kaggle_json_path.read_text()
    except OSError as exc:
        return f"~/.kaggle/kaggle.json exists but cannot be read: {exc}"

    try:
        creds = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            f"~/.kaggle/kaggle.json exists but contains invalid JSON: {exc}. "
            f"File contents preview: {raw[:120]!r}"
        )

    if not isinstance(creds, dict):
        return f"~/.kaggle/kaggle.json should be a JSON object, got {type(creds).__name__}."

    found_keys = list(creds.keys())
    has_username = "username" in creds and creds["username"]
    has_key = "key" in creds and creds["key"]

    if has_username and has_key:
        return None

    missing = []
    if not has_username:
        missing.append('"username"')
    if not has_key:
        missing.append('"key"')

    return (
        f"~/.kaggle/kaggle.json is missing required fields: {', '.join(missing)}. "
        f"Found keys: {found_keys}. "
        f'Expected format: {{"username": "your_username", "key": "your_api_key"}}'
    )


def _credentials_error() -> dict[str, Any]:
    """Build an error dict explaining how to configure Kaggle credentials.

    Returns:
        Error dict with setup instructions and diagnostics.
    """
    diagnosis = diagnose_kaggle_credentials()
    return {
        "error": "Kaggle credentials not found",
        "diagnosis": diagnosis,
        "setup_instructions": (
            "Configure Kaggle credentials using one of these methods:\n"
            "1. Set environment variables KAGGLE_USERNAME and KAGGLE_KEY\n"
            "2. Create ~/.kaggle/kaggle.json with contents:\n"
            '   {"username": "your_username", "key": "your_api_key"}\n'
            "   You can download this file from https://www.kaggle.com/settings -> API -> Create New Token"
        ),
    }


def _parse_datasets_response(
    data: list[dict[str, Any]],
    query: str,
) -> dict[str, Any]:
    """Parse the Kaggle datasets list response.

    Args:
        data: JSON response body (list of dataset objects).
        query: Original query string.

    Returns:
        Formatted results dict.
    """
    results = [
        {
            "title": ds.get("title", ds.get("ref", "")),
            "description": ds.get("subtitle", ds.get("description", "")),
            "url": f"https://www.kaggle.com/datasets/{ds['ref']}" if "ref" in ds else "",
            "size": _format_size(ds.get("totalBytes", 0)),
            "download_count": ds.get("downloadCount", 0),
        }
        for ds in data
    ]

    return {
        "results": results,
        "total_results": len(results),
        "query": query,
    }


def _parse_notebooks_response(
    data: list[dict[str, Any]],
    query: str,
) -> dict[str, Any]:
    """Parse the Kaggle kernels list response.

    Args:
        data: JSON response body (list of kernel objects).
        query: Original query string.

    Returns:
        Formatted results dict.
    """
    results = [
        {
            "title": nb.get("title", ""),
            "author": nb.get("author", ""),
            "url": f"https://www.kaggle.com/code/{nb['ref']}" if "ref" in nb else "",
            "score": nb.get("totalVotes", 0.0),
            "language": nb.get("language", "unknown"),
        }
        for nb in data
    ]

    return {
        "results": results,
        "query": query,
    }


def _format_size(total_bytes: int | float) -> str:
    """Format byte count into a human-readable size string.

    Args:
        total_bytes: Size in bytes.

    Returns:
        Formatted string like '1.5 MB' or '320 KB'.
    """
    if not total_bytes or total_bytes <= 0:
        return "unknown"

    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]
    for label, divisor in units:
        if total_bytes >= divisor:
            value = total_bytes / divisor
            return f"{value:.1f} {label}"

    return f"{int(total_bytes)} B"
