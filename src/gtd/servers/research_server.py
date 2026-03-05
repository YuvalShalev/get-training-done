"""MCP server exposing research API tools for arXiv, Kaggle, and Papers with Code."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from gtd.research.arxiv_client import search_arxiv as _search_arxiv
from gtd.research.kaggle_client import (
    search_kaggle_datasets as _search_kaggle_datasets,
    search_kaggle_notebooks as _search_kaggle_notebooks,
)
from gtd.research.pwc_client import search_papers_with_code as _search_papers_with_code

mcp = FastMCP("gtd-research")


@mcp.tool()
def search_arxiv(query: str, max_results: int = 10) -> str:
    """Search arXiv for academic papers.

    Args:
        query: Search query string (supports arXiv query syntax).
        max_results: Maximum number of results to return (default 10).

    Returns:
        JSON string with search results or error information.
    """
    try:
        result = _search_arxiv(query=query, max_results=max_results)
        return _to_json(result)
    except Exception as exc:
        return _to_json({"error": f"Unexpected error searching arXiv: {exc}", "query": query})


@mcp.tool()
def search_kaggle_datasets(query: str, max_results: int = 10) -> str:
    """Search Kaggle for datasets.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        JSON string with search results or error information.
    """
    try:
        result = _search_kaggle_datasets(query=query, max_results=max_results)
        return _to_json(result)
    except Exception as exc:
        return _to_json({"error": f"Unexpected error searching Kaggle datasets: {exc}", "query": query})


@mcp.tool()
def search_kaggle_notebooks(query: str, sort_by: str = "relevance", max_results: int = 10) -> str:
    """Search Kaggle for notebooks (kernels).

    Args:
        query: Search query string.
        sort_by: Sort order - 'relevance', 'hotness', 'dateCreated', 'dateRun',
                 'scoreAscending', 'scoreDescending', or 'voteCount'.
        max_results: Maximum number of results to return (default 10).

    Returns:
        JSON string with search results or error information.
    """
    try:
        result = _search_kaggle_notebooks(query=query, sort_by=sort_by, max_results=max_results)
        return _to_json(result)
    except Exception as exc:
        return _to_json({"error": f"Unexpected error searching Kaggle notebooks: {exc}", "query": query})


@mcp.tool()
def search_papers_with_code(query: str, task_type: str | None = None, max_results: int = 10) -> str:
    """Search Papers with Code for papers with implementations.

    Args:
        query: Search query string.
        task_type: Optional task filter (e.g. 'image-classification').
        max_results: Maximum number of results to return (default 10).

    Returns:
        JSON string with search results or error information.
    """
    try:
        result = _search_papers_with_code(query=query, task_type=task_type, max_results=max_results)
        return _to_json(result)
    except Exception as exc:
        return _to_json({"error": f"Unexpected error searching Papers with Code: {exc}", "query": query})


def _to_json(data: dict[str, Any]) -> str:
    """Serialize a dict to a JSON string.

    Args:
        data: Dictionary to serialize.

    Returns:
        Formatted JSON string.
    """
    return json.dumps(data, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
