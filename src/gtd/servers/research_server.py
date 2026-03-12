"""MCP server exposing research API tools for arXiv, Kaggle, and Papers with Code."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from gtd.research.arxiv_client import search_arxiv as _search_arxiv
from gtd.research.insight_extractor import extract_insights
from gtd.research.kaggle_client import (
    search_kaggle_datasets as _search_kaggle_datasets,
)
from gtd.research.kaggle_client import (
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
        return _to_json({
            "error": f"Unexpected error searching Kaggle datasets: {exc}",
            "query": query,
        })


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
        result = _search_kaggle_notebooks(
            query=query, sort_by=sort_by, max_results=max_results,
        )
        return _to_json(result)
    except Exception as exc:
        return _to_json({
            "error": f"Unexpected error searching Kaggle notebooks: {exc}",
            "query": query,
        })


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
        result = _search_papers_with_code(
            query=query, task_type=task_type, max_results=max_results,
        )
        return _to_json(result)
    except Exception as exc:
        return _to_json({
            "error": f"Unexpected error searching Papers with Code: {exc}",
            "query": query,
        })


@mcp.tool()
def research_and_extract(
    query: str,
    task_type: str = "",
    dataset_profile_json: str = "",
    sources: str = "arxiv,kaggle",
    max_results: int = 5,
) -> str:
    """Search multiple sources and extract structured insights in one call.

    Combines search_arxiv, search_kaggle_notebooks, and search_papers_with_code
    results into ~200 tokens of actionable insights instead of ~2000 tokens
    of raw results.

    Args:
        query: Search query string.
        task_type: Task type (e.g., "binary_classification").
        dataset_profile_json: JSON string of dataset profile for context-aware
            recommendations. Keys: n_rows, n_cols, n_numeric, n_categorical.
        sources: Comma-separated sources to search. Options: "arxiv", "kaggle",
            "pwc". Default: "arxiv,kaggle".
        max_results: Maximum results per source (default 5).

    Returns:
        JSON string with recommended_models, hp_hints, feature_tips,
        competition_strategies, and summary.
    """
    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]

    dataset_profile: dict[str, Any] | None = None
    if dataset_profile_json:
        try:
            dataset_profile = json.loads(dataset_profile_json)
        except (json.JSONDecodeError, TypeError):
            dataset_profile = None

    arxiv_results: dict[str, Any] | None = None
    kaggle_results: dict[str, Any] | None = None
    pwc_results: dict[str, Any] | None = None

    if "arxiv" in source_list:
        try:
            arxiv_results = _search_arxiv(query=query, max_results=max_results)
        except Exception as exc:
            arxiv_results = {"error": str(exc), "results": []}

    if "kaggle" in source_list:
        try:
            kaggle_results = _search_kaggle_notebooks(
                query=query, max_results=max_results,
            )
        except Exception as exc:
            kaggle_results = {"error": str(exc), "results": []}

    if "pwc" in source_list:
        try:
            pwc_results = _search_papers_with_code(
                query=query, max_results=max_results,
            )
        except Exception as exc:
            pwc_results = {"error": str(exc), "results": []}

    try:
        insights = extract_insights(
            arxiv_results=arxiv_results,
            kaggle_results=kaggle_results,
            pwc_results=pwc_results,
            task_type=task_type,
            dataset_profile=dataset_profile,
        )
        return _to_json(insights)
    except Exception as exc:
        return _to_json({
            "error": f"Insight extraction failed: {exc}",
            "query": query,
        })


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
