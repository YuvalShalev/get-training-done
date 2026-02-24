"""MCP server exposing data profiling tools for dataset analysis."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from bbopt.core import data_profiler

mcp = FastMCP("bbopt-data")


def _json_response(data: dict[str, Any]) -> str:
    """Serialize a result dict to a JSON string."""
    return json.dumps(data, indent=2, default=str)


def _error_response(error: Exception) -> str:
    """Serialize an error to a JSON string."""
    return json.dumps({
        "error": str(error),
        "error_type": type(error).__name__,
    }, indent=2)


@mcp.tool()
async def profile_dataset(
    path: str,
    target_column: str,
    task_type: str = "auto",
) -> str:
    """Profile a dataset with comprehensive first-look analysis.

    Returns shape, dtypes, per-column distributions, missing percentages,
    class balance (for classification), feature types, preprocessing
    recommendations, outlier counts, and cardinality.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target/label column.
        task_type: 'classification', 'regression', or 'auto' to infer.
    """
    try:
        result = data_profiler.profile_dataset(path, target_column, task_type)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def get_column_stats(path: str, column: str) -> str:
    """Deep dive into a single column's statistics.

    Returns distribution stats (mean, std, min, max, percentiles for numeric;
    value_counts for categorical), unique count, missing percentage, dtype,
    and is_numeric flag.

    Args:
        path: Path to the CSV file.
        column: Name of the column to analyze.
    """
    try:
        result = data_profiler.get_column_stats(path, column)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def detect_data_issues(path: str, target_column: str) -> str:
    """Identify data quality problems and potential pitfalls.

    Checks for class imbalance, multicollinearity, high cardinality columns,
    constant/near-constant features, data leakage suspects, and columns
    with heavy missing values.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target/label column.
    """
    try:
        result = data_profiler.detect_data_issues(path, target_column)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def compute_correlations(
    path: str,
    target_column: str,
    method: str = "pearson",
) -> str:
    """Compute statistical relationships between features and with the target.

    Returns feature-target correlations, top correlated feature pairs,
    and the full correlation matrix.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target/label column.
        method: Correlation method - 'pearson', 'spearman', or 'kendall'.
    """
    try:
        result = data_profiler.compute_correlations(path, target_column, method)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def preview_data(path: str, n_rows: int = 5) -> str:
    """Quick look at the first N rows and basic metadata.

    Returns the first N rows as a list of dicts, along with dtypes,
    shape, and column names.

    Args:
        path: Path to the CSV file.
        n_rows: Number of rows to preview (default 5).
    """
    try:
        result = data_profiler.preview_data(path, n_rows)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


if __name__ == "__main__":
    mcp.run()
