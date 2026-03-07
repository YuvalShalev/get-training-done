"""MCP server exposing data profiling tools for dataset analysis."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from gtd.core import data_profiler, data_splitter

mcp = FastMCP("gtd-data")


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


@mcp.tool()
async def create_data_split(
    workspace_path: str,
    data_path: str,
    target_column: str,
    task_type: str,
    strategy: str = "stratified",
    validation_fraction: float = 0.2,
    temporal_column: str | None = None,
    group_column: str | None = None,
    random_state: int = 42,
) -> str:
    """Split data into train and validation partitions for proper HPO evaluation.

    The validation set is held out for ALL evaluation — it is NEVER used for training.
    Call this before any training to ensure proper train/validation separation.

    Args:
        workspace_path: Path to the workspace directory.
        data_path: Path to the source CSV file.
        target_column: Name of the target column.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.
        strategy: Split strategy — 'random', 'stratified', 'temporal', or 'group'.
                  Default: 'stratified'.
        validation_fraction: Fraction of data for validation (default 0.2).
        temporal_column: Column name for temporal sorting (required for 'temporal').
        group_column: Column name for group splitting (required for 'group').
        random_state: Random seed (default 42).

    Returns:
        JSON string with train_data_path, validation_data_path, train_size,
        validation_size, strategy, and split_info.
    """
    try:
        result = data_splitter.create_data_split(
            workspace_path=workspace_path,
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            strategy=strategy,
            validation_fraction=validation_fraction,
            temporal_column=temporal_column,
            group_column=group_column,
            random_state=random_state,
        )
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


if __name__ == "__main__":
    mcp.run()
