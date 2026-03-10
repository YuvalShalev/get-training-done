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
async def compute_mutual_information(
    path: str,
    target_column: str,
    task_type: str = "auto",
    top_n: int = 0,
) -> str:
    """Compute mutual information scores for all features vs the target.

    Works on both numeric and categorical features. Useful for detecting
    nonlinear relationships that Pearson/Spearman correlations miss.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target column.
        task_type: 'classification', 'regression', or 'auto' to infer.
        top_n: Return only top N features (0 = all).
    """
    try:
        result = data_profiler.compute_mutual_information(path, target_column, task_type, top_n)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def compute_cramers_v(
    path: str,
    target_column: str = "",
) -> str:
    """Compute Cramér's V for categorical-categorical associations.

    If target_column is given, computes associations between each categorical
    feature and the target. Otherwise computes all categorical pairs.

    Args:
        path: Path to the CSV file.
        target_column: Optional target column for directed associations.
    """
    try:
        result = data_profiler.compute_cramers_v(path, target_column or None)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def compute_anova_scores(
    path: str,
    target_column: str,
) -> str:
    """Compute ANOVA F-test scores for numeric features vs a categorical target.

    Returns F-statistic and p-value per numeric feature, sorted by F-stat.

    Args:
        path: Path to the CSV file.
        target_column: Name of the categorical target column.
    """
    try:
        result = data_profiler.compute_anova_scores(path, target_column)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def compute_vif(
    path: str,
    target_column: str,
    top_n: int = 20,
) -> str:
    """Compute Variance Inflation Factor for feature redundancy detection.

    High VIF (>10) indicates multicollinearity. Severe VIF (>50) means
    near-perfect linear dependence between features.

    Args:
        path: Path to the CSV file.
        target_column: Target column (excluded from VIF computation).
        top_n: Max features to analyze (default 20).
    """
    try:
        result = data_profiler.compute_vif(path, target_column, top_n)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def detect_timestamp_columns(path: str) -> str:
    """Auto-detect columns containing temporal/timestamp data.

    Tries pd.to_datetime on non-numeric columns. If >80% of values parse
    successfully, marks the column as a timestamp.

    Args:
        path: Path to the CSV file.
    """
    try:
        result = data_profiler.detect_timestamp_columns(path)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def analyze_missing_patterns(path: str) -> str:
    """Classify missing data patterns as MCAR, MAR, or MNAR.

    Computes missingness indicator correlations to determine whether missing
    data is random (MCAR), depends on observed variables (MAR), or depends
    on the missing values themselves (MNAR).

    Args:
        path: Path to the CSV file.
    """
    try:
        result = data_profiler.analyze_missing_patterns(path)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def test_normality(
    path: str,
    columns: str = "",
) -> str:
    """Test normality of numeric features using Shapiro-Wilk or Anderson-Darling.

    Uses Shapiro-Wilk for datasets < 5000 rows, Anderson-Darling otherwise.

    Args:
        path: Path to the CSV file.
        columns: Comma-separated column names to test. Empty = all numeric.
    """
    try:
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        result = data_profiler.test_normality(path, col_list)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def analyze_temporal_patterns(
    path: str,
    temporal_column: str,
) -> str:
    """Analyze trend, stationarity, and autocorrelation for a timestamp column.

    Checks for temporal trends in numeric features and computes lag-1
    autocorrelation. Recommends temporal split if significant patterns found.

    Args:
        path: Path to the CSV file.
        temporal_column: Name of the temporal/date column.
    """
    try:
        result = data_profiler.analyze_temporal_patterns(path, temporal_column)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def compute_separability_score(
    path: str,
    target_column: str,
) -> str:
    """Compute Fisher's discriminant ratio for classification difficulty assessment.

    Measures class separability per feature using (μ1-μ2)² / (σ1²+σ2²).
    Higher scores mean easier classification.

    Args:
        path: Path to the CSV file.
        target_column: Name of the classification target column.
    """
    try:
        result = data_profiler.compute_separability_score(path, target_column)
        return _json_response(result)
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
async def compute_dataset_fingerprint(
    path: str,
    target_column: str,
    task_type: str = "auto",
    eda_results: str = "",
) -> str:
    """Compute a rich dataset fingerprint combining profiling with EDA results.

    Always computes core fingerprint fields from the data. If eda_results is
    provided as a JSON string, enriches the fingerprint with signal analysis,
    complexity scoring, and quality metrics.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target column.
        task_type: 'classification', 'regression', or 'auto'.
        eda_results: Optional JSON string with EDA tool outputs.
    """
    try:
        import json as _json
        eda_dict = _json.loads(eda_results) if eda_results else None
        result = data_profiler.compute_dataset_fingerprint(path, target_column, task_type, eda_dict)
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
