"""Comprehensive data profiling engine for dataset analysis and issue detection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def profile_dataset(path: str, target_column: str, task_type: str = "auto") -> dict[str, Any]:
    """Comprehensive first-look profile of a dataset.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target/label column.
        task_type: One of 'classification', 'regression', or 'auto' (inferred).

    Returns:
        Dict with shape, dtypes, distributions, missing info, feature types,
        class balance, preprocessing recommendations, outlier counts, and cardinality.
    """
    df = _load_csv(path)
    _validate_column_exists(df, target_column)

    resolved_task = _resolve_task_type(df, target_column, task_type)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    distributions = _compute_distributions(df, numeric_cols, categorical_cols)
    missing_pct = _compute_missing_pct(df)
    outlier_counts = _compute_outlier_counts(df, numeric_cols)
    cardinality = {col: int(df[col].nunique()) for col in df.columns}

    class_balance = None
    if resolved_task in ("classification", "binary_classification", "multiclass_classification"):
        class_balance = _compute_class_balance(df, target_column)

    recommendations = _generate_recommendations(
        df, numeric_cols, categorical_cols, missing_pct, outlier_counts, cardinality, class_balance,
    )

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "feature_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
        "distributions": distributions,
        "missing_pct": missing_pct,
        "class_balance": class_balance,
        "task_type": resolved_task,
        "outlier_counts": outlier_counts,
        "cardinality": cardinality,
        "recommended_preprocessing": recommendations,
    }


def get_column_stats(path: str, column: str) -> dict[str, Any]:
    """Deep dive into a single column's statistics.

    Args:
        path: Path to the CSV file.
        column: Name of the column to analyze.

    Returns:
        Dict with distribution stats, unique count, missing %, dtype, and is_numeric flag.
    """
    df = _load_csv(path)
    _validate_column_exists(df, column)

    series = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    missing_count = int(series.isna().sum())
    total = len(series)
    missing_pct = _safe_divide(missing_count, total) * 100

    stats: dict[str, Any] = {
        "column": column,
        "dtype": str(series.dtype),
        "is_numeric": is_numeric,
        "unique_count": int(series.nunique()),
        "missing_count": missing_count,
        "missing_pct": _round_val(missing_pct),
        "total_count": total,
    }

    if is_numeric:
        clean = series.dropna()
        stats["distribution"] = {
            "mean": _safe_stat(clean.mean),
            "std": _safe_stat(clean.std),
            "min": _safe_stat(clean.min),
            "max": _safe_stat(clean.max),
            "median": _safe_stat(clean.median),
            "p25": _safe_stat(lambda: clean.quantile(0.25)),
            "p50": _safe_stat(lambda: clean.quantile(0.50)),
            "p75": _safe_stat(lambda: clean.quantile(0.75)),
            "skew": _safe_stat(clean.skew),
            "kurtosis": _safe_stat(clean.kurtosis),
        }
    else:
        vc = series.value_counts()
        top_n = min(20, len(vc))
        stats["distribution"] = {
            "value_counts": {str(k): int(v) for k, v in vc.head(top_n).items()},
            "top_value": str(vc.index[0]) if len(vc) > 0 else None,
            "top_frequency": int(vc.iloc[0]) if len(vc) > 0 else 0,
        }

    return stats


def detect_data_issues(path: str, target_column: str) -> dict[str, Any]:
    """Identify data quality problems and potential pitfalls.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target/label column.

    Returns:
        Dict describing class imbalance, multicollinearity, high cardinality,
        constant features, near-constant features, leakage suspects, and
        missing-heavy columns.
    """
    df = _load_csv(path)
    _validate_column_exists(df, target_column)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    feature_cols = [c for c in df.columns if c != target_column]

    # Class imbalance
    class_imbalance = _detect_class_imbalance(df, target_column)

    # Multicollinearity (numeric feature pairs with |corr| > 0.9)
    multicollinearity = _detect_multicollinearity(df, numeric_cols, target_column, threshold=0.9)

    # High cardinality categorical columns
    high_cardinality = [
        {"column": col, "unique_values": int(df[col].nunique())}
        for col in categorical_cols
        if col != target_column and df[col].nunique() > 50
    ]

    # Constant features (0 or 1 unique values)
    constant_features = [
        col for col in feature_cols if df[col].nunique() <= 1
    ]

    # Near-constant features (top value > 95% frequency)
    near_constant = _detect_near_constant(df, feature_cols, threshold=0.95)

    # Data leakage suspects (features with > 0.95 corr to target)
    leakage_suspects = _detect_leakage(df, numeric_cols, target_column, threshold=0.95)

    # Missing-heavy columns (> 50% missing)
    missing_heavy = [
        {"column": col, "missing_pct": _round_val(df[col].isna().mean() * 100)}
        for col in df.columns
        if df[col].isna().mean() > 0.5
    ]

    return {
        "class_imbalance": class_imbalance,
        "multicollinearity": multicollinearity,
        "high_cardinality_columns": high_cardinality,
        "constant_features": constant_features,
        "near_constant_features": near_constant,
        "data_leakage_suspects": leakage_suspects,
        "missing_heavy_columns": missing_heavy,
    }


def compute_correlations(
    path: str,
    target_column: str,
    method: str = "pearson",
) -> dict[str, Any]:
    """Compute statistical relationships between features and with the target.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target/label column.
        method: Correlation method - 'pearson', 'spearman', or 'kendall'.

    Returns:
        Dict with feature-target correlations, top correlated pairs,
        and the full correlation matrix as nested dicts.
    """
    df = _load_csv(path)
    _validate_column_exists(df, target_column)

    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"Unsupported correlation method '{method}'. Use 'pearson', 'spearman', or 'kendall'.")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return {
            "feature_target_correlations": {},
            "top_correlated_pairs": [],
            "correlation_matrix": {},
        }

    corr_matrix = numeric_df.corr(method=method)

    # Feature-target correlations
    feature_target: dict[str, float] = {}
    if target_column in corr_matrix.columns:
        for col in corr_matrix.columns:
            if col != target_column:
                val = corr_matrix.loc[col, target_column]
                feature_target[col] = _to_native(val)

    # Top correlated feature pairs (excluding self-correlation)
    pairs: list[tuple[str, str, float]] = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not _is_nan(val):
                pairs.append((cols[i], cols[j], _to_native(val)))

    sorted_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    # Full matrix as dict of dicts
    matrix_dict: dict[str, dict[str, float]] = {}
    for col in cols:
        matrix_dict[col] = {
            other: _to_native(corr_matrix.loc[col, other])
            for other in cols
        }

    return {
        "feature_target_correlations": feature_target,
        "top_correlated_pairs": [
            {"feature_1": f1, "feature_2": f2, "correlation": c}
            for f1, f2, c in sorted_pairs
        ],
        "correlation_matrix": matrix_dict,
    }


def preview_data(path: str, n_rows: int = 5) -> dict[str, Any]:
    """Quick look at the first N rows and basic metadata.

    Args:
        path: Path to the CSV file.
        n_rows: Number of rows to preview.

    Returns:
        Dict with rows (list of dicts), dtypes, shape, and column names.
    """
    df = _load_csv(path)
    preview_df = df.head(n_rows)

    rows = _dataframe_to_native_records(preview_df)

    return {
        "rows": rows,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_names": df.columns.tolist(),
    }


# ─── Private Helpers ──────────────────────────────────────────────────────────


def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with basic validation."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not filepath.suffix.lower() == ".csv":
        raise ValueError(f"Expected a CSV file, got: {filepath.suffix}")
    return pd.read_csv(filepath)


def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if column is not in the dataframe."""
    if column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")


def _resolve_task_type(df: pd.DataFrame, target_column: str, task_type: str) -> str:
    """Infer task type from target column if set to 'auto'."""
    if task_type != "auto":
        return task_type

    target = df[target_column]
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
        return "regression"
    if target.nunique() == 2:
        return "binary_classification"
    return "multiclass_classification"


def _compute_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Compute per-column distribution summaries."""
    distributions: dict[str, Any] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        distributions[col] = {
            "type": "numeric",
            "mean": _safe_stat(series.mean),
            "std": _safe_stat(series.std),
            "min": _safe_stat(series.min),
            "max": _safe_stat(series.max),
            "median": _safe_stat(series.median),
        }

    for col in categorical_cols:
        vc = df[col].value_counts()
        top_n = min(10, len(vc))
        distributions[col] = {
            "type": "categorical",
            "top_values": {str(k): int(v) for k, v in vc.head(top_n).items()},
            "num_unique": int(df[col].nunique()),
        }

    return distributions


def _compute_missing_pct(df: pd.DataFrame) -> dict[str, float]:
    """Compute missing percentage per column."""
    return {
        col: _round_val(df[col].isna().mean() * 100)
        for col in df.columns
    }


def _compute_outlier_counts(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, int]:
    """Count outliers per numeric column using the IQR method."""
    outlier_counts: dict[str, int] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            outlier_counts[col] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_counts[col] = count
    return outlier_counts


def _compute_class_balance(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Compute class balance statistics for classification targets."""
    vc = df[target_column].value_counts()
    if len(vc) == 0:
        return {"distribution": {}, "minority_ratio": 0.0, "severity": "severe"}

    distribution = {str(k): int(v) for k, v in vc.items()}
    majority = int(vc.iloc[0])
    minority = int(vc.iloc[-1])
    ratio = _safe_divide(minority, majority)

    if ratio >= 0.8:
        severity = "none"
    elif ratio >= 0.5:
        severity = "mild"
    elif ratio >= 0.2:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "distribution": distribution,
        "minority_ratio": _round_val(ratio),
        "majority_class": str(vc.index[0]),
        "minority_class": str(vc.index[-1]),
        "num_classes": len(vc),
        "severity": severity,
    }


def _detect_class_imbalance(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Detect class imbalance in the target column."""
    if not pd.api.types.is_numeric_dtype(df[target_column]) or df[target_column].nunique() <= 20:
        return _compute_class_balance(df, target_column)
    return {"distribution": {}, "minority_ratio": 1.0, "severity": "none", "note": "target is continuous"}


def _detect_multicollinearity(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Find pairs of numeric features with absolute correlation above threshold."""
    feature_cols = [c for c in numeric_cols if c != target_column]
    if len(feature_cols) < 2:
        return []

    corr = df[feature_cols].corr().abs()
    pairs: list[dict[str, Any]] = []

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            val = corr.iloc[i, j]
            if not _is_nan(val) and val > threshold:
                pairs.append({
                    "feature_1": feature_cols[i],
                    "feature_2": feature_cols[j],
                    "correlation": _round_val(float(val)),
                })

    return sorted(pairs, key=lambda x: x["correlation"], reverse=True)


def _detect_near_constant(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """Find features where the most frequent value exceeds a frequency threshold."""
    results: list[dict[str, Any]] = []
    for col in feature_cols:
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > threshold:
            results.append({
                "column": col,
                "top_value": str(vc.index[0]),
                "frequency_pct": _round_val(vc.iloc[0] * 100),
            })
    return results


def _detect_leakage(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Find numeric features with very high correlation to the target."""
    if target_column not in numeric_cols:
        return []

    feature_cols = [c for c in numeric_cols if c != target_column]
    if not feature_cols:
        return []

    correlations = df[feature_cols].corrwith(df[target_column]).abs()
    suspects: list[dict[str, Any]] = []

    for col, corr_val in correlations.items():
        if not _is_nan(corr_val) and corr_val > threshold:
            suspects.append({
                "column": str(col),
                "correlation_with_target": _round_val(float(corr_val)),
            })

    return sorted(suspects, key=lambda x: x["correlation_with_target"], reverse=True)


def _generate_recommendations(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    missing_pct: dict[str, float],
    outlier_counts: dict[str, int],
    cardinality: dict[str, int],
    class_balance: dict[str, Any] | None,
) -> list[str]:
    """Generate preprocessing recommendations based on data characteristics."""
    recommendations: list[str] = []

    # Missing values
    high_missing = [col for col, pct in missing_pct.items() if pct > 50]
    moderate_missing = [col for col, pct in missing_pct.items() if 5 < pct <= 50]
    any_missing = any(pct > 0 for pct in missing_pct.values())

    if high_missing:
        recommendations.append(
            f"Consider dropping columns with >50% missing: {', '.join(high_missing)}"
        )
    if moderate_missing:
        recommendations.append(
            f"Impute missing values for: {', '.join(moderate_missing)}"
        )
    if any_missing and not high_missing and not moderate_missing:
        recommendations.append("Handle minor missing values with imputation")

    # Outliers
    high_outlier_cols = [col for col, count in outlier_counts.items() if count > len(df) * 0.05]
    if high_outlier_cols:
        recommendations.append(
            f"Investigate outliers in: {', '.join(high_outlier_cols)}"
        )

    # Scaling
    if numeric_cols:
        recommendations.append("Apply StandardScaler or RobustScaler to numeric features")

    # High cardinality
    high_card = [col for col in categorical_cols if cardinality.get(col, 0) > 50]
    if high_card:
        recommendations.append(
            f"Use target encoding for high-cardinality categoricals: {', '.join(high_card)}"
        )

    # Low cardinality categoricals
    low_card = [col for col in categorical_cols if 2 <= cardinality.get(col, 0) <= 10]
    if low_card:
        recommendations.append(
            f"One-hot encode low-cardinality categoricals: {', '.join(low_card)}"
        )

    # Class imbalance
    if class_balance and class_balance.get("severity") in ("moderate", "severe"):
        recommendations.append(
            f"Address class imbalance (severity: {class_balance['severity']}) - "
            "consider SMOTE, class weights, or stratified sampling"
        )

    return recommendations


def _dataframe_to_native_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of dicts with native Python types."""
    records = df.to_dict(orient="records")
    return [
        {k: _to_native(v) for k, v in row.items()}
        for row in records
    ]


def _to_native(value: Any) -> Any:
    """Convert numpy/pandas scalar types to native Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _safe_stat(fn: Any) -> Any:
    """Safely compute a statistic, returning None on failure."""
    try:
        result = fn()
        return _to_native(result)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safe division, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round_val(value: float, decimals: int = 4) -> float:
    """Round a float value for cleaner output."""
    return round(float(value), decimals)


def _is_nan(value: Any) -> bool:
    """Check if a value is NaN (handles numpy and Python floats)."""
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False
