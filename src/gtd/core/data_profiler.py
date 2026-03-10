"""Comprehensive data profiling engine for dataset analysis and issue detection."""

from __future__ import annotations

import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    df = load_csv(path)
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
    df = load_csv(path)
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
    df = load_csv(path)
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
    df = load_csv(path)
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
    df = load_csv(path)
    preview_df = df.head(n_rows)

    rows = _dataframe_to_native_records(preview_df)

    return {
        "rows": rows,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_names": df.columns.tolist(),
    }


# ─── New Statistical Tools ────────────────────────────────────────────────────


def compute_mutual_information(
    path: str,
    target_column: str,
    task_type: str = "auto",
    top_n: int = 0,
) -> dict[str, Any]:
    """Compute mutual information scores for all features vs the target.

    Works on both numeric and categorical features (categoricals are encoded
    via ``pd.factorize``).

    Args:
        path: Path to the CSV file.
        target_column: Name of the target column.
        task_type: 'classification', 'regression', or 'auto' (inferred).
        top_n: If > 0, return only the top N features. 0 = all.

    Returns:
        Dict with per-feature MI scores (sorted desc), max_mi, and n_informative.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    df = load_csv(path)
    _validate_column_exists(df, target_column)
    resolved_task = _resolve_task_type(df, target_column, task_type)

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        return {"scores": {}, "max_mi": 0.0, "n_informative": 0}

    X = df[feature_cols].copy()
    y = df[target_column].copy()

    # Encode categoricals and track which are discrete
    discrete_mask = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col], _ = pd.factorize(X[col])
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)

    # Fill NaN with column medians for numeric, -1 for categorical
    for i, col in enumerate(feature_cols):
        if discrete_mask[i]:
            X[col] = X[col].fillna(-1)
        else:
            X[col] = X[col].fillna(X[col].median())

    # Encode target if categorical
    if not pd.api.types.is_numeric_dtype(y):
        codes, _ = pd.factorize(y)
        y = pd.Series(codes).fillna(0)
    else:
        y = y.fillna(0)

    is_classification = resolved_task in (
        "classification", "binary_classification", "multiclass_classification",
    )
    mi_func = mutual_info_classif if is_classification else mutual_info_regression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mi_func(X, y, discrete_features=discrete_mask, random_state=42)

    scores = {col: _round_val(float(s)) for col, s in zip(feature_cols, mi_scores)}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    if top_n > 0:
        scores = dict(list(scores.items())[:top_n])

    mi_values = list(scores.values())
    return {
        "scores": scores,
        "max_mi": max(mi_values) if mi_values else 0.0,
        "n_informative": sum(1 for v in mi_values if v > 0.01),
    }


def compute_cramers_v(
    path: str,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Compute Cramér's V for categorical ↔ categorical associations.

    If target_column is given, computes associations between each categorical
    feature and the target. Otherwise, computes all categorical×categorical pairs.

    Args:
        path: Path to the CSV file.
        target_column: Optional target column for directed associations.

    Returns:
        Dict with association pairs sorted by strength, plus summary stats.
    """
    from scipy.stats import chi2_contingency

    df = load_csv(path)
    if target_column:
        _validate_column_exists(df, target_column)

    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    associations: list[dict[str, Any]] = []

    if target_column and target_column in categorical_cols:
        # Compute each categorical feature vs target
        feature_cats = [c for c in categorical_cols if c != target_column]
        for col in feature_cats:
            v = _cramers_v_pair(df, col, target_column, chi2_contingency)
            if v is not None:
                associations.append({
                    "feature_1": col,
                    "feature_2": target_column,
                    "cramers_v": _round_val(v),
                })
    else:
        # All pairs
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                v = _cramers_v_pair(
                    df, categorical_cols[i], categorical_cols[j], chi2_contingency,
                )
                if v is not None:
                    associations.append({
                        "feature_1": categorical_cols[i],
                        "feature_2": categorical_cols[j],
                        "cramers_v": _round_val(v),
                    })

    associations.sort(key=lambda x: x["cramers_v"], reverse=True)

    return {
        "associations": associations,
        "n_pairs": len(associations),
        "max_v": associations[0]["cramers_v"] if associations else 0.0,
        "n_strong": sum(1 for a in associations if a["cramers_v"] > 0.3),
    }


def compute_anova_scores(
    path: str,
    target_column: str,
) -> dict[str, Any]:
    """Compute ANOVA F-test scores for numeric features vs a categorical target.

    Args:
        path: Path to the CSV file.
        target_column: Name of the categorical target column.

    Returns:
        Dict with per-feature F-statistic and p-value, sorted by F-stat desc.
    """
    from sklearn.feature_selection import f_classif

    df = load_csv(path)
    _validate_column_exists(df, target_column)

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_column
    ]
    if not numeric_cols:
        return {"scores": {}, "n_significant": 0}

    X = df[numeric_cols].copy()
    y = df[target_column].copy()

    # Fill NaN
    X = X.fillna(X.median())
    if not pd.api.types.is_numeric_dtype(y):
        codes, _ = pd.factorize(y)
        y = pd.Series(codes).fillna(0)
    else:
        y = y.fillna(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_stats, p_values = f_classif(X, y)

    scores = {}
    for col, f_stat, p_val in zip(numeric_cols, f_stats, p_values):
        f_val = _to_native(f_stat)
        p_val_native = _to_native(p_val)
        scores[col] = {
            "f_statistic": _round_val(f_val) if f_val is not None else None,
            "p_value": _round_val(p_val_native, 6) if p_val_native is not None else None,
        }

    # Sort by F-statistic descending
    scores = dict(
        sorted(
            scores.items(),
            key=lambda x: x[1]["f_statistic"] if x[1]["f_statistic"] is not None else 0,
            reverse=True,
        )
    )

    n_significant = sum(
        1 for v in scores.values()
        if v["p_value"] is not None and v["p_value"] < 0.05
    )

    return {"scores": scores, "n_significant": n_significant}


def compute_vif(
    path: str,
    target_column: str,
    top_n: int = 20,
) -> dict[str, Any]:
    """Compute Variance Inflation Factor for numeric features.

    Uses ``numpy.linalg.lstsq`` to compute VIF for each feature. High VIF
    (>10) indicates multicollinearity.

    Args:
        path: Path to the CSV file.
        target_column: Target column (excluded from VIF computation).
        top_n: Maximum number of features to analyze (by variance, desc).

    Returns:
        Dict with per-feature VIF, n_high_vif (>10), n_severe_vif (>50).
    """
    df = load_csv(path)
    _validate_column_exists(df, target_column)

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_column
    ]
    if len(numeric_cols) < 2:
        return {"vif_scores": {}, "n_high_vif": 0, "n_severe_vif": 0}

    X = df[numeric_cols].fillna(df[numeric_cols].median()).values

    # Limit to top_n features by variance
    if len(numeric_cols) > top_n:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-top_n:]
        numeric_cols = [numeric_cols[i] for i in top_indices]
        X = X[:, top_indices]

    # Standardize to avoid numerical issues
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X = (X - np.mean(X, axis=0)) / std

    vif_scores = {}
    for i, col in enumerate(numeric_cols):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)

        if X_others.shape[1] == 0:
            vif_scores[col] = 1.0
            continue

        # Add intercept
        X_aug = np.column_stack([np.ones(X_others.shape[0]), X_others])
        coeffs, residuals, _, _ = np.linalg.lstsq(X_aug, y_i, rcond=None)
        y_pred = X_aug @ coeffs
        ss_res = np.sum((y_i - y_pred) ** 2)
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)

        if ss_tot == 0:
            vif_scores[col] = float("inf")
        else:
            r_squared = 1 - ss_res / ss_tot
            r_squared = min(r_squared, 0.9999)  # cap to avoid inf
            vif_scores[col] = _round_val(1.0 / (1.0 - r_squared))

    vif_scores = dict(sorted(vif_scores.items(), key=lambda x: x[1], reverse=True))

    return {
        "vif_scores": vif_scores,
        "n_high_vif": sum(1 for v in vif_scores.values() if v > 10),
        "n_severe_vif": sum(1 for v in vif_scores.values() if v > 50),
    }


def detect_timestamp_columns(path: str) -> dict[str, Any]:
    """Auto-detect columns that contain temporal/timestamp data.

    Tries ``pd.to_datetime`` on non-numeric columns. If >80% of values
    parse successfully, marks the column as a timestamp.

    Args:
        path: Path to the CSV file.

    Returns:
        Dict with detected timestamp columns and sample values.
    """
    df = load_csv(path)
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()

    detected: list[dict[str, Any]] = []

    for col in non_numeric:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        try:
            parsed = pd.to_datetime(series, format="mixed", errors="coerce")
            success_rate = parsed.notna().mean()
            if success_rate > 0.8:
                valid_dates = parsed.dropna()
                detected.append({
                    "column": col,
                    "success_rate": _round_val(success_rate),
                    "sample_values": [str(v) for v in series.head(3).tolist()],
                    "min_date": str(valid_dates.min()) if len(valid_dates) > 0 else None,
                    "max_date": str(valid_dates.max()) if len(valid_dates) > 0 else None,
                })
        except (ValueError, TypeError, OverflowError):
            continue

    return {
        "timestamp_columns": detected,
        "n_detected": len(detected),
    }


def analyze_missing_patterns(path: str) -> dict[str, Any]:
    """Classify missing data patterns as MCAR, MAR, or MNAR.

    Computes a missingness indicator matrix and correlates indicators with
    each other and with observed features. High correlations suggest MAR/MNAR.

    Args:
        path: Path to the CSV file.

    Returns:
        Dict with overall pattern classification, per-column missing %,
        and correlation details.
    """
    df = load_csv(path)

    missing_pct = {col: _round_val(df[col].isna().mean() * 100) for col in df.columns}
    cols_with_missing = [col for col, pct in missing_pct.items() if pct > 0]

    if not cols_with_missing:
        return {
            "pattern": "none",
            "missing_pct": missing_pct,
            "columns_with_missing": [],
            "correlations": [],
        }

    # Build missingness indicator matrix
    indicators = pd.DataFrame({
        f"missing_{col}": df[col].isna().astype(int) for col in cols_with_missing
    })

    # Check correlations between missingness indicators and observed numeric features
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    correlations: list[dict[str, Any]] = []
    max_corr = 0.0

    for miss_col in cols_with_missing:
        indicator = indicators[f"missing_{miss_col}"]
        if indicator.std() == 0:
            continue

        for num_col in numeric_cols:
            if num_col == miss_col:
                continue
            observed = df[num_col].dropna()
            common_idx = indicator.index.intersection(observed.index)
            if len(common_idx) < 10:
                continue

            corr_val = indicator.loc[common_idx].corr(df[num_col].loc[common_idx])
            if corr_val is not None and not _is_nan(corr_val):
                abs_corr = abs(float(corr_val))
                max_corr = max(max_corr, abs_corr)
                if abs_corr > 0.1:
                    correlations.append({
                        "missing_column": miss_col,
                        "correlated_with": num_col,
                        "correlation": _round_val(float(corr_val)),
                    })

    # Classify pattern
    if max_corr < 0.1:
        pattern = "MCAR"
    elif max_corr < 0.5:
        pattern = "MAR"
    else:
        pattern = "MNAR"

    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "pattern": pattern,
        "missing_pct": missing_pct,
        "columns_with_missing": cols_with_missing,
        "correlations": correlations[:20],
        "max_correlation": _round_val(max_corr),
    }


def test_normality(
    path: str,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Test normality of numeric features using Shapiro-Wilk or Anderson-Darling.

    Uses Shapiro-Wilk for n < 5000 rows, Anderson-Darling otherwise.

    Args:
        path: Path to the CSV file.
        columns: Optional list of columns to test. If None, tests all numeric.

    Returns:
        Dict with per-column test statistic, p-value, and is_normal flag.
    """
    from scipy.stats import anderson, shapiro

    df = load_csv(path)

    if columns:
        for col in columns:
            _validate_column_exists(df, col)
        test_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        test_cols = df.select_dtypes(include="number").columns.tolist()

    results: dict[str, Any] = {}
    n_rows = len(df)

    for col in test_cols:
        series = df[col].dropna()
        if len(series) < 8:
            results[col] = {
                "test": "insufficient_data",
                "is_normal": None,
                "n_valid": len(series),
            }
            continue

        try:
            if n_rows < 5000:
                stat, p_value = shapiro(series)
                results[col] = {
                    "test": "shapiro_wilk",
                    "statistic": _round_val(float(stat), 6),
                    "p_value": _round_val(float(p_value), 6),
                    "is_normal": p_value > 0.05,
                    "n_valid": len(series),
                }
            else:
                result = anderson(series)
                # Use 5% significance level (index 2)
                is_normal = result.statistic < result.critical_values[2]
                results[col] = {
                    "test": "anderson_darling",
                    "statistic": _round_val(float(result.statistic), 6),
                    "critical_value_5pct": _round_val(float(result.critical_values[2]), 6),
                    "is_normal": is_normal,
                    "n_valid": len(series),
                }
        except (ValueError, RuntimeWarning):
            results[col] = {
                "test": "error",
                "is_normal": None,
                "n_valid": len(series),
            }

    n_normal = sum(1 for r in results.values() if r.get("is_normal") is True)
    n_tested = sum(1 for r in results.values() if r.get("is_normal") is not None)

    return {
        "results": results,
        "n_normal": n_normal,
        "n_tested": n_tested,
        "pct_normal": _round_val(n_normal / n_tested * 100) if n_tested > 0 else 0.0,
    }


def analyze_temporal_patterns(
    path: str,
    temporal_column: str,
) -> dict[str, Any]:
    """Analyze trend, stationarity, and autocorrelation for a timestamp column.

    Args:
        path: Path to the CSV file.
        temporal_column: Name of the temporal/date column.

    Returns:
        Dict with trend strength, stationarity assessment, and autocorrelation.
    """
    from scipy.stats import linregress

    df = load_csv(path)
    _validate_column_exists(df, temporal_column)

    # Parse dates and sort
    dates = pd.to_datetime(df[temporal_column], errors="coerce")
    valid_mask = dates.notna()
    if valid_mask.sum() < 10:
        return {
            "temporal_column": temporal_column,
            "error": "insufficient_valid_dates",
            "n_valid": int(valid_mask.sum()),
        }

    df_sorted = df.loc[valid_mask].copy()
    df_sorted["_parsed_date"] = dates[valid_mask]
    df_sorted = df_sorted.sort_values("_parsed_date")

    # Analyze numeric columns for temporal trends
    numeric_cols = [
        c for c in df_sorted.select_dtypes(include="number").columns
        if c != temporal_column
    ]

    trends: dict[str, Any] = {}
    time_index = np.arange(len(df_sorted), dtype=float)

    for col in numeric_cols[:10]:  # limit to 10 columns
        series = df_sorted[col].values.astype(float)
        valid = ~np.isnan(series)
        if valid.sum() < 10:
            continue

        try:
            slope, intercept, r_value, p_value, std_err = linregress(
                time_index[valid], series[valid],
            )
            trends[col] = {
                "slope": _round_val(float(slope), 6),
                "r_squared": _round_val(float(r_value ** 2), 4),
                "p_value": _round_val(float(p_value), 6),
                "has_trend": abs(r_value) > 0.3 and p_value < 0.05,
            }
        except (ValueError, FloatingPointError):
            continue

    # Simple autocorrelation (lag-1) for numeric columns
    autocorrelations: dict[str, float] = {}
    for col in numeric_cols[:10]:
        series = df_sorted[col].dropna()
        if len(series) < 20:
            continue
        values = series.values.astype(float)
        mean = np.mean(values)
        var = np.var(values)
        if var == 0:
            continue
        autocorr = np.mean((values[:-1] - mean) * (values[1:] - mean)) / var
        autocorrelations[col] = _round_val(float(autocorr))

    has_trend = any(t.get("has_trend", False) for t in trends.values())
    has_autocorrelation = any(abs(v) > 0.3 for v in autocorrelations.values())

    return {
        "temporal_column": temporal_column,
        "n_valid_dates": int(valid_mask.sum()),
        "date_range": {
            "min": str(df_sorted["_parsed_date"].min()),
            "max": str(df_sorted["_parsed_date"].max()),
        },
        "trends": trends,
        "autocorrelations": autocorrelations,
        "has_trend": has_trend,
        "has_autocorrelation": has_autocorrelation,
        "recommendation": (
            "temporal_split_recommended" if (has_trend or has_autocorrelation)
            else "temporal_split_not_needed"
        ),
    }


def compute_separability_score(
    path: str,
    target_column: str,
) -> dict[str, Any]:
    """Compute Fisher's discriminant ratio for classification tasks.

    Measures (μ1-μ2)² / (σ1²+σ2²) per feature for each class pair.

    Args:
        path: Path to the CSV file.
        target_column: Name of the classification target column.

    Returns:
        Dict with mean separability, per-feature scores, and difficulty assessment.
    """
    df = load_csv(path)
    _validate_column_exists(df, target_column)

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_column
    ]
    if not numeric_cols:
        return {"scores": {}, "mean_separability": 0.0, "difficulty": "unknown"}

    classes = df[target_column].dropna().unique()
    if len(classes) < 2:
        return {"scores": {}, "mean_separability": 0.0, "difficulty": "trivial"}

    per_feature: dict[str, float] = {}

    for col in numeric_cols:
        pair_ratios: list[float] = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                group_i = df.loc[df[target_column] == classes[i], col].dropna()
                group_j = df.loc[df[target_column] == classes[j], col].dropna()
                if len(group_i) < 2 or len(group_j) < 2:
                    continue
                mu_diff_sq = (group_i.mean() - group_j.mean()) ** 2
                var_sum = group_i.var() + group_j.var()
                if var_sum > 0:
                    pair_ratios.append(float(mu_diff_sq / var_sum))

        if pair_ratios:
            per_feature[col] = _round_val(float(np.mean(pair_ratios)))

    per_feature = dict(sorted(per_feature.items(), key=lambda x: x[1], reverse=True))
    mean_sep = float(np.mean(list(per_feature.values()))) if per_feature else 0.0

    if mean_sep > 2.0:
        difficulty = "easy"
    elif mean_sep > 0.5:
        difficulty = "moderate"
    elif mean_sep > 0.1:
        difficulty = "hard"
    else:
        difficulty = "very_hard"

    return {
        "scores": per_feature,
        "mean_separability": _round_val(mean_sep),
        "difficulty": difficulty,
        "n_classes": len(classes),
    }


def compute_dataset_fingerprint(
    path: str,
    target_column: str,
    task_type: str = "auto",
    eda_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a rich dataset fingerprint combining profiling with EDA results.

    Always computes core fingerprint fields from the data. If ``eda_results``
    is provided (output from various EDA tools), enriches the fingerprint
    with signal analysis, complexity scoring, and quality metrics.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target column.
        task_type: 'classification', 'regression', or 'auto'.
        eda_results: Optional dict mapping tool names to their outputs,
            e.g. ``{"mutual_information": {...}, "vif": {...}}``.

    Returns:
        Rich fingerprint dict with core fields plus optional EDA-derived fields.
    """
    df = load_csv(path)
    _validate_column_exists(df, target_column)
    resolved_task = _resolve_task_type(df, target_column, task_type)

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    n_numeric = len([c for c in numeric_cols if c != target_column])
    n_categorical = len([c for c in categorical_cols if c != target_column])
    total_features = n_numeric + n_categorical

    # Size class
    if n_rows < 1000:
        size_class = "small"
    elif n_rows < 100_000:
        size_class = "medium"
    else:
        size_class = "large"

    # Feature mix
    if total_features == 0:
        feature_mix = "unknown"
    elif n_categorical <= 0:
        feature_mix = "all_numeric"
    elif n_numeric <= 0:
        feature_mix = "all_categorical"
    elif n_numeric / total_features > 0.7:
        feature_mix = "mostly_numeric"
    elif n_categorical / total_features > 0.7:
        feature_mix = "mostly_categorical"
    else:
        feature_mix = "mixed"

    # Core issues
    missing_pct_overall = _round_val(df.isnull().mean().mean() * 100)
    issues: list[str] = []
    if missing_pct_overall > 5:
        issues.append("missing_values")

    # Classification-specific
    n_classes = None
    minority_ratio = None
    target_entropy = None
    if resolved_task in ("classification", "binary_classification", "multiclass_classification"):
        vc = df[target_column].value_counts()
        n_classes = len(vc)
        if len(vc) >= 2:
            minority_ratio = _round_val(int(vc.iloc[-1]) / int(vc.iloc[0]))
            if minority_ratio < 0.2:
                issues.append("class_imbalance")
        # Target entropy
        from scipy.stats import entropy as sp_entropy
        probs = vc.values / vc.values.sum()
        target_entropy = _round_val(float(sp_entropy(probs)))

    # Regression-specific
    target_skewness = None
    if resolved_task == "regression" and pd.api.types.is_numeric_dtype(df[target_column]):
        target_skewness = _round_val(float(df[target_column].skew()))

    # High cardinality ratio
    high_card_cols = [
        c for c in categorical_cols if c != target_column and df[c].nunique() > 50
    ]
    high_cardinality_ratio = (
        _round_val(len(high_card_cols) / n_categorical)
        if n_categorical > 0 else 0.0
    )

    fingerprint: dict[str, Any] = {
        "size_class": size_class,
        "task": resolved_task,
        "feature_mix": feature_mix,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "issues": issues,
        "numeric_ratio": _round_val(n_numeric / total_features) if total_features > 0 else 0.0,
        "missing_pct_overall": missing_pct_overall,
        "n_classes": n_classes,
        "minority_ratio": minority_ratio,
        "target_entropy": target_entropy,
        "target_skewness": target_skewness,
        "high_cardinality_ratio": high_cardinality_ratio,
    }

    # Enrich from EDA results if provided
    if eda_results:
        fingerprint.update(_extract_eda_fingerprint_fields(eda_results))

    # Compute complexity score
    fingerprint["complexity_score"] = _compute_complexity_score(fingerprint)

    # Compute data quality score
    fingerprint["data_quality_score"] = _compute_quality_score(fingerprint)

    return fingerprint


def _extract_eda_fingerprint_fields(eda_results: dict[str, Any]) -> dict[str, Any]:
    """Extract enrichment fields from EDA tool outputs."""
    fields: dict[str, Any] = {}

    # From correlations
    corr = eda_results.get("correlations", {})
    if corr:
        feat_target = corr.get("feature_target_correlations", {})
        if feat_target:
            max_abs = max(abs(v) for v in feat_target.values()) if feat_target else 0
            fields["max_linear_signal"] = _round_val(max_abs)

    # From mutual information
    mi = eda_results.get("mutual_information", {})
    if mi:
        fields["max_nonlinear_signal"] = mi.get("max_mi", 0)
        fields["n_informative_features"] = mi.get("n_informative", 0)

    # Determine signal type
    max_linear = fields.get("max_linear_signal", 0)
    max_nonlinear = fields.get("max_nonlinear_signal", 0)
    if max_linear > 0.5 and max_nonlinear > 0.1:
        fields["signal_type"] = "mixed"
    elif max_linear > 0.5:
        fields["signal_type"] = "linear"
    elif max_nonlinear > 0.1:
        fields["signal_type"] = "nonlinear"
    else:
        fields["signal_type"] = "weak"

    # From VIF
    vif = eda_results.get("vif", {})
    if vif:
        n_high = vif.get("n_high_vif", 0)
        n_severe = vif.get("n_severe_vif", 0)
        fields["n_redundant_pairs"] = n_high
        if n_severe > 0:
            fields["redundancy_level"] = "severe"
        elif n_high > 0:
            fields["redundancy_level"] = "moderate"
        else:
            fields["redundancy_level"] = "low"

    # From missing patterns
    missing = eda_results.get("missing_patterns", {})
    if missing:
        fields["missing_pattern"] = missing.get("pattern", "unknown")
    else:
        fields["missing_pattern"] = "unknown"

    # From temporal analysis
    temporal = eda_results.get("temporal", {})
    if temporal:
        fields["is_temporal"] = temporal.get("has_trend", False) or temporal.get(
            "has_autocorrelation", False,
        )
    else:
        fields["is_temporal"] = False

    return fields


def _compute_complexity_score(fp: dict[str, Any]) -> int:
    """Compute a 1-5 complexity score from fingerprint fields."""
    score = 1

    # Size
    if fp.get("n_cols", 0) > 50:
        score += 1
    if fp.get("n_rows", 0) > 100_000:
        score += 1

    # Signal weakness
    signal_type = fp.get("signal_type", "")
    if signal_type == "weak":
        score += 1
    elif signal_type == "nonlinear":
        score += 1

    # Data quality issues
    if fp.get("missing_pct_overall", 0) > 10:
        score += 1
    if fp.get("high_cardinality_ratio", 0) > 0.3:
        score += 1

    # Class imbalance
    minority = fp.get("minority_ratio")
    if minority is not None and minority < 0.2:
        score += 1

    return min(score, 5)


def _compute_quality_score(fp: dict[str, Any]) -> float:
    """Compute a 0-1 data quality score from fingerprint fields."""
    score = 1.0

    # Penalize missing data
    missing = fp.get("missing_pct_overall", 0)
    score -= min(missing / 50, 0.3)

    # Penalize class imbalance
    minority = fp.get("minority_ratio")
    if minority is not None and minority < 0.5:
        score -= (0.5 - minority) * 0.4

    # Penalize high cardinality
    hc = fp.get("high_cardinality_ratio", 0)
    score -= min(hc * 0.2, 0.1)

    return _round_val(max(0.0, score))


def _cramers_v_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    chi2_func: Any,
) -> float | None:
    """Compute Cramér's V between two categorical columns."""
    try:
        contingency = pd.crosstab(df[col_a], df[col_b])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None
        chi2, _, _, _ = chi2_func(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
        if min_dim == 0 or n == 0:
            return None
        return float(np.sqrt(chi2 / (n * min_dim)))
    except (ValueError, ZeroDivisionError):
        return None


# ─── Private Helpers ──────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with auto-detected separator and encoding."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not filepath.suffix.lower() == ".csv":
        raise ValueError(f"Expected a CSV file, got: {filepath.suffix}")
    # Detect separator from first line (latin-1 always succeeds)
    sample = filepath.read_bytes()[:4096].decode("latin-1")
    first_line = sample.split("\n")[0]
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    # Try utf-8 first, fall back to latin-1
    for encoding in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(filepath, sep=sep, encoding="latin-1")


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
