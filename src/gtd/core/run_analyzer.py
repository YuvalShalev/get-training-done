"""Error analysis, segment identification, and statistical significance testing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gtd.core import workspace

logger = logging.getLogger(__name__)


def analyze_errors(
    workspace_path: str,
    run_id: str,
    data_path: str = "",
    target_column: str = "",
    task_type: str = "",
    top_features: list[str] | None = None,
    n_bins: int = 4,
) -> dict[str, Any]:
    """Analyze model errors by feature segment.

    For classification: computes error rate per feature bin, confusion patterns,
    and confidence-band error rates.
    For regression: computes residual statistics and MAE per feature bin.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to analyze.
        data_path: Path to the CSV data file. If omitted, uses the
                   validation partition from the workspace split.
        target_column: Name of the target column. If omitted, read from run config.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'. If omitted, read from run config.
        top_features: Feature names to analyze. If None, uses top 5 by importance.
        n_bins: Number of bins for feature segmentation.

    Returns:
        Dict with error_by_segment, and either confusion_patterns +
        confidence_analysis (classification) or residual_stats (regression).
    """
    from gtd.core.evaluator import _resolve_defaults

    data_path, target_column, task_type = _resolve_defaults(
        workspace_path, run_id, data_path, target_column, task_type,
    )
    model, X, y, feature_columns, df = _load_analysis_context(
        workspace_path, run_id, data_path, target_column,
    )

    y_pred = model.predict(X)

    # Determine which features to analyze
    if top_features is None:
        top_features = _get_top_features(model, feature_columns, n=5)

    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        is_error = (y_pred != y).astype(int)
        overall_error_rate = float(is_error.mean())

        # Error rate by feature segment
        error_by_segment = []
        for feat in top_features:
            if feat not in df.columns:
                continue
            segments = _compute_segment_metrics(
                df[feat].values, is_error, n_bins, metric_name="error_rate",
            )
            for seg in segments:
                error_by_segment.append({"feature": feat, **seg})

        # Confusion patterns: most common (true, predicted) pairs for errors
        error_mask = is_error.astype(bool)
        confusion_patterns = []
        if error_mask.any():
            pairs = list(zip(y[error_mask], y_pred[error_mask]))
            from collections import Counter
            pair_counts = Counter(pairs)
            for (true_val, pred_val), count in pair_counts.most_common(5):
                confusion_patterns.append({
                    "true": _to_serializable(true_val),
                    "predicted": _to_serializable(pred_val),
                    "count": count,
                })

        # Confidence analysis
        confidence_analysis = {}
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)
                max_conf = y_prob.max(axis=1)
                bands = _bin_values(max_conf, bins=[0.0, 0.5, 0.8, 1.01],
                                    labels=["low", "medium", "high"])
                conf_results = {}
                for band_name in ["low", "medium", "high"]:
                    mask = bands == band_name
                    if mask.any():
                        conf_results[band_name] = {
                            "count": int(mask.sum()),
                            "error_rate": float(is_error[mask].mean()),
                        }
                confidence_analysis = conf_results
            except Exception as exc:
                logger.warning("Could not compute confidence analysis: %s", exc)

        return {
            "overall_error_rate": overall_error_rate,
            "error_by_segment": error_by_segment,
            "confusion_patterns": confusion_patterns,
            "confidence_analysis": confidence_analysis,
        }

    else:
        # Regression
        residuals = y - y_pred
        abs_residuals = np.abs(residuals)

        residual_stats = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "skew": float(pd.Series(residuals).skew()),
            "mae": float(abs_residuals.mean()),
        }

        error_by_segment = []
        for feat in top_features:
            if feat not in df.columns:
                continue
            segments = _compute_segment_metrics(
                df[feat].values, abs_residuals, n_bins, metric_name="mae",
            )
            for seg in segments:
                error_by_segment.append({"feature": feat, **seg})

        return {
            "residual_stats": residual_stats,
            "error_by_segment": error_by_segment,
        }


def identify_segments(
    workspace_path: str,
    run_id: str,
    data_path: str = "",
    target_column: str = "",
    task_type: str = "",
    threshold_pct: float = 5.0,
    n_bins: int = 4,
) -> dict[str, Any]:
    """Identify high-performing and low-performing data segments.

    Computes the overall metric, then finds feature bins where performance
    differs from the overall by more than threshold_pct.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the CSV data file. If omitted, uses the
                   validation partition from the workspace split.
        target_column: Name of the target column. If omitted, read from run config.
        task_type: Task type string. If omitted, read from run config.
        threshold_pct: Minimum percentage difference to flag a segment.
        n_bins: Number of bins for feature segmentation.

    Returns:
        Dict with overall_metric, high_performing segments, and
        low_performing segments.
    """
    from gtd.core.evaluator import _resolve_defaults

    data_path, target_column, task_type = _resolve_defaults(
        workspace_path, run_id, data_path, target_column, task_type,
    )
    model, X, y, feature_columns, df = _load_analysis_context(
        workspace_path, run_id, data_path, target_column,
    )

    y_pred = model.predict(X)
    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        is_correct = (y_pred == y).astype(int)
        overall_metric = float(is_correct.mean())
        metric_values = is_correct
        metric_name = "accuracy"
    else:
        from sklearn.metrics import r2_score
        overall_metric = float(r2_score(y, y_pred))
        # Per-sample: use negative absolute error (higher = better)
        abs_errors = np.abs(y - y_pred)
        metric_values = -abs_errors  # Negate so higher = better for comparison
        metric_name = "neg_mae"

    # Get top features
    top_features = _get_top_features(model, feature_columns, n=5)

    high_performing: list[dict[str, Any]] = []
    low_performing: list[dict[str, Any]] = []

    for feat in top_features:
        if feat not in df.columns:
            continue
        segments = _compute_segment_metrics(
            df[feat].values, metric_values, n_bins, metric_name=metric_name,
        )
        for seg in segments:
            seg_metric = seg["metric"]
            if is_classification:
                delta_pct = (seg_metric - overall_metric) * 100
            else:
                # For regression, compare against overall mean of metric_values
                overall_mean = float(metric_values.mean())
                delta_pct = ((seg_metric - overall_mean) / (abs(overall_mean) + 1e-10)) * 100

            entry = {
                "feature": feat,
                "segment": seg["segment"],
                "metric": round(seg_metric, 4),
                "delta_pct": round(delta_pct, 2),
                "sample_count": seg["count"],
            }

            if delta_pct > threshold_pct:
                high_performing.append(entry)
            elif delta_pct < -threshold_pct:
                low_performing.append(entry)

    return {
        "overall_metric": round(overall_metric, 4),
        "metric_name": metric_name,
        "high_performing": sorted(high_performing, key=lambda x: x["delta_pct"], reverse=True),
        "low_performing": sorted(low_performing, key=lambda x: x["delta_pct"]),
    }


def test_significance(
    cv_scores_a: list[float],
    cv_scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Test whether two sets of CV scores differ significantly.

    Uses the Wilcoxon signed-rank test which is robust for small n (e.g.,
    5-fold CV). Falls back to a paired t-test if Wilcoxon cannot be computed.

    Args:
        cv_scores_a: CV scores from run A.
        cv_scores_b: CV scores from run B.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with test_name, p_value, is_significant, mean_diff,
        ci_lower, ci_upper, and recommendation.
    """
    a = np.array(cv_scores_a, dtype=float)
    b = np.array(cv_scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"CV score arrays must have equal length (got {len(a)} and {len(b)})"
        )

    diffs = b - a
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0

    # Confidence interval for the mean difference
    n = len(diffs)
    if n > 1 and std_diff > 0:
        from scipy.stats import t as t_dist
        t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
        margin = t_crit * std_diff / np.sqrt(n)
    else:
        margin = 0.0

    ci_lower = float(mean_diff - margin)
    ci_upper = float(mean_diff + margin)

    # Statistical test — use paired t-test (more powerful for small n like CV folds)
    # Wilcoxon's minimum p-value with n=5 is 0.0625, which can never reach alpha=0.05.
    # The paired t-test is more appropriate for paired CV score comparisons.
    test_name = "paired_t_test"
    p_value = 1.0
    try:
        from scipy.stats import ttest_rel
        if std_diff > 0:
            stat, p_value = ttest_rel(a, b)
            p_value = float(p_value)
        else:
            # All differences identical — use sign test logic
            p_value = 0.0 if mean_diff != 0 else 1.0
    except (ValueError, ImportError):
        p_value = 1.0
        test_name = "none"

    is_significant = p_value < alpha

    # Recommendation
    if not is_significant:
        recommendation = "No significant difference. Keep exploring."
    elif mean_diff > 0:
        recommendation = "B is significantly better. Adopt B."
    else:
        recommendation = "A is significantly better. Keep A."

    return {
        "test_name": test_name,
        "p_value": round(p_value, 6),
        "is_significant": is_significant,
        "mean_diff": round(mean_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "recommendation": recommendation,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _load_analysis_context(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
) -> tuple[Any, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Load model, data, feature columns, and raw DataFrame for analysis.

    Returns:
        Tuple of (model, X, y, feature_columns, df).
    """
    ws = Path(workspace_path)
    run_dir = ws / "runs" / run_id

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = workspace.load_run_artifact(ws, run_id, "config.json")
    feature_columns: list[str] = config["feature_columns"]

    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    from gtd.core.data_profiler import load_csv
    df = load_csv(str(source))
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from data: {missing}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    model = joblib.load(model_path)
    X = df[feature_columns].values
    y = df[target_column].values

    return model, X, y, feature_columns, df


def _get_top_features(
    model: Any,
    feature_columns: list[str],
    n: int = 5,
) -> list[str]:
    """Get top N features by importance from the model."""
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw = np.abs(model.coef_).flatten()
        if len(raw) != len(feature_columns):
            raw = np.abs(model.coef_).mean(axis=0)
    else:
        # Fallback: return first n features
        return feature_columns[:n]

    indices = np.argsort(raw)[::-1][:n]
    return [feature_columns[i] for i in indices]


def _compute_segment_metrics(
    feature_values: np.ndarray,
    metric_values: np.ndarray,
    n_bins: int,
    metric_name: str,
) -> list[dict[str, Any]]:
    """Bin a feature and compute the mean metric per bin.

    Returns:
        List of dicts with segment label, metric value, and sample count.
    """
    series = pd.Series(feature_values)

    # Skip non-numeric features
    if not pd.api.types.is_numeric_dtype(series):
        unique_vals = series.unique()
        segments = []
        for val in unique_vals[:n_bins]:
            mask = series == val
            if mask.sum() < 2:
                continue
            segments.append({
                "segment": str(val),
                "metric": float(metric_values[mask].mean()),
                "count": int(mask.sum()),
            })
        return segments

    try:
        bins = pd.qcut(series, q=n_bins, duplicates="drop")
    except (ValueError, TypeError):
        return []

    segments = []
    for interval in bins.cat.categories:
        mask = bins == interval
        if mask.sum() < 2:
            continue
        segments.append({
            "segment": str(interval),
            "metric": float(metric_values[mask].mean()),
            "count": int(mask.sum()),
        })
    return segments


def _bin_values(
    values: np.ndarray,
    bins: list[float],
    labels: list[str],
) -> np.ndarray:
    """Bin numeric values into labeled categories."""
    result = np.full(len(values), labels[-1], dtype=object)
    for i in range(len(labels)):
        mask = (values >= bins[i]) & (values < bins[i + 1])
        result[mask] = labels[i]
    return result


def _to_serializable(val: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val
