"""Deep model analysis for reflexion checkpoints.

Produces ranked, actionable insights from error profiling, slice discovery,
confidence calibration, and threshold optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from gtd.core.run_analyzer import _load_analysis_context

logger = logging.getLogger(__name__)


def analyze_run_deep(
    workspace_path: str,
    run_id: str,
    data_path: str = "",
    target_column: str = "",
    task_type: str = "",
    top_n: int = 15,
) -> dict[str, Any]:
    """Run comprehensive model analysis and return ranked actionable insights.

    Orchestrates error profiling, slice discovery, confidence calibration,
    and threshold optimization. Call at reflexion checkpoints (every 3 runs).

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to analyze.
        data_path: Path to the CSV data file. If omitted, uses the
                   validation partition from the workspace split.
        target_column: Name of the target column. If omitted, read from run config.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'. If omitted, read from run config.
        top_n: Maximum number of insights to return.

    Returns:
        Dict with 'insights' (ranked list), 'summary' (2-3 sentences),
        and 'top_recommendation' (single most impactful next action).
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
    is_binary = task_type == "binary_classification"

    all_insights: list[dict[str, Any]] = []

    if is_classification:
        is_error = (y_pred != y).astype(int)
        overall_error_rate = float(is_error.mean())

        all_insights.extend(
            _error_profiling_classification(model, X, y, y_pred, feature_columns, df)
        )
        all_insights.extend(
            _slice_discovery(X, y, y_pred, feature_columns, task_type, overall_error_rate)
        )
        all_insights.extend(
            _confidence_analysis(model, X, y, y_pred)
        )
        if is_binary:
            all_insights.extend(
                _threshold_optimization(model, X, y)
            )
    else:
        residuals = y - y_pred
        abs_residuals = np.abs(residuals)

        all_insights.extend(
            _error_profiling_regression(model, X, y, y_pred, feature_columns, df)
        )
        all_insights.extend(
            _slice_discovery(
                X, y, y_pred, feature_columns, task_type,
                float((abs_residuals > np.percentile(abs_residuals, 75)).mean()),
            )
        )
        all_insights.extend(
            _prediction_range_analysis(model, X, y, y_pred)
        )

    ranked = _rank_insights(all_insights, top_n)
    summary, top_rec = _generate_summary(ranked)

    return {
        "insights": ranked,
        "summary": summary,
        "top_recommendation": top_rec,
    }


# ─── Classification analysis passes ──────────────────────────────────────────


def _error_profiling_classification(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    feature_columns: list[str],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compute point-biserial correlation between features and error indicator."""
    from scipy.stats import pointbiserialr

    is_error = (y_pred != y).astype(int)
    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return []

    insights: list[dict[str, Any]] = []

    for i, feat in enumerate(feature_columns):
        col = X[:, i]
        if not np.issubdtype(col.dtype, np.number):
            continue
        valid = np.isfinite(col)
        if valid.sum() < 10:
            continue

        try:
            r, p = pointbiserialr(is_error[valid], col[valid])
        except Exception:
            continue

        if abs(r) < 0.1 or p > 0.05:
            continue

        # Compute median for errors vs correct
        median_err = float(np.median(col[valid & (is_error == 1)]))
        median_ok = float(np.median(col[valid & (is_error == 0)]))

        direction = "higher" if r > 0 else "lower"
        insights.append({
            "rank": 0,
            "category": "error_profiling",
            "description": (
                f"Errors have {direction} {feat} "
                f"(r={r:.2f}, median {median_err:.3g} vs {median_ok:.3g})"
            ),
            "current_metric": float(abs(r)),
            "target_metric": None,
            "sample_count": int(is_error.sum()),
            "estimated_impact": float(abs(r) * is_error.mean()),
            "recommendation": f"Engineer or transform feature '{feat}' to reduce error correlation",
            "confidence": "high" if abs(r) > 0.3 else "medium",
        })

    return insights


def _confidence_analysis(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Find confidence bands where model is overconfident."""
    if not hasattr(model, "predict_proba"):
        return []

    try:
        y_prob = model.predict_proba(X)
    except Exception:
        return []

    max_conf = y_prob.max(axis=1)
    is_error = (y_pred != y).astype(int)
    overall_error_rate = float(is_error.mean())

    bands = [
        ("very_low", 0.0, 0.5),
        ("low", 0.5, 0.65),
        ("medium", 0.65, 0.8),
        ("high", 0.8, 0.95),
        ("very_high", 0.95, 1.01),
    ]

    insights: list[dict[str, Any]] = []
    for name, lo, hi in bands:
        mask = (max_conf >= lo) & (max_conf < hi)
        count = int(mask.sum())
        if count < 5:
            continue

        band_error = float(is_error[mask].mean())
        if band_error <= overall_error_rate * 1.2:
            continue

        excess = band_error - overall_error_rate
        insights.append({
            "rank": 0,
            "category": "confidence",
            "description": (
                f"{name.replace('_', '-')}-confidence band ({lo:.2f}-{hi:.2f}) "
                f"has {band_error:.0%} error rate vs {overall_error_rate:.0%} overall"
            ),
            "current_metric": band_error,
            "target_metric": overall_error_rate,
            "sample_count": count,
            "estimated_impact": float(excess * count / len(is_error)),
            "recommendation": (
                f"Investigate {count} samples in {name} confidence band — "
                f"consider calibration or additional features"
            ),
            "confidence": "high" if count > 20 else "medium",
        })

    return insights


def _threshold_optimization(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict[str, Any]]:
    """Sweep thresholds to find optimal F1 for binary classification."""
    if not hasattr(model, "predict_proba"):
        return []

    try:
        y_prob = model.predict_proba(X)
    except Exception:
        return []

    if y_prob.shape[1] != 2:
        return []

    from sklearn.metrics import f1_score

    pos_prob = y_prob[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)

    best_t = 0.5
    best_f1 = 0.0
    f1_at_half = 0.0

    for t in thresholds:
        preds = (pos_prob >= t).astype(int)
        f1 = float(f1_score(y, preds, zero_division=0))
        if t == 0.5 or abs(t - 0.5) < 0.01:
            f1_at_half = f1
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    if abs(best_t - 0.5) < 0.05 or (best_f1 - f1_at_half) < 0.005:
        return []

    return [{
        "rank": 0,
        "category": "threshold",
        "description": (
            f"Optimal threshold is {best_t:.2f} (F1={best_f1:.3f}) "
            f"vs default 0.5 (F1={f1_at_half:.3f})"
        ),
        "current_metric": f1_at_half,
        "target_metric": best_f1,
        "sample_count": len(y),
        "estimated_impact": float(best_f1 - f1_at_half),
        "recommendation": (
            f"Set classification threshold to {best_t:.2f} "
            f"for +{best_f1 - f1_at_half:.3f} F1 improvement"
        ),
        "confidence": "high",
    }]


# ─── Regression analysis passes ──────────────────────────────────────────────


def _error_profiling_regression(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    feature_columns: list[str],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Correlate features with absolute residuals."""
    abs_residuals = np.abs(y - y_pred)

    insights: list[dict[str, Any]] = []
    for i, feat in enumerate(feature_columns):
        col = X[:, i]
        if not np.issubdtype(col.dtype, np.number):
            continue
        valid = np.isfinite(col)
        if valid.sum() < 10:
            continue

        corr = np.corrcoef(col[valid], abs_residuals[valid])[0, 1]
        if not np.isfinite(corr) or abs(corr) < 0.1:
            continue

        direction = "high" if corr > 0 else "low"
        insights.append({
            "rank": 0,
            "category": "error_profiling",
            "description": (
                f"{direction.title()} {feat} values correlate with larger residuals "
                f"(r={corr:.2f})"
            ),
            "current_metric": float(abs(corr)),
            "target_metric": None,
            "sample_count": int(valid.sum()),
            "estimated_impact": float(abs(corr) * abs_residuals.mean()),
            "recommendation": f"Transform or bin feature '{feat}' to reduce residual correlation",
            "confidence": "high" if abs(corr) > 0.3 else "medium",
        })

    return insights


def _prediction_range_analysis(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Bin by predicted value and find ranges where MAE is highest."""
    abs_residuals = np.abs(y - y_pred)
    overall_mae = float(abs_residuals.mean())

    try:
        bins = pd.qcut(pd.Series(y_pred), q=4, duplicates="drop")
    except (ValueError, TypeError):
        return []

    insights: list[dict[str, Any]] = []
    for interval in bins.cat.categories:
        mask = bins == interval
        count = int(mask.sum())
        if count < 5:
            continue

        bin_mae = float(abs_residuals[mask].mean())
        if bin_mae <= overall_mae * 1.3:
            continue

        excess = bin_mae - overall_mae
        insights.append({
            "rank": 0,
            "category": "error_profiling",
            "description": (
                f"Prediction range {interval} has MAE={bin_mae:.3g} "
                f"vs overall MAE={overall_mae:.3g}"
            ),
            "current_metric": bin_mae,
            "target_metric": overall_mae,
            "sample_count": count,
            "estimated_impact": float(excess * count / len(y)),
            "recommendation": (
                f"Model struggles in prediction range {interval} — "
                f"consider specialized features or ensemble"
            ),
            "confidence": "high" if count > 20 else "medium",
        })

    return insights


# ─── Shared analysis passes ──────────────────────────────────────────────────


def _slice_discovery(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    feature_columns: list[str],
    task_type: str,
    overall_error_rate: float,
) -> list[dict[str, Any]]:
    """Fit a shallow decision tree on error indicator to find weak subpopulations."""
    from sklearn.tree import DecisionTreeClassifier

    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        is_error = (y_pred != y).astype(int)
    else:
        abs_residuals = np.abs(y - y_pred)
        is_error = (abs_residuals > np.percentile(abs_residuals, 75)).astype(int)

    if is_error.sum() < 5 or (len(is_error) - is_error.sum()) < 5:
        return []

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, len(X) // 50))
    try:
        tree.fit(X, is_error)
    except Exception:
        return []

    # Extract leaf nodes with error rate above overall
    leaf_ids = tree.apply(X)
    unique_leaves = np.unique(leaf_ids)

    insights: list[dict[str, Any]] = []
    for leaf in unique_leaves:
        mask = leaf_ids == leaf
        count = int(mask.sum())
        if count < 10:
            continue

        leaf_error = float(is_error[mask].mean())
        if leaf_error <= overall_error_rate * 1.15:
            continue

        # Build rule path for this leaf
        rule = _extract_rule(tree, leaf, feature_columns)
        excess = leaf_error - overall_error_rate

        insights.append({
            "rank": 0,
            "category": "slice_discovery",
            "description": (
                f"Subpopulation [{rule}] has {leaf_error:.0%} error rate "
                f"vs {overall_error_rate:.0%} overall ({count} samples)"
            ),
            "current_metric": leaf_error,
            "target_metric": overall_error_rate,
            "sample_count": count,
            "estimated_impact": float(excess * count / len(is_error)),
            "recommendation": (
                f"Target subpopulation [{rule}] "
                "with specialized features or handling"
            ),
            "confidence": "high" if count > 30 else "medium" if count > 15 else "low",
        })

    return insights


def _extract_rule(
    tree: Any,
    leaf_id: int,
    feature_columns: list[str],
) -> str:
    """Extract the decision path to a leaf node as a human-readable rule."""
    tree_obj = tree.tree_
    feature = tree_obj.feature
    threshold = tree_obj.threshold

    # Trace the path manually from root to leaf
    path: list[str] = []
    node = 0  # root
    while node != leaf_id:
        if feature[node] < 0:
            break
        feat_name = feature_columns[feature[node]]
        thresh = threshold[node]

        left = tree_obj.children_left[node]
        right = tree_obj.children_right[node]

        # Check which child leads to the leaf
        if _node_contains_leaf(tree_obj, left, leaf_id):
            path.append(f"{feat_name} <= {thresh:.2f}")
            node = left
        elif _node_contains_leaf(tree_obj, right, leaf_id):
            path.append(f"{feat_name} > {thresh:.2f}")
            node = right
        else:
            break

    return " AND ".join(path) if path else "unknown"


def _node_contains_leaf(tree_obj: Any, node: int, leaf_id: int) -> bool:
    """Check if a tree node is or contains the target leaf."""
    if node == leaf_id:
        return True
    if tree_obj.children_left[node] == -1:  # leaf node
        return node == leaf_id

    return (
        _node_contains_leaf(tree_obj, tree_obj.children_left[node], leaf_id)
        or _node_contains_leaf(tree_obj, tree_obj.children_right[node], leaf_id)
    )


# ─── Ranking & summary ───────────────────────────────────────────────────────


def _rank_insights(
    all_insights: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Sort insights by estimated_impact descending and assign ranks."""
    sorted_insights = sorted(
        all_insights,
        key=lambda x: x["estimated_impact"],
        reverse=True,
    )[:top_n]

    for i, insight in enumerate(sorted_insights, 1):
        insight["rank"] = i

    return sorted_insights


def _generate_summary(
    ranked_insights: list[dict[str, Any]],
) -> tuple[str, str]:
    """Generate a 2-3 sentence summary and top recommendation."""
    if not ranked_insights:
        return (
            "No significant model weaknesses detected. "
            "The model performs uniformly across segments.",
            "Continue with current approach or try a different model family.",
        )

    categories = {}
    for ins in ranked_insights:
        cat = ins["category"]
        categories[cat] = categories.get(cat, 0) + 1

    top = ranked_insights[0]
    parts = [f"Found {len(ranked_insights)} actionable insights."]

    if "slice_discovery" in categories:
        parts.append(
            f"Identified {categories['slice_discovery']} weak subpopulation(s) "
            f"with elevated error rates."
        )
    if "confidence" in categories:
        parts.append(
            f"Detected {categories['confidence']} overconfident prediction band(s)."
        )
    if "threshold" in categories:
        parts.append("Threshold optimization can improve F1.")

    summary = " ".join(parts[:3])
    return summary, top["recommendation"]
