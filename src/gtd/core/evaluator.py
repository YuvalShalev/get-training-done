"""Full evaluation engine for trained models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gtd.core import workspace

logger = logging.getLogger(__name__)

# Use a clean plotting style
_PLOT_STYLE = "seaborn-v0_8-whitegrid"


def evaluate_model(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    task_type: str,
) -> dict[str, Any]:
    """Run full evaluation for a trained model on a dataset.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to evaluate.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.

    Returns:
        Full metrics dict (accuracy, f1, r2, rmse, etc. depending on task_type).
    """
    model, X, y, feature_columns = _load_run_context(
        workspace_path, run_id, data_path, target_column,
    )

    y_pred = model.predict(X)

    if task_type in ("binary_classification", "multiclass_classification"):
        metrics = _classification_metrics(model, X, y, y_pred, task_type)
    else:
        metrics = _regression_metrics(y, y_pred)

    # Persist metrics
    workspace.save_run_artifact(workspace_path, run_id, "eval_metrics.json", metrics)
    workspace.update_run_metrics(workspace_path, run_id, _numeric_subset(metrics))

    return metrics


def error_analysis(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    task_type: str,
) -> dict[str, Any]:
    """Run error analysis on a trained model, combining feature importance with segment errors.

    Gets top 5 features by importance, then delegates to run_analyzer.analyze_errors.
    Saves the result as an artifact.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to analyze.
        data_path: Path to the CSV data file.
        target_column: Name of the target column.
        task_type: Task type string.

    Returns:
        Error analysis dict from run_analyzer.analyze_errors.
    """
    from gtd.core import run_analyzer

    # Get top features via importance
    imp_result = get_feature_importance(workspace_path, run_id, data_path, target_column)
    top_features = sorted(
        imp_result["importances"].items(), key=lambda x: x[1], reverse=True,
    )[:5]
    top_feature_names = [f[0] for f in top_features]

    result = run_analyzer.analyze_errors(
        workspace_path=workspace_path,
        run_id=run_id,
        data_path=data_path,
        target_column=target_column,
        task_type=task_type,
        top_features=top_feature_names,
    )

    workspace.save_run_artifact(workspace_path, run_id, "error_analysis.json", result)
    return result


def get_feature_importance(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    method: str = "builtin",
) -> dict[str, Any]:
    """Compute feature importance for a trained model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.
        method: 'builtin' for model.feature_importances_, or 'permutation'.

    Returns:
        Dict with method, importances mapping, and plot_path.
    """
    model, X, y, feature_columns = _load_run_context(
        workspace_path, run_id, data_path, target_column,
    )

    if method == "builtin":
        importances = _builtin_importance(model, feature_columns)
    elif method == "permutation":
        config = workspace.load_run_artifact(workspace_path, run_id, "config.json")
        task_type = config.get("task_type", "binary_classification")
        importances = _permutation_importance(model, X, y, feature_columns, task_type)
    else:
        raise ValueError(f"Unknown importance method '{method}'. Use 'builtin' or 'permutation'.")

    # Save bar chart
    run_dir = workspace.get_run_dir(workspace_path, run_id)
    plot_path = str(run_dir / "feature_importance.png")
    _plot_feature_importance(importances, plot_path)

    result = {
        "method": method,
        "importances": importances,
        "plot_path": plot_path,
    }
    workspace.save_run_artifact(workspace_path, run_id, "feature_importance.json", result)
    return result


def get_roc_curve(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
) -> dict[str, Any]:
    """Compute and plot the ROC curve for a binary classification model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.

    Returns:
        Dict with fpr, tpr, auc, and plot_path.

    Raises:
        ValueError: If the model does not support predict_proba or task is not
                     binary classification.
    """
    from sklearn.metrics import auc, roc_curve

    model, X, y, _ = _load_run_context(
        workspace_path, run_id, data_path, target_column,
    )

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba; cannot compute ROC curve")

    y_prob = model.predict_proba(X)
    if y_prob.shape[1] != 2:
        raise ValueError("ROC curve is only supported for binary classification")

    fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
    roc_auc = float(auc(fpr, tpr))

    run_dir = workspace.get_run_dir(workspace_path, run_id)
    plot_path = str(run_dir / "roc_curve.png")
    _plot_roc_curve(fpr, tpr, roc_auc, plot_path)

    result = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc,
        "plot_path": plot_path,
    }
    workspace.save_run_artifact(workspace_path, run_id, "roc_curve.json", result)
    return result


def get_pr_curve(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
) -> dict[str, Any]:
    """Compute and plot the precision-recall curve.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.

    Returns:
        Dict with precision, recall, average_precision (ap), and plot_path.

    Raises:
        ValueError: If the model does not support predict_proba.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    model, X, y, _ = _load_run_context(
        workspace_path, run_id, data_path, target_column,
    )

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba; cannot compute PR curve")

    y_prob = model.predict_proba(X)

    # For binary: use column 1; for multiclass: use macro average
    if y_prob.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y, y_prob[:, 1])
        ap = float(average_precision_score(y, y_prob[:, 1]))
    else:
        # One-vs-rest macro average for multiclass
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y)
        y_bin = label_binarize(y, classes=classes)
        precision_sum = np.zeros(0)
        recall_sum = np.zeros(0)
        ap_scores: list[float] = []
        for i in range(len(classes)):
            p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            ap_scores.append(float(average_precision_score(y_bin[:, i], y_prob[:, i])))
        ap = float(np.mean(ap_scores))
        # Use the first class curves for the plot (simplification for multiclass)
        precision, recall, _ = precision_recall_curve(y_bin[:, 0], y_prob[:, 0])

    run_dir = workspace.get_run_dir(workspace_path, run_id)
    plot_path = str(run_dir / "pr_curve.png")
    _plot_pr_curve(precision, recall, ap, plot_path)

    result = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "ap": ap,
        "plot_path": plot_path,
    }
    workspace.save_run_artifact(workspace_path, run_id, "pr_curve.json", result)
    return result


def compare_runs(
    workspace_path: str,
    run_ids: list[str],
) -> dict[str, Any]:
    """Compare metrics across multiple runs side by side.

    Args:
        workspace_path: Path to the workspace directory.
        run_ids: List of run IDs to compare.

    Returns:
        Dict with comparison table, best_run_id, and metric deltas.
    """
    if not run_ids:
        raise ValueError("At least one run_id is required for comparison")

    rows: list[dict[str, Any]] = []
    all_metric_keys: set[str] = set()

    for rid in run_ids:
        run_meta = workspace.get_run_metadata(workspace_path, rid)
        if run_meta is None:
            logger.warning("Run '%s' not found in workspace, skipping", rid)
            continue

        row = {
            "run_id": rid,
            "model_type": run_meta.get("model_type", "unknown"),
            "created_at": run_meta.get("created_at", ""),
        }

        run_metrics = run_meta.get("metrics", {})
        row.update(run_metrics)
        all_metric_keys.update(run_metrics.keys())
        rows.append(row)

    if not rows:
        raise ValueError("No valid runs found for comparison")

    # Determine the primary scoring metric
    primary_metric = _infer_primary_metric(all_metric_keys)

    # Find the best run
    best_row = max(rows, key=lambda r: r.get(primary_metric, float("-inf")))
    best_run_id = best_row["run_id"]

    # Compute deltas relative to the best
    deltas: dict[str, dict[str, float]] = {}
    for row in rows:
        rid = row["run_id"]
        if rid == best_run_id:
            continue
        delta: dict[str, float] = {}
        for key in all_metric_keys:
            best_val = best_row.get(key)
            current_val = row.get(key)
            if isinstance(best_val, (int, float)) and isinstance(current_val, (int, float)):
                delta[key] = round(current_val - best_val, 6)
        deltas[rid] = delta

    return {
        "comparison": rows,
        "best_run_id": best_run_id,
        "primary_metric": primary_metric,
        "deltas": deltas,
    }


def get_optimization_history(
    workspace_path: str,
) -> dict[str, Any]:
    """Return all runs sorted by creation time with best-so-far tracking.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        Dict with runs list (each annotated with best_so_far) and overall best.
    """
    runs = workspace.list_runs(workspace_path)

    if not runs:
        return {"runs": [], "best_run_id": None, "best_score": None}

    # Sort by created_at
    sorted_runs = sorted(runs, key=lambda r: r.get("created_at", ""))

    # Determine primary metric
    all_keys: set[str] = set()
    for r in sorted_runs:
        all_keys.update(r.get("metrics", {}).keys())
    primary_metric = _infer_primary_metric(all_keys)

    # Track best-so-far
    best_so_far = float("-inf")
    best_run_id: str | None = None
    annotated: list[dict[str, Any]] = []

    for run in sorted_runs:
        score = run.get("metrics", {}).get(primary_metric)
        if score is not None and score > best_so_far:
            best_so_far = score
            best_run_id = run["run_id"]

        annotated.append({
            **run,
            "best_so_far": best_so_far if best_so_far > float("-inf") else None,
            "is_best": run["run_id"] == best_run_id,
        })

    return {
        "runs": annotated,
        "best_run_id": best_run_id,
        "best_score": best_so_far if best_so_far > float("-inf") else None,
        "primary_metric": primary_metric,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _load_run_context(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    """Load model, data, and feature columns for a run.

    Returns:
        Tuple of (model, X, y, feature_columns).
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
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns missing from data: {missing_features}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    model = joblib.load(model_path)
    X = df[feature_columns].values
    y = df[target_column].values

    return model, X, y, feature_columns


def _classification_metrics(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    is_binary = task_type == "binary_classification"
    average = "binary" if is_binary else "macro"

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
    }

    # Per-class F1 scores
    classes = np.unique(y)
    per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0)
    metrics["f1_per_class"] = {
        str(cls): float(score) for cls, score in zip(classes, per_class_f1)
    }

    # ROC AUC (requires probabilities)
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
            metrics["log_loss"] = float(log_loss(y, y_prob))

            if is_binary:
                metrics["roc_auc"] = float(roc_auc_score(y, y_prob[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as exc:
            logger.warning("Could not compute probability-based metrics: %s", exc)

    return metrics


def _regression_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive regression metrics."""
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # MAPE: avoid division by zero
    mask = y != 0
    if mask.any():
        mape = float(np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100)
    else:
        mape = float("inf")

    return {
        "r2": float(r2_score(y, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, y_pred)),
        "mape": mape,
        "explained_variance": float(explained_variance_score(y, y_pred)),
    }


def _builtin_importance(
    model: Any,
    feature_columns: list[str],
) -> dict[str, float]:
    """Extract feature importances from model attributes."""
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw = np.abs(model.coef_).flatten()
        if len(raw) != len(feature_columns):
            # Multiclass: average across classes
            raw = np.abs(model.coef_).mean(axis=0)
    else:
        raise ValueError(
            "Model does not expose feature_importances_ or coef_. "
            "Use method='permutation' instead."
        )

    return {col: float(imp) for col, imp in zip(feature_columns, raw)}


def _permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    task_type: str,
) -> dict[str, float]:
    """Compute permutation-based feature importance."""
    from sklearn.inspection import permutation_importance

    is_classification = task_type in ("binary_classification", "multiclass_classification")
    scoring = "accuracy" if is_classification else "r2"

    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, scoring=scoring,
    )
    return {
        col: float(imp)
        for col, imp in zip(feature_columns, result.importances_mean)
    }


def _numeric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract only numeric values from a metrics dict for workspace storage."""
    return {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _infer_primary_metric(metric_keys: set[str]) -> str:
    """Infer the primary scoring metric from available keys."""
    preference = ["accuracy", "r2", "f1_macro", "roc_auc", "rmse"]
    for pref in preference:
        if pref in metric_keys:
            return pref
    # Fallback to the first numeric key
    return next(iter(metric_keys), "score")


# ─── Plotting helpers ─────────────────────────────────────────────────────────


def _plot_feature_importance(
    importances: dict[str, float],
    output_path: str,
) -> None:
    """Save a horizontal bar chart of feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    # Limit to top 30 features for readability
    sorted_items = sorted_items[:30]

    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    try:
        plt.style.use(_PLOT_STYLE)
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center", color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_path: str,
) -> None:
    """Save an ROC curve plot."""
    try:
        plt.style.use(_PLOT_STYLE)
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    output_path: str,
) -> None:
    """Save a precision-recall curve plot."""
    try:
        plt.style.use(_PLOT_STYLE)
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#4C72B0", lw=2, label=f"PR curve (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
