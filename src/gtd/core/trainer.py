"""Training loop with cross-validation for ML models."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from gtd.core import model_registry, workspace

logger = logging.getLogger(__name__)


def train_model(
    workspace_path: str,
    data_path: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a model with cross-validation and save artifacts.

    Args:
        workspace_path: Path to the workspace directory.
        data_path: Path to the training CSV file.
        model_type: Model name from the registry (e.g. 'xgboost').
        hyperparameters: Hyperparameter overrides for the model.
        feature_columns: List of feature column names to use.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with run_id, cv_scores, mean_score, std_score, training_time,
        and model_path.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If columns are missing or model_type is invalid.
    """
    ws = Path(workspace_path)
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(source)
    _validate_columns(df, feature_columns, target_column)

    X = df[feature_columns].values
    y = df[target_column].values

    # Determine scorer
    is_classification = task_type in ("binary_classification", "multiclass_classification")
    metric_name = "accuracy" if is_classification else "r2"

    # Build cross-validation splitter
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    start = time.time()

    # Run cross-validation fold by fold
    cv_scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_model = model_registry.instantiate_model(
            model_type, task_type, hyperparameters, random_state,
        )
        fold_model.fit(X_train, y_train)

        fold_score = _score_model(fold_model, X_val, y_val, metric_name)
        cv_scores.append(fold_score)

    # Train the final model on all data
    final_model = model_registry.instantiate_model(
        model_type, task_type, hyperparameters, random_state,
    )
    final_model.fit(X, y)

    training_time = time.time() - start
    mean_score = float(np.mean(cv_scores))
    std_score = float(np.std(cv_scores))

    # Generate sequential run_id
    run_id = _generate_run_id(ws, model_type)
    run_dir = workspace.get_run_dir(ws, run_id)

    # Save model
    model_path = str(run_dir / "model.joblib")
    joblib.dump(final_model, model_path)

    # Save config.json
    config = {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "data_path": data_path,
    }
    workspace.save_run_artifact(ws, run_id, "config.json", config)

    # Save metrics.json
    metrics = {
        "cv_scores": cv_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "training_time": training_time,
        "metric_name": metric_name,
    }
    workspace.save_run_artifact(ws, run_id, "metrics.json", metrics)

    # Register run in workspace metadata
    workspace.register_run(
        ws,
        run_id,
        model_type,
        hyperparameters,
        feature_columns,
        metrics={metric_name: mean_score, "std": std_score},
    )

    # Update best run if applicable
    _maybe_update_best_run(ws, run_id, mean_score, metric_name, task_type)

    return {
        "run_id": run_id,
        "cv_scores": cv_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "training_time": training_time,
        "model_path": model_path,
    }


def predict(
    workspace_path: str,
    run_id: str,
    test_data_path: str,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Generate predictions using a saved model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        test_data_path: Path to the test CSV file.
        target_column: If present in test data, compute metrics.

    Returns:
        Dict with predictions, probabilities (if classification), and metrics
        (if target_column provided and present in data).

    Raises:
        FileNotFoundError: If model or test data not found.
        ValueError: If feature columns are missing from test data.
    """
    ws = Path(workspace_path)
    run_dir = ws / "runs" / run_id

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    config = workspace.load_run_artifact(ws, run_id, "config.json")
    feature_columns = config["feature_columns"]
    task_type = config["task_type"]

    test_source = Path(test_data_path)
    if not test_source.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    test_df = pd.read_csv(test_source)
    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from test data: {missing}")

    model = joblib.load(model_path)
    X_test = test_df[feature_columns].values
    predictions = model.predict(X_test).tolist()

    # Probabilities for classification
    probabilities: list[Any] | None = None
    is_classification = task_type in ("binary_classification", "multiclass_classification")
    if is_classification and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test).tolist()
        except Exception:
            logger.warning("Could not generate prediction probabilities")

    # Compute metrics if target is available
    metrics: dict[str, float] | None = None
    if target_column and target_column in test_df.columns:
        y_true = test_df[target_column].values
        y_pred = np.array(predictions)
        metrics = _compute_prediction_metrics(y_true, y_pred, task_type)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _validate_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> None:
    """Validate that all required columns exist in the DataFrame."""
    all_required = [*feature_columns, target_column]
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")


def _score_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_name: str,
) -> float:
    """Score a model on validation data."""
    from sklearn.metrics import accuracy_score, r2_score

    y_pred = model.predict(X)
    if metric_name == "accuracy":
        return float(accuracy_score(y, y_pred))
    if metric_name == "r2":
        return float(r2_score(y, y_pred))
    raise ValueError(f"Unsupported metric: {metric_name}")


def _compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    """Compute basic metrics for predictions."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    if task_type in ("binary_classification", "multiclass_classification"):
        average = "binary" if task_type == "binary_classification" else "macro"
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _generate_run_id(workspace_path: Path, model_type: str) -> str:
    """Generate a sequential run ID like 'run_001_xgboost'."""
    existing_runs = workspace.list_runs(workspace_path)
    next_num = len(existing_runs) + 1
    return f"run_{next_num:03d}_{model_type}"


def _maybe_update_best_run(
    workspace_path: Path,
    run_id: str,
    score: float,
    metric_name: str,
    task_type: str,
) -> None:
    """Update the best run in workspace if this run is better."""
    metadata = workspace.get_workspace_metadata(workspace_path)
    current_best = metadata.get("best_score")

    # Higher is better for accuracy and r2
    is_better = current_best is None or score > current_best

    if is_better:
        workspace.update_best_run(workspace_path, run_id, score, metric_name)
