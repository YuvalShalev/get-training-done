"""Training loop with cross-validation for ML models."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
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
    memory_dir: str = "",
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
        memory_dir: Path to auto-memory directory. When provided, enables
                    automatic strategy matching on the first run and includes
                    score trajectory in every response.

    Returns:
        Dict with run_id, cv_scores, mean_score, std_score, training_time,
        model_path, run_number, score_trajectory, and optionally
        strategy_recommendation.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If columns are missing or model_type is invalid.
    """
    ws = Path(workspace_path)
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Warn if no train/validation split exists
    from gtd.core.data_splitter import get_split_paths

    split_paths = get_split_paths(workspace_path)
    if not split_paths.get("train_data_path"):
        logger.warning(
            "No train/validation split detected. "
            "Consider calling create_data_split first."
        )

    from gtd.core.data_profiler import load_csv
    df = load_csv(str(source))
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
        "source_data_path": data_path,
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

    result: dict[str, Any] = {
        "run_id": run_id,
        "cv_scores": cv_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "training_time": training_time,
        "model_path": model_path,
    }

    # Persist memory_dir for later auto-discovery by export_model
    if memory_dir:
        _store_memory_dir(str(ws), memory_dir)

    # === Side effects: automatic data collection ===

    # A. Auto-append to structured run log
    run_entry = {
        "run_id": run_id,
        "model_type": model_type,
        "mean_score": mean_score,
        "std_score": std_score,
        "hyperparameters": hyperparameters,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_run_log(str(ws), run_entry)

    # B. Count runs and include in response
    run_count = _get_run_count(str(ws))
    result["run_number"] = run_count

    # B2. Session wall-clock time
    session_start = _load_session_start(str(ws))
    if session_start is not None:
        result["session_elapsed"] = time.time() - session_start

    # C. On first call: store dataset fingerprint and session start
    if run_count == 1:
        now = time.time()
        _store_session_start(str(ws), now)
        result["session_elapsed"] = time.time() - now
        try:
            from gtd.core import meta_learner

            fingerprint = meta_learner.compute_dataset_fingerprint_from_data(
                data_path, target_column, task_type,
            )
            _store_fingerprint(str(ws), fingerprint)

            # D. Strategy matching — surface past data for the agent
            if memory_dir:
                learnings = meta_learner.load_learnings(memory_dir)
                matches = meta_learner.match_strategies(fingerprint, learnings)
                if matches:
                    result["strategy_recommendation"] = matches[:2]

            # E. Load prior knowledge from past sessions
            if memory_dir:
                prior_knowledge = meta_learner.load_prior_knowledge(memory_dir)
                if prior_knowledge:
                    result["prior_knowledge"] = prior_knowledge
        except Exception as exc:
            logger.warning("Side effect (fingerprint/strategy) failed: %s", exc)

    # E. Include score trajectory in response
    trajectory = _load_run_log(str(ws))
    result["score_trajectory"] = [
        {"run_id": r["run_id"], "model": r["model_type"], "score": r["mean_score"]}
        for r in trajectory
    ]

    return result


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

    from gtd.core.data_profiler import load_csv
    test_df = load_csv(str(test_source))
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


# ─── Run log helpers (side-effect data collection) ────────────────────────────


def _append_run_log(workspace_path: str, entry: dict[str, Any]) -> None:
    """Append a run entry to ``workspace/run_log.jsonl``."""
    log_path = Path(workspace_path) / "run_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _load_run_log(workspace_path: str) -> list[dict[str, Any]]:
    """Load all run entries from ``workspace/run_log.jsonl``."""
    log_path = Path(workspace_path) / "run_log.jsonl"
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _get_run_count(workspace_path: str) -> int:
    """Count runs in the workspace run log."""
    return len(_load_run_log(workspace_path))


def _store_session_start(workspace_path: str, start_time: float) -> None:
    """Persist session start timestamp in workspace."""
    path = Path(workspace_path) / "session_start.txt"
    path.write_text(str(start_time))


def _load_session_start(workspace_path: str) -> float | None:
    """Load session start timestamp from workspace."""
    path = Path(workspace_path) / "session_start.txt"
    if not path.exists():
        return None
    return float(path.read_text().strip())


def _store_fingerprint(workspace_path: str, fingerprint: dict[str, Any]) -> None:
    """Store a dataset fingerprint in workspace metadata."""
    fp_path = Path(workspace_path) / "dataset_fingerprint.json"
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, default=str)


def _load_fingerprint(workspace_path: str) -> dict[str, Any] | None:
    """Load the dataset fingerprint from workspace metadata."""
    fp_path = Path(workspace_path) / "dataset_fingerprint.json"
    if not fp_path.exists():
        return None
    with open(fp_path, encoding="utf-8") as f:
        return json.load(f)


# ─── Memory dir persistence (auto-discovery) ──────────────────────────────


def _store_memory_dir(workspace_path: str, memory_dir: str) -> None:
    """Persist memory_dir in workspace for later use by export_model."""
    md_path = Path(workspace_path) / "memory_dir.txt"
    md_path.write_text(memory_dir)


def _discover_memory_dir(workspace_path: str) -> str:
    """Auto-discover memory_dir from workspace metadata or environment.

    Discovery chain:
    1. Workspace metadata (stored by a previous train_model call)
    2. Environment variable CLAUDE_AUTO_MEMORY_DIR
    3. Empty string (skip learning silently)
    """
    md_path = Path(workspace_path) / "memory_dir.txt"
    if md_path.exists():
        candidate = md_path.read_text().strip()
        if candidate and Path(candidate).is_dir():
            return candidate

    env_dir = os.environ.get("CLAUDE_AUTO_MEMORY_DIR", "")
    if env_dir and Path(env_dir).is_dir():
        return env_dir

    return ""


# ─── Export with auto-learning side effects ───────────────────────────────────


def export_model(
    workspace_path: str,
    run_id: str,
    export_name: str | None = None,
    memory_dir: str = "",
) -> dict[str, Any]:
    """Export a trained model and optionally save learnings.

    When ``memory_dir`` is provided, this function automatically:
    - Extracts the optimization history and strategy sequence
    - Saves enhanced learnings to ``gtd-learnings.md``
    - Updates the strategy library in ``gtd-strategy-library.md``
    - Records session metrics to ``gtd-meta-scores.jsonl``

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to export.
        export_name: Optional custom name for the export directory.
        memory_dir: Path to auto-memory directory for automatic learning.

    Returns:
        Dict with export_path, model_path, metadata_path, and optionally
        learning_saved, composite_score.
    """
    import shutil

    ws = Path(workspace_path)
    run_dir = ws / "runs" / run_id
    model_src = run_dir / "model.joblib"

    if not model_src.exists():
        raise FileNotFoundError(f"Model not found at {model_src}")

    name = export_name or run_id
    export_dir = ws / "exports" / name
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    model_dest = export_dir / "model.joblib"
    shutil.copy2(str(model_src), str(model_dest))

    # Build metadata
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    metadata: dict[str, Any] = {"run_id": run_id, "export_name": name}
    if config_path.exists():
        with open(config_path) as f:
            metadata["config"] = json.load(f)
    if metrics_path.exists():
        with open(metrics_path) as f:
            metadata["metrics"] = json.load(f)

    metadata_dest = export_dir / "metadata.json"
    with open(metadata_dest, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    result: dict[str, Any] = {
        "export_path": str(export_dir),
        "model_path": str(model_dest),
        "metadata_path": str(metadata_dest),
    }

    # === Side effects: automatic learning ===
    memory_dir = memory_dir or _discover_memory_dir(workspace_path)
    if memory_dir:
        try:
            from gtd.core import evaluator, meta_learner

            # A. Extract full optimization history
            history = evaluator.get_optimization_history(workspace_path)

            # B. Load workspace fingerprint (stored by train_model)
            fingerprint = _load_fingerprint(workspace_path) or {}

            # C. Extract strategy sequence
            strategy = meta_learner.extract_strategy_sequence(history)

            # D. Get workspace metadata for data path
            ws_metadata = workspace.get_workspace_metadata(ws)

            # E. Save enhanced learnings → gtd-learnings.md
            meta_learner.save_enhanced_learnings(memory_dir, {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "dataset_description": os.path.basename(
                    ws_metadata.get("data_path", "unknown"),
                ),
                "fingerprint": fingerprint,
                "strategy_sequence": strategy,
                "score_trajectory": (
                    f"{strategy.get('final_score', '?')} "
                    f"over {strategy.get('total_runs', '?')} runs"
                ),
                "best_model": strategy.get("final_model", "unknown"),
                "best_score": history.get("best_score", 0),
                "metric_name": history.get("primary_metric", "unknown"),
                "insight": f"Best via {strategy.get('final_model', '?')} "
                           f"in {strategy.get('runs_to_best', '?')} runs",
                "anti_pattern": "",
                "hp_sweet_spot": "",
            })

            # F. Update strategy library → gtd-strategy-library.md
            meta_learner.update_strategy_library(memory_dir, fingerprint, {
                "proven_path": " → ".join(strategy.get("optimization_path", [])),
                "hp_starting_points": "",
                "avoid": "",
                "sessions_count": 1,
            })

            # G. Record session metrics → gtd-meta-scores.jsonl
            total_runs = strategy.get("total_runs", 0)
            runs_to_best = strategy.get("runs_to_best", total_runs)
            composite = meta_learner.compute_composite_score(
                quality=history.get("best_score", 0),
                runs_to_best=runs_to_best,
                max_runs=max(total_runs, 1),
                tool_calls=0,
                max_tool_calls=100,
            )
            meta_learner.record_session_metrics(memory_dir, {
                "dataset_name": os.path.basename(
                    ws_metadata.get("data_path", "unknown"),
                ),
                "task_type": fingerprint.get("task", "unknown"),
                "final_score": history.get("best_score", 0),
                "metric_name": history.get("primary_metric", "unknown"),
                "total_runs": total_runs,
                "runs_to_best": runs_to_best,
                "best_model": strategy.get("final_model", "unknown"),
                "composite_score": composite,
            })

            result["learning_saved"] = True
            result["composite_score"] = composite

        except Exception as exc:
            logger.warning("Side effect (learning save) failed: %s", exc)
            result["learning_saved"] = False
            result["learning_error"] = str(exc)

    return result
