"""Ensemble strategies: stacking, hill climbing, and seed ensembles."""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from gtd.core import model_registry, workspace
from gtd.core.data_profiler import load_csv
from gtd.core.trainer import _score_model

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_run_id(run_id: str) -> None:
    """Validate that a run ID contains only safe characters.

    Args:
        run_id: The run ID to validate.

    Raises:
        ValueError: If the run ID contains unsafe characters.
    """
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r}")


def train_stacking_ensemble(
    workspace_path: str,
    data_path: str,
    base_model_configs: list[dict],
    meta_learner_type: str,
    meta_learner_params: dict,
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a stacking ensemble with out-of-fold predictions as meta-features.

    Args:
        workspace_path: Path to the workspace directory.
        data_path: Path to the training CSV file.
        base_model_configs: List of dicts with "model_type" and "hyperparameters".
        meta_learner_type: Model name for the meta-learner (e.g. "logistic_regression").
        meta_learner_params: Hyperparameters for the meta-learner.
        feature_columns: List of feature column names.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with run_id, mean_score, std_score, component_scores, training_time.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If base_model_configs is empty or columns are missing.
    """
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not base_model_configs:
        raise ValueError("base_model_configs must contain at least one model config")

    for idx, config in enumerate(base_model_configs):
        if "model_type" not in config:
            raise ValueError(
                f"base_model_configs[{idx}] missing required key 'model_type'"
            )

    ws = Path(workspace_path)
    df = load_csv(str(source))
    _validate_columns(df, feature_columns, target_column)

    X = df[feature_columns].values
    y_raw = df[target_column].values
    n_samples = X.shape[0]
    n_models = len(base_model_configs)

    is_classification = task_type in ("binary_classification", "multiclass_classification")
    metric_name = "accuracy" if is_classification else "r2"

    # Encode non-numeric targets for stacking meta-features
    if is_classification and not pd.api.types.is_numeric_dtype(y_raw):
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw

    if is_classification:
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state,
        )
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    start = time.time()

    # Step 1: Generate OOF predictions for each base model
    oof_predictions = np.zeros((n_samples, n_models))
    component_scores: list[dict[str, Any]] = []

    for model_idx, config in enumerate(base_model_configs):
        model_type = config["model_type"]
        hyperparams = config.get("hyperparameters", {})
        fold_scores: list[float] = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = model_registry.instantiate_model(
                model_type, task_type, hyperparams, random_state,
            )
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_val)
            oof_predictions[val_idx, model_idx] = preds
            fold_scores.append(_score_model(fold_model, X_val, y_val, metric_name))

        component_scores.append({
            "model_type": model_type,
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
        })

    # Step 2: Train meta-learner on OOF predictions
    meta_model = model_registry.instantiate_model(
        meta_learner_type, task_type, meta_learner_params, random_state,
    )
    meta_model.fit(oof_predictions, y)

    # Step 3: Evaluate stacking via CV on the meta-features
    meta_cv_scores: list[float] = []
    for train_idx, val_idx in cv.split(oof_predictions, y):
        oof_train = oof_predictions[train_idx]
        oof_val = oof_predictions[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        meta_fold = model_registry.instantiate_model(
            meta_learner_type, task_type, meta_learner_params, random_state,
        )
        meta_fold.fit(oof_train, y_train)
        meta_cv_scores.append(_score_model(meta_fold, oof_val, y_val, metric_name))

    # Step 4: Retrain all base models on full data
    base_models: list[Any] = []
    for config in base_model_configs:
        model_type = config["model_type"]
        hyperparams = config.get("hyperparameters", {})
        full_model = model_registry.instantiate_model(
            model_type, task_type, hyperparams, random_state,
        )
        full_model.fit(X, y)
        base_models.append(full_model)

    training_time = time.time() - start
    mean_score = float(np.mean(meta_cv_scores))
    std_score = float(np.std(meta_cv_scores))

    # Step 5: Save artifacts
    run_id = _generate_ensemble_run_id(ws, "stacking")
    run_dir = workspace.get_run_dir(ws, run_id)

    for idx, bm in enumerate(base_models):
        joblib.dump(bm, str(run_dir / f"base_model_{idx}.joblib"))
    joblib.dump(meta_model, str(run_dir / "meta_model.joblib"))

    config_artifact = {
        "ensemble_strategy": "stacking",
        "base_model_configs": base_model_configs,
        "meta_learner_type": meta_learner_type,
        "meta_learner_params": meta_learner_params,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "data_path": data_path,
    }
    workspace.save_run_artifact(ws, run_id, "config.json", config_artifact)

    metrics_artifact = {
        "mean_score": mean_score,
        "std_score": std_score,
        "cv_scores": meta_cv_scores,
        "component_scores": component_scores,
        "training_time": training_time,
        "metric_name": metric_name,
    }
    workspace.save_run_artifact(ws, run_id, "metrics.json", metrics_artifact)

    workspace.register_run(
        ws,
        run_id,
        f"stacking({'+'.join(c['model_type'] for c in base_model_configs)})",
        {"meta_learner": meta_learner_type},
        feature_columns,
        metrics={metric_name: mean_score, "std": std_score},
    )

    return {
        "run_id": run_id,
        "mean_score": mean_score,
        "std_score": std_score,
        "component_scores": component_scores,
        "training_time": training_time,
    }


def hill_climbing_ensemble(
    workspace_path: str,
    run_ids: list[str],
    data_path: str,
    target_column: str,
    task_type: str,
    max_ensemble_size: int = 5,
) -> dict[str, Any]:
    """Build an ensemble by greedily adding models that improve score.

    Args:
        workspace_path: Path to the workspace directory.
        run_ids: List of run IDs whose models to consider.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.
        max_ensemble_size: Maximum number of models in the ensemble.

    Returns:
        Dict with selected_models, weights, ensemble_score, individual_scores.

    Raises:
        FileNotFoundError: If data_path does not exist or a model is missing.
        ValueError: If run_ids is empty.
    """
    if not run_ids:
        raise ValueError("run_ids must contain at least one run ID")

    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ws = Path(workspace_path)
    df = load_csv(str(source))

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    is_classification = task_type in ("binary_classification", "multiclass_classification")
    metric_name = "accuracy" if is_classification else "r2"

    # Load predictions from each run
    predictions: dict[str, np.ndarray] = {}
    individual_scores: dict[str, float] = {}

    for rid in run_ids:
        _validate_run_id(rid)
        run_dir = ws / "runs" / rid
        model_path = run_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for run {rid}: {model_path}")

        config = workspace.load_run_artifact(ws, rid, "config.json")
        feature_columns = config["feature_columns"]
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature columns missing from data for run {rid}: {missing}"
            )

        model = joblib.load(model_path)
        X = df[feature_columns].values
        y = df[target_column].values

        individual_scores[rid] = float(
            _score_model(model, X, y, metric_name),
        )
        predictions[rid] = model.predict(X)

    y = df[target_column].values

    # Find best single model
    best_rid = max(individual_scores, key=lambda k: individual_scores[k])
    selected: list[str] = [best_rid]
    best_score = individual_scores[best_rid]

    # Greedy hill climbing — for classification, we use majority vote
    remaining = [r for r in run_ids if r != best_rid]

    while len(selected) < max_ensemble_size and remaining:
        best_candidate = None
        best_candidate_score = best_score

        for candidate in remaining:
            trial = [*selected, candidate]

            if is_classification:
                # Majority vote: take the most common prediction per sample
                all_preds = np.array([predictions[r] for r in trial])
                trial_final = np.array([
                    Counter(all_preds[:, i]).most_common(1)[0][0]
                    for i in range(all_preds.shape[1])
                ])
                score = float(accuracy_score(y, trial_final))
            else:
                trial_preds = np.mean(
                    [predictions[r].astype(float) for r in trial], axis=0,
                )
                score = float(r2_score(y, trial_preds))

            if score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = score

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score

    # Compute final weights (uniform for simplicity)
    n_selected = len(selected)
    weights = {rid: 1.0 / n_selected for rid in selected}

    return {
        "selected_models": selected,
        "weights": weights,
        "ensemble_score": best_score,
        "individual_scores": individual_scores,
    }


def train_seed_ensemble(
    workspace_path: str,
    data_path: str,
    model_type: str,
    hyperparameters: dict,
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    n_seeds: int = 5,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train the same model with different random seeds and average predictions.

    Args:
        workspace_path: Path to the workspace directory.
        data_path: Path to the training CSV file.
        model_type: Model name from the registry.
        hyperparameters: Hyperparameters for the model.
        feature_columns: List of feature column names.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.
        n_seeds: Number of random seeds to use.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with run_id, mean_score, individual_scores, training_time.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If columns are missing.
    """
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")

    ws = Path(workspace_path)
    df = load_csv(str(source))
    _validate_columns(df, feature_columns, target_column)

    X = df[feature_columns].values
    y = df[target_column].values

    is_classification = task_type in ("binary_classification", "multiclass_classification")
    metric_name = "accuracy" if is_classification else "r2"

    start = time.time()
    individual_scores: list[float] = []
    seed_models: list[Any] = []

    for seed_idx in range(n_seeds):
        seed = seed_idx + 1

        if is_classification:
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=seed,
            )
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        fold_scores: list[float] = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = model_registry.instantiate_model(
                model_type, task_type, hyperparameters, seed,
            )
            fold_model.fit(X_train, y_train)
            fold_scores.append(
                _score_model(fold_model, X_val, y_val, metric_name),
            )

        individual_scores.append(float(np.mean(fold_scores)))

        # Train final model on full data for this seed
        full_model = model_registry.instantiate_model(
            model_type, task_type, hyperparameters, seed,
        )
        full_model.fit(X, y)
        seed_models.append(full_model)

    training_time = time.time() - start
    mean_score = float(np.mean(individual_scores))

    # Save artifacts
    run_id = _generate_ensemble_run_id(ws, "seed")
    run_dir = workspace.get_run_dir(ws, run_id)

    for idx, sm in enumerate(seed_models):
        joblib.dump(sm, str(run_dir / f"seed_model_{idx}.joblib"))

    config_artifact = {
        "ensemble_strategy": "seed_ensemble",
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "n_seeds": n_seeds,
        "cv_folds": cv_folds,
        "data_path": data_path,
    }
    workspace.save_run_artifact(ws, run_id, "config.json", config_artifact)

    metrics_artifact = {
        "mean_score": mean_score,
        "individual_scores": individual_scores,
        "training_time": training_time,
        "metric_name": metric_name,
    }
    workspace.save_run_artifact(ws, run_id, "metrics.json", metrics_artifact)

    workspace.register_run(
        ws,
        run_id,
        f"seed_ensemble({model_type})",
        hyperparameters,
        feature_columns,
        metrics={metric_name: mean_score},
    )

    return {
        "run_id": run_id,
        "mean_score": mean_score,
        "individual_scores": individual_scores,
        "training_time": training_time,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _validate_columns(
    df: Any,
    feature_columns: list[str],
    target_column: str,
) -> None:
    """Validate that all required columns exist in the DataFrame.

    Raises:
        ValueError: If any required columns are missing.
    """
    all_required = [*feature_columns, target_column]
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")


def _generate_ensemble_run_id(workspace_path: Path, strategy: str) -> str:
    """Generate a unique ensemble run ID like 'ens_a1b2c3d4_stacking'.

    Args:
        workspace_path: Path to the workspace directory.
        strategy: Ensemble strategy name.

    Returns:
        A unique run ID with ens_ prefix.
    """
    short_id = uuid.uuid4().hex[:8]
    return f"ens_{short_id}_{strategy}"
