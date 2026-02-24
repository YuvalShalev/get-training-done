"""MCP server exposing training, evaluation, and feature engineering tools."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from bbopt.core import evaluator, feature_engine, model_registry, trainer, workspace

mcp = FastMCP("bbopt-training")


# ─── Training ─────────────────────────────────────────────────────────────────


@mcp.tool()
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
) -> str:
    """Train a model with cross-validation and save to the workspace.

    Args:
        workspace_path: Path to the workspace directory.
        data_path: Path to the training CSV file.
        model_type: Model name from the registry (e.g. 'xgboost', 'lightgbm').
        hyperparameters: Hyperparameter overrides for the model.
        feature_columns: List of feature column names.
        target_column: Name of the target column.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.
        cv_folds: Number of cross-validation folds (default 5).
        random_state: Random seed (default 42).

    Returns:
        JSON string with run_id, cv_scores, mean_score, std_score, training_time, model_path.
    """
    try:
        result = trainer.train_model(
            workspace_path=workspace_path,
            data_path=data_path,
            model_type=model_type,
            hyperparameters=hyperparameters,
            feature_columns=feature_columns,
            target_column=target_column,
            task_type=task_type,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def predict(
    workspace_path: str,
    run_id: str,
    test_data_path: str,
    target_column: str | None = None,
) -> str:
    """Generate predictions using a previously trained model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        test_data_path: Path to the test CSV file.
        target_column: Optional target column to compute metrics against.

    Returns:
        JSON string with predictions, probabilities, and metrics.
    """
    try:
        result = trainer.predict(
            workspace_path=workspace_path,
            run_id=run_id,
            test_data_path=test_data_path,
            target_column=target_column,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Evaluation ───────────────────────────────────────────────────────────────


@mcp.tool()
def evaluate_model(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    task_type: str,
) -> str:
    """Run full evaluation metrics for a trained model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.

    Returns:
        JSON string with full metrics (accuracy, f1, confusion_matrix, r2, rmse, etc.).
    """
    try:
        result = evaluator.evaluate_model(
            workspace_path=workspace_path,
            run_id=run_id,
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_feature_importance(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    method: str = "builtin",
) -> str:
    """Compute feature importance for a trained model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.
        method: 'builtin' or 'permutation'.

    Returns:
        JSON string with method, importances dict, and plot_path.
    """
    try:
        result = evaluator.get_feature_importance(
            workspace_path=workspace_path,
            run_id=run_id,
            data_path=data_path,
            target_column=target_column,
            method=method,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_roc_curve(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
) -> str:
    """Compute and plot the ROC curve for a binary classification model.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.

    Returns:
        JSON string with fpr, tpr, auc, and plot_path.
    """
    try:
        result = evaluator.get_roc_curve(
            workspace_path=workspace_path,
            run_id=run_id,
            data_path=data_path,
            target_column=target_column,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_pr_curve(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
) -> str:
    """Compute and plot the precision-recall curve.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the evaluation CSV file.
        target_column: Name of the target column.

    Returns:
        JSON string with precision, recall, average precision, and plot_path.
    """
    try:
        result = evaluator.get_pr_curve(
            workspace_path=workspace_path,
            run_id=run_id,
            data_path=data_path,
            target_column=target_column,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def compare_runs(
    workspace_path: str,
    run_ids: list[str],
) -> str:
    """Compare metrics across multiple training runs.

    Args:
        workspace_path: Path to the workspace directory.
        run_ids: List of run IDs to compare.

    Returns:
        JSON string with comparison table, best_run_id, and metric deltas.
    """
    try:
        result = evaluator.compare_runs(
            workspace_path=workspace_path,
            run_ids=run_ids,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_optimization_history(
    workspace_path: str,
) -> str:
    """Return all runs sorted by creation time with best-so-far tracking.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        JSON string with runs list, best_run_id, best_score, and primary_metric.
    """
    try:
        result = evaluator.get_optimization_history(
            workspace_path=workspace_path,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Model Registry ──────────────────────────────────────────────────────────


@mcp.tool()
def list_available_models(
    task_type: str | None = None,
) -> str:
    """List all available models, optionally filtered by task type.

    Args:
        task_type: Optional filter: 'binary_classification',
                   'multiclass_classification', or 'regression'.

    Returns:
        JSON string with list of model info dicts.
    """
    try:
        result = model_registry.list_available_models(task_type=task_type)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Feature Engineering ─────────────────────────────────────────────────────


@mcp.tool()
def engineer_features(
    data_path: str,
    operations: list[dict[str, Any]],
    output_path: str,
) -> str:
    """Apply feature engineering operations to a dataset.

    Args:
        data_path: Path to input CSV file.
        operations: List of operation dicts with 'type' and params.
                    Supported types: one_hot_encode, label_encode, impute_numeric,
                    impute_categorical, standard_scale, log_transform,
                    drop_columns, create_interaction.
        output_path: Path to save transformed CSV.

    Returns:
        JSON string with new_shape, new_columns, and operations_applied.
    """
    try:
        result = feature_engine.engineer_features(
            data_path=data_path,
            operations=operations,
            output_path=output_path,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Export ───────────────────────────────────────────────────────────────────


@mcp.tool()
def export_model(
    workspace_path: str,
    run_id: str,
    export_name: str | None = None,
) -> str:
    """Export a trained model to the workspace exports directory.

    Copies model.joblib and a metadata JSON into exports/.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to export.
        export_name: Optional custom name for the export directory.
                     Defaults to the run_id.

    Returns:
        JSON string with export_path, model_path, and metadata_path.
    """
    try:
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
            import json as _json
            with open(config_path) as f:
                metadata["config"] = _json.load(f)
        if metrics_path.exists():
            import json as _json
            with open(metrics_path) as f:
                metadata["metrics"] = _json.load(f)

        metadata_dest = export_dir / "metadata.json"
        with open(metadata_dest, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        result = {
            "export_path": str(export_dir),
            "model_path": str(model_dest),
            "metadata_path": str(metadata_dest),
        }
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
