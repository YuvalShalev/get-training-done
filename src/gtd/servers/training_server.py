"""MCP server exposing training, evaluation, and feature engineering tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from gtd.core import evaluator, feature_engine, meta_learner, model_registry, registry, run_analyzer, trainer, workspace
from gtd.core.trainer import _discover_memory_dir

mcp = FastMCP("gtd-training")


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
    memory_dir: str = "",
) -> str:
    """Train a model with cross-validation and save to the workspace.

    Automatically logs each run and includes score trajectory in the response.
    On the first call, computes a dataset fingerprint and (if memory_dir is
    provided) surfaces strategy recommendations from past sessions.

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
        memory_dir: Path to auto-memory directory for automatic strategy matching.

    Returns:
        JSON string with run_id, cv_scores, mean_score, std_score, training_time,
        model_path, run_number, score_trajectory, and optionally strategy_recommendation.
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
            memory_dir=memory_dir,
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

        # Auto-load prior knowledge for the agent
        memory_dir = _discover_memory_dir(workspace_path)
        if memory_dir:
            prior = meta_learner.load_prior_knowledge(memory_dir)
            if prior:
                result["prior_knowledge"] = prior

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
def analyze_errors(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    task_type: str,
) -> str:
    """Analyze model errors by feature segment to find where the model fails.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to analyze.
        data_path: Path to the CSV data file.
        target_column: Name of the target column.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.

    Returns:
        JSON string with error_by_segment, confusion_patterns, confidence_analysis
        (classification) or residual_stats (regression).
    """
    try:
        result = run_analyzer.analyze_errors(
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
def identify_segments(
    workspace_path: str,
    run_id: str,
    data_path: str,
    target_column: str,
    task_type: str,
    threshold_pct: float = 5.0,
) -> str:
    """Identify high-performing and low-performing data segments.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run.
        data_path: Path to the CSV data file.
        target_column: Name of the target column.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.
        threshold_pct: Minimum percentage difference to flag a segment (default 5.0).

    Returns:
        JSON string with overall_metric, high_performing, and low_performing segments.
    """
    try:
        result = run_analyzer.identify_segments(
            workspace_path=workspace_path,
            run_id=run_id,
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            threshold_pct=threshold_pct,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def test_significance(
    cv_scores_a: list[float],
    cv_scores_b: list[float],
    alpha: float = 0.05,
) -> str:
    """Test whether two sets of CV scores differ significantly.

    Uses the Wilcoxon signed-rank test for robustness with small sample sizes.

    Args:
        cv_scores_a: CV scores from run A.
        cv_scores_b: CV scores from run B.
        alpha: Significance level (default 0.05).

    Returns:
        JSON string with test_name, p_value, is_significant, mean_diff,
        ci_lower, ci_upper, and recommendation.
    """
    try:
        result = run_analyzer.test_significance(
            cv_scores_a=cv_scores_a,
            cv_scores_b=cv_scores_b,
            alpha=alpha,
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


# ─── Model State Registry ────────────────────────────────────────────────────


@mcp.tool()
def register_model(
    workspace_path: str,
    best_run_id: str,
    best_score: float,
    primary_metric: str,
    model_type: str,
    task_type: str,
    target_column: str,
    data_path: str,
    export_path: str,
    total_runs: int,
    registry_path: str = ".gtd-state.json",
) -> str:
    """Register a trained model in the local registry.

    Appends a new entry to .gtd-state.json with an auto-incremented ID
    and sets it as the current best model.

    Args:
        workspace_path: Path to the training workspace directory.
        best_run_id: ID of the best training run.
        best_score: Best cross-validation score achieved.
        primary_metric: Metric used for optimization (e.g. 'accuracy', 'f1_macro').
        model_type: Model name (e.g. 'lightgbm', 'xgboost').
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.
        target_column: Name of the target column.
        data_path: Path to the original training data.
        export_path: Path to the exported model directory.
        total_runs: Total number of training runs performed.
        registry_path: Path to the registry JSON file (default: .gtd-state.json).

    Returns:
        JSON string with the new registry entry including its assigned ID.
    """
    # Gate: require session synthesis before registration
    flag = Path(workspace_path) / ".session_synthesized"
    if not flag.exists():
        return json.dumps({
            "error": "Session synthesis required before registration. "
            "Call synthesize_session first to save learnings to long-term memory."
        })

    try:
        result = registry.register_model(
            registry_path=registry_path,
            workspace_path=workspace_path,
            best_run_id=best_run_id,
            best_score=best_score,
            primary_metric=primary_metric,
            model_type=model_type,
            task_type=task_type,
            target_column=target_column,
            data_path=data_path,
            export_path=export_path,
            total_runs=total_runs,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def list_registered_models(
    registry_path: str = ".gtd-state.json",
) -> str:
    """List all models in the local registry.

    Args:
        registry_path: Path to the registry JSON file (default: .gtd-state.json).

    Returns:
        JSON string with current_best ID and list of all registered models.
    """
    try:
        result = registry.list_models(registry_path=registry_path)
        return json.dumps(result, default=str)
    except FileNotFoundError:
        return json.dumps({"current_best": None, "models": []})
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


# ─── Self-Improvement (Meta-Learning) ─────────────────────────────────────────


@mcp.tool()
def save_observation(
    workspace_path: str,
    run_number: int,
    score_trajectory: list[dict[str, float]],
    actions_taken: list[str],
    diagnosis: str,
    next_strategy: str,
) -> str:
    """Save a within-run observation during Phase 4 optimization.

    Call this every 3 runs to record the agent's reflection on progress.

    Args:
        workspace_path: Path to the workspace directory.
        run_number: Current optimization run number.
        score_trajectory: List of {run_id: score} dicts so far.
        actions_taken: Description of actions in the last batch of runs.
        diagnosis: What's working, what's failing, and why.
        next_strategy: What to try next based on the reflection.

    Returns:
        JSON string confirming the observation was saved.
    """
    try:
        obs = meta_learner.create_observation(
            run_number=run_number,
            score_trajectory=score_trajectory,
            actions_taken=actions_taken,
            diagnosis=diagnosis,
            next_strategy=next_strategy,
        )
        meta_learner.save_observation(workspace_path, obs)
        return json.dumps({"status": "saved", "run_number": run_number})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def load_observations(workspace_path: str) -> str:
    """Load all observations for the current workspace.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        JSON string with list of observation dicts.
    """
    try:
        observations = meta_learner.load_observations(workspace_path)
        return json.dumps({"observations": observations, "count": len(observations)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def synthesize_session(
    workspace_path: str,
    dataset_name: str,
    task_type: str,
    synthesis: str,
) -> str:
    """Save synthesized session knowledge and archive the observation log.

    MUST be called before register_model. The synthesis should be a concise
    paragraph of general, transferable insights — not a data dump.

    Args:
        workspace_path: Path to the workspace directory.
        dataset_name: Name of the dataset (e.g. the CSV filename).
        task_type: Task type (e.g. 'binary_classification').
        synthesis: A concise paragraph (3-5 sentences) of general knowledge gained.

    Returns:
        JSON string with status and archive filename.
    """
    try:
        memory_dir = _discover_memory_dir(workspace_path)
        if not memory_dir:
            return json.dumps({
                "error": "Cannot discover memory_dir from workspace. "
                "Ensure train_model was called with memory_dir first."
            })

        meta_learner.save_session_synthesis(
            memory_dir, dataset_name, task_type, synthesis,
        )
        archive_name = meta_learner.archive_observation_log(workspace_path)

        # Write flag so register_model knows synthesis happened
        flag = Path(workspace_path) / ".session_synthesized"
        from datetime import datetime, timezone
        flag.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8",
        )

        return json.dumps({
            "status": "saved",
            "archived": archive_name,
            "memory_dir": memory_dir,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_strategy_recommendation(
    dataset_fingerprint: dict[str, Any],
    memory_dir: str,
) -> str:
    """Match current dataset against proven strategies from past sessions.

    Call this in Step 0 after profiling to find strategies that worked
    on similar datasets.

    Args:
        dataset_fingerprint: Dict with size_class, task, feature_mix, issues.
        memory_dir: Path to the auto-memory directory containing gtd-learnings.md.

    Returns:
        JSON string with matched strategies sorted by relevance.
    """
    try:
        learnings = meta_learner.load_learnings(memory_dir)
        matches = meta_learner.match_strategies(dataset_fingerprint, learnings)
        return json.dumps({
            "matches": matches,
            "count": len(matches),
            "has_recommendations": len(matches) > 0,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def record_session_metrics(
    memory_dir: str,
    dataset_name: str,
    task_type: str,
    final_score: float,
    metric_name: str,
    total_runs: int,
    runs_to_best: int,
    best_model: str,
    total_tool_calls: int,
) -> str:
    """Record session performance metrics for prompt evolution tracking.

    Call this as the last step (Step 6) to record the three optimization
    metrics: quality, efficiency, and token economy.

    Args:
        memory_dir: Path to the auto-memory directory.
        dataset_name: Name/description of the dataset.
        task_type: Classification or regression task type.
        final_score: Best achieved score.
        metric_name: Name of the primary metric.
        total_runs: Total training runs in this session.
        runs_to_best: Number of runs to reach the best score.
        best_model: Model type that achieved the best score.
        total_tool_calls: Total MCP tool calls in the session.

    Returns:
        JSON string confirming the metrics were recorded.
    """
    try:
        composite = meta_learner.compute_composite_score(
            quality=final_score,
            runs_to_best=runs_to_best,
            max_runs=max(total_runs, 1),
            tool_calls=total_tool_calls,
            max_tool_calls=max(total_tool_calls * 2, 100),
        )
        metrics = {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "final_score": final_score,
            "metric_name": metric_name,
            "total_runs": total_runs,
            "runs_to_best": runs_to_best,
            "best_model": best_model,
            "total_tool_calls": total_tool_calls,
            "composite_score": composite,
        }
        meta_learner.record_session_metrics(memory_dir, metrics)
        return json.dumps({"status": "recorded", "composite_score": composite, **metrics})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Export ───────────────────────────────────────────────────────────────────


@mcp.tool()
def export_model(
    workspace_path: str,
    run_id: str,
    export_name: str | None = None,
    memory_dir: str = "",
) -> str:
    """Export a trained model to the workspace exports directory.

    Copies model.joblib and a metadata JSON into exports/. When memory_dir
    is provided, automatically saves learnings, updates the strategy library,
    and records session metrics.

    Args:
        workspace_path: Path to the workspace directory.
        run_id: ID of the training run to export.
        export_name: Optional custom name for the export directory.
                     Defaults to the run_id.
        memory_dir: Path to auto-memory directory for automatic learning saves.

    Returns:
        JSON string with export_path, model_path, metadata_path, and optionally
        learning_saved and composite_score.
    """
    try:
        result = trainer.export_model(
            workspace_path=workspace_path,
            run_id=run_id,
            export_name=export_name,
            memory_dir=memory_dir,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    mcp.run()
