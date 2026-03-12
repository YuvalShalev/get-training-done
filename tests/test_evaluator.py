"""Tests for gtd.core.evaluator module."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core import workspace
from gtd.core.data_splitter import create_data_split
from gtd.core.evaluator import (
    compare_runs,
    error_analysis,
    evaluate_model,
    get_feature_importance,
    get_optimization_history,
)
from gtd.core.trainer import train_model

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_TARGET = "species"
IRIS_TASK = "multiclass_classification"

SMALL_RF_PARAMS = {"n_estimators": 10, "max_depth": 3}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trained_iris(tmp_path: Path, iris_csv: Path):
    """Train a Random Forest on iris and return (ws_path, run_id, data_path)."""
    ws = workspace.create_workspace(tmp_path)
    ws_path = Path(ws["workspace_path"])
    result = train_model(
        workspace_path=str(ws_path),
        data_path=str(iris_csv),
        model_type="random_forest",
        hyperparameters=SMALL_RF_PARAMS,
        feature_columns=IRIS_FEATURES,
        target_column=IRIS_TARGET,
        task_type=IRIS_TASK,
        cv_folds=2,
    )
    return ws_path, result["run_id"], iris_csv


@pytest.fixture()
def two_runs_iris(tmp_path: Path, iris_csv: Path):
    """Train two Random Forest runs on iris for comparison tests."""
    ws = workspace.create_workspace(tmp_path)
    ws_path = Path(ws["workspace_path"])

    r1 = train_model(
        workspace_path=str(ws_path),
        data_path=str(iris_csv),
        model_type="random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 2},
        feature_columns=IRIS_FEATURES,
        target_column=IRIS_TARGET,
        task_type=IRIS_TASK,
        cv_folds=2,
    )
    r2 = train_model(
        workspace_path=str(ws_path),
        data_path=str(iris_csv),
        model_type="random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 5},
        feature_columns=IRIS_FEATURES,
        target_column=IRIS_TARGET,
        task_type=IRIS_TASK,
        cv_folds=2,
    )
    return ws_path, r1["run_id"], r2["run_id"], iris_csv


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """Tests for evaluate_model."""

    def test_returns_accuracy(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        metrics = evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_returns_f1_macro(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        metrics = evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_returns_precision_and_recall(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        metrics = evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

    def test_returns_confusion_matrix(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        metrics = evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert isinstance(cm, list)
        assert len(cm) == 3  # 3 classes in iris

    def test_returns_f1_per_class(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        metrics = evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "f1_per_class" in metrics
        assert isinstance(metrics["f1_per_class"], dict)

    def test_eval_metrics_saved_as_artifact(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        artifact_path = ws_path / "runs" / run_id / "eval_metrics.json"
        assert artifact_path.exists()

    def test_workspace_metadata_updated_with_metrics(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        run_meta = workspace.get_run_metadata(ws_path, run_id)
        assert run_meta is not None
        assert "accuracy" in run_meta["metrics"]


# ---------------------------------------------------------------------------
# error_analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    """Tests for error_analysis wrapper."""

    def test_returns_result(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = error_analysis(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "error_by_segment" in result

    def test_saves_artifact(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        error_analysis(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        artifact_path = ws_path / "runs" / run_id / "error_analysis.json"
        assert artifact_path.exists()

    def test_uses_top_features_from_importance(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = error_analysis(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        # Segments should reference iris features
        features_in_result = {s["feature"] for s in result["error_by_segment"]}
        assert features_in_result.issubset(set(IRIS_FEATURES))

    def test_returns_overall_error_rate(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = error_analysis(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "overall_error_rate" in result
        assert 0.0 <= result["overall_error_rate"] <= 1.0


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------


class TestGetFeatureImportance:
    """Tests for get_feature_importance."""

    def test_returns_importances_dict(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert "importances" in result
        importances = result["importances"]
        assert isinstance(importances, dict)
        assert len(importances) == len(IRIS_FEATURES)

    def test_importances_keys_match_features(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert set(result["importances"].keys()) == set(IRIS_FEATURES)

    def test_importances_are_non_negative(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        for value in result["importances"].values():
            assert value >= 0.0

    def test_method_field_present(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert result["method"] == "builtin"

    def test_plot_path_exists(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert "plot_path" in result
        assert Path(result["plot_path"]).exists()

    def test_artifact_saved(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        artifact = ws_path / "runs" / run_id / "feature_importance.json"
        assert artifact.exists()


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    """Tests for compare_runs."""

    def test_returns_comparison_list(self, two_runs_iris) -> None:
        ws_path, run_id_1, run_id_2, data_path = two_runs_iris
        # Evaluate both runs so they have comparable metrics
        for rid in [run_id_1, run_id_2]:
            evaluate_model(
                workspace_path=str(ws_path),
                run_id=rid,
                data_path=str(data_path),
                target_column=IRIS_TARGET,
                task_type=IRIS_TASK,
            )

        result = compare_runs(
            workspace_path=str(ws_path),
            run_ids=[run_id_1, run_id_2],
        )
        assert "comparison" in result
        assert len(result["comparison"]) == 2

    def test_identifies_best_run(self, two_runs_iris) -> None:
        ws_path, run_id_1, run_id_2, data_path = two_runs_iris
        for rid in [run_id_1, run_id_2]:
            evaluate_model(
                workspace_path=str(ws_path),
                run_id=rid,
                data_path=str(data_path),
                target_column=IRIS_TARGET,
                task_type=IRIS_TASK,
            )

        result = compare_runs(
            workspace_path=str(ws_path),
            run_ids=[run_id_1, run_id_2],
        )
        assert "best_run_id" in result
        assert result["best_run_id"] in (run_id_1, run_id_2)

    def test_returns_deltas(self, two_runs_iris) -> None:
        ws_path, run_id_1, run_id_2, data_path = two_runs_iris
        for rid in [run_id_1, run_id_2]:
            evaluate_model(
                workspace_path=str(ws_path),
                run_id=rid,
                data_path=str(data_path),
                target_column=IRIS_TARGET,
                task_type=IRIS_TASK,
            )

        result = compare_runs(
            workspace_path=str(ws_path),
            run_ids=[run_id_1, run_id_2],
        )
        assert "deltas" in result
        assert isinstance(result["deltas"], dict)

    def test_error_on_empty_run_ids(self, two_runs_iris) -> None:
        ws_path, _, _, _ = two_runs_iris
        with pytest.raises(ValueError, match="At least one run_id"):
            compare_runs(workspace_path=str(ws_path), run_ids=[])

    def test_single_run_comparison(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        result = compare_runs(
            workspace_path=str(ws_path),
            run_ids=[run_id],
        )
        assert len(result["comparison"]) == 1
        assert result["best_run_id"] == run_id
        assert result["deltas"] == {}


# ---------------------------------------------------------------------------
# get_optimization_history
# ---------------------------------------------------------------------------


class TestGetOptimizationHistory:
    """Tests for get_optimization_history."""

    def test_empty_workspace_returns_empty(self, ws_path: Path) -> None:
        result = get_optimization_history(workspace_path=str(ws_path))
        assert result["runs"] == []
        assert result["best_run_id"] is None
        assert result["best_score"] is None

    def test_single_run_history(self, trained_iris) -> None:
        ws_path, run_id, _ = trained_iris
        result = get_optimization_history(workspace_path=str(ws_path))
        assert len(result["runs"]) == 1
        assert result["best_run_id"] is not None

    def test_multiple_runs_tracked(self, two_runs_iris) -> None:
        ws_path, run_id_1, run_id_2, _ = two_runs_iris
        result = get_optimization_history(workspace_path=str(ws_path))
        assert len(result["runs"]) == 2

    def test_best_so_far_annotation(self, two_runs_iris) -> None:
        ws_path, _, _, _ = two_runs_iris
        result = get_optimization_history(workspace_path=str(ws_path))
        for run in result["runs"]:
            assert "best_so_far" in run
            assert "is_best" in run

    def test_primary_metric_present(self, trained_iris) -> None:
        ws_path, _, _ = trained_iris
        result = get_optimization_history(workspace_path=str(ws_path))
        assert "primary_metric" in result

    def test_exactly_one_run_marked_best(self, two_runs_iris) -> None:
        ws_path, _, _, _ = two_runs_iris
        result = get_optimization_history(workspace_path=str(ws_path))
        best_flags = [r["is_best"] for r in result["runs"]]
        assert best_flags.count(True) == 1


# ---------------------------------------------------------------------------
# Auto-discovery (data_path="")
# ---------------------------------------------------------------------------


class TestAutoDiscovery:
    """Tests for auto-discovery of validation path when data_path is omitted."""

    @pytest.fixture()
    def split_iris(self, tmp_path: Path, iris_csv: Path):
        """Create workspace with split and train a model on the train partition."""
        ws = workspace.create_workspace(tmp_path)
        ws_path = Path(ws["workspace_path"])

        split = create_data_split(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            strategy="stratified",
        )

        result = train_model(
            workspace_path=str(ws_path),
            data_path=split["train_data_path"],
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=2,
        )
        return ws_path, result["run_id"], split

    def test_evaluate_model_auto_discovers_validation(self, split_iris) -> None:
        ws_path, run_id, split = split_iris
        metrics = evaluate_model(
            workspace_path=str(ws_path),
            run_id=run_id,
        )
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_get_feature_importance_auto_discovers(self, split_iris) -> None:
        ws_path, run_id, split = split_iris
        result = get_feature_importance(
            workspace_path=str(ws_path),
            run_id=run_id,
        )
        assert "importances" in result
        assert len(result["importances"]) == len(IRIS_FEATURES)

    def test_error_analysis_auto_discovers(self, split_iris) -> None:
        ws_path, run_id, split = split_iris
        result = error_analysis(
            workspace_path=str(ws_path),
            run_id=run_id,
        )
        assert "error_by_segment" in result
