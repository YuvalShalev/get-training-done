"""Tests for gtd.core.run_analyzer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core import workspace
from gtd.core.data_splitter import create_data_split
from gtd.core.run_analyzer import analyze_errors, identify_segments
from gtd.core.run_analyzer import test_significance as run_test_significance
from gtd.core.trainer import train_model

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_TARGET = "species"
IRIS_TASK = "multiclass_classification"

BOSTON_FEATURES = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax"]
BOSTON_TARGET = "medv"
BOSTON_TASK = "regression"

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
def trained_boston(tmp_path: Path, boston_csv: Path):
    """Train a Random Forest on boston and return (ws_path, run_id, data_path)."""
    ws = workspace.create_workspace(tmp_path)
    ws_path = Path(ws["workspace_path"])
    result = train_model(
        workspace_path=str(ws_path),
        data_path=str(boston_csv),
        model_type="random_forest",
        hyperparameters=SMALL_RF_PARAMS,
        feature_columns=BOSTON_FEATURES,
        target_column=BOSTON_TARGET,
        task_type=BOSTON_TASK,
        cv_folds=2,
    )
    return ws_path, result["run_id"], boston_csv


# ---------------------------------------------------------------------------
# test_significance
# ---------------------------------------------------------------------------


class TestTestSignificance:
    """Tests for test_significance (pure computation, no I/O)."""

    def test_same_scores_not_significant(self) -> None:
        scores = [0.8, 0.82, 0.81, 0.83, 0.79]
        result = run_test_significance(scores, scores)
        assert result["is_significant"] is False
        assert result["mean_diff"] == 0.0

    def test_different_scores_significant(self) -> None:
        a = [0.80, 0.82, 0.81, 0.83, 0.79]
        b = [0.90, 0.92, 0.91, 0.93, 0.89]
        result = run_test_significance(a, b)
        assert result["is_significant"] is True
        assert result["mean_diff"] > 0

    def test_required_fields_present(self) -> None:
        a = [0.8, 0.82, 0.81, 0.83, 0.79]
        b = [0.85, 0.87, 0.86, 0.88, 0.84]
        result = run_test_significance(a, b)
        required = {"test_name", "p_value", "is_significant", "mean_diff",
                     "ci_lower", "ci_upper", "recommendation"}
        assert required.issubset(result.keys())

    def test_identical_scores_edge_case(self) -> None:
        scores = [0.85, 0.85, 0.85, 0.85, 0.85]
        result = run_test_significance(scores, scores)
        assert result["is_significant"] is False

    def test_recommendation_for_better_b(self) -> None:
        a = [0.70, 0.72, 0.71, 0.73, 0.69]
        b = [0.90, 0.92, 0.91, 0.93, 0.89]
        result = run_test_significance(a, b)
        assert "B" in result["recommendation"] or "better" in result["recommendation"].lower()

    def test_unequal_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            run_test_significance([0.8, 0.82], [0.85, 0.87, 0.86])

    def test_p_value_between_0_and_1(self) -> None:
        a = [0.80, 0.82, 0.81, 0.83, 0.79]
        b = [0.85, 0.87, 0.86, 0.88, 0.84]
        result = run_test_significance(a, b)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_ci_bounds_order(self) -> None:
        a = [0.80, 0.82, 0.81, 0.83, 0.79]
        b = [0.85, 0.87, 0.86, 0.88, 0.84]
        result = run_test_significance(a, b)
        assert result["ci_lower"] <= result["ci_upper"]


# ---------------------------------------------------------------------------
# analyze_errors
# ---------------------------------------------------------------------------


class TestAnalyzeErrors:
    """Tests for analyze_errors."""

    def test_returns_segments_classification(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "error_by_segment" in result
        assert isinstance(result["error_by_segment"], list)

    def test_returns_overall_error_rate(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "overall_error_rate" in result
        assert 0.0 <= result["overall_error_rate"] <= 1.0

    def test_returns_confusion_patterns(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "confusion_patterns" in result
        assert isinstance(result["confusion_patterns"], list)

    def test_returns_confidence_analysis(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "confidence_analysis" in result

    def test_regression_returns_residual_stats(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        assert "residual_stats" in result
        stats = result["residual_stats"]
        assert "mean" in stats
        assert "std" in stats
        assert "skew" in stats
        assert "mae" in stats

    def test_regression_returns_segments(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        assert "error_by_segment" in result
        assert isinstance(result["error_by_segment"], list)

    def test_error_on_missing_model(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = Path(ws["workspace_path"])
        with pytest.raises(FileNotFoundError):
            analyze_errors(
                workspace_path=str(ws_path),
                run_id="nonexistent_run",
                data_path=str(iris_csv),
                target_column=IRIS_TARGET,
                task_type=IRIS_TASK,
            )

    def test_custom_top_features(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            top_features=["petal_length", "petal_width"],
        )
        features_in_result = {s["feature"] for s in result["error_by_segment"]}
        assert features_in_result.issubset({"petal_length", "petal_width"})


# ---------------------------------------------------------------------------
# identify_segments
# ---------------------------------------------------------------------------


class TestIdentifySegments:
    """Tests for identify_segments."""

    def test_returns_high_and_low_segments(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "high_performing" in result
        assert "low_performing" in result
        assert isinstance(result["high_performing"], list)
        assert isinstance(result["low_performing"], list)

    def test_overall_metric_is_reasonable(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "overall_metric" in result
        assert 0.0 <= result["overall_metric"] <= 1.0

    def test_segment_entries_have_required_fields(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            threshold_pct=1.0,  # Low threshold to ensure we get segments
        )
        all_segments = result["high_performing"] + result["low_performing"]
        for seg in all_segments:
            assert "feature" in seg
            assert "segment" in seg
            assert "metric" in seg
            assert "delta_pct" in seg
            assert "sample_count" in seg

    def test_handles_uniform_performance(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        # With a very high threshold, no segments should be flagged
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            threshold_pct=99.0,
        )
        assert len(result["high_performing"]) == 0
        assert len(result["low_performing"]) == 0

    def test_regression_segments(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        assert "overall_metric" in result
        assert "high_performing" in result
        assert "low_performing" in result

    def test_metric_name_present(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "metric_name" in result


# ---------------------------------------------------------------------------
# Auto-discovery (data_path="")
# ---------------------------------------------------------------------------


class TestAutoDiscovery:
    """Tests for auto-discovery of validation path when data_path is omitted."""

    @pytest.fixture()
    def split_iris(self, tmp_path: Path, iris_csv: Path):
        """Create workspace with split and train on the train partition."""
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
        return ws_path, result["run_id"]

    def test_analyze_errors_auto_discovers(self, split_iris) -> None:
        ws_path, run_id = split_iris
        result = analyze_errors(
            workspace_path=str(ws_path),
            run_id=run_id,
        )
        assert "error_by_segment" in result

    def test_identify_segments_auto_discovers(self, split_iris) -> None:
        ws_path, run_id = split_iris
        result = identify_segments(
            workspace_path=str(ws_path),
            run_id=run_id,
        )
        assert "high_performing" in result
        assert "low_performing" in result
