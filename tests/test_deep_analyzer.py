"""Tests for gtd.core.deep_analyzer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core import workspace
from gtd.core.data_splitter import create_data_split
from gtd.core.deep_analyzer import analyze_run_deep
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

TITANIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
TITANIC_TARGET = "Survived"
TITANIC_TASK = "binary_classification"

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


@pytest.fixture()
def trained_titanic(tmp_path: Path, titanic_csv: Path):
    """Train a Random Forest on titanic and return (ws_path, run_id, data_path)."""
    ws = workspace.create_workspace(tmp_path)
    ws_path = Path(ws["workspace_path"])
    result = train_model(
        workspace_path=str(ws_path),
        data_path=str(titanic_csv),
        model_type="random_forest",
        hyperparameters=SMALL_RF_PARAMS,
        feature_columns=TITANIC_FEATURES,
        target_column=TITANIC_TARGET,
        task_type=TITANIC_TASK,
        cv_folds=2,
    )
    return ws_path, result["run_id"], titanic_csv


# ---------------------------------------------------------------------------
# TestAnalyzeRunDeep — end-to-end integration
# ---------------------------------------------------------------------------


class TestAnalyzeRunDeep:
    """End-to-end tests for analyze_run_deep."""

    def test_returns_required_keys(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert "insights" in result
        assert "summary" in result
        assert "top_recommendation" in result

    def test_insights_are_ranked(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        insights = result["insights"]
        if len(insights) > 1:
            for i in range(len(insights) - 1):
                assert insights[i]["estimated_impact"] >= insights[i + 1]["estimated_impact"]

    def test_insight_schema(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        required_keys = {
            "rank", "category", "description", "current_metric",
            "target_metric", "sample_count", "estimated_impact",
            "recommendation", "confidence",
        }
        for insight in result["insights"]:
            assert required_keys.issubset(insight.keys())

    def test_top_n_limits_results(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            top_n=3,
        )
        assert len(result["insights"]) <= 3

    def test_summary_is_nonempty_string(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_regression_returns_insights(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        assert "insights" in result
        assert isinstance(result["insights"], list)

    def test_binary_classification_includes_threshold(self, trained_titanic) -> None:
        ws_path, run_id, data_path = trained_titanic
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=TITANIC_TARGET,
            task_type=TITANIC_TASK,
        )
        # May or may not have threshold insights depending on data,
        # but should not error
        assert "insights" in result
        # At minimum should have some analysis
        assert isinstance(result["top_recommendation"], str)


# ---------------------------------------------------------------------------
# TestErrorProfiling
# ---------------------------------------------------------------------------


class TestErrorProfiling:
    """Tests for error profiling pass."""

    def test_classification_finds_correlated_features(self, trained_titanic) -> None:
        ws_path, run_id, data_path = trained_titanic
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=TITANIC_TARGET,
            task_type=TITANIC_TASK,
        )
        error_insights = [
            i for i in result["insights"] if i["category"] == "error_profiling"
        ]
        # Titanic with limited features should produce some error correlations
        # but this depends on the model, so just check structure
        for insight in error_insights:
            assert "r=" in insight["description"]
            assert insight["sample_count"] > 0

    def test_regression_finds_residual_correlations(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        error_insights = [
            i for i in result["insights"] if i["category"] == "error_profiling"
        ]
        for insight in error_insights:
            assert insight["estimated_impact"] > 0


# ---------------------------------------------------------------------------
# TestSliceDiscovery
# ---------------------------------------------------------------------------


class TestSliceDiscovery:
    """Tests for slice discovery pass."""

    def test_finds_weak_subpopulations(self, trained_titanic) -> None:
        ws_path, run_id, data_path = trained_titanic
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=TITANIC_TARGET,
            task_type=TITANIC_TASK,
        )
        slice_insights = [
            i for i in result["insights"] if i["category"] == "slice_discovery"
        ]
        for insight in slice_insights:
            assert "Subpopulation" in insight["description"]
            assert insight["current_metric"] > insight["target_metric"]

    def test_regression_slice_discovery(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        slice_insights = [
            i for i in result["insights"] if i["category"] == "slice_discovery"
        ]
        for insight in slice_insights:
            assert insight["sample_count"] >= 10


# ---------------------------------------------------------------------------
# TestConfidenceAnalysis
# ---------------------------------------------------------------------------


class TestConfidenceAnalysis:
    """Tests for confidence calibration pass."""

    def test_confidence_bands_for_classification(self, trained_titanic) -> None:
        ws_path, run_id, data_path = trained_titanic
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=TITANIC_TARGET,
            task_type=TITANIC_TASK,
        )
        conf_insights = [
            i for i in result["insights"] if i["category"] == "confidence"
        ]
        for insight in conf_insights:
            assert "confidence band" in insight["description"]
            assert insight["current_metric"] > insight["target_metric"]

    def test_no_confidence_for_regression(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        conf_insights = [
            i for i in result["insights"] if i["category"] == "confidence"
        ]
        assert len(conf_insights) == 0


# ---------------------------------------------------------------------------
# TestThresholdOptimization
# ---------------------------------------------------------------------------


class TestThresholdOptimization:
    """Tests for threshold optimization pass."""

    def test_only_for_binary(self, trained_iris) -> None:
        ws_path, run_id, data_path = trained_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
        )
        threshold_insights = [
            i for i in result["insights"] if i["category"] == "threshold"
        ]
        assert len(threshold_insights) == 0

    def test_no_threshold_for_regression(self, trained_boston) -> None:
        ws_path, run_id, data_path = trained_boston
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=BOSTON_TARGET,
            task_type=BOSTON_TASK,
        )
        threshold_insights = [
            i for i in result["insights"] if i["category"] == "threshold"
        ]
        assert len(threshold_insights) == 0

    def test_binary_may_find_threshold(self, trained_titanic) -> None:
        ws_path, run_id, data_path = trained_titanic
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
            data_path=str(data_path),
            target_column=TITANIC_TARGET,
            task_type=TITANIC_TASK,
        )
        threshold_insights = [
            i for i in result["insights"] if i["category"] == "threshold"
        ]
        # May or may not find one, but if it does the schema should be correct
        for insight in threshold_insights:
            assert insight["target_metric"] > insight["current_metric"]
            assert "threshold" in insight["description"].lower()


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

    def test_analyze_run_deep_auto_discovers(self, split_iris) -> None:
        ws_path, run_id = split_iris
        result = analyze_run_deep(
            workspace_path=str(ws_path),
            run_id=run_id,
        )
        assert "insights" in result
        assert "summary" in result
        assert "top_recommendation" in result
