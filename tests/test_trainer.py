"""Tests for bbopt.core.trainer module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bbopt.core import workspace
from bbopt.core.trainer import predict, train_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_TARGET = "species"
IRIS_TASK = "multiclass_classification"

SMALL_RF_PARAMS = {"n_estimators": 10, "max_depth": 3}


@pytest.fixture()
def iris_workspace(tmp_path: Path, iris_csv: Path):
    """Create a workspace and return (workspace_path, iris_csv_path)."""
    ws = workspace.create_workspace(tmp_path)
    return Path(ws["workspace_path"]), iris_csv


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


class TestTrainModel:
    """Tests for train_model."""

    def test_returns_run_id(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        assert "run_id" in result
        assert result["run_id"].startswith("run_001_random_forest")

    def test_returns_cv_scores(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        assert "cv_scores" in result
        assert len(result["cv_scores"]) == 3
        for score in result["cv_scores"]:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_returns_mean_and_std_score(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        assert "mean_score" in result
        assert "std_score" in result
        assert isinstance(result["mean_score"], float)
        assert isinstance(result["std_score"], float)

    def test_returns_model_path(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        assert "model_path" in result
        model_file = Path(result["model_path"])
        assert model_file.exists()
        assert model_file.suffix == ".joblib"

    def test_model_file_exists_after_training(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        run_id = result["run_id"]
        run_dir = ws_path / "runs" / run_id
        assert (run_dir / "model.joblib").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()

    def test_run_registered_in_workspace_metadata(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        runs = workspace.list_runs(ws_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == result["run_id"]

    def test_training_time_positive(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=3,
        )
        assert result["training_time"] > 0

    def test_error_on_missing_data_file(self, iris_workspace) -> None:
        ws_path, _ = iris_workspace
        with pytest.raises(FileNotFoundError):
            train_model(
                workspace_path=str(ws_path),
                data_path="/nonexistent/data.csv",
                model_type="random_forest",
                hyperparameters=SMALL_RF_PARAMS,
                feature_columns=IRIS_FEATURES,
                target_column=IRIS_TARGET,
                task_type=IRIS_TASK,
            )

    def test_error_on_missing_feature_column(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        with pytest.raises(ValueError, match="not found"):
            train_model(
                workspace_path=str(ws_path),
                data_path=str(data_path),
                model_type="random_forest",
                hyperparameters=SMALL_RF_PARAMS,
                feature_columns=["nonexistent_col"],
                target_column=IRIS_TARGET,
                task_type=IRIS_TASK,
            )

    def test_sequential_run_ids(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        r1 = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=2,
        )
        r2 = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=2,
        )
        assert "001" in r1["run_id"]
        assert "002" in r2["run_id"]


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests for predict."""

    @pytest.fixture()
    def trained_run(self, iris_workspace):
        """Train a model and return (ws_path, run_id, data_path)."""
        ws_path, data_path = iris_workspace
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=2,
        )
        return ws_path, result["run_id"], data_path

    def test_predict_returns_predictions(self, trained_run) -> None:
        ws_path, run_id, data_path = trained_run
        result = predict(
            workspace_path=str(ws_path),
            run_id=run_id,
            test_data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert "predictions" in result
        assert len(result["predictions"]) == 30

    def test_predict_returns_probabilities_for_classification(
        self, trained_run
    ) -> None:
        ws_path, run_id, data_path = trained_run
        result = predict(
            workspace_path=str(ws_path),
            run_id=run_id,
            test_data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert result["probabilities"] is not None
        assert len(result["probabilities"]) == 30

    def test_predict_returns_metrics_when_target_provided(self, trained_run) -> None:
        ws_path, run_id, data_path = trained_run
        result = predict(
            workspace_path=str(ws_path),
            run_id=run_id,
            test_data_path=str(data_path),
            target_column=IRIS_TARGET,
        )
        assert result["metrics"] is not None
        assert "accuracy" in result["metrics"]
        assert "f1" in result["metrics"]

    def test_predict_no_metrics_when_target_missing(
        self, trained_run, tmp_path: Path
    ) -> None:
        ws_path, run_id, data_path = trained_run
        # Create a test file without the target column
        df = pd.read_csv(data_path)
        df_no_target = df.drop(columns=[IRIS_TARGET])
        no_target_path = tmp_path / "no_target.csv"
        df_no_target.to_csv(no_target_path, index=False)

        result = predict(
            workspace_path=str(ws_path),
            run_id=run_id,
            test_data_path=str(no_target_path),
            target_column=IRIS_TARGET,
        )
        assert result["metrics"] is None

    def test_predict_error_on_missing_model(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        with pytest.raises(FileNotFoundError, match="Model not found"):
            predict(
                workspace_path=str(ws_path),
                run_id="nonexistent_run",
                test_data_path=str(data_path),
            )
