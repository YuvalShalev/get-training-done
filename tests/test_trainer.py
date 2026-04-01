"""Tests for gtd.core.trainer module."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from gtd.core import workspace
from gtd.core.data_splitter import create_data_split
from gtd.core.trainer import (
    _discover_memory_dir,
    _load_session_start,
    _store_memory_dir,
    _store_session_start,
    export_model,
    predict,
    train_model,
)
from gtd.servers.training_server import (
    get_session_time,
    get_training_progress,
    poll_training_jobs,
    train_model_async,
)

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


# ---------------------------------------------------------------------------
# Memory dir auto-discovery
# ---------------------------------------------------------------------------


class TestMemoryDirDiscovery:
    """Tests for _store_memory_dir and _discover_memory_dir."""

    def test_store_and_discover_round_trip(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()

        _store_memory_dir(str(ws_dir), str(mem_dir))
        assert _discover_memory_dir(str(ws_dir)) == str(mem_dir)

    def test_discover_returns_empty_when_nothing_available(
        self, tmp_path: Path
    ) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        assert _discover_memory_dir(str(ws_dir)) == ""

    def test_discover_ignores_stale_path(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        _store_memory_dir(str(ws_dir), "/nonexistent/dir")
        assert _discover_memory_dir(str(ws_dir)) == ""

    def test_discover_falls_back_to_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        env_mem = tmp_path / "env_memory"
        env_mem.mkdir()

        monkeypatch.setenv("CLAUDE_AUTO_MEMORY_DIR", str(env_mem))
        assert _discover_memory_dir(str(ws_dir)) == str(env_mem)

    def test_workspace_metadata_takes_precedence_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        ws_mem = tmp_path / "ws_memory"
        ws_mem.mkdir()
        env_mem = tmp_path / "env_memory"
        env_mem.mkdir()

        _store_memory_dir(str(ws_dir), str(ws_mem))
        monkeypatch.setenv("CLAUDE_AUTO_MEMORY_DIR", str(env_mem))
        assert _discover_memory_dir(str(ws_dir)) == str(ws_mem)

    def test_train_model_persists_memory_dir(self, iris_workspace) -> None:
        ws_path, data_path = iris_workspace
        mem_dir = ws_path.parent / "memory"
        mem_dir.mkdir()

        train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=2,
            memory_dir=str(mem_dir),
        )

        assert _discover_memory_dir(str(ws_path)) == str(mem_dir)

    def test_export_model_auto_discovers_memory_dir(
        self, iris_workspace
    ) -> None:
        ws_path, data_path = iris_workspace
        mem_dir = ws_path.parent / "memory"
        mem_dir.mkdir()

        # Train with memory_dir (persists it)
        result = train_model(
            workspace_path=str(ws_path),
            data_path=str(data_path),
            model_type="random_forest",
            hyperparameters=SMALL_RF_PARAMS,
            feature_columns=IRIS_FEATURES,
            target_column=IRIS_TARGET,
            task_type=IRIS_TASK,
            cv_folds=2,
            memory_dir=str(mem_dir),
        )

        # Export WITHOUT memory_dir — should auto-discover it
        export_result = export_model(
            workspace_path=str(ws_path),
            run_id=result["run_id"],
        )

        # learning_saved should be True or False (not absent),
        # meaning auto-discovery was attempted
        assert "learning_saved" in export_result


class TestSessionTime:
    """Tests for session start timestamp persistence."""

    def test_store_and_load_session_start(self, tmp_path: Path) -> None:
        start = 1700000000.0
        _store_session_start(str(tmp_path), start)
        loaded = _load_session_start(str(tmp_path))
        assert loaded == start

    def test_load_session_start_missing(self, tmp_path: Path) -> None:
        assert _load_session_start(str(tmp_path)) is None

    def test_get_session_time_tool(self, tmp_path: Path) -> None:
        """Integration test for the MCP get_session_time tool."""
        _store_session_start(str(tmp_path), time.time() - 125)
        raw = get_session_time(str(tmp_path))
        result = json.loads(raw)
        assert result["session_elapsed"] >= 124
        assert "m" in result["formatted"]

    def test_get_session_time_no_session(self, tmp_path: Path) -> None:
        raw = get_session_time(str(tmp_path))
        result = json.loads(raw)
        assert "error" in result


class TestTrainingProgress:
    """Tests for per-fold training progress tracking."""

    def test_get_training_progress_with_data(self, tmp_path: Path) -> None:
        progress = {"model_type": "tabicl", "fold": 2, "total_folds": 5,
                     "fold_score": 0.9123, "elapsed": 42.5}
        (tmp_path / "training_progress_12345.json").write_text(json.dumps(progress))

        raw = get_training_progress(str(tmp_path))
        result = json.loads(raw)
        assert result["model_type"] == "tabicl"
        assert result["fold"] == 2
        assert result["total_folds"] == 5

    def test_get_training_progress_no_training(self, tmp_path: Path) -> None:
        raw = get_training_progress(str(tmp_path))
        result = json.loads(raw)
        assert result["status"] == "no_training_in_progress"


class TestAsyncTraining:
    """Tests for async training tools."""

    def test_train_model_async_returns_job_id(self) -> None:
        raw = train_model_async(
            workspace_path="/tmp/fake",
            data_path="/tmp/fake.csv",
            model_type="xgboost",
            hyperparameters={},
            feature_columns=["a"],
            target_column="y",
            task_type="binary_classification",
        )
        result = json.loads(raw)
        assert "job_id" in result
        assert result["status"] == "running"
        assert result["model_type"] == "xgboost"

    def test_poll_training_jobs_returns_list(self) -> None:
        raw = poll_training_jobs(workspace_path="/tmp/fake")
        result = json.loads(raw)
        assert "jobs" in result
        assert isinstance(result["jobs"], list)

    def test_async_job_completes(self, iris_csv: Path, ws_path: Path) -> None:
        """End-to-end: async job trains and completes."""
        split = create_data_split(
            str(ws_path), str(iris_csv), "species", "multiclass_classification",
        )
        train_path = split["train_data_path"]
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        raw = train_model_async(
            workspace_path=str(ws_path),
            data_path=train_path,
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            feature_columns=features,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=2,
        )
        job = json.loads(raw)
        job_id = job["job_id"]

        # Poll until done (max 30s)
        for _ in range(30):
            poll_raw = poll_training_jobs(workspace_path=str(ws_path))
            poll = json.loads(poll_raw)
            matching = [j for j in poll["jobs"] if j["job_id"] == job_id]
            if matching and matching[0]["status"] == "completed":
                assert "result" in matching[0]
                assert matching[0]["result"]["mean_score"] > 0
                return
            time.sleep(1)
        pytest.fail("Async training job did not complete within 30 seconds")
