"""Tests for the model registry module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gtd.core import registry


@pytest.fixture()
def reg_path(tmp_path: Path) -> str:
    """Return a registry file path in a temp directory."""
    return str(tmp_path / ".gtd-state.json")


def _register_sample(reg_path: str, **overrides) -> dict:
    """Register a model with sensible defaults, overridable via kwargs."""
    defaults = {
        "registry_path": reg_path,
        "workspace_path": "/tmp/ws_001",
        "best_run_id": "xgboost_0001",
        "best_score": 0.85,
        "primary_metric": "accuracy",
        "model_type": "xgboost",
        "task_type": "binary_classification",
        "target_column": "target",
        "data_path": "/tmp/data.csv",
        "export_path": "/tmp/ws_001/exports/xgboost_0001",
        "total_runs": 5,
    }
    defaults.update(overrides)
    return registry.register_model(**defaults)


class TestRegisterModel:
    """Tests for registry.register_model."""

    def test_creates_file_when_missing(self, reg_path: str) -> None:
        assert not Path(reg_path).exists()
        entry = _register_sample(reg_path)
        assert Path(reg_path).exists()
        assert entry["id"] == 1

    def test_auto_increments_id(self, reg_path: str) -> None:
        first = _register_sample(reg_path, best_run_id="run_001")
        second = _register_sample(reg_path, best_run_id="run_002")
        third = _register_sample(reg_path, best_run_id="run_003")
        assert first["id"] == 1
        assert second["id"] == 2
        assert third["id"] == 3

    def test_sets_current_best_to_latest(self, reg_path: str) -> None:
        _register_sample(reg_path, best_run_id="run_001")
        _register_sample(reg_path, best_run_id="run_002")
        data = registry.list_models(reg_path)
        assert data["current_best"] == 2

    def test_stores_all_fields(self, reg_path: str) -> None:
        entry = _register_sample(
            reg_path,
            workspace_path="/ws/path",
            best_run_id="lgbm_0003",
            best_score=0.92,
            primary_metric="f1_macro",
            model_type="lightgbm",
            task_type="multiclass_classification",
            target_column="species",
            data_path="/data/iris.csv",
            export_path="/ws/path/exports/lgbm_0003",
            total_runs=12,
        )
        assert entry["workspace_path"] == "/ws/path"
        assert entry["best_run_id"] == "lgbm_0003"
        assert entry["best_score"] == 0.92
        assert entry["primary_metric"] == "f1_macro"
        assert entry["model_type"] == "lightgbm"
        assert entry["task_type"] == "multiclass_classification"
        assert entry["target_column"] == "species"
        assert entry["data_path"] == "/data/iris.csv"
        assert entry["export_path"] == "/ws/path/exports/lgbm_0003"
        assert entry["total_runs"] == 12
        assert "created_at" in entry

    def test_preserves_existing_entries(self, reg_path: str) -> None:
        _register_sample(reg_path, best_run_id="run_001")
        _register_sample(reg_path, best_run_id="run_002")
        data = registry.list_models(reg_path)
        assert len(data["models"]) == 2
        assert data["models"][0]["best_run_id"] == "run_001"
        assert data["models"][1]["best_run_id"] == "run_002"


class TestListModels:
    """Tests for registry.list_models."""

    def test_raises_when_file_missing(self, reg_path: str) -> None:
        with pytest.raises(FileNotFoundError):
            registry.list_models(reg_path)

    def test_returns_all_models(self, reg_path: str) -> None:
        _register_sample(reg_path, best_run_id="run_001")
        _register_sample(reg_path, best_run_id="run_002")
        data = registry.list_models(reg_path)
        assert data["current_best"] == 2
        assert len(data["models"]) == 2


class TestGetModel:
    """Tests for registry.get_model."""

    def test_returns_correct_model(self, reg_path: str) -> None:
        _register_sample(reg_path, best_run_id="run_001")
        _register_sample(reg_path, best_run_id="run_002", model_type="lightgbm")
        model = registry.get_model(reg_path, 2)
        assert model["best_run_id"] == "run_002"
        assert model["model_type"] == "lightgbm"

    def test_raises_for_missing_id(self, reg_path: str) -> None:
        _register_sample(reg_path)
        with pytest.raises(ValueError, match="Model #99 not found"):
            registry.get_model(reg_path, 99)

    def test_raises_when_file_missing(self, reg_path: str) -> None:
        with pytest.raises(FileNotFoundError):
            registry.get_model(reg_path, 1)


class TestGetCurrentBest:
    """Tests for registry.get_current_best."""

    def test_returns_latest_registered(self, reg_path: str) -> None:
        _register_sample(reg_path, best_run_id="run_001")
        _register_sample(reg_path, best_run_id="run_002", best_score=0.95)
        best = registry.get_current_best(reg_path)
        assert best["id"] == 2
        assert best["best_score"] == 0.95

    def test_raises_when_file_missing(self, reg_path: str) -> None:
        with pytest.raises(FileNotFoundError):
            registry.get_current_best(reg_path)

    def test_raises_when_no_current_best(self, reg_path: str) -> None:
        # Manually create a registry with no current_best
        Path(reg_path).write_text(json.dumps({"current_best": None, "models": []}))
        with pytest.raises(ValueError, match="No current best"):
            registry.get_current_best(reg_path)


class TestRegistryFileFormat:
    """Tests for the JSON file structure."""

    def test_file_is_valid_json(self, reg_path: str) -> None:
        _register_sample(reg_path)
        with open(reg_path) as f:
            data = json.load(f)
        assert "current_best" in data
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_file_is_pretty_printed(self, reg_path: str) -> None:
        _register_sample(reg_path)
        content = Path(reg_path).read_text()
        # Pretty-printed JSON has newlines and indentation
        assert "\n" in content
        assert "  " in content
