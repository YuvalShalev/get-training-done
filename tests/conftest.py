"""Shared test fixtures for gtd tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core import workspace

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def titanic_csv() -> Path:
    """Path to the titanic fixture CSV (binary classification, target=Survived)."""
    path = FIXTURES_DIR / "titanic.csv"
    assert path.exists(), f"Fixture not found: {path}"
    return path


@pytest.fixture()
def iris_csv() -> Path:
    """Path to the iris fixture CSV (multiclass classification, target=species)."""
    path = FIXTURES_DIR / "iris.csv"
    assert path.exists(), f"Fixture not found: {path}"
    return path


@pytest.fixture()
def boston_csv() -> Path:
    """Path to the boston fixture CSV (regression, target=medv)."""
    path = FIXTURES_DIR / "boston.csv"
    assert path.exists(), f"Fixture not found: {path}"
    return path


@pytest.fixture()
def ws(tmp_path: Path) -> dict:
    """Create a fresh workspace in a temporary directory and return its metadata."""
    result = workspace.create_workspace(tmp_path)
    return result


@pytest.fixture()
def ws_path(ws: dict) -> Path:
    """Return the workspace path as a Path object."""
    return Path(ws["workspace_path"])
