"""Tests for gtd.core.data_splitter module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gtd.core import workspace
from gtd.core.data_splitter import create_data_split, get_split_paths

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_TARGET = "species"

TITANIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
TITANIC_TARGET = "Survived"

BOSTON_FEATURES = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax"]
BOSTON_TARGET = "medv"


# ---------------------------------------------------------------------------
# TestStratifiedSplit
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    """Tests for strategy='stratified'."""

    def test_preserves_class_distribution(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            target_column=IRIS_TARGET,
            task_type="multiclass_classification",
            strategy="stratified",
            validation_fraction=0.2,
        )

        train_df = pd.read_csv(result["train_data_path"])
        val_df = pd.read_csv(result["validation_data_path"])

        # Class proportions should be similar
        train_dist = train_df[IRIS_TARGET].value_counts(normalize=True).sort_index()
        val_dist = val_df[IRIS_TARGET].value_counts(normalize=True).sort_index()

        for cls in train_dist.index:
            assert abs(train_dist[cls] - val_dist[cls]) < 0.05

    def test_correct_sizes(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            target_column=IRIS_TARGET,
            task_type="multiclass_classification",
            strategy="stratified",
            validation_fraction=0.2,
        )

        total = pd.read_csv(iris_csv).shape[0]
        assert result["train_size"] + result["validation_size"] == total
        assert result["strategy"] == "stratified"

    def test_no_overlap(self, tmp_path: Path, titanic_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(titanic_csv),
            target_column=TITANIC_TARGET,
            task_type="binary_classification",
            strategy="stratified",
        )

        train_df = pd.read_csv(result["train_data_path"])
        val_df = pd.read_csv(result["validation_data_path"])

        assert result["train_size"] + result["validation_size"] == len(train_df) + len(val_df)


# ---------------------------------------------------------------------------
# TestRandomSplit
# ---------------------------------------------------------------------------


class TestRandomSplit:
    """Tests for strategy='random'."""

    def test_correct_sizes(self, tmp_path: Path, boston_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(boston_csv),
            target_column=BOSTON_TARGET,
            task_type="regression",
            strategy="random",
            validation_fraction=0.25,
        )

        total = pd.read_csv(boston_csv).shape[0]
        assert result["train_size"] + result["validation_size"] == total
        assert result["strategy"] == "random"

    def test_no_overlap(self, tmp_path: Path, boston_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(boston_csv),
            target_column=BOSTON_TARGET,
            task_type="regression",
            strategy="random",
        )

        train_df = pd.read_csv(result["train_data_path"])
        val_df = pd.read_csv(result["validation_data_path"])

        assert len(train_df) == result["train_size"]
        assert len(val_df) == result["validation_size"]


# ---------------------------------------------------------------------------
# TestTemporalSplit
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    """Tests for strategy='temporal'."""

    def test_validation_after_training(self, tmp_path: Path) -> None:
        # Create a simple dataset with a date column
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [i % 2 for i in range(100)],
        })
        data_path = tmp_path / "temporal_data.csv"
        df.to_csv(data_path, index=False)

        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(data_path),
            target_column="target",
            task_type="binary_classification",
            strategy="temporal",
            temporal_column="date",
            validation_fraction=0.2,
        )

        train_df = pd.read_csv(result["train_data_path"])
        val_df = pd.read_csv(result["validation_data_path"])

        # All validation dates should be after all training dates
        assert train_df["date"].max() <= val_df["date"].min()
        assert result["strategy"] == "temporal"

    def test_correct_sizes(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50, freq="D"),
            "feature": range(50),
            "target": [0] * 25 + [1] * 25,
        })
        data_path = tmp_path / "temporal_data.csv"
        df.to_csv(data_path, index=False)

        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(data_path),
            target_column="target",
            task_type="binary_classification",
            strategy="temporal",
            temporal_column="date",
            validation_fraction=0.2,
        )

        assert result["train_size"] + result["validation_size"] == 50


# ---------------------------------------------------------------------------
# TestGroupSplit
# ---------------------------------------------------------------------------


class TestGroupSplit:
    """Tests for strategy='group'."""

    def test_no_group_overlap(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "group_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10,
            "feature": range(100),
            "target": [i % 2 for i in range(100)],
        })
        data_path = tmp_path / "group_data.csv"
        df.to_csv(data_path, index=False)

        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(data_path),
            target_column="target",
            task_type="binary_classification",
            strategy="group",
            group_column="group_id",
            validation_fraction=0.3,
        )

        train_df = pd.read_csv(result["train_data_path"])
        val_df = pd.read_csv(result["validation_data_path"])

        train_groups = set(train_df["group_id"].unique())
        val_groups = set(val_df["group_id"].unique())

        assert train_groups.isdisjoint(val_groups), "Groups must not overlap"
        assert result["strategy"] == "group"


# ---------------------------------------------------------------------------
# TestGetSplitPaths
# ---------------------------------------------------------------------------


class TestGetSplitPaths:
    """Tests for get_split_paths."""

    def test_returns_none_when_no_split(self, tmp_path: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        paths = get_split_paths(ws_path)
        assert paths["train_data_path"] is None
        assert paths["validation_data_path"] is None

    def test_returns_paths_after_split(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        create_data_split(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            target_column=IRIS_TARGET,
            task_type="multiclass_classification",
        )

        paths = get_split_paths(ws_path)
        assert paths["train_data_path"] is not None
        assert paths["validation_data_path"] is not None
        assert Path(paths["train_data_path"]).exists()
        assert Path(paths["validation_data_path"]).exists()

    def test_returns_none_for_nonexistent_workspace(self, tmp_path: Path) -> None:
        paths = get_split_paths(str(tmp_path / "nonexistent"))
        assert paths["train_data_path"] is None
        assert paths["validation_data_path"] is None


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_invalid_strategy_raises(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(ValueError, match="Unknown split strategy"):
            create_data_split(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                target_column=IRIS_TARGET,
                task_type="multiclass_classification",
                strategy="invalid_strategy",
            )

    def test_temporal_without_column_raises(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(ValueError, match="temporal_column is required"):
            create_data_split(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                target_column=IRIS_TARGET,
                task_type="multiclass_classification",
                strategy="temporal",
            )

    def test_group_without_column_raises(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(ValueError, match="group_column is required"):
            create_data_split(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                target_column=IRIS_TARGET,
                task_type="multiclass_classification",
                strategy="group",
            )

    def test_missing_data_file_raises(self, tmp_path: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(FileNotFoundError):
            create_data_split(
                workspace_path=ws_path,
                data_path="/nonexistent/path.csv",
                target_column="target",
                task_type="binary_classification",
            )

    def test_missing_target_column_raises(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(ValueError, match="Target column"):
            create_data_split(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                target_column="nonexistent_column",
                task_type="multiclass_classification",
            )

    def test_missing_temporal_column_in_data_raises(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(ValueError, match="Temporal column"):
            create_data_split(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                target_column=IRIS_TARGET,
                task_type="multiclass_classification",
                strategy="temporal",
                temporal_column="nonexistent_date",
            )

    def test_missing_group_column_in_data_raises(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        with pytest.raises(ValueError, match="Group column"):
            create_data_split(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                target_column=IRIS_TARGET,
                task_type="multiclass_classification",
                strategy="group",
                group_column="nonexistent_group",
            )

    def test_workspace_metadata_updated(self, tmp_path: Path, iris_csv: Path) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        create_data_split(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            target_column=IRIS_TARGET,
            task_type="multiclass_classification",
        )

        metadata = workspace.get_workspace_metadata(ws_path)
        assert "train_data_path" in metadata
        assert "validation_data_path" in metadata
        # Default "auto" resolves to "stratified" for classification
        assert metadata["split_strategy"] == "stratified"


# ---------------------------------------------------------------------------
# TestAutoStrategy
# ---------------------------------------------------------------------------


class TestAutoStrategy:
    """Tests for strategy='auto' (default)."""

    def test_auto_resolves_to_stratified_for_classification(
        self, tmp_path: Path, iris_csv: Path,
    ) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            target_column=IRIS_TARGET,
            task_type="multiclass_classification",
            # strategy defaults to "auto"
        )

        assert result["strategy"] == "stratified"

        # Verify class distribution is preserved (stratified behaviour)
        train_df = pd.read_csv(result["train_data_path"])
        val_df = pd.read_csv(result["validation_data_path"])
        train_dist = train_df[IRIS_TARGET].value_counts(normalize=True).sort_index()
        val_dist = val_df[IRIS_TARGET].value_counts(normalize=True).sort_index()
        for cls in train_dist.index:
            assert abs(train_dist[cls] - val_dist[cls]) < 0.05

    def test_auto_resolves_to_random_for_regression(
        self, tmp_path: Path, boston_csv: Path,
    ) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(boston_csv),
            target_column=BOSTON_TARGET,
            task_type="regression",
            # strategy defaults to "auto"
        )

        assert result["strategy"] == "random"

    def test_auto_resolves_to_stratified_for_binary(
        self, tmp_path: Path, titanic_csv: Path,
    ) -> None:
        ws = workspace.create_workspace(tmp_path)
        ws_path = ws["workspace_path"]

        result = create_data_split(
            workspace_path=ws_path,
            data_path=str(titanic_csv),
            target_column=TITANIC_TARGET,
            task_type="binary_classification",
            # strategy defaults to "auto"
        )

        assert result["strategy"] == "stratified"
