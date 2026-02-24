"""Tests for bbopt.core.data_profiler module."""

from __future__ import annotations

from pathlib import Path

import pytest

from bbopt.core.data_profiler import (
    compute_correlations,
    detect_data_issues,
    get_column_stats,
    preview_data,
    profile_dataset,
)


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------


class TestProfileDataset:
    """Tests for profile_dataset."""

    def test_returns_correct_shape(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived")
        assert result["shape"]["rows"] == 30
        assert result["shape"]["columns"] == 12

    def test_returns_dtypes_for_all_columns(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived")
        dtypes = result["dtypes"]
        assert len(dtypes) == 12
        assert "PassengerId" in dtypes
        assert "Survived" in dtypes
        assert "Age" in dtypes

    def test_has_required_top_level_keys(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived")
        required_keys = {
            "shape",
            "dtypes",
            "feature_types",
            "distributions",
            "missing_pct",
            "class_balance",
            "task_type",
            "outlier_counts",
            "cardinality",
            "recommended_preprocessing",
        }
        assert required_keys.issubset(result.keys())

    def test_feature_types_split(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived")
        feature_types = result["feature_types"]
        assert "numeric" in feature_types
        assert "categorical" in feature_types
        assert "Age" in feature_types["numeric"]
        assert "Sex" in feature_types["categorical"]

    def test_class_balance_present_for_classification(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived")
        balance = result["class_balance"]
        assert balance is not None
        assert "distribution" in balance
        assert "num_classes" in balance
        assert balance["num_classes"] == 2

    def test_auto_task_type_detection(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived", task_type="auto")
        assert result["task_type"] == "binary_classification"

    def test_missing_pct_contains_all_columns(self, titanic_csv: Path) -> None:
        result = profile_dataset(str(titanic_csv), target_column="Survived")
        assert len(result["missing_pct"]) == 12

    def test_error_on_invalid_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            profile_dataset("/nonexistent/path.csv", target_column="Survived")

    def test_error_on_missing_target_column(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            profile_dataset(str(titanic_csv), target_column="nonexistent_column")


# ---------------------------------------------------------------------------
# get_column_stats
# ---------------------------------------------------------------------------


class TestGetColumnStats:
    """Tests for get_column_stats."""

    def test_numeric_column_stats(self, titanic_csv: Path) -> None:
        result = get_column_stats(str(titanic_csv), "Age")
        assert result["column"] == "Age"
        assert result["is_numeric"] is True
        assert "distribution" in result
        dist = result["distribution"]
        assert "mean" in dist
        assert "std" in dist
        assert "min" in dist
        assert "max" in dist
        assert "median" in dist

    def test_categorical_column_stats(self, titanic_csv: Path) -> None:
        result = get_column_stats(str(titanic_csv), "Sex")
        assert result["column"] == "Sex"
        assert result["is_numeric"] is False
        dist = result["distribution"]
        assert "value_counts" in dist
        assert "top_value" in dist

    def test_missing_count_for_age(self, titanic_csv: Path) -> None:
        result = get_column_stats(str(titanic_csv), "Age")
        assert result["missing_count"] > 0
        assert result["missing_pct"] > 0

    def test_unique_count(self, titanic_csv: Path) -> None:
        result = get_column_stats(str(titanic_csv), "Survived")
        assert result["unique_count"] == 2

    def test_error_on_missing_column(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_column_stats(str(titanic_csv), "nonexistent_column")


# ---------------------------------------------------------------------------
# detect_data_issues
# ---------------------------------------------------------------------------


class TestDetectDataIssues:
    """Tests for detect_data_issues."""

    def test_returns_all_issue_categories(self, titanic_csv: Path) -> None:
        result = detect_data_issues(str(titanic_csv), target_column="Survived")
        expected_keys = {
            "class_imbalance",
            "multicollinearity",
            "high_cardinality_columns",
            "constant_features",
            "near_constant_features",
            "data_leakage_suspects",
            "missing_heavy_columns",
        }
        assert expected_keys.issubset(result.keys())

    def test_finds_missing_heavy_cabin_column(self, titanic_csv: Path) -> None:
        result = detect_data_issues(str(titanic_csv), target_column="Survived")
        missing_heavy = result["missing_heavy_columns"]
        missing_col_names = [entry["column"] for entry in missing_heavy]
        assert "Cabin" in missing_col_names

    def test_class_imbalance_detected(self, titanic_csv: Path) -> None:
        result = detect_data_issues(str(titanic_csv), target_column="Survived")
        imbalance = result["class_imbalance"]
        assert "distribution" in imbalance
        assert "severity" in imbalance

    def test_error_on_missing_target_column(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            detect_data_issues(str(titanic_csv), target_column="nonexistent")


# ---------------------------------------------------------------------------
# compute_correlations
# ---------------------------------------------------------------------------


class TestComputeCorrelations:
    """Tests for compute_correlations."""

    def test_returns_proper_structure(self, titanic_csv: Path) -> None:
        result = compute_correlations(str(titanic_csv), target_column="Survived")
        assert "feature_target_correlations" in result
        assert "top_correlated_pairs" in result
        assert "correlation_matrix" in result

    def test_feature_target_correlations_are_floats(self, titanic_csv: Path) -> None:
        result = compute_correlations(str(titanic_csv), target_column="Survived")
        for value in result["feature_target_correlations"].values():
            assert isinstance(value, float)

    def test_top_correlated_pairs_structure(self, titanic_csv: Path) -> None:
        result = compute_correlations(str(titanic_csv), target_column="Survived")
        pairs = result["top_correlated_pairs"]
        assert isinstance(pairs, list)
        if pairs:
            pair = pairs[0]
            assert "feature_1" in pair
            assert "feature_2" in pair
            assert "correlation" in pair

    def test_correlation_matrix_is_symmetric(self, titanic_csv: Path) -> None:
        result = compute_correlations(str(titanic_csv), target_column="Survived")
        matrix = result["correlation_matrix"]
        for col_a, row in matrix.items():
            for col_b, val in row.items():
                if val is not None and matrix.get(col_b, {}).get(col_a) is not None:
                    assert abs(val - matrix[col_b][col_a]) < 1e-10

    def test_spearman_method(self, titanic_csv: Path) -> None:
        result = compute_correlations(
            str(titanic_csv), target_column="Survived", method="spearman"
        )
        assert "correlation_matrix" in result

    def test_invalid_method_raises(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported correlation method"):
            compute_correlations(
                str(titanic_csv), target_column="Survived", method="invalid"
            )

    def test_error_on_invalid_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            compute_correlations("/nonexistent.csv", target_column="Survived")


# ---------------------------------------------------------------------------
# preview_data
# ---------------------------------------------------------------------------


class TestPreviewData:
    """Tests for preview_data."""

    def test_returns_correct_number_of_rows(self, titanic_csv: Path) -> None:
        result = preview_data(str(titanic_csv), n_rows=3)
        assert len(result["rows"]) == 3

    def test_default_returns_five_rows(self, titanic_csv: Path) -> None:
        result = preview_data(str(titanic_csv))
        assert len(result["rows"]) == 5

    def test_shape_info(self, titanic_csv: Path) -> None:
        result = preview_data(str(titanic_csv))
        assert result["shape"]["rows"] == 30
        assert result["shape"]["columns"] == 12

    def test_column_names_present(self, titanic_csv: Path) -> None:
        result = preview_data(str(titanic_csv))
        assert "PassengerId" in result["column_names"]
        assert "Survived" in result["column_names"]

    def test_dtypes_present(self, titanic_csv: Path) -> None:
        result = preview_data(str(titanic_csv))
        assert len(result["dtypes"]) == 12

    def test_error_on_invalid_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            preview_data("/nonexistent/file.csv")
