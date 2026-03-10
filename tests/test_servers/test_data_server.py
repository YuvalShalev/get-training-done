"""Tests for data server tool logic (core data_profiler functions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core import data_profiler


class TestProfileDataset:
    """Tests for the profile_dataset tool logic."""

    def test_returns_shape(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "shape" in result
        assert result["shape"]["rows"] == 30
        assert result["shape"]["columns"] == 12

    def test_returns_dtypes(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "dtypes" in result
        assert "Survived" in result["dtypes"]
        assert "Age" in result["dtypes"]

    def test_returns_feature_types(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "feature_types" in result
        assert "numeric" in result["feature_types"]
        assert "categorical" in result["feature_types"]
        assert "Age" in result["feature_types"]["numeric"]
        assert "Sex" in result["feature_types"]["categorical"]

    def test_returns_distributions(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "distributions" in result
        assert "Age" in result["distributions"]
        assert result["distributions"]["Age"]["type"] == "numeric"
        assert "mean" in result["distributions"]["Age"]

    def test_returns_missing_pct(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "missing_pct" in result
        for col_pct in result["missing_pct"].values():
            assert isinstance(col_pct, float)
            assert 0.0 <= col_pct <= 100.0

    def test_returns_class_balance_for_classification(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert result["class_balance"] is not None
        assert "distribution" in result["class_balance"]
        assert "minority_ratio" in result["class_balance"]
        assert "severity" in result["class_balance"]

    def test_class_balance_is_none_for_regression(self, boston_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(boston_csv), "medv", "regression")
        assert result["class_balance"] is None

    def test_returns_outlier_counts(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "outlier_counts" in result
        for count in result["outlier_counts"].values():
            assert isinstance(count, int)
            assert count >= 0

    def test_returns_cardinality(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "cardinality" in result
        assert result["cardinality"]["Survived"] == 2

    def test_returns_preprocessing_recommendations(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "binary_classification")
        assert "recommended_preprocessing" in result
        assert isinstance(result["recommended_preprocessing"], list)

    def test_auto_task_type_infers_classification(self, titanic_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(titanic_csv), "Survived", "auto")
        assert result["task_type"] == "binary_classification"

    def test_auto_task_type_infers_regression(self, boston_csv: Path) -> None:
        result = data_profiler.profile_dataset(str(boston_csv), "medv", "auto")
        assert result["task_type"] == "regression"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            data_profiler.profile_dataset(str(tmp_path / "nonexistent.csv"), "target", "auto")

    def test_invalid_column_raises(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            data_profiler.profile_dataset(str(titanic_csv), "nonexistent_column", "auto")


class TestGetColumnStats:
    """Tests for the get_column_stats tool logic."""

    def test_numeric_column_stats(self, titanic_csv: Path) -> None:
        result = data_profiler.get_column_stats(str(titanic_csv), "Age")
        assert result["column"] == "Age"
        assert result["is_numeric"] is True
        assert "distribution" in result
        assert "mean" in result["distribution"]
        assert "std" in result["distribution"]
        assert "min" in result["distribution"]
        assert "max" in result["distribution"]
        assert "median" in result["distribution"]

    def test_categorical_column_stats(self, titanic_csv: Path) -> None:
        result = data_profiler.get_column_stats(str(titanic_csv), "Sex")
        assert result["column"] == "Sex"
        assert result["is_numeric"] is False
        assert "distribution" in result
        assert "value_counts" in result["distribution"]
        assert "male" in result["distribution"]["value_counts"] or "female" in result["distribution"]["value_counts"]

    def test_unique_count(self, titanic_csv: Path) -> None:
        result = data_profiler.get_column_stats(str(titanic_csv), "Survived")
        assert result["unique_count"] == 2

    def test_missing_info(self, titanic_csv: Path) -> None:
        result = data_profiler.get_column_stats(str(titanic_csv), "Age")
        assert "missing_count" in result
        assert "missing_pct" in result
        assert result["total_count"] == 30

    def test_invalid_column_raises(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            data_profiler.get_column_stats(str(titanic_csv), "does_not_exist")


class TestDetectDataIssues:
    """Tests for the detect_data_issues tool logic."""

    def test_returns_all_issue_categories(self, titanic_csv: Path) -> None:
        result = data_profiler.detect_data_issues(str(titanic_csv), "Survived")
        expected_keys = {
            "class_imbalance",
            "multicollinearity",
            "high_cardinality_columns",
            "constant_features",
            "near_constant_features",
            "data_leakage_suspects",
            "missing_heavy_columns",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_class_imbalance_structure(self, titanic_csv: Path) -> None:
        result = data_profiler.detect_data_issues(str(titanic_csv), "Survived")
        imbalance = result["class_imbalance"]
        assert "distribution" in imbalance
        assert "severity" in imbalance

    def test_multicollinearity_is_list(self, titanic_csv: Path) -> None:
        result = data_profiler.detect_data_issues(str(titanic_csv), "Survived")
        assert isinstance(result["multicollinearity"], list)

    def test_constant_features_is_list(self, titanic_csv: Path) -> None:
        result = data_profiler.detect_data_issues(str(titanic_csv), "Survived")
        assert isinstance(result["constant_features"], list)

    def test_invalid_target_raises(self, titanic_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            data_profiler.detect_data_issues(str(titanic_csv), "nonexistent")


class TestComputeCorrelations:
    """Tests for the compute_correlations tool logic."""

    def test_returns_feature_target_correlations(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "pearson")
        assert "feature_target_correlations" in result
        assert isinstance(result["feature_target_correlations"], dict)
        assert "sepal_length" in result["feature_target_correlations"]

    def test_returns_top_correlated_pairs(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "pearson")
        assert "top_correlated_pairs" in result
        assert isinstance(result["top_correlated_pairs"], list)
        if result["top_correlated_pairs"]:
            pair = result["top_correlated_pairs"][0]
            assert "feature_1" in pair
            assert "feature_2" in pair
            assert "correlation" in pair

    def test_returns_correlation_matrix(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "pearson")
        assert "correlation_matrix" in result
        matrix = result["correlation_matrix"]
        assert isinstance(matrix, dict)
        assert len(matrix) > 0
        for col_name, col_vals in matrix.items():
            assert isinstance(col_vals, dict)

    def test_spearman_method(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "spearman")
        assert "feature_target_correlations" in result

    def test_include_matrix_false_omits_matrix(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "pearson", include_matrix=False)
        assert result["correlation_matrix"] == {}
        assert len(result["top_correlated_pairs"]) > 0
        assert len(result["feature_target_correlations"]) > 0

    def test_include_matrix_true_returns_full_output(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "pearson", include_matrix=True)
        assert len(result["correlation_matrix"]) > 0
        assert len(result["top_correlated_pairs"]) > 0

    def test_default_includes_matrix(self, iris_csv: Path) -> None:
        result = data_profiler.compute_correlations(str(iris_csv), "petal_length", "pearson")
        assert len(result["correlation_matrix"]) > 0

    def test_invalid_method_raises(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported correlation method"):
            data_profiler.compute_correlations(str(iris_csv), "petal_length", "invalid_method")


class TestPreviewData:
    """Tests for the preview_data tool logic."""

    def test_returns_rows(self, titanic_csv: Path) -> None:
        result = data_profiler.preview_data(str(titanic_csv), n_rows=3)
        assert "rows" in result
        assert len(result["rows"]) == 3
        assert isinstance(result["rows"][0], dict)

    def test_returns_dtypes(self, titanic_csv: Path) -> None:
        result = data_profiler.preview_data(str(titanic_csv))
        assert "dtypes" in result
        assert "Survived" in result["dtypes"]

    def test_returns_shape(self, titanic_csv: Path) -> None:
        result = data_profiler.preview_data(str(titanic_csv))
        assert result["shape"]["rows"] == 30
        assert result["shape"]["columns"] == 12

    def test_returns_column_names(self, titanic_csv: Path) -> None:
        result = data_profiler.preview_data(str(titanic_csv))
        assert "column_names" in result
        assert "PassengerId" in result["column_names"]
        assert "Survived" in result["column_names"]

    def test_default_n_rows_is_5(self, titanic_csv: Path) -> None:
        result = data_profiler.preview_data(str(titanic_csv))
        assert len(result["rows"]) == 5

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            data_profiler.preview_data(str(tmp_path / "nonexistent.csv"))
