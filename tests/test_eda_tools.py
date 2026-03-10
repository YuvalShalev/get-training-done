"""Tests for new EDA statistical tools in gtd.core.data_profiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core.data_profiler import (
    analyze_missing_patterns,
    analyze_temporal_patterns,
    compute_anova_scores,
    compute_cramers_v,
    compute_dataset_fingerprint,
    compute_mutual_information,
    compute_separability_score,
    compute_vif,
    detect_timestamp_columns,
    test_normality as profiler_test_normality,
)


# ---------------------------------------------------------------------------
# compute_mutual_information
# ---------------------------------------------------------------------------


class TestComputeMutualInformation:
    """Tests for compute_mutual_information."""

    def test_returns_scores_for_all_features(self, iris_csv: Path) -> None:
        result = compute_mutual_information(str(iris_csv), "species")
        assert "scores" in result
        assert "sepal_length" in result["scores"]
        assert "petal_length" in result["scores"]

    def test_max_mi_is_positive_for_iris(self, iris_csv: Path) -> None:
        result = compute_mutual_information(str(iris_csv), "species")
        assert result["max_mi"] > 0

    def test_n_informative_count(self, iris_csv: Path) -> None:
        result = compute_mutual_information(str(iris_csv), "species")
        assert result["n_informative"] > 0

    def test_top_n_limits_output(self, iris_csv: Path) -> None:
        result = compute_mutual_information(str(iris_csv), "species", top_n=2)
        assert len(result["scores"]) == 2

    def test_auto_task_type(self, iris_csv: Path) -> None:
        result = compute_mutual_information(str(iris_csv), "species", task_type="auto")
        assert "scores" in result

    def test_regression_task(self, boston_csv: Path) -> None:
        result = compute_mutual_information(str(boston_csv), "medv", task_type="regression")
        assert result["max_mi"] > 0

    def test_mixed_feature_types(self, titanic_csv: Path) -> None:
        result = compute_mutual_information(str(titanic_csv), "Survived")
        assert "scores" in result
        assert "Sex" in result["scores"]  # categorical feature should be encoded

    def test_error_on_missing_target(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            compute_mutual_information(str(iris_csv), "nonexistent")


# ---------------------------------------------------------------------------
# compute_cramers_v
# ---------------------------------------------------------------------------


class TestComputeCramersV:
    """Tests for compute_cramers_v."""

    def test_returns_associations(self, titanic_csv: Path) -> None:
        result = compute_cramers_v(str(titanic_csv))
        assert "associations" in result
        assert "n_pairs" in result

    def test_with_target_column(self, titanic_csv: Path) -> None:
        result = compute_cramers_v(str(titanic_csv), target_column="Embarked")
        assert "associations" in result

    def test_all_numeric_returns_empty(self, iris_csv: Path) -> None:
        result = compute_cramers_v(str(iris_csv))
        # iris has no categorical pairs (species is the only categorical)
        assert result["n_pairs"] == 0

    def test_max_v_field(self, titanic_csv: Path) -> None:
        result = compute_cramers_v(str(titanic_csv))
        assert "max_v" in result
        assert isinstance(result["max_v"], float)


# ---------------------------------------------------------------------------
# compute_anova_scores
# ---------------------------------------------------------------------------


class TestComputeAnovaScores:
    """Tests for compute_anova_scores."""

    def test_returns_scores_for_numeric_features(self, iris_csv: Path) -> None:
        result = compute_anova_scores(str(iris_csv), "species")
        assert "scores" in result
        assert "sepal_length" in result["scores"]
        assert "f_statistic" in result["scores"]["sepal_length"]
        assert "p_value" in result["scores"]["sepal_length"]

    def test_n_significant_count(self, iris_csv: Path) -> None:
        result = compute_anova_scores(str(iris_csv), "species")
        # iris features should be highly significant for species prediction
        assert result["n_significant"] > 0

    def test_with_binary_target(self, titanic_csv: Path) -> None:
        result = compute_anova_scores(str(titanic_csv), "Survived")
        assert "scores" in result

    def test_error_on_missing_target(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            compute_anova_scores(str(iris_csv), "nonexistent")


# ---------------------------------------------------------------------------
# compute_vif
# ---------------------------------------------------------------------------


class TestComputeVif:
    """Tests for compute_vif."""

    def test_returns_vif_for_numeric_features(self, iris_csv: Path) -> None:
        result = compute_vif(str(iris_csv), "species")
        assert "vif_scores" in result
        assert len(result["vif_scores"]) == 4  # 4 numeric features

    def test_vif_values_are_positive(self, iris_csv: Path) -> None:
        result = compute_vif(str(iris_csv), "species")
        for v in result["vif_scores"].values():
            assert v >= 1.0

    def test_n_high_vif_count(self, iris_csv: Path) -> None:
        result = compute_vif(str(iris_csv), "species")
        assert "n_high_vif" in result
        assert isinstance(result["n_high_vif"], int)

    def test_top_n_limits_features(self, boston_csv: Path) -> None:
        result = compute_vif(str(boston_csv), "medv", top_n=5)
        assert len(result["vif_scores"]) <= 5

    def test_single_feature_returns_empty(self, tmp_path: Path) -> None:
        csv = tmp_path / "single.csv"
        csv.write_text("a,target\n1,0\n2,1\n3,0\n")
        result = compute_vif(str(csv), "target")
        assert result["vif_scores"] == {}


# ---------------------------------------------------------------------------
# detect_timestamp_columns
# ---------------------------------------------------------------------------


class TestDetectTimestampColumns:
    """Tests for detect_timestamp_columns."""

    def test_no_timestamps_in_iris(self, iris_csv: Path) -> None:
        result = detect_timestamp_columns(str(iris_csv))
        assert result["n_detected"] == 0

    def test_detects_date_column(self, tmp_path: Path) -> None:
        csv = tmp_path / "dates.csv"
        csv.write_text(
            "date,value\n2024-01-01,10\n2024-01-02,20\n2024-01-03,30\n"
            "2024-01-04,40\n2024-01-05,50\n"
        )
        result = detect_timestamp_columns(str(csv))
        assert result["n_detected"] == 1
        assert result["timestamp_columns"][0]["column"] == "date"

    def test_returns_sample_values(self, tmp_path: Path) -> None:
        csv = tmp_path / "dates.csv"
        csv.write_text(
            "date,value\n2024-01-01,10\n2024-01-02,20\n2024-01-03,30\n"
        )
        result = detect_timestamp_columns(str(csv))
        assert len(result["timestamp_columns"][0]["sample_values"]) > 0


# ---------------------------------------------------------------------------
# analyze_missing_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeMissingPatterns:
    """Tests for analyze_missing_patterns."""

    def test_no_missing_returns_none_pattern(self, iris_csv: Path) -> None:
        result = analyze_missing_patterns(str(iris_csv))
        assert result["pattern"] == "none"

    def test_titanic_has_missing(self, titanic_csv: Path) -> None:
        result = analyze_missing_patterns(str(titanic_csv))
        assert result["pattern"] in ("MCAR", "MAR", "MNAR")
        assert len(result["columns_with_missing"]) > 0

    def test_missing_pct_for_all_columns(self, titanic_csv: Path) -> None:
        result = analyze_missing_patterns(str(titanic_csv))
        assert "Age" in result["missing_pct"]
        assert "Cabin" in result["missing_pct"]

    def test_returns_correlations(self, titanic_csv: Path) -> None:
        result = analyze_missing_patterns(str(titanic_csv))
        assert isinstance(result["correlations"], list)


# ---------------------------------------------------------------------------
# test_normality
# ---------------------------------------------------------------------------


class TestNormalityCheck:
    """Tests for test_normality."""

    def test_returns_results_for_all_numeric(self, iris_csv: Path) -> None:
        result = profiler_test_normality(str(iris_csv))
        assert "results" in result
        assert "sepal_length" in result["results"]

    def test_each_result_has_is_normal(self, iris_csv: Path) -> None:
        result = profiler_test_normality(str(iris_csv))
        for col_result in result["results"].values():
            assert "is_normal" in col_result

    def test_specific_columns(self, iris_csv: Path) -> None:
        result = profiler_test_normality(str(iris_csv), columns=["sepal_length", "petal_width"])
        assert len(result["results"]) == 2

    def test_n_normal_count(self, iris_csv: Path) -> None:
        result = profiler_test_normality(str(iris_csv))
        assert isinstance(result["n_normal"], int)
        assert isinstance(result["n_tested"], int)

    def test_error_on_invalid_column(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            profiler_test_normality(str(iris_csv), columns=["nonexistent"])


# ---------------------------------------------------------------------------
# analyze_temporal_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeTemporalPatterns:
    """Tests for analyze_temporal_patterns."""

    def test_with_date_column(self, tmp_path: Path) -> None:
        csv = tmp_path / "temporal.csv"
        lines = ["date,value"]
        for i in range(30):
            lines.append(f"2024-01-{i+1:02d},{i * 2 + 1}")
        csv.write_text("\n".join(lines))
        result = analyze_temporal_patterns(str(csv), "date")
        assert "trends" in result
        assert "autocorrelations" in result
        assert result["n_valid_dates"] == 30

    def test_trend_detection(self, tmp_path: Path) -> None:
        csv = tmp_path / "trend.csv"
        lines = ["date,value"]
        for i in range(50):
            lines.append(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},{i * 10}")
        csv.write_text("\n".join(lines))
        result = analyze_temporal_patterns(str(csv), "date")
        assert result["has_trend"] is True

    def test_error_on_missing_column(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            analyze_temporal_patterns(str(iris_csv), "nonexistent")

    def test_insufficient_dates(self, tmp_path: Path) -> None:
        csv = tmp_path / "few.csv"
        csv.write_text("date,value\n2024-01-01,1\n2024-01-02,2\n")
        result = analyze_temporal_patterns(str(csv), "date")
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_separability_score
# ---------------------------------------------------------------------------


class TestComputeSeparabilityScore:
    """Tests for compute_separability_score."""

    def test_iris_has_high_separability(self, iris_csv: Path) -> None:
        result = compute_separability_score(str(iris_csv), "species")
        assert result["mean_separability"] > 0
        assert result["difficulty"] in ("easy", "moderate", "hard", "very_hard")

    def test_returns_per_feature_scores(self, iris_csv: Path) -> None:
        result = compute_separability_score(str(iris_csv), "species")
        assert "petal_length" in result["scores"]
        assert "sepal_length" in result["scores"]

    def test_n_classes(self, iris_csv: Path) -> None:
        result = compute_separability_score(str(iris_csv), "species")
        assert result["n_classes"] == 3

    def test_binary_classification(self, titanic_csv: Path) -> None:
        result = compute_separability_score(str(titanic_csv), "Survived")
        assert result["n_classes"] == 2

    def test_error_on_missing_target(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            compute_separability_score(str(iris_csv), "nonexistent")


# ---------------------------------------------------------------------------
# compute_dataset_fingerprint
# ---------------------------------------------------------------------------


class TestComputeDatasetFingerprint:
    """Tests for compute_dataset_fingerprint."""

    def test_core_fields_present(self, iris_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(iris_csv), "species")
        assert result["size_class"] == "small"
        assert "classification" in result["task"]
        assert result["feature_mix"] == "all_numeric"
        assert result["n_rows"] > 0
        assert result["n_cols"] > 0

    def test_classification_specific_fields(self, iris_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(iris_csv), "species")
        assert result["n_classes"] == 3
        assert result["minority_ratio"] is not None
        assert result["target_entropy"] is not None

    def test_regression_specific_fields(self, boston_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(boston_csv), "medv", task_type="regression")
        assert result["task"] == "regression"
        assert result["target_skewness"] is not None
        assert result["n_classes"] is None

    def test_complexity_score_range(self, iris_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(iris_csv), "species")
        assert 1 <= result["complexity_score"] <= 5

    def test_quality_score_range(self, iris_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(iris_csv), "species")
        assert 0 <= result["data_quality_score"] <= 1

    def test_enrichment_with_eda_results(self, iris_csv: Path) -> None:
        eda_results = {
            "correlations": {
                "feature_target_correlations": {"petal_length": 0.95},
            },
            "mutual_information": {
                "max_mi": 0.8,
                "n_informative": 4,
            },
            "vif": {
                "n_high_vif": 2,
                "n_severe_vif": 0,
            },
        }
        result = compute_dataset_fingerprint(
            str(iris_csv), "species", eda_results=eda_results,
        )
        assert result["max_linear_signal"] == 0.95
        assert result["max_nonlinear_signal"] == 0.8
        assert result["signal_type"] == "mixed"
        assert result["redundancy_level"] == "moderate"

    def test_without_eda_results(self, iris_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(iris_csv), "species")
        # Should still have core fields without enrichment
        assert "complexity_score" in result
        assert "data_quality_score" in result

    def test_mixed_features(self, titanic_csv: Path) -> None:
        result = compute_dataset_fingerprint(str(titanic_csv), "Survived")
        assert result["feature_mix"] in ("mixed", "mostly_numeric", "mostly_categorical")

    def test_error_on_missing_target(self, iris_csv: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            compute_dataset_fingerprint(str(iris_csv), "nonexistent")
