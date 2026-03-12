"""Tests for gtd.core.feature_engine module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gtd.core.feature_engine import auto_preprocess, engineer_features

# ---------------------------------------------------------------------------
# engineer_features -- individual operations
# ---------------------------------------------------------------------------


class TestEngineerFeaturesOneHotEncode:
    """Tests for the one_hot_encode operation."""

    def test_one_hot_encode_creates_dummy_columns(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "ohe_output.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "one_hot_encode", "columns": ["Sex"]},
            ],
            output_path=str(output),
        )
        assert "one_hot_encode" in result["operations_applied"]
        df = pd.read_csv(output)
        assert "Sex_male" in df.columns or "Sex_female" in df.columns
        assert "Sex" not in df.columns

    def test_one_hot_encode_preserves_row_count(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "ohe_output.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "one_hot_encode", "columns": ["Embarked"]},
            ],
            output_path=str(output),
        )
        assert result["new_shape"][0] == 30


class TestEngineerFeaturesLabelEncode:
    """Tests for the label_encode operation."""

    def test_label_encode_converts_to_integers(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "le_output.csv"
        engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "label_encode", "columns": ["Sex"]},
            ],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert pd.api.types.is_integer_dtype(df["Sex"])

    def test_label_encode_applies_to_multiple_columns(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "le_multi.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "label_encode", "columns": ["Sex", "Embarked"]},
            ],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert pd.api.types.is_integer_dtype(df["Sex"])
        assert pd.api.types.is_integer_dtype(df["Embarked"])
        assert "label_encode" in result["operations_applied"]


class TestEngineerFeaturesImputeNumeric:
    """Tests for the impute_numeric operation."""

    def test_impute_mean_removes_nulls(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "imputed.csv"
        engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "impute_numeric", "columns": ["Age"], "strategy": "mean"},
            ],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert df["Age"].isna().sum() == 0

    def test_impute_median_removes_nulls(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "imputed_median.csv"
        engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "impute_numeric", "columns": ["Age"], "strategy": "median"},
            ],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert df["Age"].isna().sum() == 0


class TestEngineerFeaturesDropColumns:
    """Tests for the drop_columns operation."""

    def test_drop_columns_removes_specified(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "dropped.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "drop_columns", "columns": ["Name", "Ticket", "Cabin"]},
            ],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert "Name" not in df.columns
        assert "Ticket" not in df.columns
        assert "Cabin" not in df.columns
        assert result["new_shape"][1] == 12 - 3


class TestEngineerFeaturesChained:
    """Tests for chaining multiple operations."""

    def test_chained_operations(self, titanic_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "chained.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[
                {"type": "impute_numeric", "columns": ["Age"], "strategy": "median"},
                {"type": "drop_columns", "columns": ["Name", "Ticket", "Cabin"]},
                {"type": "one_hot_encode", "columns": ["Sex", "Embarked"]},
            ],
            output_path=str(output),
        )
        assert result["operations_applied"] == [
            "impute_numeric",
            "drop_columns",
            "one_hot_encode",
        ]
        df = pd.read_csv(output)
        assert df["Age"].isna().sum() == 0
        assert "Name" not in df.columns


class TestEngineerFeaturesErrors:
    """Tests for error handling in engineer_features."""

    def test_error_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            engineer_features(
                data_path="/nonexistent/data.csv",
                operations=[],
                output_path=str(tmp_path / "out.csv"),
            )

    def test_error_on_unknown_operation(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Unknown operation type"):
            engineer_features(
                data_path=str(titanic_csv),
                operations=[{"type": "totally_made_up"}],
                output_path=str(tmp_path / "out.csv"),
            )

    def test_error_on_missing_column_in_operation(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="columns not found"):
            engineer_features(
                data_path=str(titanic_csv),
                operations=[
                    {"type": "drop_columns", "columns": ["nonexistent_col"]},
                ],
                output_path=str(tmp_path / "out.csv"),
            )


# ---------------------------------------------------------------------------
# auto_preprocess
# ---------------------------------------------------------------------------


class TestAutoPreprocess:
    """Tests for auto_preprocess."""

    def test_output_csv_is_created(self, titanic_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "auto_out.csv"
        auto_preprocess(
            data_path=str(titanic_csv),
            target_column="Survived",
            output_path=str(output),
        )
        assert output.exists()

    def test_preserves_row_count(self, titanic_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "auto_out.csv"
        result = auto_preprocess(
            data_path=str(titanic_csv),
            target_column="Survived",
            output_path=str(output),
        )
        assert result["new_shape"][0] == 30

    def test_target_column_still_present(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "auto_out.csv"
        result = auto_preprocess(
            data_path=str(titanic_csv),
            target_column="Survived",
            output_path=str(output),
        )
        assert "Survived" in result["new_columns"]

    def test_numeric_missing_values_imputed(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "auto_out.csv"
        auto_preprocess(
            data_path=str(titanic_csv),
            target_column="Survived",
            output_path=str(output),
        )
        df = pd.read_csv(output)
        # Age had missing values; after auto_preprocess they should be filled
        # Age column should remain (it is numeric, not dropped)
        assert "Age" in df.columns
        assert df["Age"].isna().sum() == 0

    def test_operations_applied_list(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "auto_out.csv"
        result = auto_preprocess(
            data_path=str(titanic_csv),
            target_column="Survived",
            output_path=str(output),
        )
        applied = result["operations_applied"]
        assert isinstance(applied, list)
        assert len(applied) > 0
        # Titanic has missing numeric values so impute should be present
        assert "impute_numeric_median" in applied

    def test_shape_is_list_of_two(self, titanic_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "auto_out.csv"
        result = auto_preprocess(
            data_path=str(titanic_csv),
            target_column="Survived",
            output_path=str(output),
        )
        assert len(result["new_shape"]) == 2

    def test_error_on_missing_target_column(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            auto_preprocess(
                data_path=str(titanic_csv),
                target_column="nonexistent",
                output_path=str(tmp_path / "out.csv"),
            )

    def test_error_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            auto_preprocess(
                data_path="/nonexistent/data.csv",
                target_column="Survived",
                output_path=str(tmp_path / "out.csv"),
            )


# ---------------------------------------------------------------------------
# New operations
# ---------------------------------------------------------------------------


class TestOpTargetEncode:
    """Tests for the target_encode operation."""

    def test_replaces_categories_with_smoothed_means(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "te.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[{
                "type": "target_encode",
                "columns": ["Sex"],
                "target_column": "Survived",
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert pd.api.types.is_float_dtype(df["Sex"])
        assert "target_encode" in result["operations_applied"]

    def test_missing_target_column_raises(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="columns not found"):
            engineer_features(
                data_path=str(titanic_csv),
                operations=[{
                    "type": "target_encode",
                    "columns": ["Sex"],
                    "target_column": "nonexistent",
                }],
                output_path=str(tmp_path / "out.csv"),
            )


class TestOpFrequencyEncode:
    """Tests for the frequency_encode operation."""

    def test_replaces_categories_with_counts(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "fe.csv"
        engineer_features(
            data_path=str(titanic_csv),
            operations=[{"type": "frequency_encode", "columns": ["Sex"]}],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert pd.api.types.is_integer_dtype(df["Sex"])
        assert df["Sex"].max() <= 30


class TestOpGroupbyAggregate:
    """Tests for the groupby_aggregate operation."""

    def test_creates_new_aggregate_column(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "ga.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[{
                "type": "groupby_aggregate",
                "group_column": "Pclass",
                "agg_column": "Fare",
                "agg_func": "mean",
                "new_name": "Pclass_mean_Fare",
            }],
            output_path=str(output),
        )
        assert "Pclass_mean_Fare" in result["new_columns"]

    def test_invalid_agg_func_raises(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="unknown agg_func"):
            engineer_features(
                data_path=str(titanic_csv),
                operations=[{
                    "type": "groupby_aggregate",
                    "group_column": "Pclass",
                    "agg_column": "Fare",
                    "agg_func": "bogus",
                }],
                output_path=str(tmp_path / "out.csv"),
            )


class TestOpPolynomialFeatures:
    """Tests for the polynomial_features operation."""

    def test_generates_polynomial_columns(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "poly.csv"
        engineer_features(
            data_path=str(iris_csv),
            operations=[{
                "type": "polynomial_features",
                "columns": ["sepal_length", "sepal_width"],
                "degree": 2,
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        # Polynomial features generates new columns (e.g. sepal_length^2)
        assert "sepal_length^2" in df.columns or "sepal_length sepal_width" in df.columns
        assert df.shape[1] > 4


class TestOpBinNumeric:
    """Tests for the bin_numeric operation."""

    def test_creates_ordinal_bins(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "binned.csv"
        engineer_features(
            data_path=str(iris_csv),
            operations=[{
                "type": "bin_numeric",
                "columns": ["sepal_length"],
                "n_bins": 4,
                "strategy": "quantile",
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert df["sepal_length"].nunique() <= 4


class TestOpFeatureSelect:
    """Tests for the feature_select operation."""

    def test_mutual_info_reduces_columns(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "fs.csv"
        engineer_features(
            data_path=str(iris_csv),
            operations=[{
                "type": "feature_select",
                "target_column": "species",
                "method": "mutual_info",
                "k": 2,
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        assert len(numeric_cols) == 2

    def test_variance_threshold_drops_low_variance(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "vt.csv"
        engineer_features(
            data_path=str(iris_csv),
            operations=[{
                "type": "feature_select",
                "target_column": "species",
                "method": "variance_threshold",
                "threshold": 0.5,
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert "species" in df.columns

    def test_unknown_method_raises(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            engineer_features(
                data_path=str(iris_csv),
                operations=[{
                    "type": "feature_select",
                    "target_column": "species",
                    "method": "bogus",
                }],
                output_path=str(tmp_path / "out.csv"),
            )


class TestOpRankTransform:
    """Tests for the rank_transform operation."""

    def test_replaces_values_with_ranks(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "ranked.csv"
        engineer_features(
            data_path=str(iris_csv),
            operations=[{
                "type": "rank_transform",
                "columns": ["sepal_length"],
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert df["sepal_length"].min() >= 1.0
        assert df["sepal_length"].max() <= 150.0


class TestOpPowerTransform:
    """Tests for the power_transform operation."""

    def test_transforms_distribution(
        self, iris_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "power.csv"
        engineer_features(
            data_path=str(iris_csv),
            operations=[{
                "type": "power_transform",
                "columns": ["sepal_length", "sepal_width"],
                "method": "yeo-johnson",
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert np.abs(df["sepal_length"].mean()) < 1.0


class TestOpCyclicEncode:
    """Tests for the cyclic_encode operation."""

    def test_creates_sin_cos_columns(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "cyclic.csv"
        engineer_features(
            data_path=str(titanic_csv),
            operations=[{
                "type": "cyclic_encode",
                "column": "Pclass",
                "period": 3,
            }],
            output_path=str(output),
        )
        df = pd.read_csv(output)
        assert "Pclass_sin" in df.columns
        assert "Pclass_cos" in df.columns
        assert "Pclass" not in df.columns


class TestOpRatioFeatures:
    """Tests for the ratio_features operation."""

    def test_creates_ratio_column(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "ratio.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[{
                "type": "ratio_features",
                "numerator": "Fare",
                "denominator": "Age",
                "name": "fare_per_age",
            }],
            output_path=str(output),
        )
        assert "fare_per_age" in result["new_columns"]


class TestOpCategoricalInteraction:
    """Tests for the categorical_interaction operation."""

    def test_creates_interaction_column(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "cat_int.csv"
        result = engineer_features(
            data_path=str(titanic_csv),
            operations=[{
                "type": "categorical_interaction",
                "columns": ["Sex", "Embarked"],
            }],
            output_path=str(output),
        )
        assert "Sex_x_Embarked" in result["new_columns"]
        df = pd.read_csv(output)
        assert pd.api.types.is_integer_dtype(df["Sex_x_Embarked"])

    def test_wrong_column_count_raises(
        self, titanic_csv: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="exactly 2 columns"):
            engineer_features(
                data_path=str(titanic_csv),
                operations=[{
                    "type": "categorical_interaction",
                    "columns": ["Sex"],
                }],
                output_path=str(tmp_path / "out.csv"),
            )
