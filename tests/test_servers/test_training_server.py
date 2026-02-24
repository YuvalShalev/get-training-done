"""Tests for training server tool logic (core trainer, evaluator, model_registry)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bbopt.core import evaluator, model_registry, trainer, workspace


class TestListAvailableModels:
    """Tests for the list_available_models tool logic."""

    def test_returns_nonempty_list(self) -> None:
        result = model_registry.list_available_models()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_model_has_required_fields(self) -> None:
        result = model_registry.list_available_models()
        required_fields = {"name", "display_name", "description", "task_types", "hyperparameters"}
        for model in result:
            assert required_fields.issubset(set(model.keys())), (
                f"Model {model.get('name', 'unknown')} missing fields: "
                f"{required_fields - set(model.keys())}"
            )

    def test_filter_by_binary_classification(self) -> None:
        result = model_registry.list_available_models(task_type="binary_classification")
        assert len(result) > 0
        for model in result:
            assert "binary_classification" in model["task_types"]

    def test_filter_by_regression(self) -> None:
        result = model_registry.list_available_models(task_type="regression")
        assert len(result) > 0
        for model in result:
            assert "regression" in model["task_types"]

    def test_filter_by_multiclass(self) -> None:
        result = model_registry.list_available_models(task_type="multiclass_classification")
        assert len(result) > 0
        for model in result:
            assert "multiclass_classification" in model["task_types"]

    def test_known_models_present(self) -> None:
        result = model_registry.list_available_models()
        model_names = {m["name"] for m in result}
        assert "random_forest" in model_names
        assert "logistic_regression" in model_names

    def test_hyperparameters_have_type_and_default(self) -> None:
        result = model_registry.list_available_models()
        for model in result:
            for hp in model["hyperparameters"]:
                assert "name" in hp
                assert "type" in hp
                assert "default" in hp


class TestTrainAndEvaluateFlow:
    """Integration tests for the train + evaluate pipeline using the iris fixture."""

    @pytest.fixture()
    def iris_workspace(self, tmp_path: Path) -> dict[str, Any]:
        """Create a workspace for iris training."""
        return workspace.create_workspace(tmp_path)

    def test_train_random_forest_on_iris(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        result = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        assert "run_id" in result
        assert "cv_scores" in result
        assert len(result["cv_scores"]) == 3
        assert "mean_score" in result
        assert 0.0 <= result["mean_score"] <= 1.0
        assert "std_score" in result
        assert "training_time" in result
        assert result["training_time"] > 0
        assert "model_path" in result
        assert Path(result["model_path"]).exists()

    def test_train_returns_reasonable_accuracy(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        result = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 50, "max_depth": 5},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        assert result["mean_score"] > 0.5, "Iris should be easy to classify above 50%"

    def test_evaluate_after_training(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        train_result = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        eval_result = evaluator.evaluate_model(
            workspace_path=ws_path,
            run_id=train_result["run_id"],
            data_path=str(iris_csv),
            target_column="species",
            task_type="multiclass_classification",
        )

        assert "accuracy" in eval_result
        assert 0.0 <= eval_result["accuracy"] <= 1.0
        assert "f1_macro" in eval_result
        assert "confusion_matrix" in eval_result

    def test_predict_after_training(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        train_result = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        pred_result = trainer.predict(
            workspace_path=ws_path,
            run_id=train_result["run_id"],
            test_data_path=str(iris_csv),
            target_column="species",
        )

        assert "predictions" in pred_result
        assert len(pred_result["predictions"]) == 30
        assert pred_result["metrics"] is not None
        assert "accuracy" in pred_result["metrics"]

    def test_train_regression_on_boston(
        self,
        boston_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "lstat"]

        result = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(boston_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 5},
            feature_columns=feature_cols,
            target_column="medv",
            task_type="regression",
            cv_folds=3,
            random_state=42,
        )

        assert "run_id" in result
        assert "mean_score" in result
        assert isinstance(result["mean_score"], float)

    def test_train_missing_column_raises(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]

        with pytest.raises(ValueError, match="not found"):
            trainer.train_model(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                model_type="random_forest",
                hyperparameters={},
                feature_columns=["nonexistent_column"],
                target_column="species",
                task_type="multiclass_classification",
            )

    def test_train_invalid_model_type_raises(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]

        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train_model(
                workspace_path=ws_path,
                data_path=str(iris_csv),
                model_type="totally_fake_model",
                hyperparameters={},
                feature_columns=["sepal_length"],
                target_column="species",
                task_type="multiclass_classification",
            )

    def test_train_missing_data_file_raises(
        self,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]

        with pytest.raises(FileNotFoundError):
            trainer.train_model(
                workspace_path=ws_path,
                data_path="/nonexistent/path/data.csv",
                model_type="random_forest",
                hyperparameters={},
                feature_columns=["sepal_length"],
                target_column="species",
                task_type="multiclass_classification",
            )

    def test_compare_runs(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        run_1 = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 5, "max_depth": 2},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        run_2 = trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 50, "max_depth": 5},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        comparison = evaluator.compare_runs(
            workspace_path=ws_path,
            run_ids=[run_1["run_id"], run_2["run_id"]],
        )

        assert "comparison" in comparison
        assert len(comparison["comparison"]) == 2
        assert "best_run_id" in comparison
        assert "primary_metric" in comparison

    def test_optimization_history(
        self,
        iris_csv: Path,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        trainer.train_model(
            workspace_path=ws_path,
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
            random_state=42,
        )

        history = evaluator.get_optimization_history(workspace_path=ws_path)

        assert "runs" in history
        assert len(history["runs"]) == 1
        assert "best_run_id" in history
        assert history["best_run_id"] is not None
        assert "best_score" in history

    def test_empty_optimization_history(
        self,
        iris_workspace: dict[str, Any],
    ) -> None:
        ws_path = iris_workspace["workspace_path"]

        history = evaluator.get_optimization_history(workspace_path=ws_path)

        assert history["runs"] == []
        assert history["best_run_id"] is None
        assert history["best_score"] is None
