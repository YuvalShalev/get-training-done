"""Tests for ensemble strategies: stacking, hill climbing, and seed ensembles."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtd.core import ensemble
from gtd.core.trainer import train_model


class TestStackingEnsemble:
    """Tests for the stacking ensemble strategy."""

    def test_stacking_with_two_base_models(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Stacking with logistic_regression + random_forest should return valid results."""
        base_configs = [
            {"model_type": "logistic_regression", "hyperparameters": {"max_iter": 500}},
            {"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}},
        ]
        feature_cols = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        ]

        result = ensemble.train_stacking_ensemble(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            base_model_configs=base_configs,
            meta_learner_type="logistic_regression",
            meta_learner_params={"max_iter": 500},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
        )

        assert result["run_id"].startswith("ens_")
        assert "stacking" in result["run_id"]
        assert 0.0 <= result["mean_score"] <= 1.0
        assert result["std_score"] >= 0.0
        assert len(result["component_scores"]) == 2
        assert result["training_time"] > 0.0

    def test_stacking_saves_artifacts(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Stacking should save base models and meta model in run directory."""
        base_configs = [
            {"model_type": "logistic_regression", "hyperparameters": {"max_iter": 500}},
            {"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}},
        ]
        feature_cols = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        ]

        result = ensemble.train_stacking_ensemble(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            base_model_configs=base_configs,
            meta_learner_type="logistic_regression",
            meta_learner_params={"max_iter": 500},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
        )

        run_dir = ws_path / "runs" / result["run_id"]
        assert (run_dir / "base_model_0.joblib").exists()
        assert (run_dir / "base_model_1.joblib").exists()
        assert (run_dir / "meta_model.joblib").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()

    def test_stacking_empty_configs_raises(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Stacking with empty base_model_configs should raise ValueError."""
        with pytest.raises(ValueError, match="at least one model"):
            ensemble.train_stacking_ensemble(
                workspace_path=str(ws_path),
                data_path=str(iris_csv),
                base_model_configs=[],
                meta_learner_type="logistic_regression",
                meta_learner_params={},
                feature_columns=["sepal_length"],
                target_column="species",
                task_type="multiclass_classification",
            )

    def test_stacking_missing_data_raises(
        self, ws_path: Path,
    ) -> None:
        """Stacking with nonexistent data path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ensemble.train_stacking_ensemble(
                workspace_path=str(ws_path),
                data_path="/nonexistent/data.csv",
                base_model_configs=[{"model_type": "logistic_regression"}],
                meta_learner_type="logistic_regression",
                meta_learner_params={},
                feature_columns=["a"],
                target_column="b",
                task_type="binary_classification",
            )


class TestHillClimbingEnsemble:
    """Tests for the hill climbing ensemble strategy."""

    def test_hill_climbing_selects_models(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Hill climbing should select models and return valid ensemble score."""
        feature_cols = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        ]

        # Train two models first
        run1 = train_model(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            model_type="logistic_regression",
            hyperparameters={"max_iter": 500},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
        )
        run2 = train_model(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            cv_folds=3,
        )

        result = ensemble.hill_climbing_ensemble(
            workspace_path=str(ws_path),
            run_ids=[run1["run_id"], run2["run_id"]],
            data_path=str(iris_csv),
            target_column="species",
            task_type="multiclass_classification",
            max_ensemble_size=5,
        )

        assert "selected_models" in result
        assert len(result["selected_models"]) >= 1
        assert "weights" in result
        assert 0.0 <= result["ensemble_score"] <= 1.0
        assert len(result["individual_scores"]) == 2

    def test_hill_climbing_empty_run_ids_raises(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Hill climbing with empty run_ids should raise ValueError."""
        with pytest.raises(ValueError, match="at least one run ID"):
            ensemble.hill_climbing_ensemble(
                workspace_path=str(ws_path),
                run_ids=[],
                data_path=str(iris_csv),
                target_column="species",
                task_type="multiclass_classification",
            )


class TestSeedEnsemble:
    """Tests for the seed ensemble strategy."""

    def test_seed_ensemble_with_random_forest(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Seed ensemble should train multiple seeds and return individual scores."""
        feature_cols = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        ]
        n_seeds = 3

        result = ensemble.train_seed_ensemble(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            n_seeds=n_seeds,
            cv_folds=3,
        )

        assert result["run_id"].startswith("ens_")
        assert "seed" in result["run_id"]
        assert len(result["individual_scores"]) == n_seeds
        assert 0.0 <= result["mean_score"] <= 1.0
        assert result["training_time"] > 0.0

    def test_seed_ensemble_saves_models(
        self, ws_path: Path, iris_csv: Path,
    ) -> None:
        """Seed ensemble should save one model per seed."""
        feature_cols = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        ]

        result = ensemble.train_seed_ensemble(
            workspace_path=str(ws_path),
            data_path=str(iris_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10},
            feature_columns=feature_cols,
            target_column="species",
            task_type="multiclass_classification",
            n_seeds=3,
            cv_folds=3,
        )

        run_dir = ws_path / "runs" / result["run_id"]
        assert (run_dir / "seed_model_0.joblib").exists()
        assert (run_dir / "seed_model_1.joblib").exists()
        assert (run_dir / "seed_model_2.joblib").exists()

    def test_seed_ensemble_regression(
        self, ws_path: Path, boston_csv: Path,
    ) -> None:
        """Seed ensemble should work for regression tasks."""
        feature_cols = ["crim", "zn", "indus", "rm", "age", "dis", "tax", "lstat"]

        result = ensemble.train_seed_ensemble(
            workspace_path=str(ws_path),
            data_path=str(boston_csv),
            model_type="random_forest",
            hyperparameters={"n_estimators": 10},
            feature_columns=feature_cols,
            target_column="medv",
            task_type="regression",
            n_seeds=3,
            cv_folds=3,
        )

        assert len(result["individual_scores"]) == 3
        assert result["mean_score"] != 0.0

    def test_seed_ensemble_missing_data_raises(
        self, ws_path: Path,
    ) -> None:
        """Seed ensemble with nonexistent data should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ensemble.train_seed_ensemble(
                workspace_path=str(ws_path),
                data_path="/nonexistent/data.csv",
                model_type="random_forest",
                hyperparameters={},
                feature_columns=["a"],
                target_column="b",
                task_type="binary_classification",
            )
