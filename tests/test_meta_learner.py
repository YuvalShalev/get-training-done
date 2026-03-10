"""Tests for gtd.core.meta_learner module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gtd.core.meta_learner import (
    compute_composite_score,
    compute_dataset_fingerprint,
    compute_dataset_fingerprint_from_eda,
    create_observation,
    extract_strategy_sequence,
    load_learnings,
    load_observations,
    load_session_metrics,
    match_strategies,
    record_session_metrics,
    save_enhanced_learnings,
    save_observation,
    update_strategy_library,
)


# ---------------------------------------------------------------------------
# Layer 1: Within-Run Reflection
# ---------------------------------------------------------------------------


class TestCreateObservation:
    def test_basic_fields(self):
        obs = create_observation(
            run_number=6,
            score_trajectory=[{"run_001": 0.8}, {"run_002": 0.85}],
            actions_taken=["tuned learning_rate", "increased depth"],
            diagnosis="Improving steadily",
            next_strategy="Try regularization",
        )
        assert obs["run_number"] == 6
        assert len(obs["score_trajectory"]) == 2
        assert obs["actions_taken"] == ["tuned learning_rate", "increased depth"]
        assert obs["diagnosis"] == "Improving steadily"
        assert obs["next_strategy"] == "Try regularization"
        assert "timestamp" in obs

    def test_immutable_inputs(self):
        trajectory = [{"run_001": 0.8}]
        actions = ["action1"]
        obs = create_observation(1, trajectory, actions, "diag", "strat")
        # Mutating the returned observation should not affect originals
        obs["score_trajectory"].append({"extra": 0.9})
        assert len(trajectory) == 1


class TestSaveAndLoadObservations:
    def test_roundtrip_single(self, tmp_path: Path):
        obs = create_observation(
            run_number=3,
            score_trajectory=[{"run_001": 0.82}],
            actions_taken=["tried xgboost"],
            diagnosis="Baseline established",
            next_strategy="Tune learning rate",
        )
        save_observation(str(tmp_path), obs)

        loaded = load_observations(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0]["run_number"] == 3
        assert loaded[0]["diagnosis"] == "Baseline established"
        assert loaded[0]["next_strategy"] == "Tune learning rate"

    def test_roundtrip_multiple(self, tmp_path: Path):
        for i in range(1, 4):
            obs = create_observation(
                run_number=i * 3,
                score_trajectory=[{f"run_{i:03d}": 0.8 + i * 0.01}],
                actions_taken=[f"action_{i}"],
                diagnosis=f"diagnosis_{i}",
                next_strategy=f"strategy_{i}",
            )
            save_observation(str(tmp_path), obs)

        loaded = load_observations(str(tmp_path))
        assert len(loaded) == 3
        assert loaded[0]["run_number"] == 3
        assert loaded[2]["run_number"] == 9

    def test_load_empty_workspace(self, tmp_path: Path):
        loaded = load_observations(str(tmp_path))
        assert loaded == []

    def test_file_created(self, tmp_path: Path):
        obs = create_observation(1, [], ["a"], "d", "s")
        save_observation(str(tmp_path), obs)
        assert (tmp_path / "observation-log.md").exists()


# ---------------------------------------------------------------------------
# Layer 2: Dataset Fingerprinting
# ---------------------------------------------------------------------------


class TestComputeDatasetFingerprint:
    def test_small_binary_numeric(self):
        profile = {
            "shape": {"rows": 500, "columns": 10},
            "task_type": "binary_classification",
            "feature_types": {
                "numeric": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "categorical": ["x", "y"],
            },
            "missing_pct": {"a": 0.0, "b": 0.0},
            "class_balance": {"severity": "none"},
        }
        fp = compute_dataset_fingerprint(profile)
        assert fp["size_class"] == "small"
        assert fp["task"] == "binary_classification"
        assert fp["feature_mix"] == "mostly_numeric"
        assert fp["n_rows"] == 500
        assert fp["n_cols"] == 10
        assert fp["issues"] == []

    def test_medium_regression_with_issues(self):
        profile = {
            "shape": {"rows": 5000, "columns": 20},
            "task_type": "regression",
            "feature_types": {
                "numeric": ["a", "b"],
                "categorical": ["c", "d", "e", "f", "g", "h", "i", "j"],
            },
            "missing_pct": {"a": 12.0, "b": 0.0},
            "class_balance": None,
        }
        fp = compute_dataset_fingerprint(profile)
        assert fp["size_class"] == "medium"
        assert fp["task"] == "regression"
        assert fp["feature_mix"] == "mostly_categorical"
        assert "missing_values" in fp["issues"]

    def test_large_with_class_imbalance(self):
        profile = {
            "shape": {"rows": 200_000, "columns": 5},
            "task_type": "binary_classification",
            "feature_types": {
                "numeric": ["a", "b", "c"],
                "categorical": [],
            },
            "missing_pct": {},
            "class_balance": {"severity": "severe"},
        }
        fp = compute_dataset_fingerprint(profile)
        assert fp["size_class"] == "large"
        assert fp["feature_mix"] == "all_numeric"
        assert "class_imbalance" in fp["issues"]

    def test_all_categorical(self):
        profile = {
            "shape": {"rows": 100, "columns": 4},
            "task_type": "multiclass_classification",
            "feature_types": {"numeric": [], "categorical": ["a", "b", "c"]},
            "missing_pct": {},
            "class_balance": {"severity": "none"},
        }
        fp = compute_dataset_fingerprint(profile)
        assert fp["feature_mix"] == "all_categorical"

    def test_mixed_features(self):
        profile = {
            "shape": {"rows": 100, "columns": 4},
            "task_type": "regression",
            "feature_types": {"numeric": ["a", "b"], "categorical": ["c", "d"]},
            "missing_pct": {},
        }
        fp = compute_dataset_fingerprint(profile)
        assert fp["feature_mix"] == "mixed"


# ---------------------------------------------------------------------------
# Layer 2: Strategy Matching
# ---------------------------------------------------------------------------


class TestMatchStrategies:
    def test_exact_match(self):
        fingerprint = {
            "task": "binary_classification",
            "size_class": "medium",
            "feature_mix": "mixed",
        }
        learnings = {
            "strategies": [
                {
                    "fingerprint": {
                        "task": "binary_classification",
                        "size_class": "medium",
                        "feature_mix": "mixed",
                    },
                    "best": "lightgbm 0.89",
                },
                {
                    "fingerprint": {
                        "task": "regression",
                        "size_class": "medium",
                        "feature_mix": "mixed",
                    },
                    "best": "xgboost 0.92",
                },
            ],
        }
        matches = match_strategies(fingerprint, learnings)
        assert len(matches) == 2
        # Best match (all three fields match) should be first
        assert matches[0]["best"] == "lightgbm 0.89"

    def test_no_match(self):
        fingerprint = {"task": "regression", "size_class": "large", "feature_mix": "all_numeric"}
        learnings = {
            "strategies": [
                {
                    "fingerprint": {
                        "task": "binary_classification",
                        "size_class": "small",
                        "feature_mix": "all_categorical",
                    },
                    "best": "lr 0.7",
                },
            ],
        }
        matches = match_strategies(fingerprint, learnings)
        assert matches == []

    def test_empty_learnings(self):
        matches = match_strategies({"task": "regression"}, {"strategies": []})
        assert matches == []

    def test_partial_match_sorted(self):
        fingerprint = {
            "task": "binary_classification",
            "size_class": "small",
            "feature_mix": "mixed",
        }
        learnings = {
            "strategies": [
                {
                    "fingerprint": {
                        "task": "binary_classification",
                        "size_class": "large",
                        "feature_mix": "all_numeric",
                    },
                    "best": "task_only",
                },
                {
                    "fingerprint": {
                        "task": "binary_classification",
                        "size_class": "small",
                        "feature_mix": "all_numeric",
                    },
                    "best": "task+size",
                },
            ],
        }
        matches = match_strategies(fingerprint, learnings)
        assert len(matches) == 2
        assert matches[0]["best"] == "task+size"


# ---------------------------------------------------------------------------
# Layer 2: Strategy Sequence Extraction
# ---------------------------------------------------------------------------


class TestExtractStrategySequence:
    def test_basic_extraction(self):
        history = {
            "runs": [
                {"run_id": "run_001_rf", "model_type": "random_forest"},
                {"run_id": "run_002_xgb", "model_type": "xgboost"},
                {"run_id": "run_003_lgbm", "model_type": "lightgbm"},
            ],
            "best_run_id": "run_002_xgb",
            "best_score": 0.89,
            "primary_metric": "accuracy",
        }
        seq = extract_strategy_sequence(history)
        assert seq["baseline_model"] == "random_forest"
        assert seq["final_model"] == "xgboost"
        assert seq["final_score"] == 0.89
        assert seq["total_runs"] == 3
        assert seq["runs_to_best"] == 2
        assert seq["optimization_path"] == ["random_forest", "xgboost", "lightgbm"]

    def test_empty_history(self):
        seq = extract_strategy_sequence({"runs": [], "best_run_id": "", "best_score": 0.0})
        assert seq["baseline_model"] == ""
        assert seq["total_runs"] == 0
        assert seq["optimization_path"] == []


# ---------------------------------------------------------------------------
# Layer 2: Enhanced Learnings I/O
# ---------------------------------------------------------------------------


class TestEnhancedLearnings:
    @pytest.fixture(autouse=True)
    def _isolate_global_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "gtd.core.meta_learner._GLOBAL_GTD_DIR",
            tmp_path / "global_gtd",
        )

    def test_save_and_load(self, tmp_path: Path):
        summary = {
            "date": "2026-02-28",
            "dataset_description": "Titanic survival prediction",
            "fingerprint": {
                "n_rows": 891,
                "n_cols": 12,
                "task": "binary_classification",
                "feature_mix": "mixed",
                "issues": ["missing_values"],
            },
            "strategy_sequence": {
                "baseline_model": "logistic_regression",
                "optimization_path": ["xgboost", "lightgbm"],
                "final_model": "lightgbm",
            },
            "score_trajectory": "0.78 → 0.85 → 0.87",
            "best_model": "lightgbm",
            "best_score": 0.87,
            "metric_name": "accuracy",
            "insight": "LightGBM with low learning rate works best",
            "anti_pattern": "Deep trees overfit on small data",
            "hp_sweet_spot": "lr=0.05, depth=4, leaves=31",
        }
        save_enhanced_learnings(str(tmp_path), summary)

        learnings = load_learnings(str(tmp_path))
        assert len(learnings["entries"]) == 1

        entry = learnings["entries"][0]
        assert entry["date"] == "2026-02-28"
        assert entry["dataset_description"] == "Titanic survival prediction"
        assert "891x12" in entry.get("fingerprint_raw", "")
        assert entry["insight"] == "LightGBM with low learning rate works best"

    def test_multiple_entries(self, tmp_path: Path):
        for i in range(3):
            summary = {
                "date": f"2026-02-0{i + 1}",
                "dataset_description": f"dataset_{i}",
                "fingerprint": {
                    "n_rows": 100 * (i + 1),
                    "n_cols": 5,
                    "task": "regression",
                    "feature_mix": "all_numeric",
                    "issues": [],
                },
                "strategy_sequence": {
                    "baseline_model": "lr",
                    "optimization_path": [],
                    "final_model": "xgb",
                },
                "score_trajectory": "0.5 → 0.6",
                "best_model": "xgb",
                "best_score": 0.6 + i * 0.1,
                "metric_name": "r2",
                "insight": f"insight_{i}",
                "anti_pattern": f"anti_{i}",
                "hp_sweet_spot": f"hp_{i}",
            }
            save_enhanced_learnings(str(tmp_path), summary)

        learnings = load_learnings(str(tmp_path))
        assert len(learnings["entries"]) == 3
        assert len(learnings["strategies"]) == 3

    def test_load_nonexistent(self, tmp_path: Path):
        learnings = load_learnings(str(tmp_path))
        assert learnings == {"entries": [], "strategies": []}

    def test_strategies_have_fingerprints(self, tmp_path: Path):
        summary = {
            "date": "2026-01-01",
            "dataset_description": "test",
            "fingerprint": {
                "n_rows": 5000,
                "n_cols": 20,
                "task": "binary_classification",
                "feature_mix": "mixed",
                "issues": [],
            },
            "strategy_sequence": {
                "baseline_model": "lr",
                "optimization_path": ["xgb"],
                "final_model": "xgb",
            },
            "score_trajectory": "0.7 → 0.9",
            "best_model": "xgb",
            "best_score": 0.9,
            "metric_name": "accuracy",
            "insight": "boosting wins",
            "anti_pattern": "none",
            "hp_sweet_spot": "lr=0.1",
        }
        save_enhanced_learnings(str(tmp_path), summary)
        learnings = load_learnings(str(tmp_path))

        assert len(learnings["strategies"]) == 1
        strat = learnings["strategies"][0]
        assert strat["fingerprint"]["task"] == "binary_classification"
        assert strat["fingerprint"]["size_class"] == "medium"


# ---------------------------------------------------------------------------
# Layer 2: Strategy Library
# ---------------------------------------------------------------------------


class TestStrategyLibrary:
    @pytest.fixture(autouse=True)
    def _isolate_global_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "gtd.core.meta_learner._GLOBAL_GTD_DIR",
            tmp_path / "global_gtd",
        )

    def test_create_new(self, tmp_path: Path):
        fp = {"size_class": "medium", "task": "binary_classification", "feature_mix": "mixed"}
        strategy = {
            "proven_path": "lr → xgb → lgbm",
            "hp_starting_points": "lr=0.05, depth=4",
            "avoid": "deep trees",
            "sessions_count": 1,
        }
        update_strategy_library(str(tmp_path), fp, strategy)

        lib_path = tmp_path / "gtd-strategy-library.md"
        assert lib_path.exists()
        content = lib_path.read_text()
        assert "Medium binary classification, mixed features" in content
        assert "lr=0.05, depth=4" in content

    def test_update_existing(self, tmp_path: Path):
        fp = {"size_class": "small", "task": "regression", "feature_mix": "all_numeric"}
        strategy1 = {
            "proven_path": "lr → xgb",
            "hp_starting_points": "lr=0.1",
            "avoid": "none",
            "sessions_count": 1,
        }
        update_strategy_library(str(tmp_path), fp, strategy1)

        strategy2 = {
            "proven_path": "lr → xgb → lgbm",
            "hp_starting_points": "lr=0.05",
            "avoid": "catboost",
            "sessions_count": 2,
        }
        update_strategy_library(str(tmp_path), fp, strategy2)

        content = (tmp_path / "gtd-strategy-library.md").read_text()
        # Should only have one section for this archetype
        assert content.count("Small regression, all numeric features") == 1
        assert "lr=0.05" in content
        assert "Sessions: 2" in content


# ---------------------------------------------------------------------------
# Layer 3: Session Metrics
# ---------------------------------------------------------------------------


class TestSessionMetrics:
    def test_record_and_load(self, tmp_path: Path):
        metrics = {
            "dataset_name": "titanic",
            "task_type": "binary_classification",
            "final_score": 0.87,
            "metric_name": "accuracy",
            "total_runs": 15,
            "runs_to_best": 8,
            "best_model": "lightgbm",
            "total_tool_calls": 45,
        }
        record_session_metrics(str(tmp_path), metrics)

        loaded = load_session_metrics(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0]["dataset_name"] == "titanic"
        assert loaded[0]["final_score"] == 0.87
        assert "timestamp" in loaded[0]

    def test_multiple_records(self, tmp_path: Path):
        for i in range(5):
            record_session_metrics(str(tmp_path), {
                "dataset_name": f"ds_{i}",
                "final_score": 0.7 + i * 0.05,
            })

        loaded = load_session_metrics(str(tmp_path))
        assert len(loaded) == 5
        assert loaded[0]["dataset_name"] == "ds_0"
        assert loaded[4]["dataset_name"] == "ds_4"

    def test_load_nonexistent(self, tmp_path: Path):
        loaded = load_session_metrics(str(tmp_path))
        assert loaded == []

    def test_jsonl_format(self, tmp_path: Path):
        record_session_metrics(str(tmp_path), {"score": 0.9})
        record_session_metrics(str(tmp_path), {"score": 0.8})

        lines = (tmp_path / "gtd-meta-scores.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "score" in parsed


class TestCompositeScore:
    def test_perfect_scores(self):
        score = compute_composite_score(
            quality=1.0,
            runs_to_best=1,
            max_runs=20,
            tool_calls=10,
            max_tool_calls=100,
        )
        # 0.6*1 + 0.25*(1 - 1/20) + 0.15*(1 - 10/100)
        expected = 0.6 + 0.25 * 0.95 + 0.15 * 0.9
        assert abs(score - expected) < 0.001

    def test_worst_case(self):
        score = compute_composite_score(
            quality=0.0,
            runs_to_best=20,
            max_runs=20,
            tool_calls=100,
            max_tool_calls=100,
        )
        assert score == 0.0

    def test_mid_range(self):
        score = compute_composite_score(
            quality=0.5,
            runs_to_best=10,
            max_runs=20,
            tool_calls=50,
            max_tool_calls=100,
        )
        expected = 0.6 * 0.5 + 0.25 * 0.5 + 0.15 * 0.5
        assert abs(score - expected) < 0.001

    def test_zero_max_runs(self):
        score = compute_composite_score(0.8, 5, 0, 10, 100)
        # efficiency = 0.0 when max_runs is 0
        expected = 0.6 * 0.8 + 0.25 * 0.0 + 0.15 * 0.9
        assert abs(score - expected) < 0.001

    def test_clamping(self):
        score = compute_composite_score(
            quality=1.5,  # > 1, should clamp
            runs_to_best=0,
            max_runs=20,
            tool_calls=0,
            max_tool_calls=100,
        )
        # quality clamped to 1.0
        expected = 0.6 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0
        assert abs(score - expected) < 0.001


# ---------------------------------------------------------------------------
# compute_dataset_fingerprint_from_eda
# ---------------------------------------------------------------------------


class TestComputeDatasetFingerprintFromEda:
    def test_extracts_core_fields(self):
        eda_result = {
            "size_class": "medium",
            "task": "binary_classification",
            "feature_mix": "mixed",
            "n_rows": 5000,
            "n_cols": 20,
            "issues": ["missing_values"],
        }
        fp = compute_dataset_fingerprint_from_eda(eda_result)
        assert fp["size_class"] == "medium"
        assert fp["task"] == "binary_classification"
        assert fp["feature_mix"] == "mixed"
        assert fp["n_rows"] == 5000
        assert fp["issues"] == ["missing_values"]

    def test_includes_enrichment_fields(self):
        eda_result = {
            "size_class": "small",
            "task": "regression",
            "feature_mix": "all_numeric",
            "n_rows": 100,
            "n_cols": 5,
            "issues": [],
            "signal_type": "linear",
            "complexity_score": 2,
            "missing_pattern": "MCAR",
            "redundancy_level": "low",
        }
        fp = compute_dataset_fingerprint_from_eda(eda_result)
        assert fp["signal_type"] == "linear"
        assert fp["complexity_score"] == 2
        assert fp["missing_pattern"] == "MCAR"
        assert fp["redundancy_level"] == "low"

    def test_missing_enrichment_defaults_to_none(self):
        eda_result = {
            "size_class": "small",
            "task": "regression",
            "feature_mix": "all_numeric",
            "n_rows": 100,
            "n_cols": 5,
            "issues": [],
        }
        fp = compute_dataset_fingerprint_from_eda(eda_result)
        assert fp["signal_type"] is None
        assert fp["complexity_score"] is None

    def test_compatible_with_match_strategies(self):
        eda_result = {
            "size_class": "medium",
            "task": "binary_classification",
            "feature_mix": "mixed",
            "n_rows": 5000,
            "n_cols": 20,
            "issues": [],
            "signal_type": "nonlinear",
            "complexity_score": 3,
        }
        fp = compute_dataset_fingerprint_from_eda(eda_result)
        learnings = {
            "strategies": [{
                "fingerprint": {
                    "task": "binary_classification",
                    "size_class": "medium",
                    "feature_mix": "mixed",
                },
                "best": "xgboost 0.9",
            }],
        }
        matches = match_strategies(fp, learnings)
        assert len(matches) == 1
        assert matches[0]["match_score"] >= 7


# ---------------------------------------------------------------------------
# Enhanced match_strategies scoring
# ---------------------------------------------------------------------------


class TestEnhancedMatchStrategies:
    def test_classification_cross_match(self):
        """binary_classification should partially match multiclass_classification."""
        fingerprint = {"task": "binary_classification", "size_class": "medium", "feature_mix": "mixed"}
        learnings = {
            "strategies": [{
                "fingerprint": {
                    "task": "multiclass_classification",
                    "size_class": "medium",
                    "feature_mix": "mixed",
                },
                "best": "xgb 0.85",
            }],
        }
        matches = match_strategies(fingerprint, learnings)
        assert len(matches) == 1
        # Should get partial task match (2) + size (2) + feature_mix (1) = 5
        assert matches[0]["match_score"] == 5

    def test_adjacent_size_class_bonus(self):
        """Adjacent size classes (small↔medium) should get partial credit."""
        fingerprint = {"task": "regression", "size_class": "small", "feature_mix": "all_numeric"}
        learnings = {
            "strategies": [{
                "fingerprint": {
                    "task": "regression",
                    "size_class": "medium",
                    "feature_mix": "all_numeric",
                },
                "best": "lr 0.8",
            }],
        }
        matches = match_strategies(fingerprint, learnings)
        assert len(matches) == 1
        # task (4) + adjacent_size (1) + feature_mix (1) = 6
        assert matches[0]["match_score"] == 6

    def test_enrichment_field_scoring(self):
        """Matching enrichment fields should increase score."""
        fingerprint = {
            "task": "binary_classification",
            "size_class": "medium",
            "feature_mix": "mixed",
            "signal_type": "nonlinear",
            "complexity_score": 3,
            "missing_pattern": "MAR",
            "redundancy_level": "moderate",
        }
        learnings = {
            "strategies": [{
                "fingerprint": {
                    "task": "binary_classification",
                    "size_class": "medium",
                    "feature_mix": "mixed",
                    "signal_type": "nonlinear",
                    "complexity_score": 3,
                    "missing_pattern": "MAR",
                    "redundancy_level": "moderate",
                },
                "best": "lgbm 0.92",
            }],
        }
        matches = match_strategies(fingerprint, learnings)
        assert len(matches) == 1
        # core: 4+2+1=7, enrichment: 1+1+0.5+0.5=3, total=10
        assert matches[0]["match_score"] == 10

    def test_old_fingerprint_still_works(self):
        """Old fingerprints without enrichment fields should still match."""
        fingerprint = {
            "task": "regression",
            "size_class": "large",
            "feature_mix": "all_numeric",
            "signal_type": "linear",
            "complexity_score": 2,
        }
        learnings = {
            "strategies": [{
                "fingerprint": {
                    "task": "regression",
                    "size_class": "large",
                    "feature_mix": "all_numeric",
                    # No enrichment fields
                },
                "best": "lr 0.95",
            }],
        }
        matches = match_strategies(fingerprint, learnings)
        assert len(matches) == 1
        # Only core fields match: 4+2+1 = 7
        assert matches[0]["match_score"] == 7
