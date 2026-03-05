"""Tests for gtd.core.prompt_evolver module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _dspy_available() -> bool:
    """Check if dspy is importable."""
    try:
        import dspy  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Composite Metric ─────────────────────────────────────────────────────────


class TestGTDCompositeMetric:
    def test_float_mode_no_trace(self):
        from gtd.core.prompt_evolver import gtd_composite_metric

        example = MagicMock()
        example.final_score = "0.85"
        example.runs_to_best = "5"
        example.total_runs = "20"
        example.get = lambda k, default="100": "50" if k == "total_tool_calls" else default

        score = gtd_composite_metric(example, pred=None, trace=None)
        assert isinstance(score, float)

        # quality=0.85, efficiency=1-5/20=0.75, economy=1-50/200=0.75
        expected = 0.6 * 0.85 + 0.25 * 0.75 + 0.15 * 0.75
        assert abs(score - expected) < 0.01

    def test_bool_mode_with_trace(self):
        from gtd.core.prompt_evolver import gtd_composite_metric

        # High quality example → should return True
        example = MagicMock()
        example.final_score = "0.95"
        example.runs_to_best = "3"
        example.total_runs = "10"
        example.get = lambda k, default="100": "30" if k == "total_tool_calls" else default

        result = gtd_composite_metric(example, pred=None, trace="some_trace")
        assert isinstance(result, bool)
        assert result is True

    def test_bool_mode_low_score(self):
        from gtd.core.prompt_evolver import gtd_composite_metric

        example = MagicMock()
        example.final_score = "0.3"
        example.runs_to_best = "15"
        example.total_runs = "15"
        example.get = lambda k, default="100": "180" if k == "total_tool_calls" else default

        result = gtd_composite_metric(example, pred=None, trace="some_trace")
        assert result is False


# ─── Training Example Collection ──────────────────────────────────────────────


class TestCollectTrainingExamples:
    @pytest.fixture()
    def memory_with_data(self, tmp_path: Path):
        """Create a memory dir with matching metrics and learnings."""
        # Write session metrics
        metrics = [
            {
                "dataset_name": "titanic",
                "task_type": "binary_classification",
                "final_score": 0.87,
                "metric_name": "accuracy",
                "total_runs": 15,
                "runs_to_best": 8,
                "best_model": "lightgbm",
                "total_tool_calls": 45,
            },
            {
                "dataset_name": "iris",
                "task_type": "multiclass_classification",
                "final_score": 0.96,
                "metric_name": "accuracy",
                "total_runs": 10,
                "runs_to_best": 5,
                "best_model": "xgboost",
                "total_tool_calls": 30,
            },
        ]
        scores_path = tmp_path / "gtd-meta-scores.jsonl"
        with open(scores_path, "w") as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")

        # Write learnings that match
        learnings_content = """# GTD Learnings

### 2026-02-28 — Titanic survival
- Fingerprint: 891x12, binary_classification, mixed, issues=['missing_values']
- Strategy sequence: lr → xgb → lgbm
- Score trajectory: 0.78 → 0.87
- Best: lightgbm 0.87 (accuracy)
- Insight: LightGBM with low lr works best
- Anti-pattern: Deep trees overfit
- HP sweet spot: lr=0.05, depth=4

### 2026-03-01 — Iris species
- Fingerprint: 150x5, multiclass_classification, all_numeric, issues=[]
- Strategy sequence: lr → xgb
- Score trajectory: 0.93 → 0.96
- Best: xgboost 0.96 (accuracy)
- Insight: XGBoost with default params near optimal
- Anti-pattern: none
- HP sweet spot: default
"""
        (tmp_path / "gtd-learnings.md").write_text(learnings_content)
        return tmp_path

    @pytest.mark.skipif(
        not _dspy_available(),
        reason="dspy not installed",
    )
    def test_collects_matching_examples(self, memory_with_data):
        from gtd.core.prompt_evolver import collect_training_examples

        examples = collect_training_examples(str(memory_with_data))
        assert len(examples) >= 1
        # Each example should have required fields
        for ex in examples:
            assert hasattr(ex, "final_score")
            assert hasattr(ex, "runs_to_best")
            assert hasattr(ex, "total_runs")

    @pytest.mark.skipif(
        not _dspy_available(),
        reason="dspy not installed",
    )
    def test_empty_memory(self, tmp_path):
        from gtd.core.prompt_evolver import collect_training_examples

        examples = collect_training_examples(str(tmp_path))
        assert examples == []


# ─── Extract Decision Instructions ────────────────────────────────────────────


class TestExtractDecisionInstructions:
    def test_extracts_from_mock_program(self):
        from gtd.core.prompt_evolver import extract_decision_instructions

        # Mock a DSPy program with named_predictors
        mock_sig = MagicMock()
        mock_sig.instructions = "Always try XGBoost first for tabular data"

        mock_predictor = MagicMock()
        mock_predictor.extended_signature = mock_sig
        mock_predictor.demos = [
            {"trajectory": "0.8 → 0.85", "next_action": "tune_hp"},
            {"trajectory": "0.85 → 0.85", "next_action": "switch_model"},
        ]

        mock_program = MagicMock()
        mock_program.named_predictors.return_value = [
            ("decide_step", mock_predictor),
        ]

        instructions = extract_decision_instructions(mock_program)
        assert "decide_step" in instructions
        assert "Always try XGBoost" in instructions["decide_step"]
        assert "decide_step_examples" in instructions
        assert len(instructions["decide_step_examples"]) == 2

    def test_handles_no_demos(self):
        from gtd.core.prompt_evolver import extract_decision_instructions

        mock_sig = MagicMock()
        mock_sig.instructions = "some instruction"

        mock_predictor = MagicMock()
        mock_predictor.extended_signature = mock_sig
        mock_predictor.demos = []

        mock_program = MagicMock()
        mock_program.named_predictors.return_value = [
            ("select_model", mock_predictor),
        ]

        instructions = extract_decision_instructions(mock_program)
        assert "select_model" in instructions
        assert "select_model_examples" not in instructions


# ─── Inject Into train.md ─────────────────────────────────────────────────────


class TestInjectIntoTrainMd:
    def test_basic_injection(self, tmp_path: Path):
        from gtd.core.prompt_evolver import inject_into_train_md

        train_md = tmp_path / "train.md"
        train_md.write_text(
            "# Phase 4\n\nSome content here.\n\n### Stopping Criteria\n\nStop when done.\n"
        )

        instructions = {
            "decide_step": "Always start with LightGBM for medium datasets",
            "decide_step_examples": [
                {"trajectory": "0.8 → 0.82", "next_action": "tune_hp"},
            ],
        }

        result = inject_into_train_md(instructions, str(train_md), backup=True)
        assert result["status"] == "injected"
        assert result["backup_version"] == 1

        content = train_md.read_text()
        assert "DSPy-Optimized Decision Guidance" in content
        assert "Always start with LightGBM" in content
        assert "### Stopping Criteria" in content

    def test_creates_backup(self, tmp_path: Path):
        from gtd.core.prompt_evolver import inject_into_train_md

        train_md = tmp_path / "train.md"
        original = "# Phase 4\n\n### Stopping Criteria\n\nStop.\n"
        train_md.write_text(original)

        inject_into_train_md({"decide_step": "test"}, str(train_md), backup=True)

        backup = tmp_path / "train.v1.md"
        assert backup.exists()
        assert backup.read_text() == original

    def test_no_marker_found(self, tmp_path: Path):
        from gtd.core.prompt_evolver import inject_into_train_md

        train_md = tmp_path / "train.md"
        train_md.write_text("# Phase 4\n\nNo marker here.\n")

        result = inject_into_train_md({"decide_step": "test"}, str(train_md))
        assert result["status"] == "marker_not_found"

    def test_file_not_found(self, tmp_path: Path):
        from gtd.core.prompt_evolver import inject_into_train_md

        result = inject_into_train_md({}, str(tmp_path / "nonexistent.md"))
        assert result["status"] == "error"

    def test_no_backup(self, tmp_path: Path):
        from gtd.core.prompt_evolver import inject_into_train_md

        train_md = tmp_path / "train.md"
        train_md.write_text("# Phase 4\n\n### Stopping Criteria\n\nStop.\n")

        result = inject_into_train_md(
            {"decide_step": "test"}, str(train_md), backup=False,
        )
        assert result["status"] == "injected"
        assert result["backup_version"] is None
        assert not list(tmp_path.glob("train.v*"))


# ─── Optimize Prompts (Integration) ──────────────────────────────────────────


class TestOptimizePrompts:
    def test_insufficient_data(self, tmp_path: Path):
        """optimize_prompts returns early when < 10 sessions available."""
        # Write just 3 sessions
        scores_path = tmp_path / "gtd-meta-scores.jsonl"
        with open(scores_path, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "dataset_name": f"ds_{i}",
                    "final_score": 0.8,
                    "best_model": "xgboost",
                }) + "\n")

        (tmp_path / "gtd-learnings.md").write_text("# GTD Learnings\n")

        # This should work even without dspy since it returns early
        try:
            from gtd.core.prompt_evolver import optimize_prompts
            result = optimize_prompts(str(tmp_path))
            assert result["status"] == "insufficient_data"
            assert result["count"] < 10
        except ImportError:
            pytest.skip("dspy not installed")


# ─── Helper ───────────────────────────────────────────────────────────────────


def _dspy_available() -> bool:
    """Check if dspy is importable."""
    try:
        import dspy  # noqa: F401
        return True
    except ImportError:
        return False
