"""DSPy-based prompt evolution for the GTD decision protocol.

Uses accumulated session data (fingerprints, trajectories, metrics) to
optimize the instructions that guide model selection and hyperparameter
tuning decisions.

Requires the ``evolve`` optional dependency: ``pip install get-training-done[evolve]``
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _import_dspy():
    """Lazy import of dspy with a helpful error message."""
    try:
        import dspy
        return dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy is required for prompt evolution. "
            "Install it with: pip install get-training-done[evolve]"
        ) from exc


# ─── DSPy Signatures ─────────────────────────────────────────────────────────


def _build_signatures():
    """Build DSPy signature classes (requires dspy import)."""
    dspy = _import_dspy()

    class ModelSelector(dspy.Signature):
        """Select the best starting model and hyperparameters for a dataset."""
        dataset_profile: str = dspy.InputField(
            desc="Dataset fingerprint: size, task, features, issues",
        )
        past_strategies: str = dspy.InputField(
            desc="Proven strategies for similar datasets",
        )
        model_choice: str = dspy.OutputField(
            desc="Model type to start with",
        )
        initial_hyperparameters: str = dspy.OutputField(
            desc="Starting HP config as JSON",
        )
        reasoning: str = dspy.OutputField(
            desc="Why this model for this data type",
        )

    class OptimizationStep(dspy.Signature):
        """Decide the next optimization action given current state."""
        trajectory: str = dspy.InputField(
            desc="Score trajectory with run details",
        )
        current_best: str = dspy.InputField(
            desc="Best model config and score so far",
        )
        error_analysis: str = dspy.InputField(
            desc="Where the model fails (segments, patterns)",
        )
        remaining_budget: int = dspy.InputField(
            desc="Remaining training runs",
        )
        next_action: str = dspy.OutputField(
            desc="tune_hp | switch_model | engineer_features",
        )
        parameter_changes: str = dspy.OutputField(
            desc="Specific changes as JSON",
        )
        expected_impact: str = dspy.OutputField(
            desc="Why this should help",
        )

    return ModelSelector, OptimizationStep


# ─── DSPy Module ──────────────────────────────────────────────────────────────


def _build_optimizer_module():
    """Build the GTDOptimizer DSPy module."""
    dspy = _import_dspy()
    ModelSelector, OptimizationStep = _build_signatures()

    class GTDOptimizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.select_model = dspy.Predict(ModelSelector)
            self.decide_step = dspy.ChainOfThought(OptimizationStep)

        def forward(self, dataset_profile, past_strategies, trajectory,
                    current_best, error_analysis, remaining_budget):
            if not trajectory:
                return self.select_model(
                    dataset_profile=dataset_profile,
                    past_strategies=past_strategies,
                )
            return self.decide_step(
                trajectory=trajectory,
                current_best=current_best,
                error_analysis=error_analysis,
                remaining_budget=remaining_budget,
            )

    return GTDOptimizer


# ─── Composite Metric ─────────────────────────────────────────────────────────


def gtd_composite_metric(example, pred, trace=None) -> float | bool:
    """Dual-mode metric: float for evaluation, bool for bootstrapping.

    Components:
    - quality (60%): normalized final score
    - efficiency (25%): fewer runs to best is better
    - economy (15%): fewer total tool calls is better

    Args:
        example: DSPy Example with final_score, runs_to_best, total_runs,
                 total_tool_calls fields.
        pred: Prediction (unused in metric computation).
        trace: If None, return float; otherwise return bool (for bootstrapping).

    Returns:
        Float composite score (0-1) or bool (composite > 0.7).
    """
    quality = float(example.final_score)
    total_runs = max(int(example.total_runs), 1)
    efficiency = 1.0 - (int(example.runs_to_best) / total_runs)
    tool_calls = int(example.get("total_tool_calls", "100"))
    economy = 1.0 - min(tool_calls / 200, 1.0)

    composite = 0.6 * quality + 0.25 * efficiency + 0.15 * economy

    if trace is None:
        return composite
    return composite > 0.7


# ─── Training Example Collection ──────────────────────────────────────────────


def collect_training_examples(memory_dir: str) -> list:
    """Convert accumulated session data into DSPy training examples.

    Reads ``gtd-meta-scores.jsonl`` and ``gtd-learnings.md`` to build
    examples pairing dataset characteristics with optimization outcomes.

    Args:
        memory_dir: Path to the auto-memory directory.

    Returns:
        List of DSPy Example objects.
    """
    dspy = _import_dspy()
    from gtd.core import meta_learner

    metrics = meta_learner.load_session_metrics(memory_dir)
    learnings = meta_learner.load_learnings(memory_dir)

    examples = []
    for session in metrics:
        matching = [
            e for e in learnings.get("entries", [])
            if e.get("best", "").startswith(session.get("best_model", "\x00"))
        ]
        if not matching:
            continue

        learning = matching[-1]
        examples.append(dspy.Example(
            dataset_profile=str(learning.get("fingerprint_raw", "")),
            past_strategies=str(learning.get("strategy_sequence_raw", "")),
            trajectory=str(learning.get("score_trajectory", "")),
            final_score=str(session.get("final_score", "0")),
            runs_to_best=str(session.get("runs_to_best", "0")),
            total_runs=str(session.get("total_runs", "0")),
            total_tool_calls=str(session.get("total_tool_calls", "100")),
            best_model=str(session.get("best_model", "")),
        ).with_inputs("dataset_profile", "past_strategies", "trajectory"))

    return examples


# ─── Evolution Pipeline ───────────────────────────────────────────────────────


def optimize_prompts(
    memory_dir: str,
    model_name: str = "anthropic/claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Run DSPy optimization on the GTD decision protocol.

    Requires at least 10 completed training sessions. Uses GEPA or
    MIPROv2 to generate improved instructions based on accumulated
    session trajectories.

    Args:
        memory_dir: Path to the auto-memory directory.
        model_name: LM model identifier for DSPy.

    Returns:
        Dict with status, optimized_program (if successful), and metadata.
    """
    dspy = _import_dspy()

    examples = collect_training_examples(memory_dir)
    if len(examples) < 10:
        return {
            "status": "insufficient_data",
            "count": len(examples),
            "message": f"Need 10+ sessions, have {len(examples)}. Keep training.",
        }

    # Split 20/80 (DSPy recommendation for prompt optimizers)
    split = max(2, len(examples) // 5)
    trainset = examples[:split]
    valset = examples[split:]

    dspy.configure(lm=dspy.LM(model_name))

    GTDOptimizer = _build_optimizer_module()
    program = GTDOptimizer()

    # Try GEPA first (best quality), fall back to MIPROv2
    try:
        optimizer = dspy.GEPA(
            metric=gtd_composite_metric,
            max_iters=8,
            num_candidates=5,
        )
    except AttributeError:
        optimizer = dspy.MIPROv2(
            metric=gtd_composite_metric,
            auto="medium",
        )

    optimized = optimizer.compile(program.deepcopy(), trainset=trainset)

    return {
        "status": "success",
        "optimized_program": optimized,
        "train_size": len(trainset),
        "val_size": len(valset),
    }


def extract_decision_instructions(optimized_program) -> dict[str, Any]:
    """Extract human-readable instructions from an optimized DSPy program.

    Args:
        optimized_program: A compiled DSPy Module.

    Returns:
        Dict mapping predictor names to their optimized instructions
        and demo examples.
    """
    instructions: dict[str, Any] = {}

    for name, predictor in optimized_program.named_predictors():
        if hasattr(predictor, "extended_signature"):
            instructions[name] = predictor.extended_signature.instructions
        elif hasattr(predictor, "signature"):
            instructions[name] = getattr(predictor.signature, "instructions", "")

        if hasattr(predictor, "demos") and predictor.demos:
            instructions[f"{name}_examples"] = [
                {k: str(v) for k, v in demo.items()}
                for demo in predictor.demos[:3]
            ]

    return instructions


def inject_into_train_md(
    instructions: dict[str, Any],
    train_md_path: str,
    backup: bool = True,
) -> dict[str, Any]:
    """Inject optimized instructions into train.md Phase 4.

    Creates a versioned backup before modifying the file.

    Args:
        instructions: Dict from :func:`extract_decision_instructions`.
        train_md_path: Path to the train.md file.
        backup: Whether to create a versioned backup.

    Returns:
        Dict with status, backup_version (if backed up).
    """
    path = Path(train_md_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {train_md_path}"}

    next_v = 0
    if backup:
        versions = list(path.parent.glob(f"{path.stem}.v*"))
        next_v = len(versions) + 1
        shutil.copy2(path, path.parent / f"{path.stem}.v{next_v}{path.suffix}")

    content = path.read_text(encoding="utf-8")

    # Build injection block
    injection = "\n### DSPy-Optimized Decision Guidance\n\n"
    if "decide_step" in instructions:
        injection += f"**Optimization Strategy**: {instructions['decide_step']}\n\n"
    if "decide_step_examples" in instructions:
        injection += "**Proven Examples**:\n"
        for ex in instructions["decide_step_examples"]:
            injection += (
                f"- Trajectory: {ex.get('trajectory', '?')} → "
                f"Action: {ex.get('next_action', '?')}\n"
            )
    injection += "\n"

    # Insert before "### Stopping Criteria"
    marker = "### Stopping Criteria"
    if marker in content:
        content = content.replace(marker, injection + marker)
        path.write_text(content, encoding="utf-8")
        return {"status": "injected", "backup_version": next_v if backup else None}

    return {"status": "marker_not_found", "message": f"'{marker}' not found in {train_md_path}"}
