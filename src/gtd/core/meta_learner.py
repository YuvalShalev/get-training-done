"""Self-improvement engine for the ML optimization agent.

Layer 1 — Within-Run Reflection (Reflexion pattern):
    Observations recorded every 3 runs during optimization.

Layer 2 — Cross-Run Learning (ExpeL pattern):
    Dataset fingerprints, strategy matching, and a strategy library.

Layer 3 — Prompt Evolution prep (OPRO pattern):
    Session metrics stored for future prompt self-editing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─── Layer 1: Within-Run Reflection ──────────────────────────────────────────


def create_observation(
    run_number: int,
    score_trajectory: list[dict[str, float]],
    actions_taken: list[str],
    diagnosis: str,
    next_strategy: str,
) -> dict[str, Any]:
    """Build a structured observation dict for within-run reflection.

    Args:
        run_number: Current optimization run number.
        score_trajectory: List of {run_id: score} dicts so far.
        actions_taken: Description of actions in the last batch of runs.
        diagnosis: What's working, what's failing, and why.
        next_strategy: What to try next based on the reflection.

    Returns:
        Observation dict with timestamp and all fields.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_number": run_number,
        "score_trajectory": list(score_trajectory),
        "actions_taken": list(actions_taken),
        "diagnosis": diagnosis,
        "next_strategy": next_strategy,
    }


def save_observation(workspace_path: str, observation: dict[str, Any]) -> None:
    """Append an observation to the workspace observation log.

    Creates ``observation-log.md`` if it doesn't exist.

    Args:
        workspace_path: Path to the workspace directory.
        observation: Observation dict from :func:`create_observation`.
    """
    log_path = Path(workspace_path) / "observation-log.md"

    entry_lines = [
        f"### Observation — Run #{observation['run_number']}",
        f"- **Time**: {observation['timestamp']}",
        f"- **Score trajectory**: {json.dumps(observation['score_trajectory'])}",
        f"- **Actions**: {'; '.join(observation['actions_taken'])}",
        f"- **Diagnosis**: {observation['diagnosis']}",
        f"- **Next strategy**: {observation['next_strategy']}",
        "",
    ]
    entry = "\n".join(entry_lines)

    if not log_path.exists():
        header = "# Observation Log\n\n"
        log_path.write_text(header + entry, encoding="utf-8")
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)


def load_observations(workspace_path: str) -> list[dict[str, Any]]:
    """Parse all observations from the workspace observation log.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        List of observation dicts (newest last).  Empty list if no log exists.
    """
    log_path = Path(workspace_path) / "observation-log.md"
    if not log_path.exists():
        return []

    text = log_path.read_text(encoding="utf-8")
    observations: list[dict[str, Any]] = []

    # Split on observation headers
    blocks = re.split(r"(?=^### Observation)", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block.startswith("### Observation"):
            continue

        obs: dict[str, Any] = {}
        run_match = re.search(r"Run #(\d+)", block)
        obs["run_number"] = int(run_match.group(1)) if run_match else 0

        time_match = re.search(r"\*\*Time\*\*:\s*(.+)", block)
        obs["timestamp"] = time_match.group(1).strip() if time_match else ""

        traj_match = re.search(r"\*\*Score trajectory\*\*:\s*(.+)", block)
        if traj_match:
            try:
                obs["score_trajectory"] = json.loads(traj_match.group(1).strip())
            except json.JSONDecodeError:
                obs["score_trajectory"] = []
        else:
            obs["score_trajectory"] = []

        actions_match = re.search(r"\*\*Actions\*\*:\s*(.+)", block)
        if actions_match:
            obs["actions_taken"] = [
                a.strip() for a in actions_match.group(1).split(";") if a.strip()
            ]
        else:
            obs["actions_taken"] = []

        diag_match = re.search(r"\*\*Diagnosis\*\*:\s*(.+)", block)
        obs["diagnosis"] = diag_match.group(1).strip() if diag_match else ""

        strat_match = re.search(r"\*\*Next strategy\*\*:\s*(.+)", block)
        obs["next_strategy"] = strat_match.group(1).strip() if strat_match else ""

        observations.append(obs)

    return observations


# ─── Layer 2: Cross-Run Learning ─────────────────────────────────────────────


def compute_dataset_fingerprint_from_data(
    data_path: str,
    target_column: str,
    task_type: str,
) -> dict[str, Any]:
    """Compute a lightweight fingerprint directly from a data file.

    This avoids requiring a full ``profile_dataset`` call and is used
    by the ``train_model`` side effect on the first training run.

    Args:
        data_path: Path to the CSV file.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.

    Returns:
        Fingerprint dict with size_class, task, feature_mix, n_rows, n_cols,
        and issues list.
    """
    import pandas as pd

    df = pd.read_csv(data_path)
    n_rows, n_cols = df.shape
    n_numeric = df.select_dtypes(include="number").shape[1]
    n_categorical = n_cols - n_numeric - 1  # minus target

    if n_rows < 1000:
        size_class = "small"
    elif n_rows < 100_000:
        size_class = "medium"
    else:
        size_class = "large"

    total = n_numeric + n_categorical
    if total == 0:
        feature_mix = "unknown"
    elif n_categorical <= 0:
        feature_mix = "all_numeric"
    elif n_numeric <= 0:
        feature_mix = "all_categorical"
    elif n_numeric / total > 0.7:
        feature_mix = "mostly_numeric"
    elif n_categorical / total > 0.7:
        feature_mix = "mostly_categorical"
    else:
        feature_mix = "mixed"

    missing_pct = df.isnull().mean().mean()
    issues: list[str] = []
    if missing_pct > 0.05:
        issues.append("missing_values")

    return {
        "size_class": size_class,
        "task": task_type,
        "feature_mix": feature_mix,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "issues": issues,
    }


def compute_dataset_fingerprint(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Derive a compact fingerprint from a ``profile_dataset`` result.

    The fingerprint captures size class, task type, feature mix, and key
    issues so that similar datasets can be matched later.

    Args:
        profile_result: Dict returned by ``data_profiler.profile_dataset``.

    Returns:
        Fingerprint dict with size_class, task, feature_mix, n_rows, n_cols,
        and issues list.
    """
    rows = profile_result.get("shape", {}).get("rows", 0)
    cols = profile_result.get("shape", {}).get("columns", 0)

    if rows < 1000:
        size_class = "small"
    elif rows < 100_000:
        size_class = "medium"
    else:
        size_class = "large"

    task = profile_result.get("task_type", "unknown")

    feat_types = profile_result.get("feature_types", {})
    n_numeric = len(feat_types.get("numeric", []))
    n_categorical = len(feat_types.get("categorical", []))
    total = n_numeric + n_categorical
    if total == 0:
        feature_mix = "unknown"
    elif n_categorical == 0:
        feature_mix = "all_numeric"
    elif n_numeric == 0:
        feature_mix = "all_categorical"
    elif n_numeric / total > 0.7:
        feature_mix = "mostly_numeric"
    elif n_categorical / total > 0.7:
        feature_mix = "mostly_categorical"
    else:
        feature_mix = "mixed"

    # Detect key issues
    issues: list[str] = []
    missing_pct = profile_result.get("missing_pct", {})
    if any(v > 5 for v in missing_pct.values()):
        issues.append("missing_values")

    class_bal = profile_result.get("class_balance")
    if class_bal and class_bal.get("severity") in ("moderate", "severe"):
        issues.append("class_imbalance")

    return {
        "size_class": size_class,
        "task": task,
        "feature_mix": feature_mix,
        "n_rows": rows,
        "n_cols": cols,
        "issues": issues,
    }


def match_strategies(
    fingerprint: dict[str, Any],
    learnings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find past strategies matching the current dataset fingerprint.

    Matching priority: task type > size class > feature mix.

    Args:
        fingerprint: Dict from :func:`compute_dataset_fingerprint`.
        learnings: Parsed learnings dict from :func:`load_learnings`.

    Returns:
        List of matching strategy entries sorted by relevance (best first).
    """
    strategies = learnings.get("strategies", [])
    if not strategies:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for strat in strategies:
        score = 0
        fp = strat.get("fingerprint", {})
        if fp.get("task") == fingerprint.get("task"):
            score += 4
        if fp.get("size_class") == fingerprint.get("size_class"):
            score += 2
        if fp.get("feature_mix") == fingerprint.get("feature_mix"):
            score += 1
        if score > 0:
            scored.append((score, strat))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


def extract_strategy_sequence(optimization_history: dict[str, Any]) -> dict[str, Any]:
    """Extract the winning strategy path from a completed optimization session.

    Args:
        optimization_history: Dict from ``evaluator.get_optimization_history``
            with keys ``runs``, ``best_run_id``, ``best_score``, ``primary_metric``.

    Returns:
        Dict with baseline_model, optimization_path (list of model+change),
        final_model, final_score, total_runs, and runs_to_best.
    """
    runs = optimization_history.get("runs", [])
    best_run_id = optimization_history.get("best_run_id", "")
    best_score = optimization_history.get("best_score", 0.0)

    baseline_model = ""
    optimization_path: list[str] = []
    runs_to_best = len(runs)

    for idx, run in enumerate(runs):
        run_id = run.get("run_id", "")
        model_type = run.get("model_type", "unknown")
        if idx == 0:
            baseline_model = model_type
        optimization_path.append(model_type)
        if run_id == best_run_id:
            runs_to_best = idx + 1

    final_model = ""
    if runs:
        best_run = next(
            (r for r in runs if r.get("run_id") == best_run_id),
            runs[-1],
        )
        final_model = best_run.get("model_type", "unknown")

    return {
        "baseline_model": baseline_model,
        "optimization_path": optimization_path,
        "final_model": final_model,
        "final_score": best_score,
        "total_runs": len(runs),
        "runs_to_best": runs_to_best,
    }


# ─── Layer 2: Enhanced Learnings I/O ─────────────────────────────────────────


def save_enhanced_learnings(memory_dir: str, session_summary: dict[str, Any]) -> None:
    """Append an enhanced learning entry to ``gtd-learnings.md``.

    Args:
        memory_dir: Path to the auto-memory directory.
        session_summary: Dict with keys: date, dataset_description, fingerprint,
            strategy_sequence, score_trajectory, best_model, best_score,
            metric_name, insight, anti_pattern, hp_sweet_spot.
    """
    learnings_path = Path(memory_dir) / "gtd-learnings.md"

    fp = session_summary.get("fingerprint", {})
    fp_str = (
        f"{fp.get('n_rows', '?')}x{fp.get('n_cols', '?')}, "
        f"{fp.get('task', '?')}, {fp.get('feature_mix', '?')}, "
        f"issues={fp.get('issues', [])}"
    )

    seq = session_summary.get("strategy_sequence", {})
    seq_str = (
        f"{seq.get('baseline_model', '?')} → "
        f"{' → '.join(seq.get('optimization_path', []))} → "
        f"{seq.get('final_model', '?')}"
    )

    traj = session_summary.get("score_trajectory", "")
    entry_lines = [
        f"### {session_summary.get('date', 'unknown')} — {session_summary.get('dataset_description', 'dataset')}",
        f"- Fingerprint: {fp_str}",
        f"- Strategy sequence: {seq_str}",
        f"- Score trajectory: {traj}",
        f"- Best: {session_summary.get('best_model', '?')} "
        f"{session_summary.get('best_score', '?')} ({session_summary.get('metric_name', '?')})",
        f"- Insight: {session_summary.get('insight', '')}",
        f"- Anti-pattern: {session_summary.get('anti_pattern', '')}",
        f"- HP sweet spot: {session_summary.get('hp_sweet_spot', '')}",
        "",
    ]
    entry = "\n".join(entry_lines)

    if not learnings_path.exists():
        header = "# GTD Learnings\n\n"
        learnings_path.write_text(header + entry, encoding="utf-8")
    else:
        with open(learnings_path, "a", encoding="utf-8") as f:
            f.write(entry)


def load_learnings(memory_dir: str) -> dict[str, Any]:
    """Parse ``gtd-learnings.md`` into structured data.

    Args:
        memory_dir: Path to the auto-memory directory.

    Returns:
        Dict with ``entries`` (list of parsed learning dicts) and
        ``strategies`` (list extracted from entries for matching).
    """
    learnings_path = Path(memory_dir) / "gtd-learnings.md"
    if not learnings_path.exists():
        return {"entries": [], "strategies": []}

    text = learnings_path.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []
    strategies: list[dict[str, Any]] = []

    blocks = re.split(r"(?=^### )", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block.startswith("###"):
            continue

        entry: dict[str, Any] = {}

        # Parse header
        header_match = re.match(r"### (.+?) — (.+)", block)
        if header_match:
            entry["date"] = header_match.group(1).strip()
            entry["dataset_description"] = header_match.group(2).strip()
        else:
            continue

        # Parse key-value lines
        for field, key in [
            ("Fingerprint", "fingerprint_raw"),
            ("Strategy sequence", "strategy_sequence_raw"),
            ("Score trajectory", "score_trajectory"),
            ("Best", "best"),
            ("Insight", "insight"),
            ("Anti-pattern", "anti_pattern"),
            ("HP sweet spot", "hp_sweet_spot"),
            ("HP note", "hp_note"),
            ("Data", "data_raw"),
            ("Avoid", "avoid"),
        ]:
            match = re.search(rf"- {re.escape(field)}:\s*(.+)", block)
            if match:
                entry[key] = match.group(1).strip()

        entries.append(entry)

        # Build strategy entry for matching
        fp = _parse_fingerprint_raw(entry.get("fingerprint_raw", ""))
        if fp:
            strategies.append({
                "fingerprint": fp,
                "date": entry.get("date", ""),
                "dataset_description": entry.get("dataset_description", ""),
                "best": entry.get("best", ""),
                "insight": entry.get("insight", ""),
                "anti_pattern": entry.get("anti_pattern", entry.get("avoid", "")),
                "hp_sweet_spot": entry.get("hp_sweet_spot", entry.get("hp_note", "")),
                "strategy_sequence_raw": entry.get("strategy_sequence_raw", ""),
            })

    return {"entries": entries, "strategies": strategies}


def _parse_fingerprint_raw(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of a fingerprint string into a dict."""
    if not raw:
        return None

    fp: dict[str, Any] = {}

    # Try to extract NxM
    dim_match = re.search(r"(\d+)\s*x\s*(\d+)", raw)
    if dim_match:
        n_rows = int(dim_match.group(1))
        fp["n_rows"] = n_rows
        fp["n_cols"] = int(dim_match.group(2))
        if n_rows < 1000:
            fp["size_class"] = "small"
        elif n_rows < 100_000:
            fp["size_class"] = "medium"
        else:
            fp["size_class"] = "large"

    # Task type
    for task in (
        "binary_classification",
        "multiclass_classification",
        "regression",
    ):
        if task in raw:
            fp["task"] = task
            break

    # Feature mix
    for mix in (
        "all_numeric",
        "all_categorical",
        "mostly_numeric",
        "mostly_categorical",
        "mixed",
    ):
        if mix in raw:
            fp["feature_mix"] = mix
            break

    return fp if fp else None


def update_strategy_library(
    memory_dir: str,
    fingerprint: dict[str, Any],
    strategy: dict[str, Any],
) -> None:
    """Update ``gtd-strategy-library.md`` with a new or updated archetype entry.

    Args:
        memory_dir: Path to the auto-memory directory.
        fingerprint: Dataset fingerprint dict.
        strategy: Dict with proven_path, hp_starting_points, avoid, sessions_count.
    """
    lib_path = Path(memory_dir) / "gtd-strategy-library.md"

    archetype = (
        f"{fingerprint.get('size_class', '?').title()} "
        f"{fingerprint.get('task', '?').replace('_', ' ')}, "
        f"{fingerprint.get('feature_mix', '?').replace('_', ' ')} features"
    )

    entry_lines = [
        f"## {archetype}",
        f"- Proven path: {strategy.get('proven_path', '?')}",
        f"- HP starting points: {strategy.get('hp_starting_points', '?')}",
        f"- Avoid: {strategy.get('avoid', '?')}",
        f"- Sessions: {strategy.get('sessions_count', 1)}",
        "",
    ]
    entry = "\n".join(entry_lines)

    if not lib_path.exists():
        header = "# GTD Strategy Library\n\n"
        lib_path.write_text(header + entry, encoding="utf-8")
    else:
        existing = lib_path.read_text(encoding="utf-8")
        # Check if archetype section already exists and replace it
        pattern = re.compile(
            rf"## {re.escape(archetype)}\n(?:- .+\n)*\n?",
            re.MULTILINE,
        )
        if pattern.search(existing):
            updated = pattern.sub(entry, existing)
            lib_path.write_text(updated, encoding="utf-8")
        else:
            with open(lib_path, "a", encoding="utf-8") as f:
                f.write(entry)


# ─── Layer 3: Session Metrics ────────────────────────────────────────────────


def record_session_metrics(memory_dir: str, metrics: dict[str, Any]) -> None:
    """Append a session metrics record to ``gtd-meta-scores.jsonl``.

    Args:
        memory_dir: Path to the auto-memory directory.
        metrics: Dict with dataset_name, task_type, final_score, metric_name,
            total_runs, runs_to_best, best_model, total_tool_calls, and
            optionally composite_score.
    """
    scores_path = Path(memory_dir) / "gtd-meta-scores.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }

    with open(scores_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_session_metrics(memory_dir: str) -> list[dict[str, Any]]:
    """Read all session metrics from ``gtd-meta-scores.jsonl``.

    Args:
        memory_dir: Path to the auto-memory directory.

    Returns:
        List of metric dicts (oldest first).  Empty list if file doesn't exist.
    """
    scores_path = Path(memory_dir) / "gtd-meta-scores.jsonl"
    if not scores_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def compute_composite_score(
    quality: float,
    runs_to_best: int,
    max_runs: int,
    tool_calls: int,
    max_tool_calls: int,
) -> float:
    """Compute the weighted composite metric.

    ``composite = 0.6 * quality + 0.25 * (1 - runs/max) + 0.15 * (1 - calls/max)``

    Args:
        quality: Normalized quality score (0-1).
        runs_to_best: Number of runs to reach best score.
        max_runs: Maximum run budget.
        tool_calls: Total MCP tool calls in the session.
        max_tool_calls: Maximum expected tool calls (for normalization).

    Returns:
        Composite score between 0 and 1.
    """
    efficiency = 1.0 - (runs_to_best / max_runs) if max_runs > 0 else 0.0
    economy = 1.0 - (tool_calls / max_tool_calls) if max_tool_calls > 0 else 0.0

    # Clamp components to [0, 1]
    efficiency = max(0.0, min(1.0, efficiency))
    economy = max(0.0, min(1.0, economy))
    quality = max(0.0, min(1.0, quality))

    return 0.6 * quality + 0.25 * efficiency + 0.15 * economy
