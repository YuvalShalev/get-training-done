"""Workspace/session filesystem manager for optimization runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_workspace(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Create a new optimization workspace with standard directory structure.

    Args:
        base_dir: Parent directory for the workspace. Defaults to current directory.

    Returns:
        Dict with workspace_path, workspace_id, and created_at.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workspace_id = f"gtd_workspace_{timestamp}"
    workspace_path = base / workspace_id

    dirs = [
        workspace_path / "data",
        workspace_path / "runs",
        workspace_path / "reports",
        workspace_path / "exports",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    metadata = {
        "workspace_id": workspace_id,
        "workspace_path": str(workspace_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
        "best_run_id": None,
        "best_score": None,
        "primary_metric": None,
        "task_type": None,
        "target_column": None,
        "data_path": None,
    }
    _write_json(workspace_path / "metadata.json", metadata)

    return {
        "workspace_path": str(workspace_path),
        "workspace_id": workspace_id,
        "created_at": metadata["created_at"],
    }


def get_workspace_metadata(workspace_path: str | Path) -> dict[str, Any]:
    """Read workspace metadata."""
    return _read_json(Path(workspace_path) / "metadata.json")


def update_workspace_metadata(
    workspace_path: str | Path, updates: dict[str, Any],
) -> dict[str, Any]:
    """Update workspace metadata with new values (immutable merge)."""
    ws = Path(workspace_path)
    metadata_path = ws / "metadata.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
    else:
        ws.mkdir(parents=True, exist_ok=True)
        metadata = {
            "workspace_id": ws.name,
            "workspace_path": str(ws),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": [],
            "best_run_id": None,
            "best_score": None,
            "primary_metric": None,
            "task_type": None,
            "target_column": None,
            "data_path": None,
        }
    merged = {**metadata, **updates}
    _write_json(metadata_path, merged)
    return merged


def register_run(
    workspace_path: str | Path,
    run_id: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    feature_columns: list[str],
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Register a training run in workspace metadata.

    Returns:
        The run entry that was added.
    """
    ws = Path(workspace_path)
    metadata = _read_json(ws / "metadata.json")

    run_entry = {
        "run_id": run_id,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "feature_columns": feature_columns,
        "metrics": metrics or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    updated_runs = [*metadata.get("runs", []), run_entry]
    _write_json(ws / "metadata.json", {**metadata, "runs": updated_runs})

    return run_entry


def update_run_metrics(
    workspace_path: str | Path,
    run_id: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Update metrics for an existing run."""
    ws = Path(workspace_path)
    metadata = _read_json(ws / "metadata.json")

    updated_runs = []
    updated_entry = None
    for run in metadata.get("runs", []):
        if run["run_id"] == run_id:
            updated_entry = {**run, "metrics": {**run.get("metrics", {}), **metrics}}
            updated_runs.append(updated_entry)
        else:
            updated_runs.append(run)

    if updated_entry is None:
        raise ValueError(f"Run '{run_id}' not found in workspace")

    _write_json(ws / "metadata.json", {**metadata, "runs": updated_runs})
    return updated_entry


def update_best_run(
    workspace_path: str | Path,
    run_id: str,
    score: float,
    metric_name: str,
) -> dict[str, Any]:
    """Update the best run tracker in workspace metadata."""
    return update_workspace_metadata(workspace_path, {
        "best_run_id": run_id,
        "best_score": score,
        "primary_metric": metric_name,
    })


def get_run_dir(workspace_path: str | Path, run_id: str) -> Path:
    """Get the directory for a specific run, creating it if needed."""
    run_dir = Path(workspace_path) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def list_runs(workspace_path: str | Path) -> list[dict[str, Any]]:
    """List all runs in the workspace with their metadata."""
    metadata = _read_json(Path(workspace_path) / "metadata.json")
    return metadata.get("runs", [])


def get_run_metadata(workspace_path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Get metadata for a specific run."""
    for run in list_runs(workspace_path):
        if run["run_id"] == run_id:
            return run
    return None


def save_run_artifact(
    workspace_path: str | Path,
    run_id: str,
    filename: str,
    data: Any,
) -> str:
    """Save a JSON artifact to a run directory.

    Returns:
        Path to the saved file.
    """
    run_dir = get_run_dir(workspace_path, run_id)
    filepath = run_dir / filename
    _write_json(filepath, data)
    return str(filepath)


def load_run_artifact(
    workspace_path: str | Path,
    run_id: str,
    filename: str,
) -> Any:
    """Load a JSON artifact from a run directory."""
    filepath = Path(workspace_path) / "runs" / run_id / filename
    return _read_json(filepath)


def save_report(workspace_path: str | Path, filename: str, data: Any) -> str:
    """Save a report file to the workspace reports directory.

    Returns:
        Path to the saved file.
    """
    filepath = Path(workspace_path) / "reports" / filename
    if filename.endswith(".json"):
        _write_json(filepath, data)
    else:
        filepath.write_text(data if isinstance(data, str) else json.dumps(data, indent=2))
    return str(filepath)


def copy_data_to_workspace(
    workspace_path: str | Path,
    source_path: str | Path,
    filename: str = "original.csv",
) -> str:
    """Copy a data file into the workspace data directory.

    Returns:
        Path to the copied file.
    """
    dest = Path(workspace_path) / "data" / filename
    shutil.copy2(str(source_path), str(dest))
    return str(dest)


def delete_workspace(workspace_path: str | Path) -> bool:
    """Delete a workspace and all its contents.

    Returns:
        True if deleted, False if workspace didn't exist.
    """
    ws = Path(workspace_path)
    if ws.exists():
        shutil.rmtree(ws)
        return True
    return False


def _read_json(filepath: Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def _write_json(filepath: Path, data: Any) -> None:
    """Write data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
