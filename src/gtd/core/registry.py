"""Lightweight JSON registry for trained models.

Stores model metadata in a `.gtd-state.json` file in the working directory.
Each training session appends an entry with an auto-incremented ID.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_registry(registry_path: str) -> dict[str, Any]:
    """Read registry file, returning empty structure if missing."""
    path = Path(registry_path)
    if not path.exists():
        return {"current_best": None, "models": []}
    with open(path) as f:
        return json.load(f)


def _write_registry(registry_path: str, data: dict[str, Any]) -> None:
    """Write registry data to file."""
    with open(registry_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def register_model(
    registry_path: str,
    workspace_path: str,
    best_run_id: str,
    best_score: float,
    primary_metric: str,
    model_type: str,
    task_type: str,
    target_column: str,
    data_path: str,
    export_path: str,
    total_runs: int,
) -> dict[str, Any]:
    """Append a new model entry to the registry.

    Returns the new entry with its auto-incremented ID.
    """
    registry = _read_registry(registry_path)
    models = registry.get("models", [])

    next_id = max((m["id"] for m in models), default=0) + 1

    entry: dict[str, Any] = {
        "id": next_id,
        "workspace_path": workspace_path,
        "best_run_id": best_run_id,
        "best_score": best_score,
        "primary_metric": primary_metric,
        "model_type": model_type,
        "task_type": task_type,
        "target_column": target_column,
        "data_path": data_path,
        "export_path": export_path,
        "total_runs": total_runs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    models.append(entry)
    registry = {"current_best": next_id, "models": models}
    _write_registry(registry_path, registry)

    return entry


def list_models(registry_path: str) -> dict[str, Any]:
    """List all registered models.

    Returns dict with ``current_best`` (int | None) and ``models`` (list).
    Raises ``FileNotFoundError`` if the registry file does not exist.
    """
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    return _read_registry(registry_path)


def get_model(registry_path: str, model_id: int) -> dict[str, Any]:
    """Get a specific model by ID.

    Raises ``FileNotFoundError`` if registry is missing.
    Raises ``ValueError`` if the model ID is not found.
    """
    registry = list_models(registry_path)
    for model in registry["models"]:
        if model["id"] == model_id:
            return model
    raise ValueError(f"Model #{model_id} not found in registry")


def get_current_best(registry_path: str) -> dict[str, Any]:
    """Get the current best model entry.

    Raises ``FileNotFoundError`` if registry is missing.
    Raises ``ValueError`` if no current best is set.
    """
    registry = list_models(registry_path)
    current_id = registry.get("current_best")
    if current_id is None:
        raise ValueError("No current best model set in registry")
    return get_model(registry_path, current_id)
