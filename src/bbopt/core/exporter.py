"""Model export and serialization utilities."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def export_model(
    workspace_path: str | Path,
    run_id: str,
    output_path: str | Path | None = None,
    fmt: str = "joblib",
) -> dict[str, Any]:
    """Export a trained model from a run to the exports directory.

    Args:
        workspace_path: Path to the workspace.
        run_id: ID of the run to export.
        output_path: Custom output path. Defaults to workspace exports dir.
        fmt: Export format ('joblib' or 'pickle').

    Returns:
        Dict with file_path, file_size_bytes, format, model_metadata.
    """
    ws = Path(workspace_path)
    run_dir = ws / "runs" / run_id

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    if output_path is None:
        exports_dir = ws / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        ext = "joblib" if fmt == "joblib" else "pkl"
        output_path = exports_dir / f"best_model.{ext}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "joblib":
        shutil.copy2(str(model_path), str(output_path))
    elif fmt == "pickle":
        import pickle

        model = joblib.load(model_path)
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Use 'joblib' or 'pickle'.")

    metadata = {
        "run_id": run_id,
        "model_type": config.get("model_type", "unknown"),
        "hyperparameters": config.get("hyperparameters", {}),
        "feature_columns": config.get("feature_columns", []),
        "metrics": metrics,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format": fmt,
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    file_size = output_path.stat().st_size

    return {
        "file_path": str(output_path),
        "metadata_path": str(metadata_path),
        "file_size_bytes": file_size,
        "format": fmt,
        "model_metadata": metadata,
    }
