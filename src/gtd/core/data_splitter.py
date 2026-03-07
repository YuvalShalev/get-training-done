"""Train/validation data partitioning for HPO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def create_data_split(
    workspace_path: str,
    data_path: str,
    target_column: str,
    task_type: str,
    strategy: str = "stratified",
    validation_fraction: float = 0.2,
    temporal_column: str | None = None,
    group_column: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split data into train and validation partitions.

    Args:
        workspace_path: Path to the workspace directory.
        data_path: Path to the source CSV file.
        target_column: Name of the target column.
        task_type: One of 'binary_classification', 'multiclass_classification',
                   or 'regression'.
        strategy: Split strategy — 'random', 'stratified', 'temporal', or 'group'.
        validation_fraction: Fraction of data for validation (default 0.2).
        temporal_column: Column name for temporal sorting (required for 'temporal').
        group_column: Column name for group splitting (required for 'group').
        random_state: Random seed for reproducibility.

    Returns:
        Dict with train_data_path, validation_data_path, train_size,
        validation_size, strategy, and split_info.

    Raises:
        ValueError: If strategy is invalid or required columns are missing.
        FileNotFoundError: If data_path does not exist.
    """
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    from gtd.core.data_profiler import load_csv

    df = load_csv(str(source))

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    ws = Path(workspace_path)
    data_dir = ws / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, split_info = _split_by_strategy(
        df=df,
        target_column=target_column,
        task_type=task_type,
        strategy=strategy,
        validation_fraction=validation_fraction,
        temporal_column=temporal_column,
        group_column=group_column,
        random_state=random_state,
    )

    train_path = str(data_dir / "train.csv")
    val_path = str(data_dir / "validation.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Update workspace metadata
    from gtd.core import workspace as ws_mod

    ws_mod.update_workspace_metadata(workspace_path, {
        "train_data_path": train_path,
        "validation_data_path": val_path,
        "split_strategy": strategy,
        "split_info": split_info,
    })

    return {
        "train_data_path": train_path,
        "validation_data_path": val_path,
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "strategy": strategy,
        "split_info": split_info,
    }


def get_split_paths(workspace_path: str) -> dict[str, str | None]:
    """Return train/validation paths from workspace metadata, or None if no split.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        Dict with 'train_data_path' and 'validation_data_path' (both may be None).
    """
    metadata_path = Path(workspace_path) / "metadata.json"
    if not metadata_path.exists():
        return {"train_data_path": None, "validation_data_path": None}

    with open(metadata_path) as f:
        metadata = json.load(f)

    train_path = metadata.get("train_data_path")
    val_path = metadata.get("validation_data_path")

    # Verify files still exist
    if train_path and not Path(train_path).exists():
        train_path = None
    if val_path and not Path(val_path).exists():
        val_path = None

    return {"train_data_path": train_path, "validation_data_path": val_path}


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _split_by_strategy(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    strategy: str,
    validation_fraction: float,
    temporal_column: str | None,
    group_column: str | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Dispatch to the appropriate split strategy.

    Returns:
        Tuple of (train_df, val_df, split_info).
    """
    if strategy == "stratified":
        return _stratified_split(df, target_column, validation_fraction, random_state)
    if strategy == "random":
        return _random_split(df, validation_fraction, random_state)
    if strategy == "temporal":
        if not temporal_column:
            raise ValueError("temporal_column is required for strategy='temporal'")
        if temporal_column not in df.columns:
            raise ValueError(f"Temporal column '{temporal_column}' not found in data")
        return _temporal_split(df, temporal_column, validation_fraction)
    if strategy == "group":
        if not group_column:
            raise ValueError("group_column is required for strategy='group'")
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in data")
        return _group_split(df, group_column, validation_fraction, random_state)

    raise ValueError(
        f"Unknown split strategy '{strategy}'. "
        "Use 'random', 'stratified', 'temporal', or 'group'."
    )


def _stratified_split(
    df: pd.DataFrame,
    target_column: str,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stratified split preserving class distribution."""
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=validation_fraction,
        stratify=df[target_column],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        {"method": "stratified", "target_column": target_column},
    )


def _random_split(
    df: pd.DataFrame,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Simple random split without stratification."""
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=validation_fraction,
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        {"method": "random"},
    )


def _temporal_split(
    df: pd.DataFrame,
    temporal_column: str,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Temporal split: earlier data for train, later for validation."""
    sorted_df = df.sort_values(temporal_column).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - validation_fraction))

    train_df = sorted_df.iloc[:split_idx].reset_index(drop=True)
    val_df = sorted_df.iloc[split_idx:].reset_index(drop=True)

    return (
        train_df,
        val_df,
        {
            "method": "temporal",
            "temporal_column": temporal_column,
            "split_index": split_idx,
        },
    )


def _group_split(
    df: pd.DataFrame,
    group_column: str,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Group split: no group appears in both partitions."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )

    groups = df[group_column]
    train_idx, val_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_groups = set(df.iloc[train_idx][group_column].unique())
    val_groups = set(df.iloc[val_idx][group_column].unique())

    return (
        train_df,
        val_df,
        {
            "method": "group",
            "group_column": group_column,
            "train_groups": len(train_groups),
            "validation_groups": len(val_groups),
        },
    )
