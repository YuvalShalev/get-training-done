"""Feature engineering operations for data preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def engineer_features(
    data_path: str,
    operations: list[dict[str, Any]],
    output_path: str,
) -> dict[str, Any]:
    """Apply a sequence of feature engineering operations to a dataset.

    Args:
        data_path: Path to input CSV file.
        operations: List of operation dicts, each with "type" and params.
        output_path: Path to save the transformed CSV.

    Returns:
        Dict with new_shape, new_columns, and operations_applied.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If an operation type is unknown or columns are missing.
    """
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(source)
    applied: list[str] = []

    for op in operations:
        op_type = op.get("type", "")
        df = _apply_operation(df, op_type, op)
        applied.append(op_type)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    return {
        "new_shape": [df.shape[0], df.shape[1]],
        "new_columns": list(df.columns),
        "operations_applied": applied,
    }


def auto_preprocess(
    data_path: str,
    target_column: str,
    output_path: str,
) -> dict[str, Any]:
    """Automatically apply sensible preprocessing to a dataset.

    Steps applied:
    - Impute missing numeric columns with median.
    - Impute missing categorical columns with mode.
    - One-hot encode categoricals with < 15 unique values.
    - Label encode categoricals with >= 15 unique values.
    - Drop constant columns (excluding target).

    Args:
        data_path: Path to input CSV file.
        target_column: Name of the target column (excluded from transforms).
        output_path: Path to save the transformed CSV.

    Returns:
        Dict with new_shape, new_columns, operations_applied, and dropped_columns.
    """
    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(source)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    applied: list[str] = []
    dropped_columns: list[str] = []

    feature_cols = [c for c in df.columns if c != target_column]

    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()

    # Impute missing numeric with median
    numeric_with_missing = [c for c in numeric_cols if df[c].isna().any()]
    if numeric_with_missing:
        for col in numeric_with_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        applied.append("impute_numeric_median")

    # Impute missing categorical with mode
    cat_with_missing = [c for c in categorical_cols if df[c].isna().any()]
    if cat_with_missing:
        for col in cat_with_missing:
            mode_val = df[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
            df[col] = df[col].fillna(fill)
        applied.append("impute_categorical_mode")

    # Drop constant columns (single unique value, excluding target)
    constant_cols = [
        c for c in feature_cols if df[c].nunique(dropna=False) <= 1
    ]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        dropped_columns.extend(constant_cols)
        applied.append("drop_constant_columns")
        # Refresh column lists after drops
        feature_cols = [c for c in df.columns if c != target_column]
        categorical_cols = [c for c in categorical_cols if c not in constant_cols]

    # Separate categoricals by cardinality
    low_cardinality = [c for c in categorical_cols if df[c].nunique() < 15]
    high_cardinality = [c for c in categorical_cols if df[c].nunique() >= 15]

    # One-hot encode low cardinality
    if low_cardinality:
        df = pd.get_dummies(df, columns=low_cardinality, drop_first=False, dtype=int)
        applied.append("one_hot_encode")

    # Label encode high cardinality
    if high_cardinality:
        for col in high_cardinality:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        applied.append("label_encode")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    return {
        "new_shape": [df.shape[0], df.shape[1]],
        "new_columns": list(df.columns),
        "operations_applied": applied,
        "dropped_columns": dropped_columns,
    }


# ─── Internal operation dispatch ──────────────────────────────────────────────


_OPERATION_HANDLERS: dict[str, Any] = {}


def _register_op(name: str):
    """Decorator to register an operation handler."""
    def decorator(fn):
        _OPERATION_HANDLERS[name] = fn
        return fn
    return decorator


def _apply_operation(
    df: pd.DataFrame,
    op_type: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch a single operation to the appropriate handler.

    Returns a new DataFrame (original is not mutated).
    """
    handler = _OPERATION_HANDLERS.get(op_type)
    if handler is None:
        supported = ", ".join(sorted(_OPERATION_HANDLERS.keys()))
        raise ValueError(
            f"Unknown operation type '{op_type}'. Supported: {supported}"
        )
    return handler(df.copy(), params)


def _validate_columns(df: pd.DataFrame, columns: list[str], op_name: str) -> None:
    """Raise ValueError if any column is missing from the DataFrame."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Operation '{op_name}': columns not found in data: {missing}"
        )


@_register_op("one_hot_encode")
def _op_one_hot_encode(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns", [])
    _validate_columns(df, columns, "one_hot_encode")
    return pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)


@_register_op("label_encode")
def _op_label_encode(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns", [])
    _validate_columns(df, columns, "label_encode")
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


@_register_op("impute_numeric")
def _op_impute_numeric(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    strategy = params.get("strategy", "mean")
    columns = params.get("columns", [])
    _validate_columns(df, columns, "impute_numeric")

    for col in columns:
        if strategy == "mean":
            fill_value = df[col].mean()
        elif strategy == "median":
            fill_value = df[col].median()
        elif strategy == "zero":
            fill_value = 0
        else:
            raise ValueError(
                f"impute_numeric: unknown strategy '{strategy}'. "
                "Use 'mean', 'median', or 'zero'."
            )
        df[col] = df[col].fillna(fill_value)
    return df


@_register_op("impute_categorical")
def _op_impute_categorical(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    strategy = params.get("strategy", "mode")
    columns = params.get("columns", [])
    fill_value = params.get("fill_value", "missing")
    _validate_columns(df, columns, "impute_categorical")

    for col in columns:
        if strategy == "mode":
            mode_val = df[col].mode()
            fv = mode_val.iloc[0] if len(mode_val) > 0 else fill_value
        elif strategy == "constant":
            fv = fill_value
        else:
            raise ValueError(
                f"impute_categorical: unknown strategy '{strategy}'. "
                "Use 'mode' or 'constant'."
            )
        df[col] = df[col].fillna(fv)
    return df


@_register_op("standard_scale")
def _op_standard_scale(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns", [])
    _validate_columns(df, columns, "standard_scale")

    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns].values)
    return df


@_register_op("log_transform")
def _op_log_transform(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns", [])
    _validate_columns(df, columns, "log_transform")

    for col in columns:
        df[col] = np.log1p(df[col])
    return df


@_register_op("drop_columns")
def _op_drop_columns(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns", [])
    _validate_columns(df, columns, "drop_columns")
    return df.drop(columns=columns)


@_register_op("create_interaction")
def _op_create_interaction(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    col_a = params.get("column_a", "")
    col_b = params.get("column_b", "")
    name = params.get("name", f"{col_a}_x_{col_b}")
    _validate_columns(df, [col_a, col_b], "create_interaction")

    df[name] = df[col_a] * df[col_b]
    return df
