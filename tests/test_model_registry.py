"""Tests for model_registry, focusing on TabPFN support."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gtd.core.model_registry import (
    ALL_MODELS,
    get_models_for_task,
    instantiate_model,
)


class TestTabPFNRegistry:
    """Tests for TabPFN model in the registry."""

    def test_tabpfn_in_all_models(self) -> None:
        """TabPFN should be registered in ALL_MODELS."""
        assert "tabpfn" in ALL_MODELS

    def test_tabpfn_spec_metadata(self) -> None:
        """TabPFN spec should have correct metadata."""
        spec = ALL_MODELS["tabpfn"]
        assert spec.display_name == "TabPFN"
        assert spec.supports_feature_importance is False
        assert "transformer" in spec.tags
        assert "pretrained" in spec.tags

    def test_tabpfn_returned_for_binary_classification(self) -> None:
        """get_models_for_task should include TabPFN for binary classification."""
        models = get_models_for_task("binary_classification")
        names = [m.name for m in models]
        assert "tabpfn" in names

    def test_tabpfn_returned_for_multiclass_classification(self) -> None:
        """get_models_for_task should include TabPFN for multiclass classification."""
        models = get_models_for_task("multiclass_classification")
        names = [m.name for m in models]
        assert "tabpfn" in names

    def test_tabpfn_not_returned_for_regression(self) -> None:
        """get_models_for_task should NOT include TabPFN for regression."""
        models = get_models_for_task("regression")
        names = [m.name for m in models]
        assert "tabpfn" not in names

    def test_instantiate_raises_import_error_when_not_installed(self) -> None:
        """instantiate_model should raise ImportError with clear message."""
        with patch(
            "importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'tabpfn'"),
        ):
            with pytest.raises(ImportError, match="TabPFN not installed"):
                instantiate_model("tabpfn", "binary_classification")

    def test_instantiate_rejects_regression_task(self) -> None:
        """instantiate_model should reject regression for TabPFN."""
        with pytest.raises(ValueError, match="does not support task type"):
            instantiate_model("tabpfn", "regression")
