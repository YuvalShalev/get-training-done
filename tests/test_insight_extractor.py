"""Tests for gtd.research.insight_extractor module."""

from __future__ import annotations

from gtd.research.insight_extractor import (
    _build_model_recommendations,
    _extract_model_mentions,
    extract_insights,
)

# ---------------------------------------------------------------------------
# Fixtures / mock data
# ---------------------------------------------------------------------------

MOCK_ARXIV = {
    "results": [
        {
            "title": "XGBoost wins tabular benchmarks",
            "abstract": (
                "We show XGBoost achieves state-of-the-art performance "
                "on 30 tabular datasets with learning_rate 0.1 and "
                "max_depth 6."
            ),
        },
        {
            "title": "LightGBM for large-scale classification",
            "abstract": (
                "LightGBM scales to millions of rows with gradient boosting "
                "and feature engineering techniques."
            ),
        },
        {
            "title": "Neural network approaches to tabular data",
            "abstract": (
                "We compare transformer-based models against ensemble "
                "methods including random forest."
            ),
        },
    ],
}

MOCK_KAGGLE = {
    "results": [
        {
            "title": "Top 1% solution with XGBoost and CatBoost ensemble",
            "score": 25,
        },
        {
            "title": "Simple LightGBM baseline",
            "score": 5,
        },
        {
            "title": "Feature engineering masterclass",
            "score": 50,
        },
    ],
}

MOCK_PWC = {
    "results": [
        {
            "title": "TabPFN: A Transformer That Solves Small Tabular",
            "abstract": "TabPFN is a state-of-the-art model for small data.",
            "tasks": ["tabular-classification"],
        },
        {
            "title": "Gradient boosting benchmarks",
            "abstract": "XGBoost and LightGBM compared on 50 datasets.",
            "tasks": ["classification", "regression"],
        },
    ],
}

SMALL_DATASET_PROFILE = {
    "n_rows": 500,
    "n_cols": 20,
    "n_numeric": 15,
    "n_categorical": 5,
}

LARGE_DATASET_PROFILE = {
    "n_rows": 100_000,
    "n_cols": 50,
    "n_numeric": 40,
    "n_categorical": 10,
}

HIGH_CAT_DATASET_PROFILE = {
    "n_rows": 50_000,
    "n_cols": 30,
    "n_numeric": 5,
    "n_categorical": 25,
}


# ---------------------------------------------------------------------------
# TestExtractModelMentions
# ---------------------------------------------------------------------------


class TestExtractModelMentions:
    """Test regex extraction of model names from text."""

    def test_xgboost_detected(self) -> None:
        assert "XGBoost" in _extract_model_mentions("Using XGBoost for tabular data")

    def test_lightgbm_detected(self) -> None:
        assert "LightGBM" in _extract_model_mentions("LightGBM baseline model")

    def test_catboost_detected(self) -> None:
        assert "CatBoost" in _extract_model_mentions("CatBoost handles categoricals")

    def test_random_forest_detected(self) -> None:
        result = _extract_model_mentions("Random Forest classifier")
        assert "Random Forest" in result

    def test_ensemble_detected(self) -> None:
        assert "Ensemble" in _extract_model_mentions("An ensemble of models")

    def test_stacking_detected(self) -> None:
        assert "Stacking" in _extract_model_mentions("Stacking approach works well")

    def test_neural_network_detected(self) -> None:
        result = _extract_model_mentions("neural network for tabular")
        assert "Neural Network" in result

    def test_transformer_detected(self) -> None:
        assert "Transformer" in _extract_model_mentions("Transformer model")

    def test_tabpfn_detected(self) -> None:
        assert "TabPFN" in _extract_model_mentions("TabPFN for small tables")

    def test_no_matches(self) -> None:
        assert _extract_model_mentions("simple linear regression") == []

    def test_multiple_models(self) -> None:
        text = "XGBoost and LightGBM ensemble"
        result = _extract_model_mentions(text)
        assert "XGBoost" in result
        assert "LightGBM" in result
        assert "Ensemble" in result

    def test_case_insensitive(self) -> None:
        assert "XGBoost" in _extract_model_mentions("xgboost model")
        assert "LightGBM" in _extract_model_mentions("lightgbm baseline")


# ---------------------------------------------------------------------------
# TestExtractInsights
# ---------------------------------------------------------------------------


class TestExtractInsights:
    """Test full insight extraction with mock research results."""

    def test_with_all_sources(self) -> None:
        result = extract_insights(
            arxiv_results=MOCK_ARXIV,
            kaggle_results=MOCK_KAGGLE,
            pwc_results=MOCK_PWC,
            task_type="binary_classification",
            dataset_profile=SMALL_DATASET_PROFILE,
        )

        assert "recommended_models" in result
        assert "hp_hints" in result
        assert "feature_tips" in result
        assert "competition_strategies" in result
        assert "summary" in result

        assert len(result["recommended_models"]) > 0
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_model_names_in_recommendations(self) -> None:
        result = extract_insights(
            arxiv_results=MOCK_ARXIV,
            kaggle_results=MOCK_KAGGLE,
            task_type="binary_classification",
        )
        model_names = [m["name"] for m in result["recommended_models"]]
        assert "XGBoost" in model_names

    def test_recommendation_structure(self) -> None:
        result = extract_insights(arxiv_results=MOCK_ARXIV)
        for rec in result["recommended_models"]:
            assert "name" in rec
            assert "reason" in rec
            assert "confidence" in rec
            assert rec["confidence"] in ("high", "medium", "low")

    def test_hp_hints_extracted_from_arxiv(self) -> None:
        result = extract_insights(
            arxiv_results=MOCK_ARXIV,
            task_type="binary_classification",
        )
        assert len(result["hp_hints"]) > 0
        hp_params = [h["param"] for h in result["hp_hints"]]
        assert "learning_rate" in hp_params or "max_depth" in hp_params

    def test_feature_tips_extracted(self) -> None:
        result = extract_insights(
            arxiv_results=MOCK_ARXIV,
            task_type="binary_classification",
        )
        assert len(result["feature_tips"]) > 0

    def test_competition_strategies_from_kaggle(self) -> None:
        result = extract_insights(
            kaggle_results=MOCK_KAGGLE,
            task_type="binary_classification",
        )
        assert len(result["competition_strategies"]) > 0

    def test_arxiv_only(self) -> None:
        result = extract_insights(arxiv_results=MOCK_ARXIV)
        assert len(result["recommended_models"]) > 0

    def test_kaggle_only(self) -> None:
        result = extract_insights(kaggle_results=MOCK_KAGGLE)
        assert len(result["recommended_models"]) > 0

    def test_pwc_only(self) -> None:
        result = extract_insights(pwc_results=MOCK_PWC)
        assert len(result["recommended_models"]) > 0


# ---------------------------------------------------------------------------
# TestExtractInsightsEmpty
# ---------------------------------------------------------------------------


class TestExtractInsightsEmpty:
    """Test with no results returns empty structure."""

    def test_no_inputs(self) -> None:
        result = extract_insights()
        assert result["recommended_models"] == []
        assert result["hp_hints"] == []
        assert result["feature_tips"] == []
        assert result["competition_strategies"] == []
        assert isinstance(result["summary"], str)

    def test_empty_results_lists(self) -> None:
        result = extract_insights(
            arxiv_results={"results": []},
            kaggle_results={"results": []},
            pwc_results={"results": []},
        )
        assert result["recommended_models"] == []

    def test_error_results_ignored(self) -> None:
        result = extract_insights(
            arxiv_results={"error": "timeout"},
            kaggle_results={"error": "no credentials"},
        )
        assert result["recommended_models"] == []

    def test_none_inputs(self) -> None:
        result = extract_insights(
            arxiv_results=None,
            kaggle_results=None,
            pwc_results=None,
        )
        assert result["recommended_models"] == []


# ---------------------------------------------------------------------------
# TestExtractInsightsDatasetProfile
# ---------------------------------------------------------------------------


class TestExtractInsightsDatasetProfile:
    """Test that dataset profile influences recommendations."""

    def test_small_dataset_recommends_tabpfn(self) -> None:
        result = extract_insights(
            arxiv_results={"results": []},
            dataset_profile=SMALL_DATASET_PROFILE,
            task_type="binary_classification",
        )
        model_names = [m["name"] for m in result["recommended_models"]]
        assert "TabPFN" in model_names

    def test_large_dataset_recommends_gbms(self) -> None:
        result = extract_insights(
            arxiv_results={"results": []},
            dataset_profile=LARGE_DATASET_PROFILE,
            task_type="binary_classification",
        )
        model_names = [m["name"] for m in result["recommended_models"]]
        assert "XGBoost" in model_names or "LightGBM" in model_names

    def test_high_cardinality_cats_recommends_catboost(self) -> None:
        result = extract_insights(
            arxiv_results={"results": []},
            dataset_profile=HIGH_CAT_DATASET_PROFILE,
            task_type="binary_classification",
        )
        model_names = [m["name"] for m in result["recommended_models"]]
        assert "CatBoost" in model_names

    def test_small_dataset_tabpfn_has_high_confidence(self) -> None:
        result = extract_insights(
            arxiv_results={"results": []},
            dataset_profile=SMALL_DATASET_PROFILE,
        )
        tabpfn_recs = [
            m for m in result["recommended_models"] if m["name"] == "TabPFN"
        ]
        assert len(tabpfn_recs) == 1
        assert tabpfn_recs[0]["confidence"] == "high"


# ---------------------------------------------------------------------------
# TestBuildModelRecommendations
# ---------------------------------------------------------------------------


class TestBuildModelRecommendations:
    """Test the ranking logic for model recommendations."""

    def test_most_mentioned_first(self) -> None:
        mentions = ["XGBoost", "XGBoost", "XGBoost", "LightGBM"]
        result = _build_model_recommendations(mentions, None, "")
        assert result[0]["name"] == "XGBoost"

    def test_empty_mentions(self) -> None:
        assert _build_model_recommendations([], None, "") == []

    def test_dataset_boosts_applied(self) -> None:
        mentions = ["XGBoost"]
        profile = {"n_rows": 500, "n_cols": 10, "n_categorical": 0}
        result = _build_model_recommendations(
            mentions, profile, "binary_classification",
        )
        model_names = [m["name"] for m in result]
        # TabPFN should be boosted for small dataset
        assert "TabPFN" in model_names

    def test_confidence_levels(self) -> None:
        mentions = ["XGBoost"] * 6 + ["LightGBM"] * 2 + ["CatBoost"]
        result = _build_model_recommendations(mentions, None, "")
        confidence_map = {m["name"]: m["confidence"] for m in result}
        assert confidence_map["XGBoost"] == "high"
        assert confidence_map["LightGBM"] == "medium"
        assert confidence_map["CatBoost"] == "low"

    def test_reason_includes_mention_count(self) -> None:
        mentions = ["XGBoost"] * 5
        result = _build_model_recommendations(mentions, None, "")
        assert "5x" in result[0]["reason"]

    def test_reason_includes_dataset_context(self) -> None:
        result = _build_model_recommendations(
            ["CatBoost"],
            {"n_rows": 5000, "n_cols": 10, "n_categorical": 5},
            "",
        )
        catboost_rec = [r for r in result if r["name"] == "CatBoost"]
        assert len(catboost_rec) == 1
        assert "categorical" in catboost_rec[0]["reason"]
