"""Structured insight extraction from research results."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Model families to detect, ordered by specificity (longer patterns first)
_MODEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("TabPFN", re.compile(r"\bTabPFN\b", re.IGNORECASE)),
    ("CatBoost", re.compile(r"\bCatBoost\b", re.IGNORECASE)),
    ("LightGBM", re.compile(r"\bLightGBM\b", re.IGNORECASE)),
    ("XGBoost", re.compile(r"\bXGBoost\b", re.IGNORECASE)),
    ("Random Forest", re.compile(
        r"\brandom\s*forest\b", re.IGNORECASE,
    )),
    ("Gradient Boosting", re.compile(
        r"\bgradient\s*boost(?:ing|ed)?\b", re.IGNORECASE,
    )),
    ("Neural Network", re.compile(
        r"\bneural\s*net(?:work)?s?\b", re.IGNORECASE,
    )),
    ("Transformer", re.compile(r"\btransformer\b", re.IGNORECASE)),
    ("Ensemble", re.compile(r"\bensemble\b", re.IGNORECASE)),
    ("Stacking", re.compile(r"\bstacking\b", re.IGNORECASE)),
]

_SOTA_PATTERN = re.compile(
    r"\b(?:state[- ]of[- ]the[- ]art|SOTA|benchmark|best[- ]performing)\b",
    re.IGNORECASE,
)

_HP_PATTERNS: list[tuple[str, str, str]] = [
    ("learning_rate", r"\blearning[_ ]rate\b.*?(\d+\.?\d*(?:e[+-]?\d+)?)", "0.01-0.3"),
    ("max_depth", r"\bmax[_ ]depth\b.*?(\d+)", "3-10"),
    ("n_estimators", r"\bn[_ ]estimators?\b.*?(\d+)", "100-1000"),
    ("num_leaves", r"\bnum[_ ]leaves\b.*?(\d+)", "31-255"),
]


def extract_insights(
    arxiv_results: dict | None = None,
    kaggle_results: dict | None = None,
    pwc_results: dict | None = None,
    task_type: str = "",
    dataset_profile: dict | None = None,
) -> dict[str, Any]:
    """Extract structured, actionable insights from research results.

    Combines findings from arXiv papers, Kaggle notebooks, and Papers with Code
    into a compact, structured format suitable for informing model selection
    and hyperparameter tuning.

    Args:
        arxiv_results: Results from search_arxiv (dict with "results" list).
        kaggle_results: Results from search_kaggle_notebooks
            (dict with "results" list).
        pwc_results: Results from search_papers_with_code
            (dict with "results" list).
        task_type: Task type string (e.g., "binary_classification").
        dataset_profile: Dataset profile dict with keys like "n_rows", "n_cols",
            "n_numeric", "n_categorical".

    Returns:
        Dict with keys: recommended_models, hp_hints, feature_tips,
        competition_strategies, and summary.
    """
    all_model_mentions: list[str] = []
    all_hp_hints: list[dict[str, str]] = []
    all_feature_tips: list[dict[str, str]] = []
    all_strategies: list[dict[str, str]] = []

    if arxiv_results and "results" in arxiv_results:
        models, hp_hints, feature_tips = _extract_from_arxiv(arxiv_results)
        all_model_mentions.extend(models)
        all_hp_hints.extend(hp_hints)
        all_feature_tips.extend(feature_tips)

    if kaggle_results and "results" in kaggle_results:
        models, strategies = _extract_from_kaggle(kaggle_results)
        all_model_mentions.extend(models)
        all_strategies.extend(strategies)

    if pwc_results and "results" in pwc_results:
        models, strategies = _extract_from_pwc(pwc_results)
        all_model_mentions.extend(models)
        all_strategies.extend(strategies)

    recommended_models = _build_model_recommendations(
        all_model_mentions, dataset_profile, task_type,
    )

    # Deduplicate HP hints by (model, param)
    seen_hp: set[tuple[str, str]] = set()
    unique_hp_hints: list[dict[str, str]] = []
    for hint in all_hp_hints:
        key = (hint.get("model", ""), hint.get("param", ""))
        if key not in seen_hp:
            seen_hp.add(key)
            unique_hp_hints.append(hint)

    # Deduplicate feature tips by technique
    seen_tips: set[str] = set()
    unique_feature_tips: list[dict[str, str]] = []
    for tip in all_feature_tips:
        technique = tip.get("technique", "")
        if technique not in seen_tips:
            seen_tips.add(technique)
            unique_feature_tips.append(tip)

    summary = _build_summary(
        recommended_models, unique_hp_hints, all_strategies, task_type,
    )

    return {
        "recommended_models": recommended_models,
        "hp_hints": unique_hp_hints,
        "feature_tips": unique_feature_tips,
        "competition_strategies": all_strategies,
        "summary": summary,
    }


def _extract_model_mentions(text: str) -> list[str]:
    """Extract model family names from text using regex patterns.

    Args:
        text: Free-form text (title, abstract, description).

    Returns:
        List of model family names found in the text (may contain duplicates).
    """
    mentions: list[str] = []
    for name, pattern in _MODEL_PATTERNS:
        if pattern.search(text):
            mentions.append(name)
    return mentions


def _extract_from_arxiv(
    results: dict,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    """Extract model mentions, HP hints, and feature tips from arXiv results.

    Args:
        results: Dict with "results" list of arXiv paper dicts.

    Returns:
        Tuple of (model_mentions, hp_hints, feature_tips).
    """
    model_mentions: list[str] = []
    hp_hints: list[dict[str, str]] = []
    feature_tips: list[dict[str, str]] = []

    for paper in results.get("results", []):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        combined = f"{title} {abstract}"

        mentions = _extract_model_mentions(combined)
        model_mentions.extend(mentions)

        # Weight SOTA papers more heavily
        if _SOTA_PATTERN.search(combined):
            model_mentions.extend(mentions)

        # Extract HP hints from abstract
        for param_name, pattern_str, default_range in _HP_PATTERNS:
            match = re.search(pattern_str, abstract, re.IGNORECASE)
            if match and mentions:
                hp_hints.append({
                    "model": mentions[0],
                    "param": param_name,
                    "suggested_range": default_range,
                })

        # Extract feature engineering tips
        if re.search(
            r"\bfeature\s+(?:engineering|selection|importance)\b",
            combined,
            re.IGNORECASE,
        ):
            feature_tips.append({
                "technique": "feature_engineering",
                "reason": f"Referenced in: {title[:80]}",
            })
        if re.search(r"\btarget\s+encoding\b", combined, re.IGNORECASE):
            feature_tips.append({
                "technique": "target_encoding",
                "reason": f"Referenced in: {title[:80]}",
            })

    return model_mentions, hp_hints, feature_tips


def _extract_from_kaggle(
    results: dict,
) -> tuple[list[str], list[dict[str, str]]]:
    """Extract model mentions and competition strategies from Kaggle results.

    Args:
        results: Dict with "results" list of Kaggle notebook dicts.

    Returns:
        Tuple of (model_mentions, competition_strategies).
    """
    model_mentions: list[str] = []
    strategies: list[dict[str, str]] = []

    for notebook in results.get("results", []):
        title = notebook.get("title", "")
        mentions = _extract_model_mentions(title)

        # Use vote count as a quality signal: repeat mentions for popular notebooks
        score = notebook.get("score", 0)
        weight = 2 if (isinstance(score, (int, float)) and score >= 10) else 1
        model_mentions.extend(mentions * weight)

        if mentions:
            strategies.append({
                "approach": f"{', '.join(mentions)} approach",
                "source": f"Kaggle: {title[:80]}",
            })

    return model_mentions, strategies


def _extract_from_pwc(
    results: dict,
) -> tuple[list[str], list[dict[str, str]]]:
    """Extract model mentions and strategies from Papers with Code results.

    Args:
        results: Dict with "results" list of PwC paper dicts.

    Returns:
        Tuple of (model_mentions, strategies).
    """
    model_mentions: list[str] = []
    strategies: list[dict[str, str]] = []

    for paper in results.get("results", []):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        combined = f"{title} {abstract}"

        mentions = _extract_model_mentions(combined)
        model_mentions.extend(mentions)

        tasks = paper.get("tasks", [])
        if mentions and tasks:
            strategies.append({
                "approach": f"{', '.join(mentions)} for {', '.join(tasks[:2])}",
                "source": f"PwC: {title[:80]}",
            })

    return model_mentions, strategies


def _build_model_recommendations(
    model_mentions: list[str],
    dataset_profile: dict | None,
    task_type: str,
) -> list[dict[str, str]]:
    """Build ranked model recommendations from mentions and dataset characteristics.

    Args:
        model_mentions: List of model names extracted from all sources.
        dataset_profile: Dataset profile dict with size/column info.
        task_type: Task type string.

    Returns:
        List of dicts with "name", "reason", and "confidence" keys,
        sorted by frequency then dataset-based boosts.
    """
    counts = Counter(model_mentions)
    recommendations: list[dict[str, str]] = []

    # Dataset-based recommendations
    if dataset_profile:
        n_rows = dataset_profile.get("n_rows", 0)
        n_cols = dataset_profile.get("n_cols", 0)
        n_categorical = dataset_profile.get("n_categorical", 0)

        if n_rows < 10000 and n_cols < 100:
            counts["TabPFN"] = counts.get("TabPFN", 0) + 5
        if n_rows >= 10000:
            counts["XGBoost"] = counts.get("XGBoost", 0) + 3
            counts["LightGBM"] = counts.get("LightGBM", 0) + 3
        if n_categorical > 0 and n_categorical >= n_cols * 0.3:
            counts["CatBoost"] = counts.get("CatBoost", 0) + 3

    if not counts:
        return []

    sorted_models = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for name, count in sorted_models:
        if count <= 0:
            continue

        reason = _reason_for_model(name, dataset_profile, count)
        if count >= 5:
            confidence = "high"
        elif count >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        recommendations.append({
            "name": name,
            "reason": reason,
            "confidence": confidence,
        })

    return recommendations


def _reason_for_model(
    name: str,
    dataset_profile: dict | None,
    mention_count: int,
) -> str:
    """Generate a human-readable reason for recommending a model.

    Args:
        name: Model family name.
        dataset_profile: Dataset profile dict.
        mention_count: Number of times the model was mentioned.

    Returns:
        Short reason string.
    """
    parts: list[str] = []

    if mention_count >= 3:
        parts.append(f"mentioned {mention_count}x in research")

    if dataset_profile:
        n_rows = dataset_profile.get("n_rows", 0)
        n_cols = dataset_profile.get("n_cols", 0)
        n_categorical = dataset_profile.get("n_categorical", 0)

        if name == "TabPFN" and n_rows < 10000 and n_cols < 100:
            parts.append("strong on small tabular datasets")
        elif name in ("XGBoost", "LightGBM") and n_rows >= 10000:
            parts.append("scales well to large datasets")
        elif name == "CatBoost" and n_categorical > 0:
            parts.append("handles categorical features natively")

    if not parts:
        parts.append("referenced in related research")

    return "; ".join(parts)


def _build_summary(
    recommended_models: list[dict[str, str]],
    hp_hints: list[dict[str, str]],
    strategies: list[dict[str, str]],
    task_type: str,
) -> str:
    """Build a 2-3 sentence summary of the research insights.

    Args:
        recommended_models: List of model recommendation dicts.
        hp_hints: List of hyperparameter hint dicts.
        strategies: List of competition strategy dicts.
        task_type: Task type string.

    Returns:
        Summary string of 2-3 sentences.
    """
    if not recommended_models:
        return "No specific model recommendations found from research."

    top_models = [m["name"] for m in recommended_models[:3]]
    model_str = ", ".join(top_models)

    parts = [f"Research suggests {model_str} for this {task_type or 'task'}."]

    if hp_hints:
        parts.append(
            f"Found {len(hp_hints)} hyperparameter hint(s) to guide tuning.",
        )

    if strategies:
        parts.append(
            f"Identified {len(strategies)} competition strateg"
            f"{'y' if len(strategies) == 1 else 'ies'} from practitioners.",
        )

    return " ".join(parts)
