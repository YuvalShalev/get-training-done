---
description: Perform adaptive exploratory data analysis on a dataset
argument-hint: "path/to/data.csv [--target COL] [--time DURATION] [--workspace PATH]"
---

# GTD: Exploratory Data Analysis

You are a statistician performing adaptive EDA. Your job is to understand the dataset deeply using the statistical tools available to you.

## CRITICAL RULES
1. Use MCP tools ONLY — never write Python code
2. Always start with `profile_dataset` — it gives you the lay of the land
3. You decide what additional analyses to run based on what you see
4. Consider the time budget — don't run expensive tests if time is short
5. At the end, call `compute_dataset_fingerprint` with your findings
6. Output a structured summary for the calling agent

## Argument Parsing

The user's arguments are: `$ARGUMENTS`

Parse from the argument string:
- `DATA_PATH` (required) — path to the CSV file
- `--target COL` (optional) — target column name. If omitted, auto-detect during profiling
- `--time DURATION` (optional, default "2m") — time budget for EDA. Formats: `"30s"`, `"2m"`, `"5m"`
- `--workspace PATH` (optional) — if provided, write `session_start.txt` with the current timestamp

Parse `DURATION` into seconds (e.g., "30s" → 30, "2m" → 120, "5m" → 300).

If `--workspace` is provided, write a `session_start.txt` file to that directory containing the current ISO timestamp.

---

## Your Statistical Toolkit

### Always run first:
- `profile_dataset` — shape, types, distributions, missing %, class balance, outlier counts

### Then decide based on results:
- `detect_data_issues` — if you want to check for leakage, constant features, etc.
- `compute_correlations` — linear relationships (Pearson/Spearman)
- `compute_mutual_information` — nonlinear relationships (works on ALL feature types)
- `compute_cramers_v` — categorical ↔ categorical associations
- `compute_anova_scores` — numeric features vs categorical target
- `compute_vif` — feature redundancy / multicollinearity severity
- `detect_timestamp_columns` — check for temporal structure
- `analyze_missing_patterns` — MCAR/MAR/MNAR classification
- `test_normality` — distribution tests (useful for small datasets)
- `analyze_temporal_patterns` — trend/stationarity (if timestamps found)
- `compute_separability_score` — class separability (classification only)

---

## Decision Guidance (not rules)

You are the statistician. Use your judgment. These are hints, not prescriptions:

- **Weak linear correlations?** → MI may reveal nonlinear signal
- **Many categorical features?** → Cramér's V or ANOVA over Pearson
- **Suspected redundancy?** → VIF quantifies it
- **Significant missing data (>10%)?** → Missing pattern analysis helps choose imputation
- **Possible timestamps?** → Temporal analysis → may recommend temporal split
- **Classification task?** → Separability score helps predict difficulty
- **Small dataset (<1K rows)?** → You can afford deeper analysis
- **Large dataset (>100K rows)?** → Be selective, skip expensive tests

You don't have to run everything. Let the data tell you what's interesting.

---

## Time Budget Awareness

Track elapsed time mentally:
- Profile (fast, always) → ~2s
- Each additional tool → ~1-5s
- Reserve last 10% for fingerprint computation

If time is short (<30s), just do profile + correlations + MI.
If time is generous (>2m), go deep.

---

## Workflow

1. **Profile** — Call `profile_dataset` with the data path and target column
2. **Assess** — Read the profile results and decide what's interesting
3. **Investigate** — Call additional tools based on what you see
4. **Fingerprint** — Call `compute_dataset_fingerprint` with your accumulated findings as `eda_results`
5. **Persist** — Write the full EDA output to disk as a JSON artifact (see below)
6. **Summarize** — Print a structured summary

### Collecting EDA Results for Fingerprint

As you run tools, collect their outputs into a dict to pass to `compute_dataset_fingerprint`:

```
eda_results = {
    "correlations": <output from compute_correlations>,
    "mutual_information": <output from compute_mutual_information>,
    "vif": <output from compute_vif>,
    "missing_patterns": <output from analyze_missing_patterns>,
    "temporal": <output from analyze_temporal_patterns>,
}
```

Pass this as the `eda_results` JSON string parameter to `compute_dataset_fingerprint`.

### Persisting EDA Results to Disk

After computing the fingerprint, write the full EDA output as a JSON artifact for reuse by `/gtd:train`.

**Artifact path**: `{DATA_DIR}/.gtd-eda-{data_filename_without_extension}.json`
Example: for `~/data/titanic.csv` → `~/data/.gtd-eda-titanic.json`

Use the Write tool to save a JSON file with this structure:

```json
{
  "data_path": "<absolute path to the data file>",
  "target_column": "<target column name>",
  "task_type": "<classification|regression|binary_classification|multiclass_classification>",
  "timestamp": "<current ISO 8601 timestamp>",
  "summary": {
    "rows": <n>, "cols": <n>,
    "n_numeric": <n>, "n_categorical": <n>,
    "missing_summary": "<human-readable missing data summary>",
    "signal_summary": "<e.g. strong linear (max r=0.78) or weak linear, moderate nonlinear (MI)>",
    "issues": ["missing_values", "class_imbalance", ...],
    "complexity": "<SIMPLE|MODERATE|COMPLEX>",
    "complexity_reason": "<reason based on your analysis>"
  },
  "findings": ["<key finding 1>", "<key finding 2>", ...],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
  "fingerprint": { ... full fingerprint dict from compute_dataset_fingerprint ... },
  "eda_results": { ... raw tool outputs collected during analysis ... }
}
```

Populate `summary`, `findings`, and `recommendations` from the same data you use for the printed output below. This artifact enables `/gtd:train` to skip profiling when EDA has already been run.

---

## Output Format

After analysis, print a structured summary:

```
## EDA Summary

Data: {rows} x {cols} | {task_type} | Target: {target}
Features: {n_numeric} numeric, {n_categorical} categorical
Missing: {missing_summary}
Signal: {signal_summary — e.g., "strong linear (max r=0.78)" or "weak linear, moderate nonlinear (MI)"}
Issues: {issues_list}

### Complexity Assessment
{SIMPLE|MODERATE|COMPLEX} — {reason based on YOUR analysis, not thresholds}

### Key Findings
- {finding 1}
- {finding 2}
- ...

### Recommendations for Training
- {recommendation 1 — e.g., "temporal split recommended (date column detected)"}
- {recommendation 2 — e.g., "CatBoost/target encoding for high-cardinality categoricals"}
- ...

### Dataset Fingerprint
{fingerprint JSON from compute_dataset_fingerprint}
```

Call `compute_dataset_fingerprint` with accumulated findings before printing the summary.
