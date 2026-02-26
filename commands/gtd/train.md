---
description: Train and optimize ML models on a dataset
argument-hint: "path/to/data.csv [--target COL] [--max-runs N] [--target-metric METRIC>VALUE]"
---

# GTD: Train & Optimize

You are running the **Get Training Done** workflow. Your job is to profile the data, research approaches, train baselines, iteratively optimize, and export the best model — all using the gtd MCP tools.

## Argument Parsing

The user's arguments are: `$ARGUMENTS`

Parse from the argument string:

- `DATA_PATH` (required) — path to the CSV file
- `--target COL` (optional) — target column name. If omitted, auto-detect during profiling
- `--max-runs N` (optional, default **20**) — total training run budget (baselines + optimization)
- `--target-metric METRIC>VALUE` (optional) — e.g. `accuracy>0.95` or `f1_macro>0.8`. Stop early if achieved

Store these as variables for use throughout the workflow.

---

## Phase 1: Data Understanding

Print: `## Phase 1: Data Understanding`
Print: `Profiling dataset... (usually takes ~30 seconds)`

1. Call `profile_dataset` (gtd-data server) with the data path to get a comprehensive overview
2. Call `detect_data_issues` to identify class imbalance, leakage, multicollinearity, high cardinality
3. Call `compute_correlations` to understand feature-target and feature-feature relationships

Print a summary table:

```
| Property          | Value                    |
|-------------------|--------------------------|
| Rows              | 10,000                   |
| Columns           | 15                       |
| Task type         | Binary Classification    |
| Target column     | churn                    |
| Numeric features  | 10                       |
| Categorical       | 4                        |
| Missing values    | Age (12%), Income (3%)   |
| Class balance     | 70/30 (mild imbalance)   |
| Issues found      | 1 leakage suspect        |
```

Ask user to confirm target column and task type before proceeding.

---

## Phase 2: Research

Print: `## Phase 2: Research`
Print: `Searching for approaches that work on similar data...`

1. Call `search_arxiv` (gtd-research server) with a query describing the dataset characteristics (e.g., "tabular classification with imbalanced classes")
2. Call `search_kaggle_notebooks` with a query about similar datasets or problem types

Print 2-3 bullet points summarizing recommended approaches and why.

---

## Phase 3: Baseline Models

Print: `## Phase 3: Baseline Models`
Print: `Training 3 diverse baselines with default hyperparameters...`

1. Call `train_model` (gtd-training server) with the **first baseline** — this creates the workspace. Use the returned `workspace_path` for all subsequent calls
2. Train 2 more models using the same workspace:
   - A gradient boosting model (e.g., `xgboost` or `lightgbm`)
   - A random forest (`random_forest`)
   - A simple model (`logistic_regression` for classification, `linear_regression` for regression)

After each run, print a one-line summary. After all 3, print:

```
| #  | Model               | Score (CV)     | Time   |
|----|---------------------|----------------|--------|
| 1  | XGBoost             | 0.872 +/- 0.023| 4.2s   |
| 2  | Random Forest       | 0.861 +/- 0.018| 2.1s   |
| 3  | Logistic Regression | 0.834 +/- 0.015| 0.3s   |
```

Call `compare_runs` and report: "Best baseline: XGBoost (accuracy 0.872). Moving to optimization."

---

## Phase 4: Iterative Optimization

Print: `## Phase 4: Iterative Optimization`
Print: `Budget: {max_runs - 3} remaining runs | Patience: 3 runs without improvement | Target: {target_metric if set, else "maximize primary metric"}`

At the start of Phase 4, call `list_available_models` to load the full hyperparameter spaces (ranges, defaults, scales) for reference.

### Optimization Decision Protocol

After each training run, follow this structured protocol to decide the next configuration:

#### Step 1: Diagnose

Examine the results from `train_model`:
- `cv_scores` array: compute std. If std > 5% of mean -> **overfitting signal**
- `mean_score` vs best previous run: delta > +0.5% -> **improving**, |delta| < 0.5% -> **plateau**, delta < -1% -> **degrading**
- If best score is still close to baseline after 3+ optimization runs -> **model family may be wrong**

#### Step 2: Classify & Act

Based on diagnosis, pick ONE action from this menu:

| Diagnosis | Action | Parameter Changes |
|-----------|--------|-------------------|
| **Overfitting** (high CV variance, train >> val) | Increase regularization | Boosting: increase reg_alpha/reg_lambda (2-5x), decrease max_depth (-2), increase min_child_weight (+5). RF: increase min_samples_leaf (+5), decrease max_depth (-3). Neural: increase alpha (5x) |
| **Underfitting** (low score, low CV variance) | Increase capacity | Boosting: increase n_estimators (+50%), increase max_depth (+2), decrease learning_rate (divide by 2). RF: increase n_estimators (+100), increase max_depth (+5). Neural: larger hidden layers |
| **Improving** (delta > +0.5%) | Continue same direction | Push the same parameter further in the same direction (moderate step: 1.5-2x) |
| **Plateau on current model** (2 runs, no improvement) | Switch strategy | Try one of: (a) different model family, (b) `engineer_features` to add interactions/transforms, (c) big parameter jump (not small tweak) |
| **Degrading** (score dropped significantly) | Revert and try different axis | Go back to best config, change a DIFFERENT parameter than the one that caused degradation |

#### Step 3: Apply

For each parameter change:
- First change: moderate step (2x increase or divide-by-2 decrease)
- If moderate didn't help: aggressive step (5-10x or try boundary values from the hyperparameter space)
- Use the hyperparameter ranges from `list_available_models` to stay within valid bounds
- Change at most 1-2 parameters per run (to attribute improvement correctly)

#### Step 4: Log

After each run, print one-line update and keep a running table:

```
| #  | Model     | Change              | Score      | Delta vs best | Status      |
|----|-----------|---------------------|------------|---------------|-------------|
| 4  | XGBoost   | lower learning_rate | 0.879      | +0.007        | New best    |
| 5  | XGBoost   | higher max_depth    | 0.876      | -0.003        |             |
| 6  | LightGBM  | Try alternative     | 0.884      | +0.005        | New best    |
| 7  | LightGBM  | lower learning_rate | 0.885      | +0.001        | New best    |
| 8  | LightGBM  | Feature engineering | 0.886      | +0.001        | New best    |
| 9  | LightGBM  | higher num_leaves   | 0.884      | -0.002        |             |
| 10 | LightGBM  | more regularization | 0.885      | -0.001        |             |
```

### Recommended Tuning Order (for boosting models)

When optimizing a gradient boosting model, tune parameters in this priority order:
1. `learning_rate` (most impactful — try lower values like 0.01, 0.05 with proportionally more `n_estimators`)
2. `max_depth` / `num_leaves` (controls complexity)
3. `subsample` + `colsample_bytree` (reduces overfitting)
4. `reg_alpha` + `reg_lambda` (L1/L2 regularization)
5. `min_child_weight` / `min_child_samples` (leaf-level regularization)

### Stopping Criteria

Stop optimization when any of these conditions is met, and print the reason:

- **Patience exhausted**: "Stopped: No improvement >0.5% for 3 consecutive runs (patience exhausted)"
- **Target achieved**: "Stopped: Target metric achieved (accuracy 0.952 > 0.95)"
- **Budget exhausted**: "Stopped: Budget exhausted ({max_runs}/{max_runs} runs used)"

---

## Phase 5: Export & Report

Print: `## Phase 5: Export & Report`

1. Identify the best run based on the primary metric
2. Call `export_model` (gtd-training server) with the best run ID
3. Call `evaluate_model` for final comprehensive metrics
4. Call `get_feature_importance` on the best run
5. Call `get_roc_curve` (for binary classification) and `get_pr_curve` (for classification tasks)
6. Call `register_model` (gtd-training server) to add this training session to the `.gtd-state.json` registry. Pass all required fields: workspace_path, best_run_id, best_score, primary_metric, model_type, task_type, target_column, data_path (the original DATA_PATH), export_path (from export_model result), and total_runs

Print the final report:

```
## Results

**Best Model**: LightGBM (run_008_lightgbm)
**Registered as**: Model #1 (current default)
**Score**: accuracy = 0.886 +/- 0.012 (5-fold CV)
**Training runs**: 10 total (3 baselines + 7 optimization)

### Hyperparameters
| Parameter      | Value |
|----------------|-------|
| n_estimators   | 500   |
| learning_rate  | 0.05  |
| num_leaves     | 31    |
| ...            | ...   |

### Top Features
| Feature         | Importance |
|-----------------|------------|
| tenure          | 0.234      |
| MonthlyCharges  | 0.189      |
| ...             | ...        |

### Files
- Model: <export_path>/model.joblib
- Metadata: <export_path>/metadata.json
- Workspace: <workspace_path>

### Next Steps
- Predict on new data: `/gtd:inference path/to/test.csv`
- Evaluate on labeled data: `/gtd:evaluate path/to/labeled.csv`
- List all trained models: `/gtd:models`
```
