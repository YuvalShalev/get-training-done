---
description: Train and optimize ML models on a dataset
argument-hint: "path/to/data.csv [--target COL] [--max-runs N] [--target-metric METRIC>VALUE]"
---

# GTD: Train & Optimize

You are running the **Get Training Done** workflow. Your job is to profile the data, research approaches, train baselines, iteratively optimize, and export the best model — all using the gtd MCP tools.

## CRITICAL RULES

You MUST follow ALL of these. Violations are unacceptable:

1. **Use MCP tools ONLY**. Call `profile_dataset`, `train_model`, `evaluate_model`, etc. via the gtd-data, gtd-training, and gtd-research MCP servers. NEVER fall back to writing Python code in Bash. If an MCP tool fails, report the error — do NOT reimplement it in Python.
2. **Zero questions in Phases 2-5**. The only question is Phase 1 target confirmation.
3. **Phase 1 confirmation**: Ask in plain text — "Target: `{target}`, task: {task_type}. Correct?" Do NOT use the AskUserQuestion tool.
4. **Never use AskUserQuestion tool** during this workflow. All communication is plain text.
5. **Compact output**: Single-line formats. No tables unless the user asks for one.
6. **No reasoning before decisions**. Act, then report the result in one line.
7. **Session synthesis is mandatory**. You MUST call `synthesize_session` before `register_model` in Phase 5. `register_model` will reject if you skip this.

## Self-Learning

Learning happens at three levels:

- **Automatic**: `train_model` with `memory_dir` stores dataset fingerprints and surfaces past strategy recommendations. `export_model` auto-discovers `memory_dir` and saves learnings, updates strategy library, and records session metrics.
- **Within-run reflection**: Call `save_observation` every 3 optimization runs in Phase 4 to record what's working and what to try next. Future sessions can query these via `load_observations`.
- **Passive extraction**: At session end, transcript-level pattern detectors extract model insights, HP discoveries, and data handling patterns into `~/.claude/rules/learned-patterns.md` (automatic, no action needed).
- **Long-term knowledge**: `train_model` (first call) returns `prior_knowledge` from past sessions. `synthesize_session` saves your insights to `high-level-observations.md` and archives the observation log.

**`memory_dir`**: Pass your auto-memory directory path to `train_model` on the FIRST call. This persists it so `export_model` auto-discovers it later.

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
Print: `Profiling dataset...`

1. Call `profile_dataset` (gtd-data server) with the data path
2. Call `detect_data_issues` to identify class imbalance, leakage, multicollinearity, high cardinality
3. Call `compute_correlations` to understand feature-target and feature-feature relationships

Print a compact 2-line summary:

```
Data: {rows} rows x {cols} cols | {task_type} | Target: {target}
Features: {n_numeric} numeric, {n_categorical} categorical | Missing: {missing_summary} | Issues: {issues_summary}
```

Confirmation: "Target: `{target}`, task: {task_type}. Correct?" (yes/no only)

**CONTEXT RULE**: From this point forward, reference ONLY the 2-line summary above. Do NOT repeat or include raw profiling JSON from this phase in any subsequent message.

---

## Phase 1.5: Data Partitioning

Print: `## Phase 1.5: Data Partitioning`

Before any training, split the data into train and validation partitions.
The validation set is held out for ALL evaluation — it is NEVER used for training.

1. Decide the split strategy based on Phase 1 profiling results:
   - If data has a date/timestamp column → `strategy="temporal"` (prevents temporal leakage)
   - If classification with class imbalance → `strategy="stratified"` (preserves class distribution)
   - If data has a group/ID column (e.g., customer_id, patient_id) → `strategy="group"` (prevents group leakage)
   - Regression without special structure → `strategy="random"`
   - Default → `strategy="stratified"` for classification, `"random"` for regression

2. Call `create_data_split` (gtd-data server)
3. Use `train_data_path` for all `train_model` calls (Phase 3-4)
4. Use `validation_data_path` for all evaluation/analysis calls, or omit `data_path` to auto-discover

Print: `Split: {strategy} | Train: {train_rows} rows | Validation: {val_rows} rows`

**Note on feature engineering**: If `engineer_features` is used later, it must be applied to train and validation CSVs separately using the same operations, after the split.

---

## Phase 2: Research (Optional)

Ask the user in plain text: "Run external research (arXiv + Kaggle)? (yes/no)"

**If the user says no** (or any negative response): skip this phase entirely and proceed to Phase 3.

**If the user says yes**:

Print: `## Phase 2: Research`
Print: `Searching for approaches...`

1. Call `search_arxiv` (gtd-research server) with a query describing the dataset characteristics
2. Call `search_kaggle_notebooks` with a query about similar datasets or problem types

If either call returns an error (e.g., missing Kaggle credentials, network timeout), print the error on one line and continue. Do NOT retry or block on research failures.

**Kaggle setup**: If the Kaggle call fails with a credentials error, tell the user: "To enable Kaggle research, set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`, or create `~/.kaggle/kaggle.json` (download from https://www.kaggle.com/settings → API → Create New Token)"

Print at most 3 compact bullets:

```
Research: (1) {model_families} dominate for {data_type} (2) {technique} for {issue} (3) {insight_from_kaggle}
```

**CONTEXT RULE**: From this point forward, reference ONLY the research summary above. Do NOT repeat or include full arXiv/Kaggle results from this phase in any subsequent message.

---

## Phase 3: Baseline Models

Print: `## Phase 3: Baseline Models`
Print: `Training 3 baselines...`

1. Call `train_model` (gtd-training server) with the **first baseline** using `train_data_path` from Phase 1.5 — this creates the workspace. **Pass `memory_dir`** with your auto-memory directory path on this first call. This automatically checks for proven strategies from past sessions and includes recommendations in the response.
2. Train 2 more models using the same workspace (always pass `train_data_path` as `data_path`):
   - A gradient boosting model (e.g., `xgboost` or `lightgbm`)
   - A random forest (`random_forest`)
   - A simple model (`logistic_regression` for classification, `linear_regression` for regression)

If the first `train_model` response includes `strategy_recommendation`, use those strategies as starting points in Phase 4 instead of defaults.

If the first `train_model` response includes `prior_knowledge`, read it carefully. This contains synthesized insights from past sessions. Use these to inform your model selection and hyperparameter choices throughout Phase 4.

After all 3, print a compact one-line summary:

```
Baselines: {Model1} {score1} | {Model2} {score2} | {Model3} {score3} → Best: {best_model}
```

**CONTEXT RULE**: From this point forward, reference ONLY the one-line baselines summary above. Do NOT repeat or include individual run JSONs from this phase in any subsequent message.

---

## Phase 4: Iterative Optimization

Print: `## Phase 4: Iterative Optimization`
Print: `Budget: {max_runs - 3} remaining | Patience: 3 | Target: {target_metric if set, else "maximize"}`

At the start of Phase 4, call `list_available_models` to load the full hyperparameter spaces.

### Important: Phase 4 is Autonomous

Do NOT ask the user to choose between models or hyperparameters. Follow the decision protocol below and report results. The user should only see compact per-run output lines.

Every `train_model` response now includes `score_trajectory` (all runs so far) and `run_number`. Use this trajectory data to inform your decisions — no need to track it manually.

### Optimization Decision Protocol

After each training run, follow this structured protocol:

#### Step 0: Error-Informed Decision

After each run, call `analyze_errors` with the current best run to understand where the model fails (omit `data_path` to auto-use the validation partition). Use this to guide the next action:
- If error analysis shows a specific segment with high error rate → suggest feature engineering or model change targeting that segment
- If improvement is not statistically significant (call `test_significance` with CV scores) → don't count it toward patience, keep exploring

If deep analysis insights are available from the last reflexion (Step 5), use the `top_recommendation` to prioritize your next action.

#### Step 1: Diagnose

Examine the results from `train_model`:
- `cv_scores` array: compute std. If std > 5% of mean → **overfitting signal**
- `mean_score` vs best previous run: delta > +0.5% → **improving**, |delta| < 0.5% → **plateau**, delta < -1% → **degrading**
- If best score is still close to baseline after 3+ optimization runs → **model family may be wrong**

Use `score_trajectory` from the response to identify trends without needing separate calls.

#### Step 2: Classify & Act

Based on diagnosis, pick ONE action:

| Diagnosis | Action | Parameter Changes |
|-----------|--------|-------------------|
| **Overfitting** | Increase regularization | Boosting: reg_alpha/reg_lambda 2-5x, max_depth -2, min_child_weight +5. RF: min_samples_leaf +5, max_depth -3 |
| **Underfitting** | Increase capacity | Boosting: n_estimators +50%, max_depth +2, learning_rate /2. RF: n_estimators +100, max_depth +5 |
| **Improving** | Continue same direction | Push same parameter further (1.5-2x step) |
| **Plateau** (2 runs) | Switch strategy | Different model family, feature engineering, or big parameter jump |
| **Degrading** | Revert and try different axis | Return to best config, change a DIFFERENT parameter |

#### Step 3: Apply

- First change: moderate step (2x or /2)
- If moderate didn't help: aggressive step (5-10x or boundary values)
- Use hyperparameter ranges from `list_available_models`
- Change at most 1-2 parameters per run

#### Step 4: Log

Per-run output — single line each:

```
#{n} {Model} {change_description} → {score} ({delta}) {significance_note} {best_marker}
```

Examples:
```
#4 XGBoost lr=0.05 → 0.879 (+0.007) ★ new best
#5 XGBoost depth=8 → 0.876 (-0.003)
#6 LightGBM default → 0.884 (+0.012, p=0.02*) ★ new best
#7 LightGBM lr=0.02 → 0.885 (+0.001) | errors: Age>60 (32% vs 12%)
```

When comparing runs, call `test_significance` and include significance in the line if p < 0.05.
After each run, call `analyze_errors` and include the top error segment if notably different from overall.

No growing table. Runs >5 old get summarized: `Runs 1-5: best #4 at 0.879`

### Recommended Tuning Order (for boosting models)

1. `learning_rate` (most impactful — try 0.01, 0.05 with proportionally more `n_estimators`)
2. `max_depth` / `num_leaves` (controls complexity)
3. `subsample` + `colsample_bytree` (reduces overfitting)
4. `reg_alpha` + `reg_lambda` (L1/L2 regularization)
5. `min_child_weight` / `min_child_samples` (leaf-level regularization)

### Stopping Criteria

Stop when any condition is met:

- **Patience exhausted**: No improvement >0.5% for 3 consecutive runs
- **Target achieved**: Target metric exceeded
- **Budget exhausted**: All runs used

Print the reason on one line.

**CONTEXT RULE (every 3 runs)**: Compact runs older than the 3 most recent into a single line: `Runs {start}-{end}: best was #{n} at {score} ({model})`. From this point forward, reference ONLY that summary for older runs. Do NOT repeat their individual run details.

#### Step 5: Reflect (every 3 optimization runs)

Every 3 optimization runs:

1. Call `analyze_run_deep` (gtd-training server) on the current best run (omit `data_path` to auto-use the validation partition) to get ranked insights about model weaknesses, overconfident predictions, and weak subpopulations.

2. Call `save_observation` (gtd-training server) with:
   - `workspace_path`: current workspace
   - `run_number`: current run number
   - `score_trajectory`: the trajectory from the last `train_model` response
   - `actions_taken`: list of changes made in the last 3 runs (e.g., `["lr=0.05", "depth=8", "switched to LightGBM"]`)
   - `diagnosis`: include the top 3 insight descriptions from `analyze_run_deep` alongside your own analysis
   - `next_strategy`: use the `top_recommendation` from deep analysis to inform what to try next

This is mandatory. Do not skip it.

---

## Phase 5: Export & Report

Print: `## Phase 5: Export & Report`

1. Identify the best run based on the primary metric
1.5. Call `load_observations` to retrieve within-run reflections from Phase 4. Use these to enrich the optimization summary with key strategy shifts and diagnoses.
2. Call `export_model` (gtd-training server) with the best run ID. It auto-discovers `memory_dir` from the workspace, so learnings are saved automatically.
3. Call `evaluate_model` for final comprehensive metrics (omit `data_path` to auto-use the validation partition)
4. Call `get_feature_importance` on the best run (omit `data_path` to auto-use the validation partition)
5. Call `get_roc_curve` (for binary classification) and `get_pr_curve` (for classification tasks) — omit `data_path` to auto-use the validation partition
6. Call `synthesize_session` (gtd-training server) with:
   - `workspace_path`: current workspace
   - `dataset_name`: the dataset filename
   - `task_type`: the detected task type
   - `synthesis`: Write a concise paragraph (3-5 sentences) synthesizing **general knowledge** gained. Focus on: which model families worked and why, effective hyperparameter ranges, strategies that failed, and data-specific insights. Extract transferable patterns, not a run log summary.
7. Call `register_model` (gtd-training server) to add this training session to the `.gtd-state.json` registry

Print a compact final report:

```
Best: {model_type} {run_id} | {metric}={score}±{std} | {total_runs} runs ({n_baselines}+{n_opt})
Saved: {export_path}/model.joblib
Top features: {feat1} ({imp1}), {feat2} ({imp2}), {feat3} ({imp3})
```

Then print expanded details:

```
### Hyperparameters
| Parameter | Value |
|-----------|-------|
| ...       | ...   |

### Files
- Model: <export_path>/model.joblib
- Metadata: <export_path>/metadata.json
- Workspace: <workspace_path>

### Next Steps
- Predict: `/gtd:inference path/to/test.csv`
- Evaluate: `/gtd:evaluate path/to/labeled.csv`
- List models: `/gtd:models`
```

**CONTEXT RULE**: Before starting Phase 5, summarize the entire optimization as: `Optimization: {n} runs, {baseline} → {best} (+{delta}). Key moves: {2-3 changes}`. From this point forward, reference ONLY that summary. Do NOT repeat individual run details from Phase 4.
