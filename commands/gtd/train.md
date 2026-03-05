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

## Self-Learning (Automatic)

Learning happens automatically via two parameters — no extra tool calls needed:

- **`memory_dir`**: Pass your auto-memory directory path (shown in your system prompt under "persistent auto memory directory at") to `train_model` on the FIRST call. This persists it in the workspace so `export_model` auto-discovers it — no need to pass it again.
- `train_model` with `memory_dir` automatically: stores a dataset fingerprint, surfaces strategy recommendations from past sessions, and includes score trajectory in every response.
- `export_model` automatically discovers `memory_dir` from the workspace (set by `train_model`) and saves learnings to `gtd-learnings.md`, updates `gtd-strategy-library.md`, and records session metrics to `gtd-meta-scores.jsonl`.

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

## Phase 2: Research

Print: `## Phase 2: Research`
Print: `Searching for approaches...`

1. Call `search_arxiv` (gtd-research server) with a query describing the dataset characteristics
2. Call `search_kaggle_notebooks` with a query about similar datasets or problem types

Print at most 3 compact bullets:

```
Research: (1) {model_families} dominate for {data_type} (2) {technique} for {issue} (3) {insight_from_kaggle}
```

**CONTEXT RULE**: From this point forward, reference ONLY the research summary above. Do NOT repeat or include full arXiv/Kaggle results from this phase in any subsequent message.

---

## Phase 3: Baseline Models

Print: `## Phase 3: Baseline Models`
Print: `Training 3 baselines...`

1. Call `train_model` (gtd-training server) with the **first baseline** — this creates the workspace. **Pass `memory_dir`** with your auto-memory directory path on this first call. This automatically checks for proven strategies from past sessions and includes recommendations in the response.
2. Train 2 more models using the same workspace:
   - A gradient boosting model (e.g., `xgboost` or `lightgbm`)
   - A random forest (`random_forest`)
   - A simple model (`logistic_regression` for classification, `linear_regression` for regression)

If the first `train_model` response includes `strategy_recommendation`, use those strategies as starting points in Phase 4 instead of defaults.

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

After each run, call `analyze_errors` with the current best run to understand where the model fails. Use this to guide the next action:
- If error analysis shows a specific segment with high error rate → suggest feature engineering or model change targeting that segment
- If improvement is not statistically significant (call `test_significance` with CV scores) → don't count it toward patience, keep exploring

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

---

## Phase 5: Export & Report

Print: `## Phase 5: Export & Report`

1. Identify the best run based on the primary metric
2. Call `export_model` (gtd-training server) with the best run ID. It auto-discovers `memory_dir` from the workspace, so learnings are saved automatically.
3. Call `evaluate_model` for final comprehensive metrics
4. Call `get_feature_importance` on the best run
5. Call `get_roc_curve` (for binary classification) and `get_pr_curve` (for classification tasks)
6. Call `register_model` (gtd-training server) to add this training session to the `.gtd-state.json` registry

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
