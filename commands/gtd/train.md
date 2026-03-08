---
description: Train and optimize ML models on a dataset
argument-hint: "path/to/data.csv [--target COL] [--time DURATION] [--target-metric METRIC>VALUE]"
---

# GTD: Train & Optimize

You are running the **Get Training Done** workflow. Your job is to profile the data, research approaches, train baselines, iteratively optimize, and export the best model — all using the gtd MCP tools.

## CRITICAL RULES

You MUST follow ALL of these. Violations are unacceptable:

1. **Use MCP tools ONLY**. Call `profile_dataset`, `train_model`, `evaluate_model`, etc. via the gtd-data, gtd-training, and gtd-research MCP servers. NEVER fall back to writing Python code in Bash. If an MCP tool fails, report the error — do NOT reimplement it in Python.
2. **Zero questions in Phases 3-5**. Questions are only allowed in Phase 1 (target confirmation) and Phase 2 (research opt-in, credential setup).
3. **All questions to the user MUST use the AskUserQuestion tool.** Never ask questions in plain text.
4. **Phase 1 confirmation**: Use AskUserQuestion — "Target: `{target}`, task: {task_type}. Correct?"
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
- `--time DURATION` (optional, default **"10m"**) — time budget for optimization. Formats: `"5m"`, `"30m"`, `"1h"`, `"1.5h"`
- `--target-metric METRIC>VALUE` (optional) — e.g. `accuracy>0.95` or `f1_macro>0.8`. Stop early if achieved

Parse `DURATION` into `TIME_BUDGET_SECONDS` (e.g., "5m" → 300, "1.5h" → 5400). Store these as variables for use throughout the workflow.

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

### Complexity Assessment

Based on profiling and correlation results, classify the problem:

**SIMPLE** — strong linear signal exists:
- Any feature-target |r| > 0.5, few features (< 20), clean data (< 5% missing), no severe class imbalance

**MODERATE** — nonlinear patterns likely needed:
- Moderate correlations (0.2 < |r| < 0.5), mix of feature types, some missing data or imbalance

**COMPLEX** — likely needs advanced approaches:
- Weak correlations (|r| < 0.2), high cardinality categoricals, many features (> 50), severe imbalance, complex interactions likely

Print: `Complexity: {SIMPLE|MODERATE|COMPLEX} — {one-line reason}`

Confirmation: Use AskUserQuestion — "Target: `{target}`, task: {task_type}. Correct?" (yes/no only)

**CONTEXT RULE**: From this point forward, reference ONLY the 2-line summary and complexity assessment above. Do NOT repeat or include raw profiling JSON from this phase in any subsequent message.

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

## Phase 3a: First Baseline + Experience Check

Print: `## Phase 3a: First Baseline`

Each `train_model` response includes `session_elapsed` — wall-clock seconds since the first training call. Use this directly for time display (no manual accumulation needed).

Print: `⏱ Time: 0s / {TIME_BUDGET_SECONDS}s remaining`

### Tier 1: Simplest baseline (always runs first)

1. Call `train_model` (gtd-training server) with the **simplest model** using `train_data_path` from Phase 1.5 — this creates the workspace. **Pass `memory_dir`** with your auto-memory directory path on this first call. This automatically checks for proven strategies from past sessions and includes recommendations in the response.
   - Classification → `logistic_regression`
   - Regression → `linear_regression`

Print: `#1 {Model} → {score} | ⏱ {elapsed} / {budget}`

### Experience Check

After the first `train_model` call, check if `strategy_recommendation` is present in the response. Each recommendation includes a `match_score` (1-7) indicating similarity to past datasets:

| Match Score | Label | Actions |
|-------------|-------|---------|
| **7** (task+size+feature) | Very similar | Skip Phase 2. Skip non-proven tree baseline in Phase 3b. Start Phase 4 with proven model+HPs. Patience = 2. |
| **5-6** (task+size OR task+feature) | Similar | Skip Phase 2. Run all baselines in Phase 3b. Use proven config as first Phase 4 attempt. Patience = 3. |
| **≤ 4** (task only) | Loosely related | Run Phase 2 normally. Use as hints only. Patience = 3. |
| **None** | No match | Run all phases normally. |

Use the highest `match_score` from the first recommendation (they are sorted best-first).

Print format:
- Score 7: `⚡ Strong match (score 7): {description} → {best}. Accelerating with proven strategy.`
- Score 5-6: `⚡ Moderate match (score {n}): {description}. Using as starting config.`
- Score ≤ 4: `📋 Weak match (score {n}): {description}. Using as hint only.`

If the first `train_model` response includes `prior_knowledge`, read it carefully. This contains synthesized insights from past sessions. Use these to inform your model selection and hyperparameter choices throughout Phase 4.

Store the experience check result (match score and recommendations) for use in Phase 2 and Phase 3b.

---

## Phase 2: Research (Conditional)

**If Experience Check score ≥ 5, skip this phase entirely.** Print: `Skipping research — strong experience match available.`

Otherwise, use AskUserQuestion: "Run external research (arXiv + Kaggle)? Kaggle requires API credentials. (yes/no)"

**If the user says no** (or any negative response): skip this phase entirely and proceed to Phase 3b.

**If the user says yes**:

Print: `## Phase 2: Research`
Print: `Searching for approaches...`

### Kaggle Credential Check

Before calling Kaggle, check if credentials exist by calling `search_kaggle_notebooks` with a test query.

**If Kaggle returns a credentials error** (the response contains `"error": "Kaggle credentials not found"` and a `"diagnosis"` field):

1. Print the `diagnosis` message so the user can see exactly what's wrong.
2. Use AskUserQuestion to ask:
   "Kaggle credentials not found. Would you like me to create ~/.kaggle/kaggle.json for you?

   You'll need your Kaggle username and API key from https://www.kaggle.com/settings → API → Create New Token.

   Alternatively, you can set env vars: `export KAGGLE_USERNAME=... KAGGLE_KEY=...`

   Options: (a) Create the file for me (b) Skip Kaggle research (c) I'll set it up myself"

3. **If (a)**: Use AskUserQuestion to ask for their Kaggle username, then their API key. Create `~/.kaggle/kaggle.json` with the Write tool containing `{"username": "...", "key": "..."}`. Retry the Kaggle search.
4. **If (b)**: Continue with arXiv only.
5. **If (c)**: Continue with arXiv only — Kaggle will work next session.

### Research Queries

1. Call `search_arxiv` (gtd-research server) with a query describing the dataset characteristics — no credentials needed
2. Call `search_kaggle_notebooks` with a query about similar datasets or problem types — requires Kaggle API credentials (skip if credentials unavailable per above)

If either call returns a non-credential error (network timeout, HTTP error), print the error on one line and continue with whatever results you got. Do NOT retry or block on research failures.

Print at most 3 compact bullets:

```
Research: (1) {model_families} dominate for {data_type} (2) {technique} for {issue} (3) {insight_from_kaggle}
```

**CONTEXT RULE**: From this point forward, reference ONLY the research summary above. Do NOT repeat or include full arXiv/Kaggle results from this phase in any subsequent message.

---

## Phase 3b: Remaining Baselines

Print: `## Phase 3b: Remaining Baselines`

### Tier 2: Tree-based baselines

Train tree-based models using the same workspace (always pass `train_data_path` as `data_path`):

- **If Experience Check score = 7** and the proven best model is tree-based: train ONLY that model family (e.g., if proven best is `xgboost`, skip `random_forest` and vice versa).
- **Otherwise**: Train both:
  - A random forest (`random_forest`) with defaults
  - Best boosting model (`xgboost` or `lightgbm`) with defaults

Print each with timer: `#2 {Model} → {score} | ⏱ {elapsed} / {budget}`

If the first `train_model` response included `strategy_recommendation`, use those strategies as starting points in Phase 4 instead of defaults.

### Evaluate tier positioning

After baselines, determine where to focus optimization:
- If Tier 1 score is within 2% of Tier 2 best → start Phase 4 at Tier 1 (optimize simple model)
- Otherwise → start Phase 4 at Tier 2 (optimize tree models)
- Phase 4 always escalates to next tier when current tier plateaus

**Experience-accelerated start for Phase 4**:
- **Score 7**: Start Phase 4 directly with proven model + HP config. Patience = 2.
- **Score 5-6**: Use proven config as the FIRST attempt in Phase 4. Patience = 3.

Print: `Baselines: {Model1} {score1} | {Model2} {score2} | {Model3} {score3}`
Print: `Starting optimization at Tier {1|2} — {reason}`

**CONTEXT RULE**: From this point forward, reference ONLY the baselines summary above. Do NOT repeat or include individual run JSONs from this phase in any subsequent message.

---

## Phase 4: Iterative Optimization

Print: `## Phase 4: Iterative Optimization`
Print: `⏱ Budget: {remaining} | Patience: 3 | Target: {target_metric if set, else "maximize"}`

At the start of Phase 4, call `list_available_models` to load the full hyperparameter spaces.

### Important: Phase 4 is Autonomous

Do NOT ask the user to choose between models or hyperparameters. Follow the decision protocol below and report results. The user should only see compact per-run output lines.

Every `train_model` response now includes `score_trajectory` (all runs so far) and `run_number`. Use this trajectory data to inform your decisions — no need to track it manually.

### Time Tracking (MANDATORY on every line)

After each `train_model` call, read time from the response:
- `ELAPSED_TIME = session_elapsed` (from the `train_model` response — wall-clock since first training call)
- `remaining` = TIME_BUDGET_SECONDS - ELAPSED_TIME
- `avg_run_time` = ELAPSED_TIME / runs_completed

Format elapsed/remaining as human-readable: e.g., 65s → "1m05s", 400s → "6m40s".

Per-run output format:
```
#{n} {Model} {change} → {score} ({delta}) ⏱ {elapsed}/{budget} [{remaining} left]
```

Examples:
```
#4 LogReg C=0.1 → 0.891 (+0.003) ★ new best ⏱ 1m05s/10m [8m55s left]
#7 XGBoost lr=0.05 → 0.912 (+0.008) ★ new best ⏱ 3m20s/10m [6m40s left]
#12 Stack(XGB+RF+LR) → 0.921 (+0.004) ★ new best ⏱ 7m15s/10m [2m45s left]
```

### Tier 1: Simple Model Optimization

Optimize the simple model (`logistic_regression` / `linear_regression` / `elasticnet`):
- Regularization tuning (C, alpha, l1_ratio)
- Feature selection (drop low-importance features)
- Feature engineering (interactions for key features found in correlations)

**Escalate to Tier 2 when**: patience exhausted (3 runs no improvement) OR complexity is COMPLEX

When escalating, print:
`→ Tier 1 plateau at {score}. Escalating to Tier 2 — tree-based optimization ⏱ {remaining} left`

### Tier 2: Tree-Based Optimization

Standard HPO for tree models (recommended tuning order):
1. `learning_rate` + `n_estimators` (most impactful — try 0.01, 0.05 with proportionally more `n_estimators`)
2. `max_depth` / `num_leaves` (controls complexity)
3. `subsample` + `colsample_bytree` (reduces overfitting)
4. `reg_alpha` + `reg_lambda` (L1/L2 regularization)

Try multiple model families: xgboost, lightgbm, catboost, random_forest, extra_trees.

**Escalate to Tier 3 when**: patience exhausted AND remaining time > avg_run_time * 3

When escalating, print:
`→ Tier 2 plateau at {score}. Escalating to Tier 3 — advanced approaches ⏱ {remaining} left`

### Tier 3: Advanced Approaches (research-driven)

This tier uses insights from Phase 2 research AND additional targeted research:

1. **Research-informed models**: If Phase 2 research found specific approaches
   (e.g., a Kaggle notebook using TabNet, a paper recommending CatBoost with
   specific settings for this data type), try those first.

2. **Stacking/Ensembling**: Build stacked ensembles combining the best models
   from Tiers 1 and 2. Use `train_model` with the best configs, then ensemble
   predictions via feature engineering + a meta-learner.

3. **Neural networks**: Try `mlp_classifier` / `mlp_regressor` with architecture
   search (hidden layer sizes, activation functions).

4. **Advanced feature engineering**: Based on error analysis insights —
   create interaction features, polynomial features, or domain-specific
   transforms that target weak segments identified by `analyze_errors`.

5. **Additional research**: If time remains, call `search_arxiv` or
   `search_kaggle_notebooks` with refined queries based on what you've
   learned about the dataset's challenges. Apply any novel techniques found.

**If time runs out during any tier**: Stop and proceed to Phase 5 with best result so far.

### Optimization Decision Protocol

After each training run within any tier, follow this structured protocol:

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

Per-run output — single line each, always including the timer:

```
#{n} {Model} {change_description} → {score} ({delta}) {significance_note} {best_marker} ⏱ {elapsed}/{budget} [{remaining} left]
```

When comparing runs, call `test_significance` and include significance in the line if p < 0.05.
After each run, call `analyze_errors` and include the top error segment if notably different from overall.

No growing table. Runs >5 old get summarized: `Runs 1-5: best #4 at 0.879`

### Stopping Criteria

Stop when any condition is met:

- **Time up**: estimated time for next run > remaining time
- **Patience exhausted at Tier 3**: No improvement after exploring advanced approaches
- **Target achieved**: Target metric exceeded

Print: `⏱ Stopping: {reason} | Total: {elapsed} | Runs: {n}`

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
Best: {model_type} {run_id} | {metric}={score}±{std} | {total_runs} runs in {elapsed}
Tier: {final_tier} | Complexity: {assessment}
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

**CONTEXT RULE**: Before starting Phase 5, summarize the entire optimization as: `Optimization: {n} runs in {elapsed}, {baseline} → {best} (+{delta}). Tier: {final_tier}. Key moves: {2-3 changes}`. From this point forward, reference ONLY that summary. Do NOT repeat individual run details from Phase 4.
