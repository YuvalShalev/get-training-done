---
description: Train and optimize ML models on a dataset
argument-hint: "path/to/data.csv [--target COL] [--time DURATION] [--target-metric METRIC>VALUE] [--deep yes/no] [--agents N]"
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
6. **Be concise**. One line per run result. Save extended reasoning for reflection checkpoints.
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
- `--deep yes/no` (optional, default **"no"**) — enable deep learning foundation models (TabICL, TabPFN). If yes, install missing dependencies automatically
- `--agents N` (optional, default **1**) — number of parallel training agents. When >1, uses `train_model_async` + `poll_training_jobs` to run N models concurrently per round

Parse `DURATION` into `TIME_BUDGET_SECONDS` (e.g., "5m" → 300, "1.5h" → 5400). Store these as variables for use throughout the workflow.

---

## Phase 1: Quick Profile & Split

Print: `## Phase 1: Quick Profile & Split`

### Load User Context

Check for a context artifact named `.gtd-context.json`. Search in this order:
1. The same directory as DATA_PATH
2. The current working directory (CWD)

Use the Read tool to check each location. Use the first one found.

**If found**:
- Parse the JSON and extract the `domain` and `keywords` fields
- Print: `Loading user context: {domain}`
- Store the `domain` and `keywords` for use in Phase 2 research queries

**If not found**: Continue without context. No message needed.

### Quick Profile

Call `profile_dataset` on DATA_PATH (with target column if provided via `--target`). This lightweight call returns: target column, task_type, column types, class balance, and basic statistics.

If the profile suggests temporal structure may be present, also call `detect_timestamp_columns` on DATA_PATH.

Extract from the profile:
- `target`, `task_type`, column types, class balance
- Whether temporal columns exist (for split strategy)

### Target Confirmation

Confirmation: Use AskUserQuestion — "Target: `{target}`, task: {task_type}. Correct?" (yes/no only)

### Data Split

Before any training or deep analysis, split the data into train and validation partitions.
The validation set is held out for ALL evaluation — it is NEVER used for training or EDA.

1. Choose a split strategy based on the quick profile. Consider temporal structure (prevents leakage), class balance (stratified preserves distribution), and group columns (prevents group leakage). Explain your choice in the print line.

2. Call `create_data_split` (gtd-data server)
3. Use `train_data_path` for all `train_model` calls (Phase 3-4)
4. Use `validation_data_path` for all evaluation/analysis calls, or omit `data_path` to auto-discover

Print: `Split: {strategy} | Train: {train_rows} rows | Validation: {val_rows} rows`

**Note on feature engineering**: If `engineer_features` is used later, it must be applied to train and validation CSVs separately using the same operations, after the split.

---

## Phase 1.5: EDA (Train Split)

Print: `## Phase 1.5: EDA (train split)`

EDA runs on the **train split only** to prevent information leakage from validation data into feature engineering or model selection decisions.

### Check for existing EDA results

Derive the artifact filename: `.gtd-eda-{data_filename_without_extension}.json` in the same directory as DATA_PATH.
Example: for `~/data/titanic.csv` → `~/data/.gtd-eda-titanic.json`

Use the Read tool to check if this file exists.

**If found**:
- Read the file and parse the JSON
- **Cache validation**: Check the `data_path` field in the artifact. Only use the cached results if `data_path` matches `train_data_path` (the train split), NOT the original DATA_PATH. If it points to the original full data path, discard the cache and re-run EDA on the train split.
- If cache is valid: Extract `task_type`, `target`, `summary` (complexity, signal, issues), `recommendations`, `fingerprint`
- Print: `Loading EDA results from prior analysis ({timestamp})...`
- Print the summary section as-is

**If not found or cache invalid**:
- Run EDA tools on `train_data_path` (NOT the original DATA_PATH):
  1. Call `profile_dataset` with `train_data_path` and target column
  2. Based on profile results, call additional EDA tools as needed (same adaptive logic as `/gtd:eda`), always passing `train_data_path`
  3. Call `compute_dataset_fingerprint` with accumulated findings as `eda_results`
- Build the EDA artifact JSON:
  ```json
  {
    "data_path": "<absolute path to TRAIN SPLIT file>",
    "target_column": "<target>",
    "task_type": "<detected task type>",
    "timestamp": "<current ISO 8601 timestamp>",
    "summary": {
      "rows": <n>, "cols": <n>,
      "n_numeric": <n>, "n_categorical": <n>,
      "missing_summary": "<e.g. Age 19.9%, Cabin 77.1%>",
      "signal_summary": "<e.g. weak linear, moderate nonlinear (MI)>",
      "issues": ["<issue1>", ...],
      "complexity": "<SIMPLE|MODERATE|COMPLEX>",
      "complexity_reason": "<reason>"
    },
    "findings": ["<finding1>", ...],
    "recommendations": ["<recommendation1>", ...],
    "fingerprint": { ... },
    "eda_results": { ... }
  }
  ```
- Write this JSON to `.gtd-eda-{data_filename_without_extension}.json` using the Write tool
- Print structured summary

Extract from the EDA output (whether loaded or freshly computed):
- Complexity, signal characteristics
- Dataset fingerprint (store for Phase 3a)

**CONTEXT RULE**: From this point forward, reference ONLY the EDA summary above. Do NOT repeat or include raw profiling JSON from this phase in any subsequent message.

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

After the first `train_model` call, check the response for:
- `past_strategies`: raw summaries of past sessions (dataset description, best model, score, insight, anti-pattern, hyperparameters). These are NOT pre-filtered or scored.
- `prior_knowledge`: synthesized high-level observations from past sessions.

Cross-reference these against the EDA findings from Phase 1.5. Consider whether past datasets had similar signal characteristics, complexity, and data quality — not just similar shape. Discard past strategies that are superficially similar but differ in signal strength, feature structure, or domain. Print a one-line summary of what you'll carry forward.

Store useful insights for Phase 2 (research decision), Phase 3b, and Phase 4.

---

## Phase 2: Research (Conditional)

Decide whether external research would add value given what you know from EDA and prior experience. Strong prior knowledge may make research redundant. If you decide to skip, print the reason. Otherwise, use AskUserQuestion: "Run external research (arXiv + Kaggle)? Kaggle requires API credentials. (yes/no)"

**If the user says no** (or any negative response): skip this phase entirely and proceed to Phase 3b.

**If the user says yes**:

Print: `## Phase 2: Research`
Print: `Searching for approaches...`

### Unified Research Call

Use the `research_and_extract` tool (gtd-research server) for a single consolidated call that searches and extracts structured insights. This replaces separate `search_arxiv` + `search_kaggle_notebooks` calls and returns ~200 tokens instead of ~2000.

When constructing the query, incorporate user context if loaded in Phase 1. Example: without context → `"tabular classification 12 features"`, with context → `"vehicle health prediction sensor data classification"`.

Pass `dataset_profile_json` with `{"n_rows": ..., "n_cols": ..., "n_numeric": ..., "n_categorical": ...}` from the EDA profile for context-aware model recommendations.

```
research_and_extract(
    query="...",
    task_type="binary_classification",
    dataset_profile_json='{"n_rows": 1000, "n_cols": 12, ...}',
    sources="arxiv,kaggle",
)
```

If Kaggle returns a credentials error in the response, retry with `sources="arxiv"` only.

**Store the returned `recommended_models` and `feature_tips` for Phases 3b and 4.**

Also save the research insights to the workspace for `train_model` to surface them:
- Write the insights JSON to `{workspace_path}/research_insights.json` using the Write tool.

Print at most 3 compact bullets from the `summary` field.

**CONTEXT RULE**: From this point forward, reference ONLY the research summary above. Do NOT repeat or include full research results from this phase in any subsequent message.

---

## Phase 3b: Remaining Baselines

Print: `## Phase 3b: Remaining Baselines`

### Baselines

Train 1-3 baseline models to establish benchmarks. Choose based on dataset characteristics, prior experience, EDA complexity, and research insights. Available levels: simple (logistic/linear), tree-based (RF, XGBoost, LightGBM, CatBoost), advanced (MLP, TabPFN, TabICL). Don't waste runs on models that prior knowledge flags as poor fits.

- If research recommended specific models, include one in baselines
- If `--deep yes`:
  - **Find the plugin root first**: Run `dirname $(dirname $(which gtd 2>/dev/null || find ~/.claude -name "install-deep.sh" -path "*/get-training-done/*" 2>/dev/null | head -1))` or simply search for the install script: `find ~/.claude -name "install-deep.sh" -path "*get-training-done*" 2>/dev/null | head -1`. Save the plugin root path (the directory containing `scripts/install-deep.sh`) as `PLUGIN_ROOT`.
  - **Install TabICL**: The MCP servers run in the **plugin's own venv**, not the user's project. Run:
    `bash $PLUGIN_ROOT/scripts/install-deep.sh`
    The script handles Python 3.12 setup, venv rebuild, tabicl install, and triggers MCP server restart automatically. After it finishes, wait 10 seconds for servers to reload, then verify by calling `get_session_time` on the gtd-training server — any successful response confirms the server restarted with the new packages. Only then proceed with TabICL training.
  - If TabPFN is also applicable (classification, <10k rows, <100 features), install it too:
    `$PLUGIN_ROOT/.venv/bin/pip install tabpfn`
  - **Check device**: After install, run `$PLUGIN_ROOT/.venv/bin/python -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')"` to detect the available accelerator. Print the result (e.g., `Deep learning device: mps`). On Apple Silicon (M1-M5), MPS is available and TabICL will use it automatically. CPU-only is fine for small datasets but will be slow on large ones
  - Include TabICL as a baseline (all task types, up to ~100k rows). TabICL is a foundation model that often matches tuned tree models out of the box
  - Include TabPFN as a baseline (classification only, <10k rows, <100 features)
  - **Run in parallel**: Deep learning models can be slow. Send MULTIPLE `train_model` calls in the SAME message — one for TabICL and one for a tree-based model (e.g., LightGBM). The MCP server handles concurrent requests. Don't wait for TabICL to finish before starting tree models. Use `get_training_progress` on the gtd-training server to check if a long-running model is still progressing (shows current fold, score, elapsed time)
- If `--deep no` (default): skip TabICL and TabPFN

Always pass `train_data_path` as `data_path`.

Print each with timer: `#2 {Model} → {score} | ⏱ {elapsed} / {budget}`

### Optimization Focus

Based on baseline results, decide which model family to optimize. Consider relative scores, prior experience, and dataset complexity.

Print: `Baselines: {Model1} {score1} | {Model2} {score2} | {Model3} {score3}`
Print: `Optimization focus: {model} — {reason}`

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

If you need elapsed time between training runs (e.g., during feature engineering or error analysis), call `get_session_time` on the gtd-training server.

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

### Optimization Protocol

You are a data scientist optimizing a model. You have the full context: EDA findings,
prior experience, research insights, baseline scores, and the `score_trajectory` from
each `train_model` response.

After each run:

1. **Analyze**: Call `analyze_errors` on the current best run. Read the results,
   error analysis, and score trajectory. What is working? What is not?
2. **Decide**: Choose your next move — hyperparameter tuning, model switch, feature
   engineering, regularization, ensemble/stacking, or research-informed techniques.
3. **Act**: Make 1-2 changes per run. More changes make it hard to attribute improvements.
4. **Report**: Print the result in the standard per-run format.

When comparing runs, call `test_significance` and note significance if p < 0.05.

#### Guidance (not rules)

These are patterns experienced data scientists use. They are not prescriptions:
- High CV variance often signals overfitting — consider regularization or simpler models
- Plateaus after several runs may mean the model family is exhausted — try a different
  family or feature engineering
- Error analysis revealing specific weak segments suggests targeted feature engineering
- Prior experience that strongly matches this dataset deserves significant trust
- Ensemble approaches are most valuable when diverse base models exist
- Diminishing returns in one approach suggest pivoting strategy, not stopping — try a different model family, feature engineering, or ensembles

#### Research-Driven Decisions

When research insights are available (from `train_model` response `research_hints` or stored from Phase 2):
- Prioritize model families that research recommends
- Try HP ranges suggested by competition solutions
- Apply feature engineering techniques from winning notebooks

#### Ensemble Strategy (after run 8+ or score plateau)

When you have 3+ trained models with diverse architectures:
1. Try `train_ensemble` with strategy="stacking" using top 2-3 models
2. If improvement < 0.2%, try strategy="hill_climbing" with all runs
3. Consider strategy="seed_ensemble" on best single model (3-5 seeds)

Ensemble should be the LAST optimization step.

#### Stopping

You MUST keep optimizing until the time budget is nearly exhausted. Only stop early if:
- The target metric (if set) has been achieved
- Remaining time < avg_run_time (not enough time for another run)

If the current approach has plateaued but time remains:
- Switch model family (e.g., tree-based → linear, or vice versa)
- Try feature engineering (interactions, binning, target encoding)
- Attempt ensembles/stacking with diverse base models
- Revisit research hints you haven't tried yet

A plateau in one model family is NOT a reason to stop — it's a reason to pivot.

Print: `Stopping: {reason} | Total: {elapsed} | Runs: {n}`

#### Parallel Optimization Protocol (when --agents > 1)

When `--agents N` is set to more than 1, Phase 4 runs in **synchronized rounds** instead of sequential runs. Use `train_model_async` and `poll_training_jobs` on the gtd-training server.

**Round structure:**

1. **Plan**: Based on current results, assign one model+config to each of the N agent slots.
   - Round 1: Spread across model families (e.g., slot 1=LightGBM, slot 2=XGBoost, slot 3=CatBoost)
   - Later rounds: Focus on promising families with different HPs, or try feature engineering variants

2. **Execute**: Call `train_model_async` N times — one per agent slot. These run concurrently in background threads.
   Print: `Round {r}: Launching {N} parallel jobs...`

3. **Wait**: Call `poll_training_jobs` repeatedly (every 15 seconds) until all jobs show status "completed" or "failed". Print progress as jobs finish:
   `Round {r}: {completed}/{N} done — {model} finished with {score}`

4. **Discuss**: Analyze ALL round results collectively:
   - Which model improved most vs previous round?
   - Which approach plateaued? (3+ rounds without improvement → reassign that slot)
   - Any slot's model clearly worse? → Switch it to untried family or ensemble
   - Share insights across slots: if LightGBM found a good HP range, try similar on XGBoost
   Print: `Round {r} results: [{slot1}: {model} {score}] [{slot2}: {model} {score}] ...`
   Print: `Discussion: {key insight} → Next round: {assignments}`

5. **Reflect**: Every 3 rounds (not every 3 individual runs), do the standard reflection checkpoint (analyze_run_deep + save_observation)

6. **Stop**: Same criteria — time budget nearly exhausted or target met

When `--agents 1` (default), use the standard sequential `train_model` protocol below.

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
