---
description: Evaluate model performance on labeled data
argument-hint: "path/to/labeled.csv [--model N]"
---

# GTD: Evaluate

You are evaluating a trained model on labeled data. The argument passed to this command is the path to a labeled CSV file that must contain the target column used during training.

## Argument Parsing

The user's arguments are: `$ARGUMENTS`

Parse from the argument string:

- `EVAL_DATA_PATH` (required) — path to the labeled CSV file
- `--model N` (optional) — model ID from the registry. If omitted, uses the current default model

---

## Step 1: Load Model Reference

Call `list_registered_models` (gtd-training server) to read the model registry.

**If the registry is empty** (no models or `current_best` is null):
> No trained models found. Run `/gtd:train path/to/data.csv` first.

Then stop.

**Select the model**:
- If `--model N` was provided: find the model entry with `id == N`. If not found, list available model IDs and stop with an error
- If no `--model` flag: use the model entry matching `current_best`

Extract from the selected model entry:
- `workspace_path` — where the trained model lives
- `best_run_id` — which run to evaluate
- `task_type` — classification or regression
- `target_column` — the label column name
- `primary_metric` — the metric used during optimization
- `model_type` — the model name
- `best_score` — the training score
- `id` — the model ID

---

## Step 2: Evaluate Model

Call `evaluate_model` (gtd-training server) with:
- `workspace_path`: from model entry
- `run_id`: the `best_run_id` from model entry
- `data_path`: the EVAL_DATA_PATH from the argument
- `target_column`: from model entry
- `task_type`: from model entry

This returns comprehensive metrics.

---

## Step 3: Generate Visualizations

Call these tools:

1. `get_feature_importance` (gtd-training server) with workspace, run ID, data path, and target column
2. `get_roc_curve` (gtd-training server) — for binary classification only (skip for multiclass and regression)
3. `get_pr_curve` (gtd-training server) — for classification tasks only (skip for regression)

---

## Step 4: Present Evaluation Report

Print the evaluation report:

```
## Evaluation Report

**Model**: {model_type} (model #{id}, {best_run_id})
**Training score**: {primary_metric} = {best_score} (CV)
**Evaluated on**: {filename} ({row_count} rows)
```

### For classification tasks:

```
### Metrics
| Metric          | Score  |
|-----------------|--------|
| Accuracy        | 0.891  |
| F1 (macro)      | 0.874  |
| Precision       | 0.882  |
| Recall          | 0.867  |
| ROC-AUC         | 0.943  |
| PR-AUC          | 0.912  |

### Confusion Matrix
|            | Pred: 0 | Pred: 1 |
|------------|---------|---------|
| Actual: 0  | 1,234   | 166     |
| Actual: 1  | 78      | 522     |
```

### For regression tasks:

```
### Metrics
| Metric   | Score   |
|----------|---------|
| R2       | 0.876   |
| RMSE     | 2.341   |
| MAE      | 1.823   |
| MAPE     | 12.4%   |
```

### Always include:

```
### Top Features
| Feature         | Importance |
|-----------------|------------|
| tenure          | 0.234      |
| MonthlyCharges  | 0.189      |
| ...             | ...        |

### Visualizations
- ROC curve: {path} (if applicable)
- PR curve: {path} (if applicable)
- Feature importance: {path}

### Assessment
- Overall model performance assessment for this type of problem
- Where the model is making mistakes (analyze confusion matrix or residuals)
- Suggestions for improvement (more data, feature engineering, different models)
```
