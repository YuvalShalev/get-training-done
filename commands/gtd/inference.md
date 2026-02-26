---
description: Run predictions on new data using a trained model
argument-hint: "path/to/test.csv [--model N]"
---

# GTD: Inference

You are running inference on new data using a model trained by `/gtd:train`.

## Argument Parsing

The user's arguments are: `$ARGUMENTS`

Parse from the argument string:

- `TEST_DATA_PATH` (required) — path to the CSV file to predict on
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
- `best_run_id` — which run to use for predictions
- `task_type` — what kind of predictions to expect
- `target_column` — the target column name
- `model_type` — the model name (e.g., "lightgbm")
- `best_score` — the training score
- `primary_metric` — the metric name
- `id` — the model ID

---

## Step 2: Preview Test Data

Call `profile_dataset` (gtd-data server) with the test data path to understand the data structure.

Check if the `target_column` exists in the test data. If it does, you will pass it to `predict` so metrics can be computed against the true labels.

---

## Step 3: Run Predictions

Call `predict` (gtd-training server) with:
- `workspace_path`: from model entry
- `run_id`: the `best_run_id` from model entry
- `test_data_path`: the TEST_DATA_PATH from the argument
- `target_column`: only pass this if the target column exists in the test data

---

## Step 4: Present Results

Print the inference report:

```
## Inference Results

**Model**: {model_type} (model #{id}, {best_run_id}, {primary_metric}={best_score})
**Input**: {filename} ({row_count} rows)

### Prediction Summary
- Total predictions: {count}
- Class distribution: {0: 3,412 (68.2%), 1: 1,588 (31.8%)}
- Mean probability: 0.32 (class 1)

### Output
- Predictions saved to: {path}

### Other Available Models
Run `/gtd:models` to see all trained models.
Use `--model N` to predict with a specific model.

### Next Steps
- If your data has labels, evaluate: `/gtd:evaluate path/to/labeled.csv`
```

**For regression**, replace the class distribution with summary statistics: mean, median, std, min, max of predicted values.

**If true labels were available** and metrics were returned by `predict`, include them in an additional "Metrics on Test Set" section.
