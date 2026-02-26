---
description: List all trained models from the local registry
---

# GTD: Trained Models

You are listing all trained models from the local registry.

## Step 1: Read Registry

Call `list_registered_models` (gtd-training server) to read the `.gtd-state.json` registry.

**If the registry is empty** (no models or the call returns an empty models list):
> No trained models found. Run `/gtd:train path/to/data.csv` first to train a model.

Then stop.

## Step 2: Display Models

Extract `current_best` and the `models` list from the result.

Print a table of all models. Mark the current default model with `*`:

```
## Trained Models

| #   | Model     | Score      | Metric   | Target   | Data       | Runs | Created    |
|-----|-----------|------------|----------|----------|------------|------|------------|
| 1   | LightGBM  | 0.886      | accuracy | churn    | churn.csv  | 10   | 2026-02-25 |
| 2*  | XGBoost   | 0.912      | f1_macro | fraud    | fraud.csv  | 15   | 2026-02-24 |

`*` = current default (used by `/gtd:inference` and `/gtd:evaluate`)

### Usage
- Predict with default model: `/gtd:inference path/to/test.csv`
- Predict with specific model: `/gtd:inference path/to/test.csv --model 1`
- Evaluate with specific model: `/gtd:evaluate path/to/labeled.csv --model 1`
- Train a new model: `/gtd:train path/to/data.csv`
```

For the "Data" column, show just the filename (not the full path) extracted from the model's `data_path`.
For the "Created" column, show just the date portion of `created_at`.
