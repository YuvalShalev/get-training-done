# Quick Start

Get up and running with bbopt in under five minutes. This guide walks you through
installation, your first model optimization, and how to interpret the results.

## Installation

Install bbopt and its dependencies:

```bash
pip install -e .
```

If you plan to use the research tools (arXiv, Kaggle, Papers with Code), install
the optional research dependencies:

```bash
pip install -e ".[research]"
```

### Verify the installation

```python
from bbopt.core import model_registry

models = model_registry.list_available_models()
print(f"Available models: {len(models)}")
# Available models: 14
```

## Your First Optimization

This example trains a Random Forest on the classic Iris dataset using the bbopt
training pipeline with cross-validation.

### Step 1: Profile your data

Before training, profile the dataset to understand its structure, missing values,
and potential issues.

```python
from bbopt.core import data_profiler

profile = data_profiler.profile_dataset(
    path="data/iris.csv",
    target_column="species",
    task_type="auto",
)

print(f"Shape: {profile['shape']}")
print(f"Task type: {profile['task_type']}")
print(f"Class balance: {profile['class_balance']}")
print(f"Recommendations: {profile['recommended_preprocessing']}")
```

The profiler automatically detects the task type (classification vs regression),
flags class imbalance, identifies outliers, and recommends preprocessing steps.

### Step 2: Create a workspace

A workspace is a directory that stores all training runs, models, metrics, and
reports for a single optimization session.

```python
from bbopt.core import workspace

ws = workspace.create_workspace("./my_optimization")
ws_path = ws["workspace_path"]
print(f"Workspace created at: {ws_path}")
```

### Step 3: Train a model

Train a Random Forest classifier with cross-validation:

```python
from bbopt.core import trainer

result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/iris.csv",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 5,
    },
    feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    target_column="species",
    task_type="multiclass_classification",
    cv_folds=5,
    random_state=42,
)

print(f"Run ID: {result['run_id']}")
print(f"CV Accuracy: {result['mean_score']:.4f} +/- {result['std_score']:.4f}")
print(f"Training time: {result['training_time']:.2f}s")
```

### Step 4: Evaluate the model

Run a full evaluation to get detailed metrics:

```python
from bbopt.core import evaluator

metrics = evaluator.evaluate_model(
    workspace_path=ws_path,
    run_id=result["run_id"],
    data_path="data/iris.csv",
    target_column="species",
    task_type="multiclass_classification",
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (macro): {metrics['f1_macro']:.4f}")
print(f"Confusion matrix:\n{metrics['confusion_matrix']}")
```

### Step 5: Try another model

Compare with XGBoost using different hyperparameters:

```python
result_xgb = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/iris.csv",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
    },
    feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    target_column="species",
    task_type="multiclass_classification",
    cv_folds=5,
    random_state=42,
)

print(f"XGBoost CV Accuracy: {result_xgb['mean_score']:.4f}")
```

### Step 6: Compare runs

See how your models stack up against each other:

```python
comparison = evaluator.compare_runs(
    workspace_path=ws_path,
    run_ids=[result["run_id"], result_xgb["run_id"]],
)

print(f"Best model: {comparison['best_run_id']}")
for row in comparison["comparison"]:
    print(f"  {row['run_id']}: {row.get('accuracy', 'N/A')}")
```

## Understanding Results

### Training result

When you call `train_model`, the returned dict contains:

| Field           | Description                                         |
|-----------------|-----------------------------------------------------|
| `run_id`        | Unique identifier for this training run             |
| `cv_scores`     | List of scores from each cross-validation fold      |
| `mean_score`    | Average score across all folds                      |
| `std_score`     | Standard deviation of scores across folds           |
| `training_time` | Wall-clock seconds for the full training process    |
| `model_path`    | Filesystem path to the saved model (joblib format)  |

### Evaluation metrics

The `evaluate_model` function returns different metrics depending on the task type.

**Classification** (binary or multiclass):
- `accuracy` -- Overall classification accuracy
- `f1_macro` -- Macro-averaged F1 score
- `precision_macro` -- Macro-averaged precision
- `recall_macro` -- Macro-averaged recall
- `confusion_matrix` -- Full confusion matrix as nested lists
- `roc_auc` (binary) / `roc_auc_ovr` (multiclass) -- Area under the ROC curve

**Regression**:
- `r2` -- R-squared (coefficient of determination)
- `rmse` -- Root mean squared error
- `mae` -- Mean absolute error
- `mape` -- Mean absolute percentage error
- `explained_variance` -- Explained variance score

### Workspace structure

After training, your workspace directory looks like this:

```
bbopt_workspace_20260223_120000/
  metadata.json          # Workspace-level tracking
  data/                  # Copied datasets
  runs/
    run_001_random_forest/
      model.joblib       # Trained model
      config.json        # Hyperparameters and settings
      metrics.json       # CV scores and timing
      eval_metrics.json  # Full evaluation results
    run_002_xgboost/
      ...
  reports/               # Generated reports
  exports/               # Exported models
```

## Next Steps

- See [Binary Classification](binary_classification.md) for a detailed
  walkthrough on a churn prediction dataset.
- See [Regression](regression.md) for optimizing a regression model.
- Use the MCP servers to integrate bbopt into an agent-driven workflow.
