# Regression: House Price Prediction

This walkthrough demonstrates the full optimization pipeline for a regression
problem -- predicting house prices. You will learn how to profile numerical
data, handle skewed features, train multiple regressors, and evaluate them
using regression-specific metrics.

## Dataset Overview

We will work with a housing dataset where the target column is `medv` (median
value of owner-occupied homes in thousands of dollars). Features include crime
rate, number of rooms, distance to employment centers, and tax rate.

Assume the dataset is saved at `data/housing.csv`.

## Step 1: Profile the Dataset

```python
from bbopt.core import data_profiler

profile = data_profiler.profile_dataset(
    path="data/housing.csv",
    target_column="medv",
    task_type="regression",
)

print(f"Rows: {profile['shape']['rows']}, Columns: {profile['shape']['columns']}")
print(f"Task type: {profile['task_type']}")
print(f"Numeric features: {len(profile['feature_types']['numeric'])}")
print(f"Categorical features: {len(profile['feature_types']['categorical'])}")
```

### Inspect distributions

Understanding feature distributions is especially important for regression.
Skewed features and outliers can significantly affect model performance.

```python
for col_name, dist in profile["distributions"].items():
    if dist["type"] == "numeric":
        print(f"{col_name}: mean={dist['mean']:.2f}, std={dist['std']:.2f}, "
              f"min={dist['min']:.2f}, max={dist['max']:.2f}")
```

### Check for outliers

```python
print("Outlier counts per feature:")
for col, count in profile["outlier_counts"].items():
    if count > 0:
        print(f"  {col}: {count} outliers")
```

### Check correlations with the target

```python
correlations = data_profiler.compute_correlations(
    path="data/housing.csv",
    target_column="medv",
    method="pearson",
)

print("Feature-target correlations:")
sorted_corrs = sorted(
    correlations["feature_target_correlations"].items(),
    key=lambda x: abs(x[1]),
    reverse=True,
)
for name, corr in sorted_corrs:
    print(f"  {name}: {corr:.4f}")
```

This helps identify which features have strong linear relationships with the
target and which might need nonlinear transformations.

### Detect potential problems

```python
issues = data_profiler.detect_data_issues(
    path="data/housing.csv",
    target_column="medv",
)

if issues["multicollinearity"]:
    print("Multicollinear feature pairs:")
    for pair in issues["multicollinearity"]:
        print(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.4f}")

if issues["missing_heavy_columns"]:
    print(f"Columns with >50% missing: {issues['missing_heavy_columns']}")
```

## Step 2: Feature Engineering

For regression, common preprocessing steps include log-transforming skewed
features, scaling, and creating interaction terms.

```python
from bbopt.core import feature_engine

result = feature_engine.engineer_features(
    data_path="data/housing.csv",
    operations=[
        # Log-transform highly skewed features
        {
            "type": "log_transform",
            "columns": ["crim", "lstat"],
        },
        # Scale all numeric features
        {
            "type": "standard_scale",
            "columns": ["crim", "zn", "indus", "nox", "rm", "age",
                        "dis", "rad", "tax", "ptratio", "lstat"],
        },
        # Create interaction between rooms and lstat
        {
            "type": "create_interaction",
            "column_a": "rm",
            "column_b": "lstat",
            "name": "rm_x_lstat",
        },
    ],
    output_path="data/housing_processed.csv",
)

print(f"New shape: {result['new_shape']}")
print(f"Operations: {result['operations_applied']}")
```

## Step 3: Create a Workspace

```python
from bbopt.core import workspace

ws = workspace.create_workspace("./housing_optimization")
ws_path = ws["workspace_path"]
```

## Step 4: Explore Available Regression Models

```python
from bbopt.core import model_registry

regression_models = model_registry.list_available_models(task_type="regression")
print(f"Available regression models: {len(regression_models)}")
for model in regression_models:
    print(f"  {model['display_name']}: {model['description']}")
```

## Step 5: Train Multiple Models

### Baseline -- Linear Regression

```python
from bbopt.core import trainer
import pandas as pd

df = pd.read_csv("data/housing_processed.csv")
feature_cols = [c for c in df.columns if c != "medv"]

linear_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="linear_regression",
    hyperparameters={},
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"Linear Regression -- R2: {linear_result['mean_score']:.4f} "
      f"+/- {linear_result['std_score']:.4f}")
```

### ElasticNet

A regularized linear model that handles multicollinearity.

```python
enet_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="elasticnet",
    hyperparameters={
        "alpha": 0.1,
        "l1_ratio": 0.5,
        "max_iter": 2000,
    },
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"ElasticNet -- R2: {enet_result['mean_score']:.4f}")
```

### Random Forest Regressor

```python
rf_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"Random Forest -- R2: {rf_result['mean_score']:.4f}")
```

### XGBoost Regressor

```python
xgb_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"XGBoost -- R2: {xgb_result['mean_score']:.4f}")
```

### LightGBM Regressor

```python
lgb_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="lightgbm",
    hyperparameters={
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"LightGBM -- R2: {lgb_result['mean_score']:.4f}")
```

## Step 6: Compare Models

```python
from bbopt.core import evaluator

comparison = evaluator.compare_runs(
    workspace_path=ws_path,
    run_ids=[
        linear_result["run_id"],
        enet_result["run_id"],
        rf_result["run_id"],
        xgb_result["run_id"],
        lgb_result["run_id"],
    ],
)

print(f"Best model: {comparison['best_run_id']}")
print(f"Primary metric: {comparison['primary_metric']}")
print()
for row in comparison["comparison"]:
    r2 = row.get("r2", row.get("accuracy", "N/A"))
    print(f"  {row['run_id']} ({row['model_type']}): score={r2}")
```

## Step 7: Evaluate the Best Model

### Full regression metrics

```python
best_run_id = comparison["best_run_id"]

eval_metrics = evaluator.evaluate_model(
    workspace_path=ws_path,
    run_id=best_run_id,
    data_path="data/housing_processed.csv",
    target_column="medv",
    task_type="regression",
)

print(f"R2: {eval_metrics['r2']:.4f}")
print(f"RMSE: {eval_metrics['rmse']:.4f}")
print(f"MAE: {eval_metrics['mae']:.4f}")
print(f"MAPE: {eval_metrics['mape']:.2f}%")
print(f"Explained Variance: {eval_metrics['explained_variance']:.4f}")
```

### Understanding regression metrics

| Metric               | Interpretation                                     | Ideal |
|----------------------|----------------------------------------------------|-------|
| R2                   | Proportion of variance explained by the model      | 1.0   |
| RMSE                 | Average prediction error in target units           | 0.0   |
| MAE                  | Average absolute prediction error                  | 0.0   |
| MAPE                 | Average error as percentage of actual values        | 0.0%  |
| Explained Variance   | Similar to R2, but not penalized for bias          | 1.0   |

**R2** is the primary metric for model comparison. An R2 of 0.85 means the model
explains 85% of the variance in the target. Values above 0.7 are generally
considered good for real-world datasets.

**RMSE vs MAE**: RMSE penalizes large errors more heavily than MAE. If RMSE is
much larger than MAE, the model is making some very large errors on certain
predictions.

### Feature importance

```python
importance = evaluator.get_feature_importance(
    workspace_path=ws_path,
    run_id=best_run_id,
    data_path="data/housing_processed.csv",
    target_column="medv",
    method="builtin",
)

print("Top features for predicting house prices:")
sorted_imp = sorted(importance["importances"].items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_imp[:10]:
    print(f"  {name}: {score:.4f}")
```

For housing datasets, you typically expect `rm` (number of rooms) and `lstat`
(lower status population %) to be the strongest predictors.

## Step 8: Hyperparameter Tuning

Once you have identified the best model type, iterate on its hyperparameters.

```python
# Iteration 1: Increase tree count, lower learning rate
tuned_v1 = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    },
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"Tuned v1 -- R2: {tuned_v1['mean_score']:.4f}")

# Iteration 2: Adjust regularization
tuned_v2 = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/housing_processed.csv",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 800,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
    },
    feature_columns=feature_cols,
    target_column="medv",
    task_type="regression",
    cv_folds=5,
    random_state=42,
)

print(f"Tuned v2 -- R2: {tuned_v2['mean_score']:.4f}")
```

## Step 9: Review Optimization History

```python
history = evaluator.get_optimization_history(workspace_path=ws_path)

print(f"Total runs: {len(history['runs'])}")
print(f"Best overall: {history['best_run_id']} (R2={history['best_score']:.4f})")
print()
print("Run history:")
for run in history["runs"]:
    marker = " <-- best" if run["is_best"] else ""
    score = run.get("metrics", {}).get("r2", "N/A")
    print(f"  {run['run_id']}: R2={score}{marker}")
```

## Step 10: Generate Predictions

```python
predictions = trainer.predict(
    workspace_path=ws_path,
    run_id=history["best_run_id"],
    test_data_path="data/housing_test.csv",
    target_column="medv",
)

print(f"Sample predictions: {predictions['predictions'][:5]}")
if predictions["metrics"]:
    print(f"Test R2: {predictions['metrics']['r2']:.4f}")
    print(f"Test RMSE: {predictions['metrics']['rmse']:.4f}")
    print(f"Test MAE: {predictions['metrics']['mae']:.4f}")
```

## Key Takeaways

1. **Log-transform skewed features.** Many housing and financial features have
   long right tails. Applying `log1p` before training helps tree-based models
   and is essential for linear models.

2. **Compare R2 with standard deviation.** A model with R2=0.85 +/- 0.10 across
   folds might be less reliable than one with R2=0.82 +/- 0.03. Low variance
   across folds indicates the model generalizes consistently.

3. **Watch for RMSE vs MAE gap.** If RMSE is substantially larger than MAE, your
   model is making some very large errors. Investigate those cases -- they may
   be outliers in the data or a systematic weakness in the model.

4. **Feature interactions matter for regression.** Creating interaction terms
   (like `rm * lstat`) can capture nonlinear relationships that even tree-based
   models benefit from when the dataset is small.

5. **Lower learning rate + more trees** is a reliable strategy for gradient
   boosting hyperparameter tuning. Start with a moderate configuration, then
   gradually decrease the learning rate while increasing the number of trees.

6. **Regularization prevents overfitting.** For XGBoost and LightGBM, increasing
   `reg_alpha` (L1) and `reg_lambda` (L2) helps when the training score is much
   higher than the cross-validation score.
