# Binary Classification: Churn Prediction

This walkthrough demonstrates the full optimization pipeline for a binary
classification problem -- predicting customer churn. You will learn how to
profile data, engineer features, train multiple models, evaluate them, and
select the best one.

## Dataset Overview

We will work with a typical telecom customer churn dataset. The target column is
`Churn` (1 = churned, 0 = retained). Features include account tenure, monthly
charges, contract type, and service usage indicators.

Assume the dataset is saved at `data/churn.csv`.

## Step 1: Profile the Dataset

Start by understanding your data before building any model.

```python
from bbopt.core import data_profiler

profile = data_profiler.profile_dataset(
    path="data/churn.csv",
    target_column="Churn",
    task_type="binary_classification",
)

print(f"Rows: {profile['shape']['rows']}, Columns: {profile['shape']['columns']}")
print(f"Numeric features: {profile['feature_types']['numeric']}")
print(f"Categorical features: {profile['feature_types']['categorical']}")
```

### Inspect class balance

```python
balance = profile["class_balance"]
print(f"Class distribution: {balance['distribution']}")
print(f"Minority ratio: {balance['minority_ratio']:.2f}")
print(f"Imbalance severity: {balance['severity']}")
```

For churn datasets, you will often see moderate to severe imbalance (e.g., 70/30
or 80/20 split). The profiler flags this and recommends strategies like SMOTE or
class weights.

### Check for data quality issues

```python
issues = data_profiler.detect_data_issues(
    path="data/churn.csv",
    target_column="Churn",
)

print(f"High cardinality columns: {issues['high_cardinality_columns']}")
print(f"Constant features: {issues['constant_features']}")
print(f"Missing-heavy columns: {issues['missing_heavy_columns']}")
print(f"Leakage suspects: {issues['data_leakage_suspects']}")
```

Pay special attention to `data_leakage_suspects`. Any feature with a correlation
above 0.95 with the target is flagged -- this often indicates a column that was
derived from the target or collected after the event.

### Drill into specific columns

```python
tenure_stats = data_profiler.get_column_stats("data/churn.csv", "tenure")
print(f"Tenure mean: {tenure_stats['distribution']['mean']:.1f}")
print(f"Tenure std: {tenure_stats['distribution']['std']:.1f}")
print(f"Missing: {tenure_stats['missing_pct']:.1f}%")
```

### Examine correlations

```python
correlations = data_profiler.compute_correlations(
    path="data/churn.csv",
    target_column="Churn",
    method="spearman",
)

print("Top features correlated with churn:")
sorted_corrs = sorted(
    correlations["feature_target_correlations"].items(),
    key=lambda x: abs(x[1]),
    reverse=True,
)
for name, corr in sorted_corrs[:10]:
    print(f"  {name}: {corr:.4f}")
```

## Step 2: Feature Engineering

Based on the profiling results, apply appropriate preprocessing.

```python
from bbopt.core import feature_engine

result = feature_engine.engineer_features(
    data_path="data/churn.csv",
    operations=[
        {
            "type": "impute_numeric",
            "columns": ["tenure", "MonthlyCharges", "TotalCharges"],
            "strategy": "median",
        },
        {
            "type": "impute_categorical",
            "columns": ["InternetService", "Contract", "PaymentMethod"],
            "strategy": "mode",
        },
        {
            "type": "one_hot_encode",
            "columns": ["InternetService", "Contract", "PaymentMethod"],
        },
        {
            "type": "standard_scale",
            "columns": ["tenure", "MonthlyCharges", "TotalCharges"],
        },
        {
            "type": "create_interaction",
            "column_a": "tenure",
            "column_b": "MonthlyCharges",
            "name": "tenure_x_monthly",
        },
    ],
    output_path="data/churn_processed.csv",
)

print(f"New shape: {result['new_shape']}")
print(f"Operations applied: {result['operations_applied']}")
print(f"Columns: {result['new_columns'][:10]}...")
```

## Step 3: Create a Workspace

```python
from bbopt.core import workspace

ws = workspace.create_workspace("./churn_optimization")
ws_path = ws["workspace_path"]
```

## Step 4: Train Multiple Models

### Baseline -- Logistic Regression

Start with an interpretable baseline.

```python
from bbopt.core import trainer

# Identify feature columns (all columns except target)
import pandas as pd
df = pd.read_csv("data/churn_processed.csv")
feature_cols = [c for c in df.columns if c != "Churn"]

baseline = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/churn_processed.csv",
    model_type="logistic_regression",
    hyperparameters={"C": 1.0, "penalty": "l2", "max_iter": 1000},
    feature_columns=feature_cols,
    target_column="Churn",
    task_type="binary_classification",
    cv_folds=5,
    random_state=42,
)

print(f"Logistic Regression -- Accuracy: {baseline['mean_score']:.4f}")
```

### Random Forest

```python
rf_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/churn_processed.csv",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    feature_columns=feature_cols,
    target_column="Churn",
    task_type="binary_classification",
    cv_folds=5,
    random_state=42,
)

print(f"Random Forest -- Accuracy: {rf_result['mean_score']:.4f}")
```

### XGBoost

```python
xgb_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/churn_processed.csv",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    feature_columns=feature_cols,
    target_column="Churn",
    task_type="binary_classification",
    cv_folds=5,
    random_state=42,
)

print(f"XGBoost -- Accuracy: {xgb_result['mean_score']:.4f}")
```

### LightGBM

```python
lgb_result = trainer.train_model(
    workspace_path=ws_path,
    data_path="data/churn_processed.csv",
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
    target_column="Churn",
    task_type="binary_classification",
    cv_folds=5,
    random_state=42,
)

print(f"LightGBM -- Accuracy: {lgb_result['mean_score']:.4f}")
```

## Step 5: Compare and Evaluate

### Side-by-side comparison

```python
from bbopt.core import evaluator

comparison = evaluator.compare_runs(
    workspace_path=ws_path,
    run_ids=[
        baseline["run_id"],
        rf_result["run_id"],
        xgb_result["run_id"],
        lgb_result["run_id"],
    ],
)

print(f"Best model: {comparison['best_run_id']}")
print(f"Primary metric: {comparison['primary_metric']}")
for row in comparison["comparison"]:
    print(f"  {row['run_id']}: accuracy={row.get('accuracy', 'N/A')}")
```

### Full evaluation of the best model

```python
best_run_id = comparison["best_run_id"]

eval_metrics = evaluator.evaluate_model(
    workspace_path=ws_path,
    run_id=best_run_id,
    data_path="data/churn_processed.csv",
    target_column="Churn",
    task_type="binary_classification",
)

print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
print(f"F1 (macro): {eval_metrics['f1_macro']:.4f}")
print(f"Precision: {eval_metrics['precision_macro']:.4f}")
print(f"Recall: {eval_metrics['recall_macro']:.4f}")
print(f"ROC AUC: {eval_metrics.get('roc_auc', 'N/A')}")
```

### ROC curve

```python
roc = evaluator.get_roc_curve(
    workspace_path=ws_path,
    run_id=best_run_id,
    data_path="data/churn_processed.csv",
    target_column="Churn",
)

print(f"AUC: {roc['auc']:.4f}")
print(f"ROC plot saved to: {roc['plot_path']}")
```

### Precision-recall curve

For imbalanced datasets, the precision-recall curve is often more informative
than the ROC curve.

```python
pr = evaluator.get_pr_curve(
    workspace_path=ws_path,
    run_id=best_run_id,
    data_path="data/churn_processed.csv",
    target_column="Churn",
)

print(f"Average Precision: {pr['ap']:.4f}")
print(f"PR plot saved to: {pr['plot_path']}")
```

### Feature importance

Understand which features drive the model's predictions.

```python
importance = evaluator.get_feature_importance(
    workspace_path=ws_path,
    run_id=best_run_id,
    data_path="data/churn_processed.csv",
    target_column="Churn",
    method="builtin",
)

print("Top 10 features:")
sorted_imp = sorted(importance["importances"].items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_imp[:10]:
    print(f"  {name}: {score:.4f}")
print(f"Plot saved to: {importance['plot_path']}")
```

## Step 6: Optimization History

Review how your optimization progressed over time.

```python
history = evaluator.get_optimization_history(workspace_path=ws_path)

print(f"Total runs: {len(history['runs'])}")
print(f"Best run: {history['best_run_id']}")
print(f"Best score: {history['best_score']:.4f}")

for run in history["runs"]:
    marker = " <-- best" if run["is_best"] else ""
    print(f"  {run['run_id']}: {run.get('metrics', {}).get('accuracy', 'N/A')}{marker}")
```

## Step 7: Generate Predictions

Use the best model to make predictions on new data.

```python
predictions = trainer.predict(
    workspace_path=ws_path,
    run_id=best_run_id,
    test_data_path="data/churn_test.csv",
    target_column="Churn",
)

print(f"Predictions: {predictions['predictions'][:10]}")
print(f"Probabilities: {predictions['probabilities'][:3]}")
if predictions["metrics"]:
    print(f"Test accuracy: {predictions['metrics']['accuracy']:.4f}")
```

## Key Takeaways

1. **Always profile first.** Understanding your data prevents wasted training
   cycles. Check for class imbalance, leakage, and missing values before
   building models.

2. **Start with a simple baseline.** Logistic regression provides a reference
   point that reveals how much value complex models add.

3. **Use cross-validation scores** for model comparison, not single train/test
   splits. The `mean_score` and `std_score` together tell you both performance
   level and reliability.

4. **For imbalanced targets**, pay more attention to the precision-recall curve
   and F1 score than raw accuracy. A model with 95% accuracy on 95/5 class
   split is not necessarily useful.

5. **Feature importance** helps you understand the model and catch potential
   issues. If a feature you did not expect is dominating, investigate whether
   it represents data leakage.
