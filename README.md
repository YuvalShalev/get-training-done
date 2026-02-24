# Agent-Driven Black-Box Optimization

An LLM-agent-powered AutoML plugin for [Claude Code](https://claude.com/claude-code) that optimizes ML models the way a senior data scientist would — with domain knowledge, research awareness, and transparent reasoning.

## Why This Beats Traditional AutoML

Traditional AutoML (Bayesian optimization, grid search, successive halving) treats hyperparameter tuning as a pure math problem. This plugin uses an LLM agent that:

- **Analyzes data first** — profiles distributions, detects class imbalance, finds data leakage before touching a model
- **Researches what works** — searches arXiv, Kaggle, and Papers with Code for approaches that succeed on similar data
- **Makes informed decisions** — chooses XGBoost over Random Forest because your data has high cardinality categoricals, not because it's next in the queue
- **Explains everything** — every model choice, hyperparameter change, and feature engineering step is justified with evidence
- **Knows when to stop** — convergence detection based on diminishing returns, not arbitrary budgets

## Quick Start

```bash
# Install
pip install agent-driven-bbopt

# Configure Claude Code automatically
bbopt setup

# Open Claude Code and start optimizing
claude
```

Then in Claude Code:

```
> /ml-optimizer

> I have a dataset at ./data/customers.csv with target column "churn".
> Find the best model to predict churn. Maximize F1 score.
```

The agent will:
1. Profile your data and identify issues
2. Research similar problems on arXiv and Kaggle
3. Train diverse baseline models
4. Iteratively optimize with evidence-based decisions
5. Export the best model with a full report

## How It Works

```
Claude Code → ML Optimizer Agent → MCP Tools
                                    ├── Data Analysis (profile, detect issues, correlations)
                                    ├── Model Training (train, evaluate, compare, export)
                                    └── Research (arXiv, Kaggle, Papers with Code)
```

The plugin provides:
- **3 MCP servers** with 20+ specialized ML tools
- **1 agent definition** with a structured optimization workflow
- **14 supported models** across classification and regression

## Supported Models

### Classification
| Model | Best For |
|-------|----------|
| XGBoost | Strong default for structured data |
| LightGBM | Fast training, native categorical support |
| CatBoost | High-cardinality categoricals |
| Random Forest | Robust baseline, handles noise |
| Extra Trees | High-dimensional data |
| Logistic Regression | Interpretable linear baseline |
| SVM | Small-medium datasets |
| KNN | Low-dimensional with clear clusters |
| MLP | Complex nonlinear patterns |

### Regression
All tree-based models above plus Linear Regression, ElasticNet, SVR, KNN Regressor, MLP Regressor.

## MCP Tools

### Data Analysis (`bbopt-data`)
- `profile_dataset` — Shape, distributions, missing values, class balance, outliers
- `get_column_stats` — Deep dive into a single column
- `detect_data_issues` — Class imbalance, multicollinearity, data leakage, high cardinality
- `compute_correlations` — Feature-target and feature-feature correlations
- `preview_data` — Quick data preview

### Model Training (`bbopt-training`)
- `train_model` — Cross-validated training with any supported model
- `evaluate_model` — Full metrics (accuracy, F1, ROC-AUC, confusion matrix, etc.)
- `get_feature_importance` — Built-in or permutation importance with plots
- `get_roc_curve` / `get_pr_curve` — Curve visualizations
- `compare_runs` — Side-by-side model comparison
- `engineer_features` — One-hot encoding, imputation, scaling, interactions
- `export_model` — Save best model for deployment
- `predict` — Score new data
- `get_optimization_history` — Full run history

### Research (`bbopt-research`)
- `search_arxiv` — Find relevant ML papers
- `search_kaggle_datasets` — Find similar datasets
- `search_kaggle_notebooks` — Find winning competition solutions
- `search_papers_with_code` — Find state-of-the-art methods

## Agent Workflow

The optimizer follows a 5-phase workflow:

**Phase 1: Data Understanding** — Profile the dataset, identify quality issues, compute correlations

**Phase 2: Research & Strategy** — Search literature for what works on similar data, formulate a strategy

**Phase 3: Baselines** — Train 2-3 diverse models with default parameters to establish a baseline

**Phase 4: Iterative Optimization** — Tune hyperparameters, try different models, engineer features. Each decision is justified by evidence from previous runs.

**Phase 5: Final Report** — Select best model, generate visualizations (ROC, PR, SHAP), export model, summarize findings

Convergence: stops when target metric is met, no improvement >0.5% for 3 consecutive runs, or budget (20 runs) is exhausted.

## Installation

### From PyPI

```bash
pip install agent-driven-bbopt
bbopt setup
```

### From Source

```bash
git clone https://github.com/yuvalshalev/agent-driven-bb-optimization.git
cd agent-driven-bb-optimization
pip install -e ".[dev]"
bbopt setup
```

### Manual Setup

If you prefer to configure Claude Code manually, add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "bbopt-data": {
      "command": "python",
      "args": ["-m", "bbopt.servers.data_server"]
    },
    "bbopt-training": {
      "command": "python",
      "args": ["-m", "bbopt.servers.training_server"]
    },
    "bbopt-research": {
      "command": "python",
      "args": ["-m", "bbopt.servers.research_server"]
    }
  }
}
```

And copy `src/bbopt/agents/ml-optimizer.md` to `~/.claude/agents/`.

## Development

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=bbopt --cov-report=term-missing
```

## Project Structure

```
src/bbopt/
├── cli.py                    # `bbopt setup` CLI
├── servers/
│   ├── data_server.py        # MCP: data analysis tools
│   ├── training_server.py    # MCP: model training/eval tools
│   └── research_server.py    # MCP: arXiv/Kaggle/PwC search
├── core/
│   ├── data_profiler.py      # Statistical analysis engine
│   ├── feature_engine.py     # Feature engineering operations
│   ├── model_registry.py     # 14 models with hyperparameter spaces
│   ├── trainer.py            # Training loop with cross-validation
│   ├── evaluator.py          # Metrics, curves, feature importance
│   ├── exporter.py           # Model serialization
│   └── workspace.py          # Session filesystem manager
├── research/
│   ├── arxiv_client.py       # arXiv API
│   ├── kaggle_client.py      # Kaggle API
│   └── pwc_client.py         # Papers with Code API
└── agents/
    └── ml-optimizer.md       # Agent definition
```

## License

MIT
