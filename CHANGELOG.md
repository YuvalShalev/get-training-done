# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-03-12

### Added
- 11 advanced feature engineering operations: target encoding, frequency encoding, groupby aggregation, polynomial features, binning, feature selection, rank transform, power transform, cyclic encoding, ratio features, categorical interaction
- Ensemble training via `train_ensemble`: stacking, hill climbing, and seed ensembling strategies
- TabPFN model support for small datasets (optional dependency: `pip install tabpfn`)
- Structured research insights via `research_and_extract` — unified search + extraction in one call
- Research-driven optimization protocol in train workflow
- Response compression for better context management

### Changed
- `auto_preprocess` now uses target encoding for high-cardinality categoricals when target column is available
- train.md updated with ensemble protocol, research integration, and tighter context rules
- `engineer_features` docstring updated with all 19 available operations

## [0.2.1] - 2026-03-11

### Added
- Architecture diagrams (Mermaid) in README
- All 7 slash commands documented
- CI status badge
- SECURITY.md for responsible disclosure

### Changed
- README restructured for GA release
- Development Status upgraded to Beta

### Fixed
- Auto strategy for data splitting (regression uses random, not stratified)
- Metric direction for lower-is-better metrics (RMSE, MAE)
- class_weight added to classifier HP specs
- Removed unused shap and seaborn dependencies

### Removed
- Exporter module (consolidated into trainer)
- Unused ml-optimizer agent definition

## [0.2.0] - 2026-02-26

### Added
- Claude Code plugin support (`.claude-plugin/plugin.json`)
- Auto-configured MCP servers via `.mcp.json`
- `/gtd:train` — full optimization workflow with argument parsing
- `/gtd:inference` — model predictions with registry integration
- `/gtd:evaluate` — comprehensive model evaluation
- `/gtd:models` — trained model registry listing
- Model registry (`.gtd-state.json`) for tracking trained models across sessions
- CI/CD with GitHub Actions
- Self-bootstrapping Python venv for zero-config MCP servers
- CONTRIBUTING.md and issue templates

### Changed
- Restructured repo to follow Claude Code plugin conventions
- README rewritten for plugin-first experience
- Removed CLI (`gtd setup`) in favor of plugin auto-configuration

## [0.1.0] - 2026-02-24

### Added
- Initial implementation with 3 MCP servers (data, training, research)
- 14 supported ML models (classification + regression)
- 5-phase optimization workflow
- Cross-validated training with convergence detection
- arXiv, Kaggle, and Papers with Code research integration
