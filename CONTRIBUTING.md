# Contributing to Get Training Done

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/yuvalshalev/get-training-done.git
cd get-training-done
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=gtd --cov-report=term-missing
```

## Linting

```bash
ruff check src/ tests/
ruff check --fix src/ tests/
```

## Adding a New Model

1. Add the model config to `src/gtd/core/model_registry.py` in the `MODEL_REGISTRY` dict
2. Include: model class import, default hyperparameters, hyperparameter search space with ranges
3. Add tests in `tests/test_registry.py`
4. Update the supported models table in `README.md`

## Modifying Commands

Commands live in `commands/gtd/`. Each `.md` file is a Claude Code slash command with YAML frontmatter:

```yaml
---
description: Short description shown in command picker
argument-hint: "expected arguments"
---
```

The body is a prompt that Claude follows when the command is invoked.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `pytest` and `ruff check` locally
4. Open a PR with a clear description of what changed and why
5. Ensure CI passes

## Code Style

- Line length: 100 characters
- Follow existing patterns in the codebase
- Type hints are encouraged but not required
- Keep functions focused and under 50 lines where possible
