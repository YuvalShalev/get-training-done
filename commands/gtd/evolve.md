---
description: Evolve the GTD decision protocol using DSPy optimization
---

# GTD: Evolve

You are running prompt evolution for the GTD optimizer. This uses DSPy to improve the Phase 4 decision protocol based on accumulated training sessions.

## Requirements

- At least 10 completed training sessions (recorded in `gtd-meta-scores.jsonl`)
- DSPy will be installed automatically into the plugin environment on first run

## Environment Setup

Before running any Python code, ensure the plugin environment is ready:

```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  echo "ERROR: CLAUDE_PLUGIN_ROOT is not set" >&2
  exit 1
fi

PLUGIN_PYTHON="$CLAUDE_PLUGIN_ROOT/.venv/bin/python"

# Bootstrap venv if missing
if [ ! -f "$PLUGIN_PYTHON" ]; then
  echo "GTD: Setting up Python environment..." >&2
  python3 -m venv "$CLAUDE_PLUGIN_ROOT/.venv"
  "$CLAUDE_PLUGIN_ROOT/.venv/bin/pip" install --quiet -e "$CLAUDE_PLUGIN_ROOT"
fi

# Ensure DSPy is installed
"$PLUGIN_PYTHON" -c "import dspy" 2>/dev/null || \
  "$CLAUDE_PLUGIN_ROOT/.venv/bin/pip" install --quiet -e "$CLAUDE_PLUGIN_ROOT[evolve]"
```

## Steps

1. Load session metrics from your auto-memory directory's `gtd-meta-scores.jsonl`
2. Print: `Sessions available: {count} | Min required: 10`
3. If < 10: Print "Need more training sessions. Run /gtd:train on more datasets." and stop.
4. If >= 10, proceed with evolution:

### Run Evolution

```bash
CLAUDE_PLUGIN_ROOT="$CLAUDE_PLUGIN_ROOT" \
TRAIN_MD_PATH="$CLAUDE_PLUGIN_ROOT/commands/gtd/train.md" \
"$CLAUDE_PLUGIN_ROOT/.venv/bin/python" <<'PYEOF'
import json, os, sys
from gtd.core.prompt_evolver import optimize_prompts, extract_decision_instructions, inject_into_train_md

memory_dir = "<your auto-memory directory>"

# Step A: Optimize
result = optimize_prompts(memory_dir=memory_dir)
if result["status"] != "success":
    print(json.dumps(result, indent=2))
    sys.exit(1)

# Step B: Extract improved instructions
instructions = extract_decision_instructions(result["optimized_program"])

# Step C: Back up and inject into train.md
inject_result = inject_into_train_md(
    instructions=instructions,
    train_md_path=os.environ["TRAIN_MD_PATH"],
    backup=True,
)

print(json.dumps({
    "status": "success",
    "train_size": result["train_size"],
    "val_size": result["val_size"],
    "instructions": {k: str(v)[:200] for k, v in instructions.items()},
    "inject_result": inject_result,
}, indent=2))
PYEOF
```

5. Print results:
   - What changed in the decision protocol
   - Expected improvement based on training/validation split
   - Backup version number for rollback

## Rollback

If the evolved instructions perform worse, restore from backup:
```bash
cp "$CLAUDE_PLUGIN_ROOT/commands/gtd/train.v{N}.md" "$CLAUDE_PLUGIN_ROOT/commands/gtd/train.md"
```
