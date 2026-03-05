---
description: Evolve the GTD decision protocol using DSPy optimization
---

# GTD: Evolve

You are running prompt evolution for the GTD optimizer. This uses DSPy to improve the Phase 4 decision protocol based on accumulated training sessions.

## Requirements

- At least 10 completed training sessions (recorded in `gtd-meta-scores.jsonl`)
- DSPy installed: `pip install get-training-done[evolve]`

## Steps

1. Load session metrics from your auto-memory directory's `gtd-meta-scores.jsonl`
2. Print: `Sessions available: {count} | Min required: 10`
3. If < 10: Print "Need more training sessions. Run /gtd:train on more datasets." and stop.
4. If >= 10, proceed with evolution:

### Run Evolution

```python
from gtd.core.prompt_evolver import optimize_prompts, extract_decision_instructions, inject_into_train_md

# Step A: Optimize
result = optimize_prompts(memory_dir="<your auto-memory directory>")

if result["status"] != "success":
    print(f"Evolution failed: {result}")
    # Stop here
```

```python
# Step B: Extract improved instructions
instructions = extract_decision_instructions(result["optimized_program"])

# Step C: Back up and inject into train.md
inject_result = inject_into_train_md(
    instructions=instructions,
    train_md_path="<path to commands/gtd/train.md>",
    backup=True,
)
```

5. Print results:
   - What changed in the decision protocol
   - Expected improvement based on training/validation split
   - Backup version number for rollback

## Rollback

If the evolved instructions perform worse, restore from backup:
```
cp commands/gtd/train.v{N}.md commands/gtd/train.md
```
