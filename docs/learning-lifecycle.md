# GTD Learning Lifecycle

This document explains **every file** read or written during a GTD training session, organized by when it happens: session start, during training, and session end.

---

## 1. SESSION START (Phase 1-2 of `/gtd:train`)

### What gets READ (knowledge loading)

On the **first `train_model()` call**, the system loads prior knowledge to bootstrap the session:

| What | Source File | Read By | Purpose |
|------|------------|---------|---------|
| High-level observations | `~/.claude/gtd/high-level-observations.md` (global, falls back to `memory_dir/`) | `meta_learner.load_prior_knowledge()` | LLM-synthesized insights from all past sessions — general ML wisdom |
| Structured learnings | `~/.claude/gtd/gtd-learnings.md` (global, falls back to `memory_dir/`) | `meta_learner.load_learnings()` | Per-session entries with fingerprints, strategies, HP sweet spots |
| Strategy matches | (derived from learnings above) | `meta_learner.match_strategies()` | Filters learnings to find entries matching current dataset fingerprint |

**First session ever:** These files don't exist yet → no prior knowledge loaded → agent starts from scratch.

**Returning session (same or different project):** Global files exist → agent gets strategy recommendations + prior knowledge injected into the first `train_model` response as `result["strategy_recommendation"]` and `result["prior_knowledge"]`.

### What gets WRITTEN at session start

| File | Location | Written By | Trigger |
|------|----------|-----------|---------|
| `memory_dir.txt` | `workspace_path/` | `trainer._store_memory_dir()` | First `train_model()` with `memory_dir` param — persists path for later auto-discovery |
| `dataset_fingerprint.json` | `workspace_path/` | `trainer._store_fingerprint()` | First `train_model()` — computes size_class, task, feature_mix, issues |
| `metadata.json` | `workspace_path/` | `workspace.create_workspace()` | Workspace creation (Phase 2) — initial empty metadata |

---

## 2. DURING SESSION (Phase 3-4: Training & Optimization)

### Every `train_model()` call writes:

| File | Location | Written By | Content |
|------|----------|-----------|---------|
| `model.joblib` | `workspace_path/runs/run_NNN_MODEL/` | `trainer.train_model()` | Pickled trained model |
| `config.json` | `workspace_path/runs/run_NNN_MODEL/` | `workspace.save_run_artifact()` | Model type, hyperparameters, features, task type |
| `metrics.json` | `workspace_path/runs/run_NNN_MODEL/` | `workspace.save_run_artifact()` | CV scores, mean/std score, training time |
| `run_log.jsonl` | `workspace_path/` | `trainer._append_run_log()` | One JSON line per run: run_id, model_type, score, params, timestamp |
| `metadata.json` | `workspace_path/` | `workspace.register_run()` | Updated: runs array, best_run_id, best_score |

### Every 3 optimization runs (within-run reflection):

| File | Location | Written By | Content |
|------|----------|-----------|---------|
| `observation-log.md` | `workspace_path/` | `meta_learner.save_observation()` | Appended markdown block: run #, score trajectory, diagnosis, next strategy |

This is the **Reflexion pattern** — the agent pauses every 3 runs to reflect on what's working and what to try next. These observations are workspace-local (not global) because they're specific to the current optimization session.

---

## 3. SESSION END (Phase 5: Export, Synthesize, Register)

Phase 5 has three steps that write learning files. The order matters.

### Step 1: `synthesize_session()` — LLM-generated insights

The agent writes a 3-5 sentence synthesis paragraph capturing transferable insights.

| File | Location | Written By | Content |
|------|----------|-----------|---------|
| `high-level-observations.md` | `~/.claude/gtd/` (GLOBAL) | `meta_learner.save_session_synthesis()` | Appended: date, dataset, task type + synthesis paragraph |
| `high-level-observations.md` | `memory_dir/` (project) | `meta_learner.save_session_synthesis()` | Same content (dual-write) |
| `observation-log-TIMESTAMP.md` | `workspace_path/` | `meta_learner.archive_observation_log()` | Archive of within-run observations before reset |
| `.session_synthesized` | `workspace_path/` | `training_server.synthesize_session()` | Flag file (ISO timestamp) — gates `register_model()` |

### Step 2: `export_model()` — Structured learning + strategy library

Triggered when the agent exports the best model. Auto-saves learning data.

| File | Location | Written By | Content |
|------|----------|-----------|---------|
| `model.joblib` | `workspace_path/exports/EXPORT_NAME/` | `trainer.export_model()` | Copy of best model |
| `metadata.json` | `workspace_path/exports/EXPORT_NAME/` | `trainer.export_model()` | Export metadata (run_id, config, metrics) |
| **`gtd-learnings.md`** | **`~/.claude/gtd/` (GLOBAL)** | `meta_learner.save_enhanced_learnings()` | Appended: fingerprint, strategy sequence, best model, HP sweet spot, anti-pattern, insight |
| **`gtd-learnings.md`** | **`memory_dir/` (project)** | `meta_learner.save_enhanced_learnings()` | Same content (dual-write) |
| **`gtd-strategy-library.md`** | **`~/.claude/gtd/` (GLOBAL)** | `meta_learner.update_strategy_library()` | Upserted archetype: proven path, HP starting points, avoid, session count |
| **`gtd-strategy-library.md`** | **`memory_dir/` (project)** | `meta_learner.update_strategy_library()` | Same content (dual-write) |
| `gtd-meta-scores.jsonl` | `memory_dir/` (project only) | `meta_learner.record_session_metrics()` | Appended JSON line: composite score (60% quality + 25% efficiency + 15% economy) |

### Step 3: `register_model()` — Final registration

| File | Location | Written By | Content |
|------|----------|-----------|---------|
| `.gtd-state.json` | Project root (cwd) | `registry.register_model()` | current_best ID + models array with scores, paths, metadata |

---

## Summary: All Files by Type

### Global files (`~/.claude/gtd/`) — shared across ALL projects

| File | Format | When Written | What It Captures |
|------|--------|-------------|-----------------|
| `high-level-observations.md` | Markdown | `synthesize_session()` | LLM-written transferable insights (general ML wisdom) |
| `gtd-learnings.md` | Markdown | `export_model()` | Structured per-session entries: fingerprint, strategy path, HP ranges, anti-patterns |
| `gtd-strategy-library.md` | Markdown | `export_model()` | Aggregated archetypes: "Medium binary classification → proven path: lgbm" |

### Project-scoped files (`memory_dir/`) — duplicate of global + project-only metrics

| File | Format | When Written | What It Captures |
|------|--------|-------------|-----------------|
| `high-level-observations.md` | Markdown | `synthesize_session()` | Same as global (dual-write) |
| `gtd-learnings.md` | Markdown | `export_model()` | Same as global (dual-write) |
| `gtd-strategy-library.md` | Markdown | `export_model()` | Same as global (dual-write) |
| `gtd-meta-scores.jsonl` | JSONL | `export_model()` | Composite scores per session (project-only, not global) |

### Workspace files (`workspace_path/`) — specific to one training session

| File | Format | When Written | What It Captures |
|------|--------|-------------|-----------------|
| `metadata.json` | JSON | Every `train_model()` | Runs list, best run, primary metric |
| `run_log.jsonl` | JSONL | Every `train_model()` | One line per run: model, score, params |
| `dataset_fingerprint.json` | JSON | First `train_model()` | Size class, task, feature mix, issues |
| `memory_dir.txt` | Text | First `train_model()` | Persisted memory_dir path for auto-discovery |
| `observation-log.md` | Markdown | Every 3 runs | Within-run reflections (Reflexion pattern) |
| `observation-log-TIMESTAMP.md` | Markdown | `synthesize_session()` | Archived copy of observation log |
| `.session_synthesized` | Text | `synthesize_session()` | Gate flag for `register_model()` |
| `runs/run_NNN_MODEL/` | Dir | Every `train_model()` | model.joblib + config.json + metrics.json |
| `exports/EXPORT_NAME/` | Dir | `export_model()` | Best model copy + metadata |

### Project root

| File | Format | When Written | What It Captures |
|------|--------|-------------|-----------------|
| `.gtd-state.json` | JSON | `register_model()` | Model registry: current best + history |

---

## Data Flow Diagram

```
SESSION START (Phase 1-2)
  READ: ~/.claude/gtd/high-level-observations.md  ->  prior_knowledge (injected into prompt)
  READ: ~/.claude/gtd/gtd-learnings.md            ->  match_strategies() -> strategy_recommendation
  WRITE: workspace_path/metadata.json              (workspace created)

DURING SESSION (Phase 3-4)
  WRITE: workspace_path/runs/run_NNN_*/            (model + config + metrics per run)
  WRITE: workspace_path/run_log.jsonl              (appended per run)
  WRITE: workspace_path/observation-log.md         (appended every 3 runs)
  WRITE: workspace_path/dataset_fingerprint.json   (first run only)
  WRITE: workspace_path/memory_dir.txt             (first run only)

SESSION END (Phase 5)
  Step 1 -- synthesize_session():
    WRITE: ~/.claude/gtd/high-level-observations.md        (GLOBAL, appended)
    WRITE: memory_dir/high-level-observations.md           (project, appended)
    WRITE: workspace_path/observation-log-TIMESTAMP.md     (archive)
    WRITE: workspace_path/.session_synthesized             (gate flag)

  Step 2 -- export_model():
    WRITE: workspace_path/exports/EXPORT_NAME/             (model copy)
    WRITE: ~/.claude/gtd/gtd-learnings.md                  (GLOBAL, appended)
    WRITE: memory_dir/gtd-learnings.md                     (project, appended)
    WRITE: ~/.claude/gtd/gtd-strategy-library.md           (GLOBAL, upserted)
    WRITE: memory_dir/gtd-strategy-library.md              (project, upserted)
    WRITE: memory_dir/gtd-meta-scores.jsonl                (project only, appended)

  Step 3 -- register_model():
    WRITE: .gtd-state.json                                 (project root)
```

## Key Source Files

| File | Role |
|------|------|
| `src/gtd/core/meta_learner.py` | All learning read/write functions |
| `src/gtd/core/trainer.py` | `train_model()` and `export_model()` — triggers learning side effects |
| `src/gtd/servers/training_server.py` | MCP server: `synthesize_session()`, `get_strategy_recommendation()` |
| `src/gtd/core/workspace.py` | Workspace creation, run registration, artifact storage |
| `src/gtd/core/registry.py` | Model registry (`.gtd-state.json`) |
| `commands/gtd/train.md` | The `/gtd:train` skill prompt that orchestrates the full workflow |
