---
name: hgt-autoresearch
description: >
  Autonomous research loop that continuously improves HGT alarm-prediction model performance
  through serial experimentation. Trigger this skill whenever the user mentions "autoresearch",
  "research loop", "experiment loop", "autonomous experiments", "improve HGT", or wants to
  start an automated hyperparameter/architecture search for the HGT model. Also trigger when the
  user says "start researching", "run experiments", or "optimize the model autonomously".
---

# HGT Autoresearch

You are an autonomous ML researcher. Your job is to run a serial experiment loop that continuously
improves the HGT alarm-prediction model by modifying config and model code, training, evaluating
on validation only, and keeping improvements or discarding regressions.

This is inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the human
writes the research program, the agent executes it indefinitely.

## Environment requirement

**All python and pytest commands MUST run inside the `miso` conda environment.** This project's
dependencies (torch, torch_geometric, transformers, sklearn, etc.) are only installed there.

Use `conda run -n miso` as a prefix for every python invocation:
```bash
conda run -n miso python main.py --config configs/config.yaml --mode train
conda run -n miso pytest -q tests/
conda run -n miso python -c "import json; ..."
```

Never use bare `python` or `pip install` — the base environment lacks the required packages and
installing into it violates the reproducibility constraint.

## Quick start

If the user says something like "start autoresearch" or "run the research loop", follow these steps:

1. Read all three reference files in `docs/autoresearch/` for full context:
   - `docs/autoresearch/program.md` — operating instructions & experiment loop
   - `docs/autoresearch/constraints.md` — hard/soft constraints checklist
   - `docs/autoresearch/metrics.md` — keep/discard decision logic
2. Read `configs/config.yaml` to understand the current configuration state.
3. Follow the **Setup** section in `program.md` to initialize a session (branch, baseline data, results.tsv, progress.md).
4. Enter the **Experiment Loop** and run indefinitely until interrupted.

The three reference files are the complete operating manual — read them carefully before the first
experiment. The rest of this skill gives you the condensed protocol for quick reference during the loop.

## CLI contract

The entrypoint is `main.py`. It accepts these arguments:

```
python main.py --config configs/config.yaml --mode train
```

- `--config` (required): path to the YAML config file.
- `--mode` (optional): `train` (default) or `inference`.

There is **no** `--run-mode` flag. The pipeline always runs in `full` mode internally.
"Research mode" is a conceptual term — it simply means you train (`--mode train`) and only
look at validation metrics, never test metrics. The pipeline writes both, but you ignore test.

## Condensed protocol

### Setup (once per session)

```
1. Agree on run tag with user (e.g. "mar12")
2. git checkout -b hgt-research/<tag>
3. Read docs/autoresearch/{program,constraints,metrics}.md
4. Verify data: wc -l data/synthetic/transformed_{train,val,test}.json → 8000/500/1000
5. Create results.tsv header (do NOT commit it)
6. Initialize progress.md (see Progress tracking section below)
7. Run baseline experiment → record in results.tsv AND progress.md
```

### Experiment loop (runs forever)

```
LOOP:
  1. Hypothesize — pick ONE change based on prior results
  2. Apply change to config.yaml and/or model code
  3. Self-check constraints (see checklist in constraints.md)
  4. **MANDATORY GIT COMMIT:** git commit -am "research: <short description of change>"
     - Each experiment MUST be a single, atomic commit
     - Commit message should clearly describe what was changed
     - This is your only checkpoint — no commits = no way to revert if things break
  5. timeout 3600 conda run -n miso python main.py --config configs/config.yaml --mode train > run.log 2>&1
  6. Extract metrics from outputs/results/validation_metrics.json
  7. Record in results.tsv
  8. Update progress.md with experiment result and reasoning
  9. KEEP (f1 improved) → leave commit, advance baseline
     DISCARD (f1 equal/worse) → git reset --hard HEAD~1
  10. GOTO 1
```

### Key rules

- **Serial only** — one experiment at a time, each builds on the last kept result.
- **Never stop** — don't pause to ask the user. If stuck, try something different.
- **Validation only** — never look at test set results. Use `--mode train` and only read validation metrics.
- **1-hour timeout** — use `timeout 3600`. Timeout = crash.
- **Single variable** — change one thing per experiment when possible.
- **Git = experiment log** — kept experiments are commits on the branch, discarded ones are reset.
- **Always miso** — every python command goes through `conda run -n miso`.

### What you CAN modify

- `configs/config.yaml` — all model/training/batching/synthetic parameters
- `src/models/` — architecture, loss function, prediction head
- `src/graph/feature_extraction.py` — feature layout (sync with model.in_dim)
- New files under `src/` — new techniques

### What you CANNOT modify

- `src/training/trainer.py` (metrics computation) — ground truth
- `src/training/config.py` — config loader infrastructure
- `main.py` — fixed entrypoint
- `tests/` — fixed verification
- `src/dataset/` — data loading pipeline
- `training_data/` — data generation code

### Keep/discard decision (condensed)

Primary metric: **val_f1_calibrated** from `validation_metrics.json → f1_calibrated`

```
f1_new > f1_best         → KEEP
f1_new ≈ f1_best (±0.001) → tie-break on graph_accuracy_calibrated, code simplicity
f1_new < f1_best         → DISCARD
```

### Results logging

Append to `results.tsv` (tab-separated, never committed):

```
commit	val_f1_cal	val_graph_acc	val_auc	status	description
a1b2c3d	0.8200	0.6800	0.9000	keep	baseline
```

### Progress tracking (progress.md)

Create and maintain `docs/autoresearch/progress.md` throughout the session. This file is the
human-readable narrative of the research session — it records not just metrics but reasoning,
observations, and important context that results.tsv cannot capture.

**Initialize at session start** with this template:

```markdown
# HGT Autoresearch Progress — <tag>

Started: <date and time>
Branch: hgt-research/<tag>
Baseline config: n_hid=64, num_layers=4, n_heads=4, lr=0.001, epochs=8, dropout=0.2

## Session summary

| # | Experiment | val_f1_cal | val_graph_acc | val_auc | Status | Commit |
|---|-----------|-----------|--------------|--------|--------|--------|
| 0 | baseline  | —         | —            | —      | —      | —      |

## Experiment log

### Exp 0: Baseline
- **Hypothesis:** Establish baseline metrics with default config.
- **Changes:** None (default config).
- **Result:** val_f1_cal=X.XXXX, val_graph_acc=X.XXXX, val_auc=X.XXXX
- **Decision:** KEEP (baseline)
- **Commit:** <short hash>
- **Observations:** <any notable patterns from the training log>
```

**After each experiment**, append a new entry to the experiment log AND update the summary table.
Each entry should include:

1. **Hypothesis** — what you expected and why
2. **Changes** — exact config diffs or code modifications
3. **Result** — the three key metrics
4. **Decision** — KEEP/DISCARD/CRASH with brief reasoning
5. **Commit** — the git short hash (if kept, the live commit; if discarded, note "reverted")
6. **Observations** — anything notable: training dynamics, loss curves, convergence speed,
   surprising results, ideas for next experiment

**Do NOT commit progress.md** — it stays untracked alongside results.tsv. But DO update it
after every single experiment. If the session is interrupted, progress.md is the user's primary
record of what happened and why.

When consecutive experiments are discarded (3+ in a row), add a **Strategy note** explaining
what pattern you see and what direction you'll try next.

### Idea priority

1. **Quick wins first:** lr sweep, n_hid scaling, num_layers, dropout, epochs
2. **Architecture:** head count, edge predictor design, skip connections, use_rte
3. **Regularization:** weight_decay, warmup_ratio, batch size
4. **Advanced:** feature engineering, loss function mods, attention modifications
