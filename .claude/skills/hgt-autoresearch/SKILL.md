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

## Quick start

If the user says something like "start autoresearch" or "run the research loop", follow these steps:

1. Read all three reference files in `docs/autoresearch/` for full context:
   - `docs/autoresearch/program.md` — operating instructions & experiment loop
   - `docs/autoresearch/constraints.md` — hard/soft constraints checklist
   - `docs/autoresearch/metrics.md` — keep/discard decision logic
2. Read `configs/config.yaml` to understand the current configuration state.
3. Follow the **Setup** section in `program.md` to initialize a session (branch, baseline data, results.tsv).
4. Enter the **Experiment Loop** and run indefinitely until interrupted.

The three reference files are the complete operating manual — read them carefully before the first
experiment. The rest of this skill gives you the condensed protocol for quick reference during the loop.

## Condensed protocol

### Setup (once per session)

```
1. Agree on run tag with user (e.g. "mar12")
2. git checkout -b hgt-research/<tag>
3. Read docs/autoresearch/{program,constraints,metrics}.md
4. Verify data: wc -l data/synthetic/transformed_{train,val,test}.json → 8000/500/1000
5. Ensure config has: reuse_existing_splits: true
6. Create results.tsv header (do NOT commit it)
7. Run baseline experiment → record in results.tsv
```

### Experiment loop (runs forever)

```
LOOP:
  1. Hypothesize — pick ONE change based on prior results
  2. Apply change to config.yaml and/or model code
  3. Self-check constraints (see checklist in constraints.md)
  4. git commit -m "research: <description>"
  5. timeout 3600 python main.py --config configs/config.yaml --run-mode research > run.log 2>&1
  6. Extract metrics from outputs/results/validation_metrics.json
  7. Record in results.tsv
  8. KEEP (f1 improved) → advance branch
     DISCARD (f1 equal/worse) → git reset --hard HEAD~1
  9. GOTO 1
```

### Key rules

- **Serial only** — one experiment at a time, each builds on the last kept result.
- **Never stop** — don't pause to ask the user. If stuck, try something different.
- **Validation only** — never look at test set results. Use `--run-mode research`.
- **1-hour timeout** — use `timeout 3600`. Timeout = crash.
- **Single variable** — change one thing per experiment when possible.
- **Git = experiment log** — kept experiments are commits on the branch, discarded ones are reset.

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

### Idea priority

1. **Quick wins first:** lr sweep, n_hid scaling, num_layers, dropout, epochs
2. **Architecture:** head count, edge predictor design, skip connections, use_rte
3. **Regularization:** weight_decay, warmup_ratio, batch size
4. **Advanced:** feature engineering, loss function mods, attention modifications
