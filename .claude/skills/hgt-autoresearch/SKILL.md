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
7. Initialize session log file (see Session log section below)
8. Run baseline experiment → record in results.tsv, progress.md, AND session log
```

### Experiment loop (runs forever)

```
LOOP:
  1. Hypothesize — pick ONE change based on prior results
  2. Update progress.md: fill "Current experiment" table + constraint checklist
  3. Apply change to config.yaml and/or model code
  4. Self-check constraints (see checklist in constraints.md)
  5. **MANDATORY GIT COMMIT:** git commit -am "research: <short description of change>"
     - Each experiment MUST be a single, atomic commit
     - Commit message should clearly describe what was changed
     - This is your only checkpoint — no commits = no way to revert if things break
  6. Update progress.md: set Status=COMMITTED, update Commit field
  7. Log experiment start to session log (hypothesis, changes, checklist, commit)
  8. timeout 3600 conda run -n miso python main.py --config configs/config.yaml --mode train > run.log 2>&1
  9. Update progress.md: paste epoch-by-epoch eval_loss, set Status=EVALUATING
  10. Extract val metrics from run.log (see Metric extraction section below)
  11. Update progress.md: fill Result section, set Status=DECIDING
  12. Record in results.tsv
  13. Log metrics and training dynamics to session log
  14. Make keep/discard decision:
      KEEP (f1 improved) → leave commit, advance baseline
      DISCARD (f1 equal/worse) → git reset --hard HEAD~1
  15. Update progress.md: fill Decision, update Current best + Experiment history, set Status=DONE
  16. Log decision to session log
  17. Clear progress.md "Current experiment" for next iteration
  18. GOTO 1
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

### Metric extraction

The pipeline does NOT write a standalone `validation_metrics.json`. Val metrics are logged by the
HuggingFace Trainer to stdout (captured in `run.log`) at each epoch with an `eval_` prefix.

**Extract val metrics from the last eval epoch in run.log:**

```bash
# Get the last eval line (final epoch metrics)
grep -o "'eval_edge_best_f1': [0-9.]*" run.log | tail -1
grep -o "'eval_graph_accuracy': [0-9.]*" run.log | tail -1
grep -o "'eval_edge_auc': [0-9.]*" run.log | tail -1
grep -o "'eval_loss': [0-9.]*" run.log | tail -1
```

**Alternatively, use a one-liner python extraction:**

```bash
conda run -n miso python -c "
import re, sys
with open('run.log') as f:
    text = f.read()
# Find last eval block
evals = re.findall(r'\{[^}]*eval_edge_best_f1[^}]*\}', text)
if not evals:
    print('ERROR: no eval metrics found in run.log'); sys.exit(1)
last = eval(evals[-1])
print(f'val_f1_cal: {last.get(\"eval_edge_best_f1\", \"N/A\")}')
print(f'val_graph_acc: {last.get(\"eval_graph_accuracy\", \"N/A\")}')
print(f'val_auc: {last.get(\"eval_edge_auc\", \"N/A\")}')
print(f'val_loss: {last.get(\"eval_loss\", \"N/A\")}')
"
```

**Key metric name mapping** (HF log key → what we track):

| run.log key | results.tsv column | Description |
|---|---|---|
| `eval_edge_best_f1` | `val_f1_cal` | F1 at best threshold (calibrated) |
| `eval_graph_accuracy` | `val_graph_acc` | Whole-graph correctness |
| `eval_edge_auc` | `val_auc` | Edge-level AUC |
| `eval_loss` | (monitored) | Validation loss |

If `run.log` has no eval metrics, the run crashed — check `tail -n 50 run.log` for the error.

### Keep/discard decision (condensed)

Primary metric: **val_f1_calibrated** = `eval_edge_best_f1` from the last eval epoch in `run.log`

```
f1_new > f1_best         → KEEP
f1_new ≈ f1_best (±0.001) → tie-break on graph_accuracy, code simplicity
f1_new < f1_best         → DISCARD
```

### Results logging

Append to `results.tsv` (tab-separated, never committed):

```
commit	val_f1_cal	val_graph_acc	val_auc	status	description
a1b2c3d	0.8200	0.6800	0.9000	keep	baseline
```

### Progress tracking (progress.md)

`docs/autoresearch/progress.md` is the **live dashboard for the current experiment**. It shows
what's happening right now — which step of the experiment lifecycle you're in, what you changed,
and what you're waiting for. The user can check this file at any time to see exactly where the
current experiment stands.

**Create immediately during setup** (before running any experiments):

```bash
# Create the file with the template below
cat > docs/autoresearch/progress.md << 'TEMPLATE'
... (see template below)
TEMPLATE
```

**Full template:**

```markdown
# HGT Autoresearch Progress — <tag>

Started: <YYYY-MM-DD HH:MM>
Branch: hgt-research/<tag>
Data: 8000 train / 500 val / 1000 test (reuse_existing_splits: true)

## Current best

| Metric | Value | Set by |
|--------|-------|--------|
| val_f1_cal | — | — |
| val_graph_acc | — | — |
| val_auc | — | — |

## Current experiment

| Field | Value |
|-------|-------|
| Exp # | 0 |
| Title | Baseline |
| Status | PENDING |
| Hypothesis | Establish baseline metrics with default config |
| Changes | None (default config) |
| Config | n_hid=64, num_layers=4, n_heads=4, lr=0.001, epochs=8, dropout=0.2 |
| Commit | — |
| Started at | — |
| Duration | — |

### Constraint checklist
- [ ] Modified files in allowed scope
- [ ] Did not touch evaluation code
- [ ] No new dependencies
- [ ] Feature dim matches model.in_dim
- [ ] num_types=3, num_relations=9 unchanged
- [ ] Using --mode train (validation only)
- [ ] Only looking at validation metrics, not test
- [ ] Single variable change (or logically related group)
- [ ] Expected training time < 1 hour (timeout 3600)

### Live training output
(Updated as training progresses — paste epoch-by-epoch eval_loss here)

### Result
- val_f1_cal: —
- val_graph_acc: —
- val_auc: —
- val_loss: —

### Decision
— (KEEP/DISCARD/CRASH + reasoning)

## Experiment history

| # | Experiment | val_f1_cal | val_graph_acc | val_auc | Status | Commit |
|---|-----------|-----------|--------------|--------|--------|--------|
```

**Update protocol — update progress.md at each step of EVERY experiment:**

1. **Before experiment:** Fill in the "Current experiment" table (Exp #, Title, Hypothesis,
   Changes, Config). Set Status to `PREPARING`. Fill in the constraint checklist — check every
   box before proceeding.
2. **After git commit:** Update Commit field. Set Status to `COMMITTED`.
3. **Training started:** Update Started at. Set Status to `TRAINING`.
4. **Training completes:** Fill in Duration. Paste epoch-by-epoch eval_loss into "Live training
   output". Set Status to `EVALUATING`.
5. **Metrics extracted:** Fill in the Result section with 4 decimal places.
   Set Status to `DECIDING`.
6. **After keep/discard:** Fill in the Decision section. Set Status to `DONE`.
   Update "Current best" table if KEEP. Add a row to the "Experiment history" table.
   Then clear the "Current experiment" section for the next experiment.

**Do NOT commit progress.md** — it stays untracked. If the session is interrupted, the user
can immediately see where things stand by reading this file.

### Session log (detailed experiment log file)

Create and maintain `docs/autoresearch/session-<tag>.log` as a detailed, append-only log file
that captures the full record of every experiment in the research session. This is the permanent
history — progress.md shows the *current* state, the session log is the *complete* record.

**Create during setup:**

```bash
echo "=== HGT Autoresearch Session: <tag> ===" > docs/autoresearch/session-<tag>.log
echo "Started: $(date)" >> docs/autoresearch/session-<tag>.log
echo "Branch: hgt-research/<tag>" >> docs/autoresearch/session-<tag>.log
echo "" >> docs/autoresearch/session-<tag>.log
```

**Append to the session log at each of these moments:**

1. **Before each experiment** — log hypothesis, changes, AND the constraint checklist result:
   ```
   --- Exp N: <title> ---
   Time: <timestamp>
   Hypothesis: <what and why>
   Changes:
     - configs/config.yaml: learning_rate: 0.001 → 0.002
     - src/models/hgt.py: added residual connection in forward() (line 45)
   Config snapshot: n_hid=64, num_layers=4, n_heads=4, lr=0.002, epochs=8, dropout=0.2
   Constraint checklist:
     [x] Modified files in allowed scope
     [x] Did not touch evaluation code
     [x] No new dependencies
     [x] Feature dim matches model.in_dim
     [x] num_types=3, num_relations=9 unchanged
     [x] Using --mode train (validation only)
     [x] Only looking at validation metrics, not test
     [x] Single variable change
     [x] Expected training time < 1 hour
   Commit: <full hash>
   ```

2. **After training completes** — log extracted metrics AND training dynamics:
   ```
   Training completed: <duration>
   Epoch-by-epoch eval_loss: [1.08, 0.69, 0.74, 0.56]
   Final metrics:
     eval_edge_best_f1: 0.8234
     eval_graph_accuracy: 0.7100
     eval_edge_auc: 0.9050
     eval_loss: 0.5561
   ```

3. **After keep/discard decision** — log decision with delta reasoning:
   ```
   Decision: KEEP — f1 improved 0.8200 → 0.8234 (+0.0034)
   New best f1: 0.8234
   Observations: <what was learned, ideas for next experiment>
   ```
   or:
   ```
   Decision: DISCARD — f1 dropped 0.8200 → 0.7900 (-0.0300)
   Reverted: git reset --hard HEAD~1
   Observations: <why it failed, what to avoid>
   ```

4. **On crash** — log the error:
   ```
   CRASH: RuntimeError: CUDA out of memory
   Last 5 lines of run.log:
     <paste from tail -5 run.log>
   Action: reverting, will reduce batch size
   ```

5. **Strategy pivots** — when changing approach direction:
   ```
   === STRATEGY PIVOT ===
   After 3 discards on lr sweep (0.002, 0.005, 0.0005), lr=0.001 appears optimal.
   Pivoting to architecture exploration: will try n_hid scaling next.
   ```

**Do NOT commit the session log** — it stays untracked. The session log combined with
progress.md gives the user a complete picture: progress.md for live status, session log for
the full history.

### Idea priority

1. **Quick wins first:** lr sweep, n_hid scaling, num_layers, dropout, epochs
2. **Architecture:** head count, edge predictor design, skip connections, use_rte
3. **Regularization:** weight_decay, warmup_ratio, batch size
4. **Advanced:** feature engineering, loss function mods, attention modifications
