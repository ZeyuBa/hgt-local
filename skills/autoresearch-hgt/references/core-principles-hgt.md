# Core Principles — HGT Autoresearch Edition (aligned with mar13)

7 universal principles adapted from Karpathy's autoresearch for this repo's HGT training workflow.

## 1. Constraint = Enabler

Autonomous iteration works because of clear boundaries, not because of unlimited freedom.

| Original autoresearch pattern | HGT mar13 adaptation |
|---|---|
| Small, fully legible scope | Explicit in-scope files: `configs/config.yaml` and allowed training/model files |
| Fixed iteration budget | Per-run hard safety cap with `timeout 3600` |
| One success metric | Primary metric locked to `val_f1_calibrated` (`eval_edge_best_f1`) |

**Why:**
- Fixed evaluation and data pipeline keep experiments comparable.
- Clear boundaries increase agent confidence and iteration speed.

**Apply in HGT before starting:**
- Define what files are in scope.
- Define the one primary metric for keep/discard.
- Define the run wrapper and time cap (`conda run -n miso` + `timeout 3600`).

## 2. Separate Strategy from Tactics

Humans set direction; agents execute experiments.

| Strategic (Human) | Tactical (Agent) |
|---|---|
| "Improve alarm link prediction quality" | Tune `learning_rate`, `n_hid`, `num_layers`, `dropout`, `num_train_epochs` |
| "Preserve evaluation integrity" | Never modify eval logic, never optimize on test |
| "Explore stronger modeling choices" | Change allowed architecture/training components inside scope |

**Why:**
- Humans optimize the WHY.
- Agents optimize the HOW through fast iteration loops.

## 3. Metrics Must Be Mechanical

If a metric cannot be extracted by command, it cannot drive autonomous decisions.

For HGT runs, extract from the final eval block in `run.log`:
- `eval_edge_best_f1` → `val_f1_cal` (primary)
- `eval_graph_accuracy` → `val_graph_acc`
- `eval_edge_auc` → `val_auc`
- `eval_loss` → `val_loss`

**Anti-pattern:** "looks better", "seems more stable", "probably improved".
These remove a deterministic keep/discard function.

## 4. Verification Must Be Fast

If verification is slower than iteration, exploration collapses.

In HGT autoresearch:
- **Fast loop (inside each experiment):** run training, parse `run.log`, decide keep/discard.
- **Slow loop (outside):** deeper human analysis, broader review, long-horizon checks.

**Apply:** use the fastest verification that preserves decision quality in the loop.

## 5. Iteration Cost Shapes Behavior

- Lower experiment cost → broader search and more bold hypotheses.
- Higher experiment cost → conservative, narrow exploration.

In this repo, lower cost by:
- Preferring single-variable experiments.
- Starting from config-level changes before larger code rewrites.
- Reverting quickly on timeout/dimension/OOM/NaN failures.

## 6. Git as Memory and Audit Trail

Commit-first experimentation creates a causal ledger:
- **Causality:** which change produced metric movement?
- **Stacking wins:** each keep becomes a new baseline.
- **Auditability:** humans can inspect the exact decision sequence.

HGT-aligned loop:
1. Commit experiment (`research: ...`).
2. Run and extract validation metrics.
3. Keep if improved by policy; otherwise discard via reset.

## 7. Honest Limitations

State capabilities and limits explicitly.

HGT-autoresearch can:
- Search training/architecture tactics within fixed constraints.

HGT-autoresearch cannot:
- Change evaluation logic to fabricate gains.
- Use test-set outcomes inside research iterations.
- Introduce new dependencies outside the project environment.
- Guarantee every experiment improves the primary metric.

When blocked, report concrete failure mode (timeout, OOM, NaN, dimension mismatch), log it, and move on.

---

## Meta-Principle

> In HGT training, autonomy scales when scope is bounded, success is singular and mechanical, verification is fast, and keep/discard discipline is enforced through Git.

This does not remove humans; it upgrades human leverage from execution to strategy.
