# HGT-autoresearch

Autonomous research loop for the HGT alarm-prediction pipeline.
Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — adapted for heterogeneous graph link prediction.

## Philosophy

Autoresearch 的核心思想：

1. **Agent 是研究员，Human 是研究主管。** Human 只编辑 `program.md`（研究组织的"源代码"），Agent 负责一切实验。
2. **固定评估、可变训练。** 评估协议和数据集锁死，Agent 只能改训练相关代码和配置。
3. **Keep/Discard 二元决策。** 每次实验要么改善了主指标就保留，要么等于或更差就回滚。没有"也许"。
4. **永不停止。** Agent 启动后自主运行直到被人打断。卡住了就换思路，不要问人。

## Setup

To set up a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `hgt-research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b hgt-research/<tag>` from current main.
3. **Read the in-scope files**: Read these for full context:
   - `docs/autoresearch/program.md` — this file, your operating instructions.
   - `docs/autoresearch/constraints.md` — experiment constraints (hard and soft).
   - `docs/autoresearch/metrics.md` — keep/discard metric definitions.
   - `configs/config.yaml` — the config you modify.
   - `src/training/config.py` — config schema (read-only, understand it).
   - `src/training/trainer.py` — training pipeline (read-only, understand it).
   - `src/models/hgt_for_link_prediction.py` — model architecture (read-only for now).
4. **Verify baseline data**: Confirm `data/synthetic/` contains transformed splits (8000 train / 500 val / 1000 test). If not, run a first pass to generate them: `conda run -n miso python main.py --config configs/config.yaml --mode train`. Then set `reuse_existing_splits: true` in config to lock the data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. Do NOT commit this file.
6. **Initialize progress.md**: Create `docs/autoresearch/progress.md` from the template in the SKILL.md. Do NOT commit this file.
7. **Initialize session log**: Create `docs/autoresearch/session-<tag>.log` as an append-only detailed log. Do NOT commit this file.
8. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

Each experiment trains the HGT model via the pipeline's **research mode**, which evaluates on validation only and never touches the test set. There is no fixed time budget per experiment — training length is controlled by `num_train_epochs` in config. A **1-hour hard timeout** exists as a safety net to kill runaway or errored experiments.

**Important:** All python commands must use the `miso` conda environment.

**Launch command:**
```bash
timeout 3600 conda run -n miso python main.py --config configs/config.yaml --mode train > run.log 2>&1
```

Note: The CLI accepts `--mode train` (not `--run-mode research`). "Research mode" is conceptual — you train normally and only look at validation metrics, ignoring test metrics.

### What you CAN do

- **Modify `configs/config.yaml`** — this is your primary lever. Everything is fair game:
  - Model architecture: `n_hid`, `num_layers`, `n_heads`, `dropout`, `conv_name`, `use_rte`
  - Training hyperparameters: `learning_rate`, `weight_decay`, `warmup_ratio`, `num_train_epochs`
  - Batching: `per_device_train_batch_size`
  - Synthetic data parameters: `num_sites`, `wl_stations_per_site`, `fault_site_count`, etc.

- **Modify `src/models/` code** — if you want to change model architecture beyond config knobs:
  - `hgt_for_link_prediction.py` — loss function, prediction head
  - `edge_predictor.py` — scoring mechanism
  - `hgt.py` — encoder wrapper

- **Modify `src/graph/feature_extraction.py`** — to change the 32-dim feature layout.

- **Add new files under `src/`** — if needed for new techniques (e.g., a new loss function, a new attention mechanism).

### What you CANNOT do

- **Modify evaluation/metrics code** in `src/training/trainer.py` (`compute_link_prediction_metrics`). This is the ground truth.
- **Modify `src/training/config.py`** — the config loader is fixed infrastructure.
- **Modify `main.py`** — the entrypoint is fixed.
- **Modify test files** — `tests/` are fixed verification.
- **Modify `src/dataset/`** — data loading/batching pipeline is fixed.
- **Modify `training_data/`** — data generation code is fixed. Change parameters via config, not code.
- **Look at or use test set results** — research mode only sees validation. Test set is reserved for humans.
- **Install new packages** — use what's already in the project.

### The goal

**Get the highest `val_f1_calibrated` (validation F1 at calibrated threshold).**

Secondary goals (in priority order):
1. Maximize `val_graph_accuracy_calibrated` (whole-graph correctness)
2. Maximize `val_auc` (ranking quality)
3. Minimize `val_loss` (training signal quality)

See `docs/autoresearch/metrics.md` for the full keep/discard decision logic.

### Baseline first

Your very first run must establish the baseline. Run the pipeline with the default config as-is.

## Output format

After each run, extract val metrics from `run.log`. The pipeline does NOT write a standalone
`validation_metrics.json` — val metrics are logged by the HF Trainer at each epoch with `eval_` prefix.

**Extract val metrics from the last eval epoch:**
```bash
conda run -n miso python -c "
import re, sys
with open('run.log') as f:
    text = f.read()
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
| `eval_edge_best_f1` | `val_f1_cal` | F1 at best threshold (calibrated) — PRIMARY |
| `eval_graph_accuracy` | `val_graph_acc` | Whole-graph correctness |
| `eval_edge_auc` | `val_auc` | Edge-level AUC |
| `eval_loss` | (monitored) | Validation loss |

If the run crashed, check:
```bash
tail -n 50 run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and 6 columns:

```
commit	val_f1_cal	val_graph_acc	val_auc	status	description
```

1. git commit hash (short, 7 chars)
2. val_f1_calibrated (e.g. 0.8500) — use 0.0000 for crashes
3. val_graph_accuracy_calibrated (e.g. 0.7200) — use 0.0000 for crashes
4. val_auc (e.g. 0.9100) — use 0.0000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:
```
commit	val_f1_cal	val_graph_acc	val_auc	status	description
a1b2c3d	0.8200	0.6800	0.9000	keep	baseline
b2c3d4e	0.8500	0.7200	0.9200	keep	increase n_hid to 128
c3d4e5f	0.8100	0.6500	0.8900	discard	switch to GAT conv
d4e5f6g	0.0000	0.0000	0.0000	crash	6 heads on 64-dim (indivisible)
```

Do NOT commit results.tsv — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `hgt-research/mar11`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. **Hypothesize**: pick an experimental idea based on prior results and domain knowledge.
3. **Update progress.md**: fill "Current experiment" table with hypothesis, changes, config. Complete the constraint checklist.
4. Apply the change — modify config and/or model code.
5. `git commit -m "research: <description>"`
6. **Update progress.md**: set Status=COMMITTED, update Commit field.
7. **Log to session log**: hypothesis, changes, config snapshot, constraint checklist, commit hash.
8. Run the experiment: `timeout 3600 conda run -n miso python main.py --config configs/config.yaml --mode train > run.log 2>&1`
9. **Update progress.md**: paste epoch-by-epoch eval_loss, set Status=EVALUATING.
10. Read out the results: extract val metrics from the last eval epoch in `run.log` (see Output format section above).
11. If metrics extraction fails, the run crashed. Read `tail -n 50 run.log` and attempt a fix.
12. **Update progress.md**: fill Result section with metrics, set Status=DECIDING.
13. Record the results in results.tsv.
14. **Log to session log**: metrics, epoch-by-epoch eval_loss, training duration.
15. **Keep/Discard decision** (see `docs/autoresearch/metrics.md`):
    - If `val_f1_calibrated` improved → **KEEP** the commit, advance the branch.
    - If `val_f1_calibrated` equal or worse → **DISCARD** via `git reset --hard HEAD~1`.
    - Exception: equal F1 but better graph_accuracy or simpler code → KEEP.
16. **Update progress.md**: fill Decision section, update "Current best" table if KEEP, add row to "Experiment history". Set Status=DONE. Clear "Current experiment" for next iteration.
17. **Log to session log**: decision with delta reasoning, observations.
18. Go to step 2.

### Idea generation strategy

When looking for experiments to try, consider these categories:

**Quick wins (try first):**
- Learning rate sweep (0.0005, 0.001, 0.002, 0.005)
- Hidden dimension scaling (64 → 128 → 256)
- Number of HGT layers (2, 4, 6, 8)
- Dropout tuning (0.1, 0.2, 0.3)
- Training epochs (8, 16, 32, 64, 128) — longer training is encouraged with 8000 train graphs

**Architecture exploration:**
- Number of attention heads vs hidden dim ratio
- Edge predictor design (bilinear → MLP, dot-product)
- Skip connections / residual patterns
- use_rte (relative temporal encoding)
- Different conv types if available

**Data & regularization:**
- Noise probability adjustment
- Weight decay sweep
- Warmup ratio sweep
- Batch size effects
- Topology complexity (num_sites, fault_site_count)

**Advanced (if quick wins plateau):**
- Feature engineering changes (feature_extraction.py)
- Loss function modifications (focal loss, label smoothing)
- Multi-scale aggregation
- Attention mechanism modifications

### Timeout

There is no strict time-per-experiment limit — longer training (more epochs, larger model) is encouraged if it improves metrics. However, a **1-hour hard timeout** is enforced via the `timeout 3600` wrapper. If a run hits the 1-hour wall, treat it as a crash: the experiment was too expensive or stuck in an error loop. Reduce epochs or model size and retry.

### Crashes

If a run crashes: fix if trivial (typo, dimension mismatch). If fundamentally broken, log "crash", revert, move on.

### NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human. The human may be away. You are autonomous. If you run out of ideas, re-read the in-scope files, recombine prior near-misses, try more radical changes. The loop runs until the human interrupts you.
