# Experiment Adaptation — Two-Phase Verification for ML Training

Adapts the standard autoresearch loop to this project's workflow: local smoke test as crash gate, then formal remote experiment (3 seeds on GPU) for authoritative metrics.

## Why Two Phases

The standard autoresearch loop assumes verification completes in seconds. ML training on full data takes hours. Running every speculative change on a GPU cluster is wasteful — most ideas crash or obviously fail.

The solution: a cheap local smoke test filters out crashes before committing GPU time. Only non-crashing changes proceed to the expensive formal experiment.

```
Modify → Smoke (local, ~30s) → Formal Experiment (remote, hours) → Keep/Discard
              |                         |
              | CRASH → fix/discard     | mean(eval_graph_accuracy) across 3 seeds
              | (skip remote)           | drives keep/discard decision
```

## Adapted Loop

The autoresearch loop phases from `autonomous-loop-protocol.md` remain the same. Phase 5 (Verify) expands into a remote execution sub-loop:

```
LOOP:
  1. Review      — read in-scope files, results log, git history, PROGRESS.md
  2. Ideate      — pick next change
  3. Modify      — make ONE focused change
  4. Commit      — git commit (before verification, for clean rollback)
  5a. Smoke Test — run smoke_verify.py locally (crash gate only)
      - CRASH (exit 1) → attempt fix (max 3 tries), else revert + log "crash"
      - NO CRASH       → proceed to 5b
  5b. Formal Launch — run via /running-experiments skill
      - Create experiment dir, configs, scripts (Phases 3-4)
      - Skip Phase 5 (local test) — smoke test already covers it
      - Sync & run remotely (Phase 6.1-6.4)
  5c. TensorBoard Start — after remote outputs appear
      - Start ONE TensorBoard for this experiment
      - Record PID, port, start time in PROGRESS.md
      - If already running for this experiment, reuse it
  5d. Formal Monitor — 10-minute health-check poll loop
      - Check run script PID / experiment process still alive
      - Check run_all.log for fatal errors and completion marker
      - Check TensorBoard PID still alive; restart once if needed
  5e. Formal Collect — collect results locally (Phase 7)
      - Sync logs to local
      - Run collect script
      - Stop TensorBoard after collection or after fatal abort
  6. Decide      — keep/discard based on formal_mean AND HC-9 time gate
      - If avg_time_h > baseline_avg_time + 0.5: discard(HC-9), revert commit (even if metric improved)
      - Else if formal_mean improved: keep
      - Else: discard
  7. Log         — update results log + PROGRESS.md (include avg_time_h column)
  8. Repeat
```

## Skill Integration: Autoresearch → Running-Experiments

Each formal experiment invokes the `/running-experiments` skill. Some phases are handled by autoresearch itself and should be skipped.

| Running-Experiments Phase | In Autoresearch | Notes |
|---------------------------|-----------------|-------|
| Phase 1: Plan & Branch | **SKIP** | Autoresearch owns planning and branching |
| Phase 2: Code Changes | **SKIP** | Autoresearch Phase 3 (Modify) handles this |
| Phase 3: Directory & Configs | **RUN** | Create experiment dir, per-seed YAML configs |
| Phase 4: Scripts & Templates | **RUN** | Create run script, report template, collect script |
| Phase 5: Local Test | **SKIP** | Autoresearch smoke test (5a) already validates |
| Phase 6.1-6.4: Sync & Remote Run | **RUN** | Sync code + configs, launch on GPU |
| Phase 6.5: Launch TensorBoard | **RUN** | Autoresearch must decide start timing, record PID/port, avoid duplicates |
| Phase 7.1-7.5: Collect Results | **RUN** | Sync logs, run collect script, fill report |
| Phase 7.6: Stop TensorBoard | **RUN** | Autoresearch must stop TensorBoard after collect or fatal abort |

## Experiment Naming

Each autoresearch iteration gets a unique experiment name:

```
experiment_<MMDD>_ar_iter_<N>
```

Examples:
- `experiment_0316_ar_iter_0` — baseline
- `experiment_0316_ar_iter_1` — first change
- `experiment_0316_ar_iter_5` — fifth iteration

`<MMDD>` is the date the autoresearch session started (not each iteration's date). This groups all iterations from one session.

## Authoritative Metric

| Layer | Metric | Purpose |
|-------|--------|---------|
| Smoke test (local) | `eval_graph_accuracy` from `smoke_verify.py` | Crash detection only — value is ignored |
| Formal experiment (remote) | `mean(eval_graph_accuracy)` across 3 seeds | **Keep/discard decision signal** |
| Formal experiment (remote) | `std(eval_graph_accuracy)` across 3 seeds | Stability tracking (informational) |
| Formal experiment (remote) | Average training time across seeds | **HC-9 gate: > baseline_avg_time + 0.5h → discard regardless of metric** |

The smoke metric does NOT influence keep/discard. The formal experiment's mean across seeds is the primary metric, subject to HC-9: if average training time exceeds (baseline avg time + 0.5h), discard the iteration even if the metric improved. The baseline avg time is recorded in iteration 0 and serves as the reference for all subsequent iterations.

## Results Log Format

Extends the standard TSV from `results-logging.md` with experiment-specific columns:

```tsv
# metric_direction: higher_is_better
# metric_column: formal_mean
iteration	commit	smoke_metric	formal_mean	formal_std	avg_time_h	status	experiment	description
0	a1b2c3d	0.0400	0.5200	0.0100	0.95	baseline	experiment_0316_ar_iter_0	initial state (baseline_time=0.95h, HC-9 limit=1.45h)
1	b2c3d4e	0.0600	0.5400	0.0080	1.02	keep	experiment_0316_ar_iter_1	increase n_hid from 64 to 128
2	-	0.0000	-	-	-	crash	-	double attention heads (OOM in smoke)
3	-	0.0500	0.5100	0.0150	1.63	discard(HC-9)	experiment_0316_ar_iter_3	increase n_hid to 1024 (avg 1.63h > limit 1.45h)
4	-	0.0500	0.5100	0.0150	1.20	discard	experiment_0316_ar_iter_4	add residual connections
```

Column definitions:

| Column | Description |
|--------|-------------|
| `iteration` | Sequential counter (0 = baseline) |
| `commit` | Short git hash, `-` if reverted |
| `smoke_metric` | Value from smoke_verify.py (informational only) |
| `formal_mean` | Mean eval_graph_accuracy across 3 seeds (`-` if smoke crashed) |
| `formal_std` | Std eval_graph_accuracy across 3 seeds (`-` if smoke crashed) |
| `avg_time_h` | Average training time in hours across successful seeds (`-` if smoke crashed) |
| `status` | `baseline`, `keep`, `discard`, `discard(HC-9)`, `crash` |
| `experiment` | Experiment directory name (`-` if smoke crashed) |
| `description` | One-sentence description of the change |

**HC-9 discard rule**: if `avg_time_h > baseline_avg_time + 0.5`, set status to `discard(HC-9)` and revert the commit, even if `formal_mean` improved. The baseline avg time comes from iteration 0's `avg_time_h` column. Record the actual avg time so the pattern is visible in history.

## Smoke Test: Crash-Only Gate

The smoke test serves ONE purpose: detect crashes before wasting GPU time.

**Pass condition:** `smoke_verify.py` exits with code 0.

**Fail condition:** `smoke_verify.py` exits with non-zero code.

The smoke test's metric output is recorded in the results log for informational purposes but is NEVER used for keep/discard decisions. A change that produces low smoke accuracy but passes crash detection still proceeds to the formal experiment.

### Crash Recovery

When smoke crashes:
1. Read stderr for the error
2. Attempt fix (max 3 tries, same as standard autoresearch crash protocol)
3. If fixed, re-run smoke
4. If unfixable after 3 attempts, revert and log `crash`

## PROGRESS.md — Resuming After Interruption

Autoresearch sessions can be interrupted by session crashes, network disconnects, or user breaks. `.autoresearch/PROGRESS.md` tracks state so the agent can resume.

**When to update:** At the start and end of every sub-phase (smoke start, smoke end, formal launch, formal complete, decision).

**How to resume:** At the start of any autoresearch session, check if `.autoresearch/PROGRESS.md` exists and has an incomplete iteration. If so:

| Interrupted During | Resume Action |
|--------------------|---------------|
| Smoke test | Re-run smoke from the beginning |
| Formal experiment (running) | Check if remote process is still alive; if yes, resume 10-minute poll loop; if no, check logs for completion or crash |
| Formal experiment (failed) | If PROGRESS.md shows `formal_failed`, the experiment crashed during a health check. Revert commit and log `crash`, then proceed to next iteration |
| Formal experiment (collecting) | Re-sync logs and re-run collect script |
| Decision phase | Re-read formal results and make the keep/discard decision |

## Commit Timing

Standard autoresearch commits BEFORE verification. With two-phase verify:

1. **Commit** after making the code change (Phase 4)
2. **Run smoke** (Phase 5a) — if crash, revert commit
3. **Run formal experiment** (Phase 5b) — if metric doesn't improve, revert commit
4. **Keep** — commit stays

This preserves the clean-rollback property: `git reset --hard HEAD~1` always undoes exactly one iteration.

## Practical Notes

### Wait Times & Health Monitoring

Formal experiments take 1-4 hours depending on GPU availability and whether seeds run in parallel. The agent should:
- Launch the experiment
- Wait until experiment outputs exist, then start TensorBoard
- Poll every **10 minutes** (`Start-Sleep -Seconds 600` — this is the maximum allowed sleep duration)
- Treat every poll as a combined process/log/TensorBoard health check
- Look for `All experiments completed!` in the log

**Health check protocol (every 10-minute poll cycle):**

Each poll is not just a completion check — it's a health check. The agent MUST verify the experiment is still running successfully:

```
POLL LOOP (every 10 minutes):
  1. Sleep: Start-Sleep -Seconds 600  (MAXIMUM allowed)
  2. Check experiment process / PID
     - If process NOT found AND log does NOT contain "All experiments completed!":
       → Experiment died unexpectedly. Read last 50 lines of log for error.
       → Log status in PROGRESS.md: "formal_failed — process died at <timestamp>"
       → Stop TensorBoard if it was started
       → Abort this iteration (revert commit, log "crash")
  3. Check log tail for errors
     - Look for CUDA OOM, Python tracebacks, "Error", "FAILED" patterns
     - If fatal error found:
       → Log error details in PROGRESS.md
       → Stop TensorBoard if it was started
       → Abort this iteration (revert commit, log "crash")
  4. Check TensorBoard PID
     - If TensorBoard PID missing but experiment still running:
       → Restart TensorBoard once, record new PID/port in PROGRESS.md
       → If restart fails, continue experiment and note "tb_down" (non-fatal)
  5. Check for completion marker in log
     - If found → exit poll loop, proceed to result collection
  6. Report: Print brief status (e.g., "Poll #3: experiment running, TB alive, last log line: Epoch 5/10...")
  7. Repeat from step 1
```

**Why fixed 10-minute intervals (not exponential backoff):**
- Exponential backoff risks missing a crash for 20-40+ minutes
- 10-minute intervals balance responsiveness with SSH overhead
- Early crash detection saves hours of wasted waiting

**`Start-Sleep` hard limit:** Never use `Start-Sleep -Seconds` with a value greater than 600 (10 minutes). This ensures the agent remains responsive and detects failures promptly.

### TensorBoard Lifecycle

TensorBoard is part of the formal experiment phase, not an optional afterthought. Autoresearch should treat it as a managed background service with explicit start, health, and stop steps.

**Start phase:**
- Start only after confirming remote outputs exist (`outputs/seed_42/runs/` contains event files or checkpoints)
- Start exactly one TensorBoard per experiment
- Record **PID**, **port**, **start time**, and **status** in `PROGRESS.md`
- If resuming from interruption, first check whether the recorded PID is still alive before starting a new one

**Monitor phase:**
- During every 10-minute poll, verify the TensorBoard PID is still alive
- If TensorBoard died but training is still healthy, restart it once and record the new PID
- TensorBoard failure is non-fatal for the experiment, but it must be noted in `PROGRESS.md`

**Stop phase:**
- Stop TensorBoard immediately after results collection completes
- Also stop it on fatal aborts (process died, fatal log error, manual discard after crash)
- Verify the PID is gone before marking the iteration complete

### PID & Log Polling Templates

Use these as the default autoresearch monitoring snippets. `running-experiments` already covers the raw launch template; the snippets below define the autoresearch health-check loop.

#### 1. Capture launch PID

```powershell
$REMOTE = "root@10.44.101.62"
$REMOTE_DIR = "/data1/b00953788/NFMs/outage_prediction_mini_batch_three_layer"
$EXP = "experiments/experiment_<MMDD>_ar_iter_<N>"

# Launch and capture PID from the remote shell
$RUN_PID = ssh $REMOTE "cd ${REMOTE_DIR}; nohup ./$EXP/run_experiment.sh > /dev/null 2>&1 & echo \$!"
```

Record `$RUN_PID` in `PROGRESS.md`. Then confirm the process exists:

```powershell
ssh $REMOTE "ps -p $RUN_PID -o pid,cmd"
```

#### 2. Start TensorBoard and capture PID

```powershell
$TB_PORT = 6006
$TB_PID = ssh $REMOTE "cd ${REMOTE_DIR}; nohup tensorboard --logdir $EXP/outputs --bind_all --port $TB_PORT > $EXP/logs/tensorboard.log 2>&1 & echo \$!"
ssh $REMOTE "ps -p $TB_PID -o pid,cmd"
```

If port `6006` is occupied, pick a free port, record it in `PROGRESS.md`, and reuse that port for the rest of the iteration.

#### 3. One poll cycle

```powershell
Start-Sleep -Seconds 600

# Process liveness
ssh $REMOTE "ps -p $RUN_PID -o pid,cmd"

# Main experiment log
ssh $REMOTE "tail -20 ${REMOTE_DIR}/$EXP/logs/run_all.log"
```

Default monitoring should poll the experiment process and `run_all.log` only. Do not poll TensorBoard status unless the user explicitly asks for TensorBoard health checks or TensorBoard itself is being debugged.

#### 4. Completion check

```powershell
ssh $REMOTE "tail -20 ${REMOTE_DIR}/$EXP/logs/run_all.log | grep 'All experiments completed!'"
```

If the completion marker is missing, keep polling. If the run PID is gone and the marker is still missing, treat it as a fatal failure.

#### 5. Fatal abort cleanup

```powershell
# Stop TensorBoard if it was started
ssh $REMOTE "kill $TB_PID 2>/dev/null || true"
ssh $REMOTE "ps -p $TB_PID -o pid,cmd 2>&1 || echo 'TensorBoard stopped'"
```

Run this cleanup before revert/log when the formal experiment crashes.

### Formal Result Collection Template

Autoresearch should prefer a small local script over ad-hoc remote grep when the goal is: "extract the best `eval_graph_accuracy` from each seed log and compute mean/std."

Use the script only after logs have been synced locally.

#### Minimal workflow

1. Sync `logs/` from remote to local
2. Save a collection script under the experiment directory, for example `scripts/collect_best_graph_accuracy.py`
3. Run it locally
4. Copy the per-seed values, `mean`, and `std` into `results.tsv`, `PROGRESS.md`, and the keep/discard decision

#### Template script

```python
from pathlib import Path
import re
import statistics

EXP_DIR = Path(r"experiments/experiment_<MMDD>_ar_iter_<N>")
LOG_DIR = EXP_DIR / "logs"

PATTERN = re.compile(
    r"""["']eval_graph_accuracy["']\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"""
)


def extract_best_eval_graph_accuracy(log_path: Path) -> float:
    best = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "eval_graph_accuracy" not in line:
                continue

            for match in PATTERN.finditer(line):
                value = float(match.group(1))
                if best is None or value > best:
                    best = value

    if best is None:
        raise ValueError(f"No eval_graph_accuracy found in {log_path}")

    return best


def main():
    seed_logs = sorted(LOG_DIR.glob("seed_*.log"))
    if not seed_logs:
        raise FileNotFoundError(f"No seed logs found in {LOG_DIR}")

    rows = []
    values = []

    for log_path in seed_logs:
        best = extract_best_eval_graph_accuracy(log_path)
        rows.append((log_path.stem, best))
        values.append(best)

    mean_v = statistics.mean(values)
    std_v = statistics.pstdev(values) if len(values) > 1 else 0.0

    print("Best eval_graph_accuracy per seed")
    for seed_name, best in rows:
        print(f"- {seed_name}: {best:.4f}")

    print()
    print(f"mean: {mean_v:.4f}")
    print(f"std:  {std_v:.4f}")


if __name__ == "__main__":
    main()
```

#### Recommended local run command

```powershell
$env:PYTHONPATH = "."; venv; python experiments\experiment_<MMDD>_ar_iter_<N>\scripts\collect_best_graph_accuracy.py
```

#### Why this template exists

- Reproducible: same script can be re-run after interruption or resume
- Local-first: avoids fragile remote one-liners over SSH
- Autoresearch-friendly: output maps directly to `formal_mean`, `formal_std`, and the per-seed evidence used in keep/discard

### Cleanup Between Iterations

Remote experiment outputs (checkpoints) consume disk space. After collecting results:
- Keep logs and results locally
- Consider removing remote checkpoints if disk space is tight (ask user first — destructive operation)

### Stuck Detection

Standard autoresearch triggers "stuck" mode after 5 consecutive discards. With expensive iterations, lower this threshold to 3 consecutive discards. When stuck:
- Re-read all in-scope files
- Review the full results log
- Consider whether the metric has plateaued (formal_mean stable across recent iterations)
- Try a radical architectural change

## Example: One Full Iteration

**Iteration 3: Add layer normalization to edge predictor**

```
1. Review
   - Read PROGRESS.md: iteration 2 was a keep (formal_mean 0.54)
   - Read results log: 2 keeps, 0 discards so far
   - Read src/models/edge_predictor.py

2. Ideate
   - Previous keeps improved hidden dim and dropout
   - Try adding LayerNorm before final prediction layer

3. Modify
   - Edit src/models/edge_predictor.py: add nn.LayerNorm(n_hid)

4. Commit
   - git commit -m "experiment: add layer norm to edge predictor"

5a. Smoke Test
   - Run: python scripts/smoke_verify.py
   - Result: exit 0, METRIC: 0.0600
   - Update PROGRESS.md: phase = smoke_passed

5b. Formal Experiment
   - Create experiment_0316_ar_iter_3/ (configs, scripts, report template)
   - Sync to remote, launch run_experiment.sh
   - Update PROGRESS.md: phase = formal_running, pid = 12345, poll loop = every 10 min check process + run_all.log
   - Wait for outputs, start TensorBoard
   - Update PROGRESS.md: tensorboard_pid = 23456, tensorboard_port = 6006
   - ⚠️ Polling rule: each poll sends exactly one SSH command from the outer PowerShell session. Do not send multiple SSH commands back-to-back, and do not inspect `seed_*.log` unless process state or `run_all.log` indicates a failure or stall.
   - Poll every 10 min from outer PowerShell (Start-Sleep -Seconds 600), then run exactly one SSH command that checks both process state and `run_all.log`:
     ```powershell
     Start-Sleep -Seconds 600
     ssh root@10.44.101.62 "ps -p 12345 -o pid=,etime=,cmd=; echo '---'; tail -n 5 /data1/b00953788/NFMs/outage_prediction_mini_batch_three_layer/experiments/experiment_0316_ar_iter_3/logs/run_all.log"
     ```
     - One poll = one SSH command
     - Poll #1: process alive, last log: "Epoch 2/10..."
     - Poll #2: process alive, last log: "Epoch 5/10..."
     - ...
     - Poll #N: "All experiments completed!" found
   - Experiment complete
   - Sync logs to local
   - Run collect_results.py
   - Stop TensorBoard and verify PID is gone
   - Result: mean(eval_graph_accuracy) = 0.56, std = 0.008
   - Update PROGRESS.md: phase = deciding

6. Decide
   - 0.56 > 0.54 (previous best) → KEEP
   - Update PROGRESS.md: best_metric = 0.56, best_commit = c3d4e5f

7. Log
   - Append to results TSV: 3  c3d4e5f  0.0600  0.5600  0.0080  keep  experiment_0316_ar_iter_3  add layer norm to edge predictor
   - Update PROGRESS.md: iteration = 4, phase = ready

8. Repeat → go to Phase 1 for iteration 4
```
