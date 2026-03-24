---
name: autoresearch
description: Use when the user asks to iterate autonomously, improve a metric, monitor ongoing work, or run an autoresearch session.
---

# Autoresearch — Autonomous Goal-directed Iteration

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Applies constraint-driven autonomous iteration to ANY work — not just ML research.

**Core idea:** You are an autonomous agent. Modify → Verify → Keep/Discard → Repeat.

## Modes

| Mode | Purpose |
|------|---------|
| Default | Run the autonomous loop |
| Plan | Interactive wizard to build Scope, Metric, Direction & Verify from a Goal |

### Plan Mode — Goal → Configuration Wizard

Converts a plain-language goal into a validated, ready-to-execute autoresearch configuration.

Read `references/plan-workflow.md` for full protocol.

**Quick summary:**

1. **Capture Goal** — ask what the user wants to improve (or accept inline text)
2. **Analyze Context** — scan codebase for tooling, test runners, build scripts
3. **Define Scope** — suggest file globs, validate they resolve to real files
4. **Define Metric** — suggest mechanical metrics, validate they output a number
5. **Define Direction** — higher or lower is better
6. **Define Verify** — construct the shell command, **dry-run it**, confirm it works
7. **Confirm & Launch** — present the complete config, offer to launch immediately

**Critical gates:**
- Metric MUST be mechanical (outputs a parseable number, not subjective)
- Verify command MUST pass a dry run on the current codebase before accepting
- Scope MUST resolve to ≥1 file

**Usage examples:**

```
autoresearch plan
Goal: Make the API respond faster

autoresearch plan: Increase test coverage to 95%
```

After the wizard completes, the user gets a ready-to-use autoresearch configuration — or can launch it directly.

## When to Activate

- User says "autoresearch", "run autoresearch" → run the loop
- User says "autoresearch plan", "help me set up autoresearch" → run the planning wizard
- User says "work autonomously", "iterate until done", "keep improving" → run the loop
- Any task requiring repeated iteration cycles with measurable outcomes → run the loop

## Optional: Bounded Iteration Count

By default, autoresearch loops until the user stops the session. Users can optionally specify a **max iteration count** to limit iterations.

### Usage

**Unlimited (default):**
```
autoresearch
Goal: Increase test coverage to 90%
```

**Bounded (N iterations):**
```
autoresearch, max 25 iterations
Goal: Increase test coverage to 90%
```

### When to Use Bounded Loops

| Scenario | Recommendation |
|----------|---------------|
| Long improvement session | Unlimited (default) |
| Quick improvement session | max 10 iterations |
| Targeted fix with known scope | max 5 iterations |
| Exploratory — see if approach works | max 15 iterations |

### Behavior with Iteration Limit

When a max iteration count is specified:
- Run exactly N iterations through the autoresearch loop
- After iteration N, print a **final summary** with baseline → current best, keeps/discards/crashes
- If the goal is achieved before N iterations, print early completion and stop
- All other rules (atomic changes, mechanical verification, auto-rollback) still apply

## Setup Phase (Do Once)

1. **Read all in-scope files** for full context before any modification
2. **Define the goal** — What does "better" mean? Extract or ask for a mechanical metric:
   - Code: tests pass, build succeeds, performance benchmark improves
   - ML: eval_loss decreases, F1/AP/AUC improves, graph accuracy increases
   - Content: word count target hit, readability score
   - If no metric exists → define one with user, or use simplest proxy (e.g. "compiles without errors")
3. **Define scope constraints** — Which files can you modify? Which are read-only?
4. **Define guard (optional)** — A command that must ALWAYS pass for a change to be kept. Use this to prevent regressions while optimizing the main metric (e.g., `pytest tests/` must pass while optimizing benchmark time). If not specified, no guard is enforced.
5. **Create a results log** — Track every iteration (see `references/results-logging.md`)
6. **Establish baseline** — Run verification on current state AND guard (if set). Record as iteration #0
7. **Confirm and go** — Show user the setup, get confirmation, then BEGIN THE LOOP

## The Loop

Read `references/autonomous-loop-protocol.md` for full protocol details.

```
LOOP (agent-driven, until goal reached or user stops you):
  1. Review: Read current state + git history + results log
  2. Ideate: Pick next change based on goal, past results, what hasn't been tried
  3. Modify: Make ONE focused change to in-scope files
  4. Commit: Git commit the change (before verification)
  5. Verify: Run the mechanical metric (tests, build, benchmark, etc.)
  6. Guard: If guard is set, run the guard command
  7. Decide:
     - IMPROVED + guard passed (or no guard) → Keep commit, log "keep", advance
     - IMPROVED + guard FAILED → Revert, then try to rework the optimization
       (max 2 attempts) so it improves the metric WITHOUT breaking the guard.
       Never modify guard/test files — adapt the implementation instead.
       If still failing → log "discard (guard failed)" and move on
     - SAME/WORSE → Git revert, log "discard"
     - CRASHED → Try to fix (max 3 attempts), else log "crash" and move on
  8. Log: Record result in results log
  9. Repeat: Go to step 1.
     - The agent issues the next command itself after inspecting the latest output.
     - If unbounded: continue the conversation and issue the next command yourself. Do not ask "should I continue?"
     - If bounded (N): Stop after N iterations, print final summary
```

## Critical Rules

1. **Loop until done** — Unbounded: loop until the user stops you. Bounded: loop N times then summarize.
2. **Read before write** — Always understand full context before modifying
3. **One change per iteration** — Atomic changes. If it breaks, you know exactly why
4. **Mechanical verification only** — No subjective "looks good". Use metrics
5. **Automatic rollback** — Failed changes revert instantly. No debates
6. **Simplicity wins** — Equal results + less code = KEEP. Tiny improvement + ugly complexity = DISCARD
7. **Git is memory** — Every kept change committed. Agent reads history to learn patterns
8. **When stuck, think harder** — Re-read files, re-read goal, combine near-misses, try radical changes. Don't ask for help unless truly blocked by missing access/permissions
9. **Agent loop, not shell loop** — For monitoring or polling tasks, issue one command at a time, inspect the output, then decide the next command. Do not replace the agent loop with `while ($true)`, `for (;;)`, `watch`, or background infinite loops. Even for 10-minute health checks, the agent must explicitly perform each poll as a normal step in the conversation.
10. **Do not hand off and disappear** — If the task is to keep watching, polling, or monitoring, keep the conversation alive with brief status updates. Emitting a loop script, background watcher, or detached polling process and ending the conversation is a failure.
11. **Minimal monitoring by default** — When monitoring a long-running formal experiment, prefer the smallest reliable signal set: poll the experiment process state and `run_all.log` only. Do not poll TensorBoard status unless the user explicitly asks for TensorBoard health checks or TensorBoard itself is the thing being debugged.

**Polling template (outer PowerShell, one SSH per poll):**

```powershell
Start-Sleep -Seconds 600
ssh root@10.44.101.62 "ps -p <remote_pid> -o pid=,etime=,cmd=; echo '---'; tail -n 5 <run_all_log>"
```

Use exactly one SSH command per poll. Do not split process checks and log checks into separate SSH calls, and do not inspect `seed_*.log` unless `run_all.log` or process state suggests a failure or stall.

## Principles Reference

See `references/core-principles.md` for the 7 generalizable principles from autoresearch.

## ML Experiment Adaptation

For ML training projects that require remote GPU experiments, see `references/experiment-adaptation.md`. This adapts the loop to a two-phase verification workflow:

1. **Smoke test** (local, ~30s) — crash gate only, filters out broken changes before committing GPU time
2. **Formal experiment** (remote, hours) — 3-seed training via `/running-experiments` skill; `mean(eval_graph_accuracy)` across seeds is the authoritative keep/discard metric

Key differences from the standard loop:
- Phase 5 splits into 5a (smoke) and 5b (formal experiment)
- Formal experiments use the `/running-experiments` skill (Phases 3-4, 6-7; Phase 5 skipped since smoke covers it)
- `.autoresearch/PROGRESS.md` tracks session state for resume after interruptions
- Results log includes `smoke_metric`, `formal_mean`, `formal_std` columns

## Adapting to Different Domains

| Domain | Metric | Scope | Verify Command | Guard |
|--------|--------|-------|----------------|-------|
| ML training | eval_loss / AP / F1 | `src/**/*.py`, `configs/` | `python main.py --config ...` | — |
| ML data pipeline | Tests pass + data integrity | `training_data/*.py` | `pytest tests/` | — |
| Python backend | Tests pass + coverage % | `src/**/*.py` | `pytest --cov` | — |
| Data analysis | Result metric (accuracy, etc.) | `src/analysis/` | Custom script | `pytest` |
| Performance | Benchmark time (ms) | Target files | `python scripts/benchmark.py` | `pytest` |
| Refactoring | Tests pass + LOC reduced | Target module | `pytest && wc -l src/**/*.py` | `mypy src/` |

Adapt the loop to your domain. The PRINCIPLES are universal; the METRICS are domain-specific.
