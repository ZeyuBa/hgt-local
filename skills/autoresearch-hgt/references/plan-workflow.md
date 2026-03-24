# Plan Workflow — Autoresearch Plan Mode

Convert a textual goal into a validated, ready-to-execute autoresearch configuration.

**Output:** A complete autoresearch configuration with Scope, Metric, Direction, and Verify — all validated before launch.

## Trigger

- User says "autoresearch plan", "help me set up autoresearch", "plan an autoresearch run", "what should my metric be"

## Workflow

### Phase 1: Capture Goal

If no goal provided, use the AskQuestion tool to ask:

- "What do you want to improve?" with options: Code quality, Performance, Content, Refactoring

If user provides goal text directly, skip to Phase 2.

### Phase 2: Analyze Context

1. Read codebase structure (requirements.txt, pyproject.toml, setup.cfg, configs/)
2. Identify domain: backend, frontend, ML, content, DevOps, etc.
3. Detect existing tooling: test runner, linter, bundler, benchmark scripts
4. Infer likely metric candidates from goal + tooling

### Phase 3: Define Scope

Use the AskQuestion tool to present scope options based on codebase analysis (inferred scope globs with file counts, plus an "Entire project" option).

**Scope validation rules:**
- Scope must resolve to at least 1 file (run glob, confirm matches)
- Warn if scope exceeds 50 files (agent context may struggle)
- Warn if scope includes test files AND source files (prefer separating)

### Phase 4: Define Metric

This is the critical step. The metric must be **mechanical** — extractable from a command output as a single number.

Use the AskQuestion tool to present metric options based on goal + tooling. Each option should show: metric name, what it measures, and the extraction command.

**Metric validation rules (CRITICAL):**

| Check | Pass | Fail |
|-------|------|------|
| Outputs a number | `87.3`, `0.95`, `42` | `PASS`, `looks good`, `✓` |
| Extractable by command | `grep`, `awk`, `jq` | Requires human judgment |
| Deterministic | Same input → same output | Random, flaky, time-dependent |
| Fast | < 30 seconds | > 2 minutes |

If metric fails validation, explain why and suggest alternatives. **Do not proceed until metric is mechanical.**

### Phase 4.5: Define Guard (Optional)

Use the AskQuestion tool to ask if the user wants a guard command to prevent regressions. Options: "Yes — run tests as guard (Recommended)", "Yes — custom guard", "No guard needed".

**Guard suggestion rules:**
- If metric is performance/benchmark → suggest `{test_command}` as guard (e.g. `pytest tests/`)
- If metric is ML training (loss/F1/AP) → suggest `{test_command}` as guard if tests exist, else "No guard needed"
- If metric is refactoring (LOC reduction) → suggest `{test_command} && {typecheck_command}` as guard (e.g. `pytest && mypy src/`)
- If metric IS tests (coverage, pass count) → suggest "No guard needed" as default
- If no test runner detected → suggest "No guard needed" with note

**Guard validation:** If guard is set, run it once to confirm it passes on current codebase. If it fails, help user fix it before proceeding.

### Phase 5: Define Direction

Use the AskQuestion tool to ask: "Is a higher or lower number better for your metric?" with options "Higher is better" and "Lower is better".

### Phase 6: Define Verify Command

Construct the verification command that:
1. Runs the tool/test/benchmark
2. Extracts the metric as a single number
3. Exits 0 on success, non-zero on crash

Use the AskQuestion tool to present the constructed command and ask "Does this verify command look right?" with options: "Looks good, use this", "Modify it", "I have my own command".

**Verify validation (MANDATORY — run before accepting):**

1. **Dry run** the verify command on current codebase
2. Confirm it exits with code 0
3. Confirm output contains a parseable number
4. Record the baseline metric value
5. If dry run fails → show error, ask user to fix, re-validate

```
Dry run result:
  Exit code: {0 or error}
  Output snippet: {relevant line}
  Extracted metric: {number}
  Baseline: {number}
  Status: ✓ VALID / ✗ INVALID — {reason}
```

**Do not proceed if verify command fails dry run.** Help user fix it.

### Phase 7: Confirm & Launch

Present the complete configuration:

```markdown
## Autoresearch Configuration

**Goal:** {user's goal}
**Scope:** {glob pattern}
**Metric:** {metric name} ({direction})
**Verify:** `{command}`
**Guard:** `{guard_command}` *(or "none")*
**Baseline:** {value from dry run}

### Ready-to-use configuration:

autoresearch
Goal: {goal}
Scope: {scope}
Metric: {metric} ({direction})
Verify: {verify_command}
Guard: {guard_command}
```

If no guard was set, omit the Guard line from the output.

Then use the AskQuestion tool to ask: "Configuration validated. How do you want to run it?" with options: "Launch now — unlimited", "Launch now — bounded", "Copy config only".

If "Launch now — unlimited": begin the autoresearch loop with the configuration.
If "Launch now — bounded": ask for iteration count, then begin with that limit.
If "Copy config only": output the ready-to-paste configuration block and stop.

## Metric Suggestion Database

Use these as starting points based on detected domain/tooling:

### ML Research
| Goal Pattern | Metric | Verify Template |
|---|---|---|
| reduce loss | eval_loss | `python main.py --config {config} 2>&1 \| grep "eval_loss" \| tail -1` |
| improve F1/AP/AUC | F1 / AP / AUC score | `python main.py --config {config} 2>&1 \| grep "eval_ap"` |
| graph accuracy | Graph Accuracy % | `python main.py --config {config} 2>&1 \| grep "graph_accuracy"` |
| training speed | Time per epoch (s) | `python main.py --config {config} 2>&1 \| grep "train_runtime"` |
| inference latency | Time in ms | `python scripts/benchmark_inference.py \| grep "latency"` |

### Code Quality
| Goal Pattern | Metric | Verify Template |
|---|---|---|
| test coverage | Coverage % | `pytest --cov={scope} \| grep "TOTAL"` |
| type safety | Error count | `mypy {scope} 2>&1 \| grep -c "error"` |
| lint errors | Error count | `flake8 {scope} 2>&1 \| wc -l` |
| build errors | Error count | `python -m py_compile {file} 2>&1 \| grep -c "Error"` |

### Performance
| Goal Pattern | Metric | Verify Template |
|---|---|---|
| response time | Time in ms | `{bench_cmd} \| grep "p95"` |
| memory usage | Peak RSS in MB | `python scripts/profile_memory.py \| grep "peak"` |
| throughput | Samples per second | `python scripts/benchmark.py \| grep "throughput"` |

### Content
| Goal Pattern | Metric | Verify Template |
|---|---|---|
| readability | Flesch score | `python scripts/readability.py {file}` |
| word count | Word count | `wc -w {scope}` |

### Refactoring
| Goal Pattern | Metric | Verify Template |
|---|---|---|
| reduce LOC | Line count | `pytest && find {scope} -name "*.py" \| xargs wc -l \| tail -1` |
| reduce complexity | Cyclomatic complexity | `radon cc {scope} -s -a \| grep "Average"` |
| eliminate pattern | Pattern count | `grep -r "{pattern}" {scope} \| wc -l` |

## Error Recovery

| Error | Recovery |
|---|---|
| No test runner detected | Ask user for test command |
| Verify command fails | Show error, suggest fix, re-validate |
| Metric not parseable | Suggest adding `grep`/`awk` to extract number |
| Scope resolves to 0 files | Show glob result, ask user to fix pattern |
| Scope too broad (>100 files) | Suggest narrowing, warn about context limits |

## Anti-Patterns

- **Do NOT accept subjective metrics** — "looks better" is not a metric
- **Do NOT skip the dry run** — always validate verify command works
- **Do NOT suggest verify commands you haven't tested** — run it first
- **Do NOT overwhelm with questions** — max 5-6 questions total across all phases
- **Do NOT auto-launch without explicit user consent** — always confirm at Phase 7
