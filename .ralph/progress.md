# Progress Log
Started: Tue Mar 10 00:43:06 CST 2026

## Codebase Patterns
- (add reusable patterns here)

---
## [2026-03-10 00:50:31 +0800] - US-001: Lock runtime config contract
Thread:
Run: 20260310-004306-16535 (iteration 1)
Run log: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-004306-16535-iter-1.log
Run summary: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-004306-16535-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: none (workspace is not a git repository, so `git commit` could not run)
- Post-commit status: not available (workspace is not a git repository)
- Verification:
  - Command: `conda run -n miso pytest -q tests/test_runtime_config.py` -> PASS
  - Command: `conda run -n miso pytest -q tests/test_runtime_config.py tests/test_trainer.py tests/test_end_to_end.py` -> PASS
  - Command: `conda run -n miso pytest -q` -> PASS
  - Command: `conda run -n miso python -c "from alarm_hgt.runtime_config import load_runtime_config; cfg = load_runtime_config('configs/alarm_hgt.yaml'); print(cfg.model.n_hid, cfg.model.num_layers, cfg.batching.per_device_train_batch_size, cfg.outputs.checkpoints_dir, cfg.outputs.results_dir)"` -> PASS
  - Command: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke` -> FAIL
- Files changed:
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/runtime_config.py
  - /Users/a1-6/Documents/projects/hgt-local/configs/alarm_hgt.yaml
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_runtime_config.py
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/activity.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/progress.md
- What was implemented
  - Added a strict YAML-backed runtime config contract with typed sections for synthetic generation, dataset paths, batching, model settings, metrics, training args, and output directories.
  - Added the canonical `configs/alarm_hgt.yaml` file and conversion helpers back into `AlarmHGTConfig`, `SyntheticGraphConfig`, and `LinkPredictionTrainerArgs`.
  - Added focused tests for valid config loading and readable validation failures when required fields are missing.
- **Learnings for future iterations:**
  - Patterns discovered
  - `load_runtime_config` can be the single entry point for later CLI/trainer wiring without copying defaults into code.
  - Gotchas encountered
  - The workspace still has no `.git` directory, so commit and clean-tree checks are impossible until the repo is initialized or remounted correctly.
  - Useful context
  - The global smoke gate currently fails at module resolution (`alarm_hgt.train` missing), which is aligned with the still-open US-002 CLI story rather than a regression in US-001.
---
## [2026-03-10 01:03:08 +0800] - US-002: Create config-driven CLI entrypoint
Thread:
Run: 20260310-004306-16535 (iteration 2)
Run log: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-004306-16535-iter-2.log
Run summary: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-004306-16535-iter-2.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 8b7b971 feat(cli): add config-driven train entrypoint
- Post-commit status: .ralph/activity.log, .ralph/progress.md, .ralph/runs/run-20260310-004306-16535-iter-2.log
- Verification:
  - Command: `conda run -n miso pytest -q tests/test_train_cli.py` -> PASS
  - Command: `conda run -n miso pytest -q` -> PASS
  - Command: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke` -> PASS
- Files changed:
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/export.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/train.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_train_cli.py
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/activity.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/progress.md
- What was implemented
  - Added the `alarm_hgt.train` module entrypoint with `--config` and `--run-mode` parsing, deterministic smoke artifact names, stage logging, and non-zero failure behavior for bad config paths and unwritable output paths.
  - Reused the existing export, dataset, model, and trainer internals to execute export, train, validation, test, and artifact save from a single config-driven command.
  - Extended export helpers to honor configured dataset file paths and added subprocess CLI tests covering smoke success, invalid run-mode rejection, missing config failure, and unwritable output failure.
- **Learnings for future iterations:**
  - Patterns discovered
  - A thin CLI wrapper can wire the current internals together cleanly without duplicating the runtime config contract.
  - Gotchas encountered
  - Relative runtime paths should follow the YAML values as invoked from the current working directory; resolving them against the config file location breaks the canonical `outputs/` contract for `configs/alarm_hgt.yaml`.
  - Useful context
  - This repository started on an unborn `main` branch, so the first commit necessarily snapshot the existing baseline plus the US-002 story changes before the progress/log follow-up commit.
---
