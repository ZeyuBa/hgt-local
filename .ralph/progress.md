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
## [2026-03-10 01:28:26 +0800] - US-003: Integrate train and validation loop
Thread:
Run: 20260310-011544-23854 (iteration 1)
Run log: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-1.log
Run summary: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: e12e1e1 feat(trainer): add epoch train validation loop
- Post-commit status: .ralph/runs/run-20260310-011544-23854-iter-1.log
- Verification:
  - Command: `conda run -n miso pytest -q tests/test_trainer.py tests/test_train_cli.py` -> PASS
  - Command: `conda run -n miso pytest -q` -> PASS
  - Command: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke` -> PASS
- Files changed:
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/train.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/trainer.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_train_cli.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_trainer.py
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/activity.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/progress.md
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-1.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-1.md
- What was implemented
  - Moved the config-driven epoch loop into `LinkPredictionTrainer`, using the custom bucket sampler and padding collator for both train and validation paths.
  - Added explicit failures when any batch has no trainable positions or any batch/epoch loss becomes NaN or inf, blocking fake-success runs.
  - Persisted `train_history.json` and `val_history.json` under `outputs/results/` with stable `split`, `metric`, and per-epoch loss keys, and wired the CLI to use configured metric `ks`.
- **Learnings for future iterations:**
  - Patterns discovered
  - Trainer-owned epoch results make it easy to persist histories and reuse the same evaluation path for validation and test.
  - Gotchas encountered
  - The run log keeps changing until the very last tool action, so the progress/log sync commit must be the final repo-touching step or `git status` will drift dirty again.
  - Useful context
  - Smoke verification now writes finite per-epoch train and validation losses to `outputs/results/train_history.json` and `outputs/results/val_history.json`, while the checkpoint/test-metric artifact contract remains in the existing CLI path for later stories.
---
## [2026-03-10 01:39:14 +0800] - US-004: Persist checkpoints and test evaluation
Thread:
Run: 20260310-011544-23854 (iteration 2)
Run log: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-2.log
Run summary: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-2.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 0bf1695 feat(training): persist test checkpoint artifacts
- Post-commit status: .ralph/runs/run-20260310-011544-23854-iter-2.log
- Verification:
  - Command: `conda run -n miso pytest -q tests/test_trainer.py tests/test_train_cli.py` -> PASS
  - Command: `conda run -n miso pytest -q` -> PASS
  - Command: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke` -> PASS
- Files changed:
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/metrics.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/train.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/trainer.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_train_cli.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_trainer.py
  - /Users/a1-6/Documents/projects/hgt-local/.agents/tasks/prd-alarm-hgt.json
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/activity.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/errors.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/progress.md
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-1.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-1.md
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-2.log
- What was implemented
  - Added best-checkpoint tracking to the trainer, saved deterministic `*-last.pt` and `*-best.pt` artifacts under `outputs/checkpoints/`, and reloaded the selected best checkpoint before test evaluation.
  - Added `outputs/results/test_metrics.json` with required finite `precision`, `recall`, and `f1` keys, and blocked completion if the checkpoint is missing or the required metric keys are absent.
  - Expanded trainer and CLI coverage to assert checkpoint artifact creation, test-metric persistence, and missing-checkpoint failure behavior.
- **Learnings for future iterations:**
  - Patterns discovered
  - A deterministic `smoke-best.pt` path is enough to serve as the stable best-checkpoint pointer without introducing extra indirection files.
  - Gotchas encountered
  - The smoke checkpoint/test-metric contract is now satisfied, but model-quality thresholds like minimum F1 belong to a later story because the current smoke run still learns poorly.
  - Useful context
  - Snapshotting the best model state in-memory during training keeps checkpoint selection simple and avoids threading run-mode-specific file naming logic into the trainer loop.
---
## [2026-03-10 01:57:53 +0800] - US-005: Enforce model-design compliance in runtime
Thread:
Run: 20260310-011544-23854 (iteration 3)
Run log: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-3.log
Run summary: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-3.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 21ba217 fix(training): enforce model design checks
- Post-commit status: .agents/tasks/prd-alarm-hgt.json, .ralph/activity.log, .ralph/errors.log, .ralph/runs/run-20260310-011544-23854-iter-2.log, .ralph/runs/run-20260310-011544-23854-iter-2.md, .ralph/runs/run-20260310-011544-23854-iter-3.log
- Verification:
  - Command: `conda run -n miso pytest -q tests/test_train_cli.py::test_module_exits_non_zero_when_model_design_drifts_from_spec tests/test_train_cli.py::test_config_driven_runtime_masks_loss_and_metrics_to_trainable_alarm_entities_only tests/test_synthetic_topology.py::test_fault_modes_keep_full_graph_export_and_describe_logical_failures` -> PASS
  - Command: `conda run -n miso pytest -q` -> PASS
  - Command: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke` -> PASS
- Files changed:
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/train.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_synthetic_topology.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_train_cli.py
  - /Users/a1-6/Documents/projects/hgt-local/.agents/tasks/prd-alarm-hgt.json
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/activity.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/errors.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/progress.md
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-2.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-2.md
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-3.log
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-011544-23854-iter-3.md
- What was implemented
  - Added a runtime design validator that fails fast when the configured feature width, node-type count, or relation count drift from the model design spec, and when non-trainable alarms leak into `trainable_mask`.
  - Added regression coverage proving smoke/CLI runtime rejects design drift, full-graph exports preserve logical failure semantics for both `mains_failure` and `link_down`, and mixed fault-mode config-driven batches only score `ne_is_disconnected` positions while still training and evaluating end to end.
- **Learnings for future iterations:**
  - Patterns discovered
  - The cleanest place to block silent model drift is `build_runtime_objects()`, after datasets exist but before the HGT model is constructed and trained.
  - Gotchas encountered
  - `num_types` and `num_relations` can drift upward without crashing pyHGT, so relying on shape errors is not enough; they need explicit runtime checks.
  - Useful context
  - A deterministic scripted forward pass inside the CLI integration test makes the mask-vs-leak metric behavior provable without weakening the real config-driven dataloader path.
---
## [2026-03-10 08:41:57 +0800] - US-006: Stabilize smoke training thresholds
Thread:
Run: 20260310-080338-45074 (iteration 1)
Run log: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-080338-45074-iter-1.log
Run summary: /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-080338-45074-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: d7a376f fix(smoke): stabilize smoke thresholds
- Post-commit status: .ralph/runs/run-20260310-080338-45074-iter-1.log
- Verification:
  - Command: `conda run -n miso pytest -q tests/test_modeling.py tests/test_export.py tests/test_train_cli.py` -> PASS
  - Command: `conda run -n miso pytest -q` -> PASS
  - Command: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke` -> PASS
- Files changed:
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/export.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/metrics.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/modeling.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/train.py
  - /Users/a1-6/Documents/projects/hgt-local/alarm_hgt/trainer.py
  - /Users/a1-6/Documents/projects/hgt-local/configs/alarm_hgt.yaml
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_export.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_modeling.py
  - /Users/a1-6/Documents/projects/hgt-local/tests/test_train_cli.py
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/runs/run-20260310-080338-45074-iter-1.md
  - /Users/a1-6/Documents/projects/hgt-local/.ralph/progress.md
- What was implemented
  - Added representative smoke sampling so the CPU smoke run exports disjoint positive train/validation/test graphs instead of collapsing into all-negative splits.
  - Rebalanced the masked BCE loss on mixed batches, calibrated test precision/recall/F1 from the best validation threshold, and enforced a fail-closed smoke gate that checks both majority loss improvement and `f1 >= 0.60`.
  - Tuned the default smoke config to `train=16`, `val=4`, `test=4`, `8` epochs, and `0.001` learning rate so the default miso smoke gate now passes and prints `<promise>COMPLETE</promise>` only on success.
- **Learnings for future iterations:**
  - Patterns discovered
  - Smoke stability depends more on deterministic sample composition than on raw epoch count; the train/val/test positive examples need deliberate spacing to avoid a fake-easy or fake-hopeless smoke split.
  - Gotchas encountered
  - Seeded manual searches are mandatory for this pipeline; unseeded experiments looked promising but lied about the real default smoke behavior.
  - Useful context
  - The passing default smoke run now records `decision_threshold = 0.82`, `f1 = 0.6667`, and improvement on all 7 epoch-to-epoch transitions in `outputs/results/`.
---
