# Research Mode Automation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a validation-only research loop that supports fixed benchmark data and configurable best-checkpoint selection without breaking existing full/smoke behavior.

**Architecture:** Extend runtime configuration with optional automation-oriented controls, thread a new `research` run mode through the CLI/runtime/artifact layer, and keep test-set evaluation reserved for `full` and `smoke`. Reuse the existing training pipeline where possible, but split validation metrics artifacts from test metrics artifacts so automated search only sees validation signals.

**Tech Stack:** Python, pytest, HuggingFace Trainer, YAML runtime config

---

### Task 1: Lock behavior with tests

**Files:**
- Modify: `tests/test_main.py`
- Modify: `tests/test_runtime_config.py`
- Modify: `tests/test_inference.py`

**Step 1:** Add failing tests for `--run-mode research` and research artifacts.

**Step 2:** Add failing config tests for optional `synthetic.reuse_existing_splits`, `training_args.metric_for_best_model`, and `training_args.greater_is_better`.

**Step 3:** Add a failing unit test proving frozen split reuse skips export when files already exist.

### Task 2: Extend configuration and CLI

**Files:**
- Modify: `src/training/config.py`
- Modify: `main.py`

**Step 1:** Add `research` to run mode choices.

**Step 2:** Add optional config parsing defaults for reusable splits and checkpoint selection metric.

**Step 3:** Thread `--run-mode` through the CLI while keeping existing defaults intact.

### Task 3: Implement runtime/artifact split

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `src/inference/predictor.py`

**Step 1:** Teach data export to reuse existing split files when configured.

**Step 2:** Split validation-metric artifacts from test-metric artifacts.

**Step 3:** Make `research` mode evaluate validation only and avoid writing test artifacts.

**Step 4:** Use configurable best-model selection in trainer arguments.

### Task 4: Verify and summarize

**Files:**
- Modify: `README.md` if needed

**Step 1:** Run targeted pytest coverage for config, CLI, and runtime behavior.

**Step 2:** Summarize why the new research loop is safer for automated optimization.
