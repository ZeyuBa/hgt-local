# Alarm HGT Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the alarm HGT pipeline to improve readability, extensibility, and configuration hygiene without changing model behavior or breaking the existing smoke workflow.

**Architecture:** Split runtime orchestration out of the CLI entrypoint into focused pipeline modules, replace implicit feature-bundle dictionaries with explicit typed structures, and centralize runtime artifact naming plus split metadata so path handling and verification logic stop leaking magic strings across the codebase.

**Tech Stack:** Python, PyTorch, pyHGT, pytest, dataclasses, pathlib, JSON

---

### Task 1: Lock Structured Feature Bundle Contracts

**Files:**
- Create: `tests/test_features.py`
- Modify: `alarm_hgt/features.py`
- Modify: `alarm_hgt/dataset.py`

**Step 1: Write the failing test**

Add tests that assert:

- feature construction returns a typed bundle object instead of an untyped dict
- alarm entity ids have direct position lookup without `list.index`
- dataset edge construction can resolve owner, AE, and alarm node indices from explicit mappings

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL because the typed feature bundle contract does not exist yet.

**Step 3: Write minimal implementation**

Introduce a `FeatureBundle` dataclass with explicit fields and lookup maps, then update dataset tensorization to use the new structure.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py tests/test_dataset.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 2: Extract Runtime Pipeline Modules

**Files:**
- Create: `alarm_hgt/runtime.py`
- Modify: `alarm_hgt/train.py`
- Test: `tests/test_train_cli.py`

**Step 1: Write the failing test**

Add tests that assert:

- runtime path resolution and artifact directory preparation can be used without importing the CLI module
- the CLI entrypoint delegates to the extracted runtime pipeline module
- run-mode, artifact naming, and verification behavior remain stable after extraction

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_cli.py -v`
Expected: FAIL because the extracted runtime module does not exist and the CLI still owns orchestration.

**Step 3: Write minimal implementation**

Move pipeline orchestration, artifact validation, and runtime dataclasses into a dedicated module. Keep `alarm_hgt/train.py` as a thin CLI shell that re-exports the public runtime helpers required by existing tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_cli.py tests/test_runtime_config.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 3: Centralize Runtime Naming And Validation Rules

**Files:**
- Modify: `alarm_hgt/runtime.py`
- Modify: `alarm_hgt/export.py`
- Modify: `alarm_hgt/trainer.py`
- Test: `tests/test_train_cli.py`
- Test: `tests/test_export.py`

**Step 1: Write the failing test**

Add tests that assert:

- split names and artifact filenames come from shared constants instead of repeated inline strings
- representative smoke export uses centralized split iteration rules
- runtime verification still enforces required metrics and smoke thresholds

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py tests/test_train_cli.py -v`
Expected: FAIL because runtime naming and verification metadata are still scattered.

**Step 3: Write minimal implementation**

Introduce shared split/artifact constants and update export, runtime verification, and trainer serialization code to use them.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_export.py tests/test_train_cli.py tests/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 4: Full Regression Verification

**Files:**
- Test: `tests/test_features.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_export.py`
- Test: `tests/test_trainer.py`
- Test: `tests/test_train_cli.py`

**Step 1: Run targeted regression suite**

Run: `pytest tests/test_features.py tests/test_dataset.py tests/test_export.py tests/test_trainer.py tests/test_train_cli.py -v`
Expected: PASS

**Step 2: Run full test suite**

Run: `pytest -q`
Expected: PASS

**Step 3: Run smoke quality gate**

Run: `python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke`
Expected: PASS with `<promise>COMPLETE</promise>`

**Step 4: Run miso quality gates**

Run: `conda run -n miso pytest -q`
Expected: PASS

Run: `conda run -n miso python -m alarm_hgt.train --config configs/alarm_hgt.yaml --run-mode smoke`
Expected: PASS with `<promise>COMPLETE</promise>`

**Step 5: Commit**

Skip: workspace is not a git repository.
