# HGT Full Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the legacy `alarm_hgt/*` package with the target project architecture built around `src/*`, `training_data/*`, and a root `main.py` that unifies training and inference.

**Architecture:** Preserve the current domain logic where it is correct, but re-home it into the target layered structure: topology generation in `training_data`, graph/feature tensorization in `src/graph` and `src/dataset`, HGT modeling in `src/models`, Trainer-backed orchestration in `src/training`, and checkpoint-driven inference in `src/inference`. The new training stack must use a real `transformers.Trainer` subclass for dataloaders, loss/eval collection, and checkpoint selection; the old `alarm_hgt/*` package is deleted after parity is verified.

**Tech Stack:** Python, PyTorch, Transformers Trainer, NetworkX, PyYAML, pytest

---

### Task 1: Lock The New Public Architecture In Tests

**Files:**
- Create: `tests/test_main.py`
- Create: `tests/test_training_trainer.py`
- Create: `tests/test_inference.py`
- Modify: `tests/test_end_to_end.py`
- Delete/replace references to legacy `alarm_hgt/*` imports in existing tests

**Step 1: Write the failing test**

Add tests that assert:

- root `main.py` exists and supports `--config`, `--run-mode`, and `--mode train|inference`
- all test imports resolve only through `src.*` modules and `training_data.*`
- `src.training.trainer.LinkPredictionTrainer` is a subclass of `transformers.Trainer`
- inference mode can load a saved checkpoint and write test metrics without running training

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py tests/test_training_trainer.py tests/test_inference.py -v`
Expected: FAIL because the new architecture and entrypoint do not exist yet.

**Step 3: Write minimal implementation**

Create the new module skeletons and root entrypoint with imports only, enough for tests to reach the missing behavior.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py tests/test_training_trainer.py tests/test_inference.py -v`
Expected: PASS

**Step 5: Commit**

Skip until the full migration is green.

### Task 2: Move Synthetic Topology Generation Into `training_data/*`

**Files:**
- Create: `training_data/topo_generator.py`
- Create: `training_data/topo_combiner.py`
- Create: `training_data/topo_complete.py`
- Create: `training_data/__init__.py`
- Modify: `configs/config.yaml`
- Modify: `tests/test_synthetic_topology.py`
- Modify: `tests/test_export.py`

**Step 1: Write the failing test**

Add tests that assert:

- `topo_generator.py` produces base topology samples with `fault_modes`, `noise_sites`, and site-level flags
- `topo_combiner.py` preserves split naming and deterministic seed behavior
- `topo_complete.py` expands complete samples with all four alarm types and complete logical labels

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_topology.py tests/test_export.py -v`
Expected: FAIL because the generation pipeline still lives in the legacy package.

**Step 3: Write minimal implementation**

Extract and split the existing synthetic/export logic into the three `training_data` stages, keeping deterministic behavior and representative smoke coverage.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_topology.py tests/test_export.py -v`
Expected: PASS

**Step 5: Commit**

Skip until the full migration is green.

### Task 3: Rebuild Graph And Dataset Layers Under `src/*`

**Files:**
- Create: `src/__init__.py`
- Create: `src/graph/__init__.py`
- Create: `src/graph/graph_builder.py`
- Create: `src/graph/feature_extraction.py`
- Create: `src/dataset/__init__.py`
- Create: `src/dataset/hgt_dataset.py`
- Create: `src/dataset/builder.py`
- Create: `src/dataset/collate.py`
- Create: `src/dataset/bucket_sampler.py`
- Modify: `tests/test_features.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_batching.py`

**Step 1: Write the failing test**

Add tests that assert:

- `FeatureExtractor` produces the documented 32-dim feature contract
- `TopologyGraph` builds heterogeneous nodes and 9 relations from completed topology samples
- `HGTDataset` returns HGT-ready tensors plus `trainable_mask`
- bucketed collation pads only the alarm-entity axis and never leaks padding into trainable positions

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py tests/test_dataset.py tests/test_batching.py -v`
Expected: FAIL because the `src.graph` and `src.dataset` layers do not exist yet.

**Step 3: Write minimal implementation**

Move the feature, graph, dataset, and batching logic into the new modules with typed bundle/data objects and explicit relation/type constants.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py tests/test_dataset.py tests/test_batching.py -v`
Expected: PASS

**Step 5: Commit**

Skip until the full migration is green.

### Task 4: Rebuild Models And Real Trainer Integration

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/hgt.py`
- Create: `src/models/edge_predictor.py`
- Create: `src/models/hgt_for_link_prediction.py`
- Create: `src/training/__init__.py`
- Create: `src/training/config.py`
- Create: `src/training/trainer.py`
- Modify: `tests/test_modeling.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_training_trainer.py`

**Step 1: Write the failing test**

Add tests that assert:

- `HGTForLinkPrediction` remains `PreTrainedModel`-compatible
- `src.training.trainer.LinkPredictionTrainer` subclasses `transformers.Trainer`
- custom bucketed dataloaders are used for train/eval/test
- `trainer.train()` and `trainer.evaluate()` produce finite losses and compute masked metrics through the Trainer API

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_modeling.py tests/test_metrics.py tests/test_training_trainer.py -v`
Expected: FAIL because the trainer still relies on a handwritten loop.

**Step 3: Write minimal implementation**

Implement the model wrappers and a real Trainer subclass that overrides dataloaders, metric adaptation, and history extraction while using Transformers training/eval flows for optimization and checkpointing.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_modeling.py tests/test_metrics.py tests/test_training_trainer.py -v`
Expected: PASS

**Step 5: Commit**

Skip until the full migration is green.

### Task 5: Add Unified Runtime, Inference, And Artifact Governance

**Files:**
- Create: `src/inference/__init__.py`
- Create: `src/inference/predictor.py`
- Create: `src/inference/test_analyzer.py`
- Create: `main.py`
- Modify: `configs/config.yaml`
- Modify: `tests/test_main.py`
- Modify: `tests/test_inference.py`

**Step 1: Write the failing test**

Add tests that assert:

- `main.py --mode train` runs export, train, validation, test, and artifact verification
- `main.py --mode inference --checkpoint-path <path>` skips training and runs val-threshold calibration plus test prediction
- checkpoints, summary, train/val history, and test metrics land in the configured output directories
- config is the single source of truth for model, data, batching, metrics, and output paths

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py tests/test_inference.py tests/test_runtime_config.py -v`
Expected: FAIL because unified runtime/inference entry and config contract are incomplete.

**Step 3: Write minimal implementation**

Build a new runtime layer around the new modules, wire it into root `main.py`, and support both train and inference-only execution using the same config file and artifact contract.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py tests/test_inference.py tests/test_runtime_config.py -v`
Expected: PASS

**Step 5: Commit**

Skip until the full migration is green.

### Task 6: Remove Legacy Package And Run Full Regression

**Files:**
- Delete: `alarm_hgt/`
- Delete or replace any legacy-path tests
- Verify: `tests/`

**Step 1: Delete legacy package after green migration**

Remove the `alarm_hgt/*` package and any leftover imports to it.

**Step 2: Run targeted regression suite**

Run: `pytest tests/test_synthetic_topology.py tests/test_features.py tests/test_dataset.py tests/test_batching.py tests/test_modeling.py tests/test_training_trainer.py tests/test_main.py tests/test_inference.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest -q`
Expected: PASS

**Step 4: Run smoke runtime**

Run: `python main.py --config configs/config.yaml --run-mode smoke --mode train`
Expected: PASS with saved artifacts and verified summary.

**Step 5: Commit**

Skip only if the repository still contains unrelated user edits that should not be bundled.
