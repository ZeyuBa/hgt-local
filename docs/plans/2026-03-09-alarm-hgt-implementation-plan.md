# Alarm HGT Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete synthetic-data-to-training pipeline for the alarm prediction model described in `model_design.md` on top of `pyHGT`.

**Architecture:** Add a synthetic graph generator and labeler, a dataset layer that constructs the 32-dim features and padding-aware mini-batches, a pyHGT-based link prediction model, and trainer/metrics utilities. Keep exported samples as full graphs and apply propagation via logical masking only. The final training path must be driven by HuggingFace Transformers `Trainer` plus an external YAML config file so model, data, export, and training behavior are configured outside code.

**Tech Stack:** Python, PyTorch, pyHGT, HuggingFace Transformers `Trainer`, PyYAML, pytest, JSONL

---

### Task 1: Project Skeleton And Constants

**Files:**
- Create: `alarm_hgt/__init__.py`
- Create: `alarm_hgt/constants.py`
- Create: `alarm_hgt/types.py`
- Test: `tests/test_constants.py`

**Step 1: Write the failing test**

Write tests that assert:

- node type ids are stable
- alarm ids and domains are stable
- relation names needed by the spec exist

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_constants.py -v`
Expected: FAIL because module does not exist.

**Step 3: Write minimal implementation**

Add the constants and typed helper structures used by the rest of the pipeline.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_constants.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 2: Synthetic Topology Generator

**Files:**
- Create: `alarm_hgt/synthetic.py`
- Test: `tests/test_synthetic_topology.py`

**Step 1: Write the failing test**

Add tests that assert:

- each sample graph is connected before logical faults
- each site owns one router and at least one wl_station
- cross-site edges only connect routers
- exported sample nodes and edges are complete, not destructively pruned

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_topology.py -v`
Expected: FAIL because generator is missing.

**Step 3: Write minimal implementation**

Implement deterministic graph generation from a seed with site backbone and backup links.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_topology.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 3: Fault Propagation And Labeling

**Files:**
- Modify: `alarm_hgt/synthetic.py`
- Test: `tests/test_labeling.py`

**Step 1: Write the failing test**

Add tests that assert:

- `mains_failure` labels local site alarms correctly
- `link_down` labels both link endpoints correctly
- downstream disconnection is determined by AN reachability on the logically masked graph
- noise mains failure does not create false downstream outages

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_labeling.py -v`
Expected: FAIL because labeling logic is incomplete.

**Step 3: Write minimal implementation**

Implement logical masks, AN reachability, and alarm entity labels.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_labeling.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 4: Dataset Export

**Files:**
- Create: `alarm_hgt/export.py`
- Test: `tests/test_export.py`

**Step 1: Write the failing test**

Add tests that assert:

- train/val/test JSONL files are produced
- one sample is written per line
- required sample keys are present

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL because exporter is missing.

**Step 3: Write minimal implementation**

Implement dataset split generation and JSONL export.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_export.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 5: Feature Builder And Dataset Loader

**Files:**
- Create: `alarm_hgt/features.py`
- Create: `alarm_hgt/dataset.py`
- Test: `tests/test_dataset.py`

**Step 1: Write the failing test**

Add tests that assert:

- the feature vector width is 32
- site-level flags are broadcast to NE and AlarmEntity nodes
- reverse edges and self edges are synthesized
- `trainable_mask` excludes anchor, AN, and padding owners

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL because loader and feature builder are missing.

**Step 3: Write minimal implementation**

Implement feature construction and sample-to-tensors conversion.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 6: Bucket Sampler And Padding Collator

**Files:**
- Create: `alarm_hgt/batching.py`
- Test: `tests/test_batching.py`

**Step 1: Write the failing test**

Add tests that assert:

- bucketed batches group similar graph sizes
- padding nodes are isolated
- padding exists only after collation
- padded positions are mask-disabled
- padding `AlarmEntity` nodes reuse a real `NE` owner index without creating any `NE-AE` edges

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_batching.py -v`
Expected: FAIL because batching utilities are missing.

**Step 3: Write minimal implementation**

Implement sampler and an `AE`-aligned padding-aware collate function. Only pad `AlarmEntity` slots up to batch-local `max_ae`; reuse the first real `NE` as the owner index for padded `AlarmEntity` slots, but do not add any `NE-AE` edges for those padding nodes.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_batching.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 7: Model Config And Link Prediction Wrapper

**Files:**
- Create: `alarm_hgt/config.py`
- Create: `alarm_hgt/modeling.py`
- Modify: `pyHGT/model.py`
- Test: `tests/test_modeling.py`

**Step 1: Write the failing test**

Add tests that assert:

- config defaults match the design
- the model produces `[B, max_ae]` logits
- loss only uses masked trainable positions
- paired NE and AE embeddings are scored bilinearly

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_modeling.py -v`
Expected: FAIL because wrapper model is missing.

**Step 3: Write minimal implementation**

Implement config, edge predictor, and `HGTForLinkPrediction`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_modeling.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 8: Metrics

**Files:**
- Create: `alarm_hgt/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test**

Add tests that assert:

- edge-level metrics ignore masked positions
- best-F1 threshold scan works
- graph-level metrics are computed per graph

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL because metrics module is missing.

**Step 3: Write minimal implementation**

Implement metric computation over flattened masked logits and per-sample slices.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 9: Trainer Utilities

**Files:**
- Create: `alarm_hgt/trainer.py`
- Test: `tests/test_trainer.py`

**Step 1: Write the failing test**

Add tests that assert:

- trainer chooses custom sampler/collator
- eval predictions can be converted into the metric input shape

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL because trainer utilities are missing.

**Step 3: Write minimal implementation**

Implement trainer helpers and metric adapters with minimal HF dependencies.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 10: End-To-End Sanity Flow

**Files:**
- Create: `tests/test_end_to_end.py`
- Optionally modify: `alarm_hgt/export.py`
- Optionally modify: `alarm_hgt/dataset.py`
- Optionally modify: `alarm_hgt/modeling.py`

**Step 1: Write the failing test**

Add a small end-to-end test that:

- generates synthetic JSONL data
- loads two samples
- collates them with padding
- runs a forward pass
- returns finite loss and correctly shaped logits

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py -v`
Expected: FAIL because one or more integration pieces are missing.

**Step 3: Write minimal implementation**

Fill the smallest missing gaps only.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.

### Task 11: Transformers Trainer Integration And YAML Runtime Config

**Files:**
- Create: `configs/alarm_hgt.yaml`
- Create: `alarm_hgt/runtime_config.py`
- Modify: `alarm_hgt/trainer.py`
- Create: `alarm_hgt/train.py`
- Test: `tests/test_runtime_config.py`
- Test: `tests/test_transformers_trainer.py`

**Step 1: Write the failing tests**

Add tests that assert:

- a single YAML file can define synthetic data, dataset paths, batching, model hyperparameters, metrics, and training args
- runtime config loading populates typed config objects without falling back to hidden hardcoded defaults in the training path
- the Transformers `Trainer` wrapper uses the custom sampler, collator, and compute-metrics adapter needed by HGT training
- a train entrypoint can build `TrainingArguments`, model, datasets, and trainer instances entirely from the YAML file path

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runtime_config.py tests/test_transformers_trainer.py -v`
Expected: FAIL because YAML config loading and full Transformers Trainer integration are missing.

**Step 3: Write minimal implementation**

Implement a YAML-driven runtime config layer and a real Transformers-Trainer-based training entrypoint.

Required behavior:

- `configs/alarm_hgt.yaml` is the single source of truth for runtime parameters
- model dimensions, dropout, layer counts, relation/type counts, dataset/export paths, synthetic generation knobs, batching sizes, checkpoint/output dirs, metric Ks, and training args must all come from YAML
- `alarm_hgt/runtime_config.py` loads YAML into typed config structures and performs minimal validation
- `alarm_hgt/trainer.py` exposes a Transformers-`Trainer`-compatible implementation for the full HGT training flow, reusing the custom dataloader, collator, and metric adapter
- `alarm_hgt/train.py` accepts a config path, loads YAML, constructs datasets/model/trainer, and can run train/eval without embedding hyperparameters in code

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runtime_config.py tests/test_transformers_trainer.py -v`
Expected: PASS

**Step 5: Commit**

Skip: workspace is not a git repository.
