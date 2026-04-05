# Alarm HGT pipeline

Synthetic telecom-style topologies are turned into heterogeneous graphs; an HGT-based link predictor (local `pyHGT` + Hugging Face `Trainer`) estimates which `ne_is_disconnected` alarm entities should fire after fault propagation. There is no real production dataset—everything is generated from `configs/config.yaml`.

## What you need

Python 3.10+ and a typical ML stack: **PyTorch**, **transformers**, **numpy**, **scikit-learn**, **networkx**, **PyYAML**. Install versions that match your CUDA/CPU setup.

### Installation (recommended)

```bash
# Clone the repo
git clone https://github.com/ZeyuBa/hgt-local
cd hgt-local

# Install in editable mode (with core dependencies)
pip install -e .

# For development (includes tests, linters, formatters)
pip install -e ".[dev]"

# For visualization app (optional)
pip install -e ".[visualization]"
```

### Quick install with requirements.txt

Alternatively, install dependencies directly:

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

Run tests from the repo root (project root on `PYTHONPATH`):

```bash
pytest
```

## Usage

Train (regenerates synthetic splits, then trains and evaluates):

```bash
python main.py --config configs/config.yaml
```

Inference (same config schema; reloads checkpoint and evaluates):

```bash
python main.py --config configs/config.yaml --mode inference
```

Optional checkpoint:

```bash
python main.py --config configs/config.yaml --mode inference --checkpoint-path outputs/checkpoints/full-best.pt
```

`main.py` always runs the **full** data path (`run_pipeline(..., "full", ...)`). On success it prints `<promise>COMPLETE</promise>` to stdout. A **smoke** run mode exists internally for fast tests but is not exposed on the root CLI.

## Layout

```text
.
├── main.py                 # CLI: --config, --mode train|inference, --checkpoint-path
├── configs/config.yaml     # Synthetic data, paths, batching, model, training, outputs
├── training_data/          # Topology generation, splits, label propagation, JSON export
├── src/
│   ├── constants.py        # Centralized type registries (node types, alarms, relations)
│   ├── graph/              # Features and graph assembly
│   ├── dataset/            # JSONL datasets, batching, collate
│   ├── models/             # HGT encoder + link head
│   ├── training/
│   │   ├── config.py       # Runtime config dataclasses
│   │   ├── metrics.py      # Link prediction metrics (edge, graph, ranking)
│   │   └── trainer.py      # HF Trainer subclass + pipeline orchestration
│   └── inference/          # Checkpoint evaluation helpers
├── pyHGT/                  # Bundled HGT implementation (conv, model, data)
├── tests/
└── visualize_app.py        # Streamlit topology visualization
```

### Module dependency graph

```text
main.py
  └── src/training/trainer.py
        ├── src/training/metrics.py
        ├── src/training/config.py
        ├── src/constants.py          ← single source of truth for all type IDs
        ├── src/dataset/
        │     ├── hgt_dataset.py      ← src/constants, src/graph/feature_extraction
        │     ├── collate.py          ← src/constants
        │     └── bucket_sampler.py
        ├── src/models/
        │     ├── hgt_for_link_prediction.py
        │     ├── hgt.py              ← pyHGT/model.py
        │     └── edge_predictor.py
        ├── src/graph/
        │     ├── feature_extraction.py ← src/constants
        │     └── graph_builder.py
        ├── src/inference/predictor.py
        └── training_data/topo_complete.py
              └── topo_generator.py   ← src/constants
```

Generation writes `topology_{train,val,test}.json` then `transformed_{train,val,test}.json` under `synthetic.output_dir` (default `data/synthetic/`). Each transformed line is one graph sample (nodes, edges, alarm entities, labels, metadata).

## Pipeline

1. **Export** — synthetic sites, faults, propagation rules, optional noise; export JSONL splits.
2. **Train** — tensorize graphs, train link predictor, save HF checkpoints plus `full-best.pt` / `full-last.pt`, tune a decision threshold on val, report test metrics.
3. **Inference** — regenerate splits from the **current** config, load checkpoint, recover threshold from val, evaluate test.

Changing synthetic seeds or topology settings between train and inference means you are not scoring the exact same graphs as an earlier run.

## Config

`configs/config.yaml` is the single source of truth. Main sections:

| Section | Role |
|--------|------|
| `synthetic` | Output dir, seeds, split sizes (including `smoke_split_sizes` for tests), topology ranges, noise, `topology_mode` |
| `dataset_paths` | Train/val/test JSON paths |
| `batching` | Per-device batch sizes, workers, `pin_memory`, `drop_last` |
| `model` | `in_dim`, `n_hid`, `num_layers`, `n_heads`, `dropout`, type/relation counts, `conv_name`, `use_rte` |
| `metrics` | Ranking cutoffs (`ks`) |
| `training_args` | Epochs, LR, weight decay, warmup, logging, training seed |
| `outputs` | `checkpoints_dir`, `results_dir` |

## Outputs

After a full training run, expect under `outputs/` (paths come from config):

- `checkpoints/full-best.pt`, `full-last.pt`, and HuggingFace checkpoint folders
- `results/full-summary.json` — run metadata and linked artifacts
- `results/test_metrics.json`, `train_history.json`, `val_history.json`

`data/` and `outputs/` are gitignored by default.

## Notes

- **Target alarms:** Training focuses on `ne_is_disconnected`; other alarms are context or anchors, not the main prediction head target (see `training_data/` labeling logic).
- **Autoresearch:** Agent-oriented workflow docs live under `.claude/skills/autoresearch-hgt/`; they are optional and not required to train or infer.
