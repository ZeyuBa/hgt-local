# Alarm HGT Pipeline

This project builds and evaluates an alarm prediction model on fully synthetic telecom-style topology data. It models each sample as a heterogeneous graph and trains an HGT-based link predictor to estimate which `ne_is_disconnected` alarm entities should fire after fault propagation.

The repo is self-contained:
- it generates synthetic train/val/test graph samples
- it tensorizes them into HGT-ready batches
- it trains a pyHGT-based model through HuggingFace `Trainer`
- it saves best and last checkpoints
- it supports checkpoint-backed inference on regenerated synthetic splits

## Background

Each sample is one topology-centered event graph with:
- `NE` nodes for physical devices
- `AlarmEntity` nodes for "alarm X on network element Y"
- `Alarm` nodes for alarm templates

The current setup predicts only `ne_is_disconnected` alarm entities. Other alarms such as `mains_failure`, `device_powered_off`, and `link_down` are treated as observed conditions or propagation anchors, not trainable targets.

The synthetic data generator creates random site topologies, injects fault or risk sites, propagates outage logic through the graph, and adds optional noise alarms. That gives the model a controlled but nontrivial supervision signal without relying on a real production dataset.

## Repo Structure

```text
.
|-- main.py                     # unified CLI entrypoint
|-- configs/
|   `-- config.yaml            # runtime config for synthetic data, model, batching, outputs
|-- training_data/
|   |-- topo_generator.py      # base topology generation
|   |-- topo_combiner.py       # split export helpers
|   `-- topo_complete.py       # label propagation and transformed JSON export
|-- src/
|   |-- graph/                 # feature extraction and graph assembly helpers
|   |-- dataset/               # JSONL dataset, bucket sampling, padding collate
|   |-- models/                # HGT encoder, edge predictor, model wrapper
|   |-- training/              # config loading, trainer, runtime orchestration
|   `-- inference/             # checkpoint loading and evaluation helpers
|-- pyHGT/                     # local pyHGT implementation used by the model
|-- tests/                     # unit and end-to-end coverage
|-- data/synthetic/            # generated transformed datasets
`-- outputs/
    |-- checkpoints/           # best/last checkpoints and HF trainer checkpoints
    `-- results/               # metrics, histories, run summaries
```

## Data Generation

Synthetic export happens before both training and inference. The generator writes:
- `data/synthetic/transformed_train.json`
- `data/synthetic/transformed_val.json`
- `data/synthetic/transformed_test.json`

Each line is one complete heterogeneous graph sample containing:
- node list
- topology edges
- alarm entities with labels
- site-level metadata such as AN sites, fault sites, and logical failures

The main generation stages are:
1. Build a synthetic topology with site count, station count, AN sites, fault sites, and optional backup links.
2. Assign fault modes such as `mains_failure` or `link_down`.
3. Propagate outage and reachability rules to derive alarm labels.
4. Export complete JSONL splits that the dataset layer later tensorizes.

The default config uses fixed seeds, so repeated runs are deterministic unless you change the seeds or generation ranges in the config.

## Train and Inference Logic

### Training flow

`main.py` calls the unified runtime pipeline in `src/training/trainer.py`.

The full train path is:
1. Load YAML config.
2. Regenerate synthetic splits.
3. Build datasets for `train`, `val`, and `test`.
4. Validate feature and relation counts against the model design.
5. Train the HGT link predictor with a custom `Trainer` subclass.
6. Save HuggingFace checkpoints plus compact `full-best.pt` and `full-last.pt`.
7. Re-evaluate the best checkpoint on validation to get a decision threshold.
8. Evaluate the best checkpoint on test and write run artifacts.

### Inference flow

Inference is checkpoint-backed, but it still regenerates synthetic splits from the current config before evaluation. That means inference assumes:
- the same config schema
- compatible synthetic generation settings
- a checkpoint trained for this model layout

The inference path:
1. Load config.
2. Regenerate synthetic splits.
3. Build datasets and model.
4. Load a saved checkpoint.
5. Evaluate validation to recover the configured test decision threshold.
6. Evaluate test and write summary artifacts.

## Config

The runtime config lives in `configs/config.yaml` and has six important sections:

- `synthetic`
  Controls output directory, RNG seed, split sizes, topology ranges, noise rate, and topology mode.
- `dataset_paths`
  Declares where transformed train/val/test JSON files are written and read.
- `batching`
  Controls per-device batch sizes and dataloader behavior.
- `model`
  Defines HGT dimensions and relation/type counts.
- `metrics`
  Declares ranking cutoffs such as `k = 5, 10, 20, 50`.
- `training_args`
  Sets epochs, learning rate, weight decay, warmup ratio, logging cadence, and training seed.
- `outputs`
  Sets checkpoint and result directories.

Default model settings:
- input dim: `32`
- hidden dim: `64`
- HGT layers: `4`
- attention heads: `4`
- node types: `3`
- relation types: `9`

Default synthetic split sizes:
- train: `128`
- val: `32`
- test: `32`

## One-Line Commands

Train on the current synthetic config:

```bash
python main.py --config configs/config.yaml
```

Run inference from the default best checkpoint:

```bash
python main.py --config configs/config.yaml --mode inference
```

Run inference from a specific checkpoint:

```bash
python main.py --config configs/config.yaml --mode inference --checkpoint-path outputs/checkpoints/full-best.pt
```

## Outputs

After a successful full run, the main artifacts are:
- `outputs/checkpoints/full-best.pt`
- `outputs/checkpoints/full-last.pt`
- `outputs/results/full-summary.json`
- `outputs/results/test_metrics.json`
- `outputs/results/train_history.json`
- `outputs/results/val_history.json`

`test_metrics.json` contains the final reported test metrics. `full-summary.json` links the config, dataset paths, checkpoint paths, and metric files used for the run.

## Notes

- There is no real-world dataset in this repo. The current workflow is synthetic-data-first.
- The CLI exposed by `main.py` runs the `full` pipeline. Smoke-mode helpers exist in the runtime layer and tests, but they are not wired into the root CLI arguments.
- Inference currently re-exports synthetic splits from config before evaluation. If you change the synthetic seed or generation rules, you are not evaluating against the exact same split as an earlier run.
