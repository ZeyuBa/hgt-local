from pathlib import Path

from src.training.trainer import run_pipeline

def _write_smoke_cli_config(config_dir, *, runtime_root):
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        f"""
synthetic:
  output_dir: {runtime_root / "generated"}
  seed: 7
  split_sizes:
    train: 4
    val: 2
    test: 2
  smoke_split_sizes:
    train: 16
    val: 4
    test: 4
  num_sites: [8, 16]
  wl_stations_per_site: [2, 4]
  fault_site_count: [1, 3]
  an_site_count: [1, 2]
  backup_link_probability: 0.18
  noise_probability: 0.08
  topology_mode: random_tree
dataset_paths:
  train: {runtime_root / "datasets" / "transformed_train.json"}
  val: {runtime_root / "datasets" / "transformed_val.json"}
  test: {runtime_root / "datasets" / "transformed_test.json"}
batching:
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  dataloader_drop_last: false
  dataloader_num_workers: 0
  dataloader_pin_memory: false
model:
  in_dim: 32
  n_hid: 64
  num_layers: 4
  n_heads: 4
  dropout: 0.2
  num_types: 3
  num_relations: 9
  conv_name: hgt
  use_rte: false
metrics:
  ks: [5, 10, 20, 50]
training_args:
  num_train_epochs: 8
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_ratio: 0.1
  logging_steps: 1
  seed: 13
outputs:
  checkpoints_dir: {runtime_root / "artifacts" / "checkpoints"}
  results_dir: {runtime_root / "artifacts" / "results"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_root_main_smoke_mode_passes_acceptance_gate(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_smoke_cli_config(tmp_path / "configs", runtime_root=runtime_root)

    artifacts = run_pipeline(config_path, "smoke", mode="train")

    assert artifacts["best_checkpoint"] == runtime_root / "artifacts" / "checkpoints" / "smoke-best.pt"
    assert artifacts["summary"] == runtime_root / "artifacts" / "results" / "smoke-summary.json"
