import json

from src.training.trainer import (
    build_runtime_objects,
    export_runtime_data,
    prepare_runtime_environment,
    resolve_runtime_paths,
    run_training_pipeline,
    run_inference_pipeline,
)
from src.training.config import load_runtime_config


def _write_cli_config(config_dir, *, generated_root, dataset_root, outputs_root):
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        f"""
synthetic:
  output_dir: {generated_root}
  seed: 7
  split_sizes:
    train: 4
    val: 2
    test: 2
  smoke_split_sizes:
    train: 8
    val: 4
    test: 4
  num_sites: [4, 4]
  wl_stations_per_site: [1, 1]
  fault_site_count: [1, 1]
  an_site_count: [1, 1]
  backup_link_probability: 0.0
  noise_probability: 0.0
  topology_mode: chain
dataset_paths:
  train: {dataset_root / "transformed_train.json"}
  val: {dataset_root / "transformed_val.json"}
  test: {dataset_root / "transformed_test.json"}
batching:
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  dataloader_drop_last: false
  dataloader_num_workers: 0
  dataloader_pin_memory: false
model:
  in_dim: 32
  n_hid: 16
  num_layers: 2
  n_heads: 4
  dropout: 0.0
  num_types: 3
  num_relations: 9
  conv_name: hgt
  use_rte: false
metrics:
  ks: [1, 2]
training_args:
  num_train_epochs: 2
  learning_rate: 0.001
  weight_decay: 0.0
  warmup_ratio: 0.0
  logging_steps: 1
  seed: 13
outputs:
  checkpoints_dir: {outputs_root / "checkpoints"}
  results_dir: {outputs_root / "results"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_inference_pipeline_can_reuse_saved_checkpoint_without_retraining(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )

    config = load_runtime_config(config_path)
    paths = resolve_runtime_paths(config)
    prepare_runtime_environment(paths)
    export_runtime_data(config, paths, "full")
    runtime = build_runtime_objects(config, paths)

    train_artifacts = run_training_pipeline(runtime, run_mode="full")
    inference_artifacts = run_inference_pipeline(
        runtime,
        run_mode="full",
        checkpoint_path=train_artifacts["best_checkpoint"],
    )

    summary = json.loads(inference_artifacts["summary"].read_text(encoding="utf-8"))

    assert inference_artifacts["best_checkpoint"].exists()
    assert inference_artifacts["test_metrics"].exists()
    assert summary["run_mode"] == "full"
    assert summary["checkpoint_path"].endswith("full-best.pt")
