import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_root_main_exposes_train_and_inference_modes():
    completed = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--mode" in completed.stdout
    assert "--run-mode" not in completed.stdout
    assert "train" in completed.stdout
    assert "inference" in completed.stdout


def test_root_main_can_run_full_training(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            str(config_path),
            "--mode",
            "train",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "<promise>COMPLETE</promise>" in completed.stdout
    assert (runtime_root / "artifacts" / "checkpoints" / "full-best.pt").exists()
    assert (runtime_root / "artifacts" / "results" / "full-summary.json").exists()


def test_root_main_can_run_inference_against_saved_checkpoint(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )

    train_completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            str(config_path),
            "--mode",
            "train",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert train_completed.returncode == 0, train_completed.stderr

    checkpoint_path = runtime_root / "artifacts" / "checkpoints" / "full-best.pt"
    inference_completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            str(config_path),
            "--mode",
            "inference",
            "--checkpoint-path",
            str(checkpoint_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert inference_completed.returncode == 0, inference_completed.stderr
    assert "<promise>COMPLETE</promise>" in inference_completed.stdout
    assert (runtime_root / "artifacts" / "results" / "test_metrics.json").exists()
