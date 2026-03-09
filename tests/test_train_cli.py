import json
import math
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_cli_config(
    config_dir: Path,
    *,
    generated_root: Path,
    dataset_root: Path,
    outputs_root: Path,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "alarm_hgt.yaml"
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
    train: 2
    val: 1
    test: 1
  num_sites: [4, 4]
  wl_stations_per_site: [1, 1]
  fault_site_count: [1, 1]
  an_site_count: [1, 1]
  backup_link_probability: 0.0
  noise_probability: 0.0
  topology_mode: chain
dataset_paths:
  train: {dataset_root / "from-config-train.json"}
  val: {dataset_root / "from-config-val.json"}
  test: {dataset_root / "from-config-test.json"}
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
  num_train_epochs: 1
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


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alarm_hgt.train", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_module_smoke_run_builds_runtime_from_yaml_path_and_writes_deterministic_artifacts(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )

    completed = _run_cli("--config", str(config_path), "--run-mode", "smoke")

    assert completed.returncode == 0, completed.stderr
    assert "stage=export" in completed.stdout
    assert "stage=train" in completed.stdout
    assert "stage=validation" in completed.stdout
    assert "stage=test" in completed.stdout
    assert "stage=artifact_save" in completed.stdout
    assert "stage=finished status=ok" in completed.stdout

    assert (runtime_root / "datasets" / "from-config-train.json").exists()
    assert (runtime_root / "datasets" / "from-config-val.json").exists()
    assert (runtime_root / "datasets" / "from-config-test.json").exists()
    assert (runtime_root / "artifacts" / "checkpoints" / "smoke-last.pt").exists()
    assert (runtime_root / "artifacts" / "results" / "smoke-summary.json").exists()
    assert (runtime_root / "artifacts" / "results" / "train_history.json").exists()
    assert (runtime_root / "artifacts" / "results" / "val_history.json").exists()

    train_history = json.loads(
        (runtime_root / "artifacts" / "results" / "train_history.json").read_text(encoding="utf-8")
    )
    val_history = json.loads(
        (runtime_root / "artifacts" / "results" / "val_history.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (runtime_root / "artifacts" / "results" / "smoke-summary.json").read_text(encoding="utf-8")
    )

    assert train_history["metric"] == "train_loss"
    assert val_history["metric"] == "val_loss"
    assert [entry["epoch"] for entry in train_history["history"]] == [1]
    assert [entry["epoch"] for entry in val_history["history"]] == [1]
    assert math.isfinite(train_history["history"][0]["train_loss"])
    assert math.isfinite(val_history["history"][0]["val_loss"])
    assert summary["train_history_path"].endswith("train_history.json")
    assert summary["val_history_path"].endswith("val_history.json")
    assert "edge_precision_at_1" in summary["test_metrics"]
    assert "edge_precision_at_2" in summary["test_metrics"]
    assert "edge_precision_at_5" not in summary["test_metrics"]


def test_module_exits_non_zero_for_missing_config_path():
    missing_path = PROJECT_ROOT / "does-not-exist.yaml"

    completed = _run_cli("--config", str(missing_path), "--run-mode", "smoke")

    assert completed.returncode != 0
    assert "status=ok" not in completed.stdout


def test_module_rejects_invalid_run_mode(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )

    completed = _run_cli("--config", str(config_path), "--run-mode", "invalid")

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr
    assert "status=ok" not in completed.stdout


def test_module_exits_non_zero_when_output_directory_is_unwritable(tmp_path):
    runtime_root = tmp_path / "runtime"
    blocked_path = runtime_root / "blocked"
    runtime_root.mkdir(parents=True, exist_ok=True)
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=blocked_path,
    )
    blocked_path.write_text("not a directory\n", encoding="utf-8")

    completed = _run_cli("--config", str(config_path), "--run-mode", "smoke")

    assert completed.returncode != 0
    assert "status=ok" not in completed.stdout
    assert "blocked" in completed.stderr
