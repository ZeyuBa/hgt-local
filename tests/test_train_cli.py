import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import alarm_hgt.train as train_module
from alarm_hgt.metrics import compute_link_prediction_metrics
from alarm_hgt.modeling import LinkPredictionOutput, masked_bce_loss
from alarm_hgt.runtime_config import load_runtime_config
from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample
from alarm_hgt.train import (
    build_runtime_objects,
    evaluate_test_split,
    export_runtime_data,
    prepare_runtime_environment,
    resolve_runtime_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_cli_config(
    config_dir: Path,
    *,
    generated_root: Path,
    dataset_root: Path,
    outputs_root: Path,
    model_overrides: dict[str, object] | None = None,
    smoke_split_sizes: tuple[int, int, int] = (2, 1, 1),
    training_overrides: dict[str, object] | None = None,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "alarm_hgt.yaml"
    model_values: dict[str, object] = {
        "in_dim": 32,
        "n_hid": 16,
        "num_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "num_types": 3,
        "num_relations": 9,
        "conv_name": "hgt",
        "use_rte": False,
    }
    if model_overrides:
        model_values.update(model_overrides)

    training_values: dict[str, object] = {
        "num_train_epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "logging_steps": 1,
        "seed": 13,
    }
    if training_overrides:
        training_values.update(training_overrides)

    def _yaml_scalar(value: object) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    model_yaml = "\n".join(
        f"  {key}: {_yaml_scalar(value)}" for key, value in model_values.items()
    )
    training_yaml = "\n".join(
        f"  {key}: {_yaml_scalar(value)}" for key, value in training_values.items()
    )
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
    train: {smoke_split_sizes[0]}
    val: {smoke_split_sizes[1]}
    test: {smoke_split_sizes[2]}
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
{model_yaml}
metrics:
  ks: [1, 2]
training_args:
{training_yaml}
outputs:
  checkpoints_dir: {outputs_root / "checkpoints"}
  results_dir: {outputs_root / "results"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_samples(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )


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
        smoke_split_sizes=(16, 4, 4),
        training_overrides={
            "num_train_epochs": 8,
            "learning_rate": 0.001,
        },
    )

    completed = _run_cli("--config", str(config_path), "--run-mode", "smoke")

    assert completed.returncode == 0, completed.stderr
    assert "stage=export" in completed.stdout
    assert "stage=train" in completed.stdout
    assert "stage=validation" in completed.stdout
    assert "stage=test" in completed.stdout
    assert "stage=artifact_save" in completed.stdout
    assert "stage=verify" in completed.stdout
    assert "stage=finished status=ok" in completed.stdout
    assert completed.stdout.rstrip().endswith("<promise>COMPLETE</promise>")

    assert (runtime_root / "datasets" / "from-config-train.json").exists()
    assert (runtime_root / "datasets" / "from-config-val.json").exists()
    assert (runtime_root / "datasets" / "from-config-test.json").exists()
    assert (runtime_root / "artifacts" / "checkpoints" / "smoke-last.pt").exists()
    assert (runtime_root / "artifacts" / "checkpoints" / "smoke-best.pt").exists()
    assert (runtime_root / "artifacts" / "results" / "smoke-summary.json").exists()
    assert (runtime_root / "artifacts" / "results" / "train_history.json").exists()
    assert (runtime_root / "artifacts" / "results" / "val_history.json").exists()
    assert (runtime_root / "artifacts" / "results" / "test_metrics.json").exists()

    train_history = json.loads(
        (runtime_root / "artifacts" / "results" / "train_history.json").read_text(encoding="utf-8")
    )
    val_history = json.loads(
        (runtime_root / "artifacts" / "results" / "val_history.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (runtime_root / "artifacts" / "results" / "smoke-summary.json").read_text(encoding="utf-8")
    )
    test_metrics = json.loads(
        (runtime_root / "artifacts" / "results" / "test_metrics.json").read_text(encoding="utf-8")
    )

    assert train_history["metric"] == "train_loss"
    assert val_history["metric"] == "val_loss"
    assert [entry["epoch"] for entry in train_history["history"]] == list(range(1, 9))
    assert [entry["epoch"] for entry in val_history["history"]] == list(range(1, 9))
    assert all(math.isfinite(entry["train_loss"]) for entry in train_history["history"])
    assert all(math.isfinite(entry["val_loss"]) for entry in val_history["history"])
    assert summary["train_history_path"].endswith("train_history.json")
    assert summary["val_history_path"].endswith("val_history.json")
    assert summary["best_checkpoint_path"].endswith("smoke-best.pt")
    assert "edge_precision_at_1" in summary["test_metrics"]
    assert "edge_precision_at_2" in summary["test_metrics"]
    assert "edge_precision_at_5" not in summary["test_metrics"]
    assert math.isfinite(test_metrics["precision"])
    assert math.isfinite(test_metrics["recall"])
    assert math.isfinite(test_metrics["f1"])
    assert test_metrics["f1"] >= 0.60


def test_evaluate_test_split_fails_when_selected_checkpoint_is_missing(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )
    config = load_runtime_config(config_path)
    paths = resolve_runtime_paths(config)
    prepare_runtime_environment(paths)
    export_runtime_data(config, paths, "smoke")
    runtime = build_runtime_objects(config, paths)

    with pytest.raises(FileNotFoundError, match="smoke-best.pt"):
        evaluate_test_split(runtime, checkpoint_path=paths.checkpoints_dir / "smoke-best.pt")


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


def test_module_smoke_run_exits_non_zero_when_smoke_thresholds_are_not_met(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
        smoke_split_sizes=(8, 4, 4),
        training_overrides={
            "num_train_epochs": 8,
            "learning_rate": 0.001,
        },
    )

    completed = _run_cli("--config", str(config_path), "--run-mode", "smoke")

    assert completed.returncode != 0
    assert "<promise>COMPLETE</promise>" not in completed.stdout
    assert "smoke acceptance failed" in completed.stderr


def test_main_exits_non_zero_without_promise_when_saved_artifact_verification_fails(
    tmp_path,
    monkeypatch,
    capsys,
):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
        smoke_split_sizes=(16, 4, 4),
        training_overrides={
            "num_train_epochs": 8,
            "learning_rate": 0.001,
        },
    )

    original_save_run_artifacts = train_module.save_run_artifacts

    def save_and_remove_test_metrics(*args, **kwargs):
        artifacts = original_save_run_artifacts(*args, **kwargs)
        Path(artifacts["test_metrics"]).unlink()
        return artifacts

    monkeypatch.setattr(train_module, "save_run_artifacts", save_and_remove_test_metrics)

    exit_code = train_module.main(["--config", str(config_path), "--run-mode", "smoke"])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "<promise>COMPLETE</promise>" not in captured.out
    assert "test_metrics.json" in captured.err


def test_main_exits_non_zero_when_saved_test_metrics_fail_smoke_thresholds(
    tmp_path,
    monkeypatch,
    capsys,
):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
        smoke_split_sizes=(16, 4, 4),
        training_overrides={
            "num_train_epochs": 8,
            "learning_rate": 0.001,
        },
    )

    original_save_run_artifacts = train_module.save_run_artifacts

    def save_and_corrupt_test_metrics(*args, **kwargs):
        artifacts = original_save_run_artifacts(*args, **kwargs)
        test_metrics_path = Path(artifacts["test_metrics"])
        payload = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        payload["f1"] = 0.10
        test_metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        summary_path = Path(artifacts["summary"])
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_payload["test_metrics"]["f1"] = 0.10
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifacts

    monkeypatch.setattr(train_module, "save_run_artifacts", save_and_corrupt_test_metrics)

    exit_code = train_module.main(["--config", str(config_path), "--run-mode", "smoke"])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "<promise>COMPLETE</promise>" not in captured.out
    assert "below required threshold" in captured.err


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


def test_module_smoke_run_emits_complete_only_after_thresholds_pass(tmp_path):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
        smoke_split_sizes=(16, 4, 4),
        training_overrides={
            "num_train_epochs": 8,
            "learning_rate": 0.001,
        },
    )

    completed = _run_cli("--config", str(config_path), "--run-mode", "smoke")

    assert completed.returncode == 0, completed.stderr
    assert "<promise>COMPLETE</promise>" in completed.stdout

    train_history = json.loads(
        (runtime_root / "artifacts" / "results" / "train_history.json").read_text(encoding="utf-8")
    )["history"]
    val_history = json.loads(
        (runtime_root / "artifacts" / "results" / "val_history.json").read_text(encoding="utf-8")
    )["history"]
    test_metrics = json.loads(
        (runtime_root / "artifacts" / "results" / "test_metrics.json").read_text(encoding="utf-8")
    )

    improving_transitions = 0
    for index in range(1, len(train_history)):
        train_improved = train_history[index]["train_loss"] < train_history[index - 1]["train_loss"]
        val_improved = val_history[index]["val_loss"] < val_history[index - 1]["val_loss"]
        improving_transitions += int(train_improved or val_improved)

    assert improving_transitions >= 4
    assert test_metrics["f1"] >= 0.60


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_fragment"),
    [
        ("in_dim", 31, "expected model.in_dim=32"),
        ("num_types", 4, "expected model.num_types=3"),
        ("num_relations", 10, "expected model.num_relations=9"),
    ],
)
def test_module_exits_non_zero_when_model_design_drifts_from_spec(
    tmp_path,
    field,
    bad_value,
    expected_fragment,
):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
        model_overrides={field: bad_value},
    )

    completed = _run_cli("--config", str(config_path), "--run-mode", "smoke")

    assert completed.returncode != 0
    assert "model design drift" in completed.stderr
    assert expected_fragment in completed.stderr
    assert "status=ok" not in completed.stdout


def test_config_driven_runtime_masks_loss_and_metrics_to_trainable_alarm_entities_only(
    tmp_path,
    monkeypatch,
):
    runtime_root = tmp_path / "runtime"
    config_path = _write_cli_config(
        tmp_path / "nested" / "configs",
        generated_root=runtime_root / "generated",
        dataset_root=runtime_root / "datasets",
        outputs_root=runtime_root / "artifacts",
    )
    config = load_runtime_config(config_path)
    paths = resolve_runtime_paths(config)
    prepare_runtime_environment(paths)

    graph_config = SyntheticGraphConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    mains_failure_sample = generate_sample(
        seed=101,
        config=graph_config,
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_001"],
        forced_fault_modes={"site_001": "mains_failure"},
        forced_noise_sites=[],
    )
    link_down_sample = generate_sample(
        seed=102,
        config=graph_config,
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_002"],
        forced_fault_modes={"site_002": "link_down"},
        forced_noise_sites=[],
    )
    for split_path in paths.dataset_paths.values():
        _write_samples(split_path, [mains_failure_sample, link_down_sample])

    runtime = build_runtime_objects(config, paths)

    observed_alarm_names = {
        alarm_entity_id.split(";")[0]
        for index in range(len(runtime.datasets["train"]))
        for alarm_entity_id in runtime.datasets["train"][index]["alarm_entity_ids"]
    }
    assert observed_alarm_names == {
        "ne_is_disconnected",
        "mains_failure",
        "device_powered_off",
        "link_down",
    }

    training_result = runtime.trainer.train_and_validate(
        num_epochs=runtime.config.training_args.num_train_epochs,
        learning_rate=runtime.config.training_args.learning_rate,
        weight_decay=runtime.config.training_args.weight_decay,
        results_dir=runtime.paths.results_dir,
    )
    test_result = runtime.trainer.evaluate(
        runtime.datasets["test"],
        split_name="test",
        epoch=runtime.config.training_args.num_train_epochs,
    )

    assert math.isfinite(training_result.train_history[-1]["train_loss"])
    assert math.isfinite(training_result.val_history[-1]["val_loss"])
    assert math.isfinite(test_result.loss)

    eval_batch = next(iter(runtime.trainer.get_eval_dataloader(runtime.datasets["val"])))

    def scripted_forward(**batch):
        labels = batch["labels"]
        trainable_mask = batch["trainable_mask"]
        correct_logits = torch.where(labels > 0.5, torch.full_like(labels, 6.0), torch.full_like(labels, -6.0))
        incorrect_logits = -correct_logits
        logits = torch.where(trainable_mask, correct_logits, incorrect_logits)
        loss = masked_bce_loss(logits, labels, trainable_mask)
        return LinkPredictionOutput(
            loss=loss,
            logits=logits,
            node_embeddings=torch.zeros(
                batch["node_features"].shape[0],
                runtime.model.config.n_hid,
                dtype=batch["node_features"].dtype,
            ),
        )

    monkeypatch.setattr(runtime.model, "forward", scripted_forward)

    scripted_output = runtime.model(**eval_batch)
    leaky_loss = masked_bce_loss(
        scripted_output.logits,
        eval_batch["labels"],
        torch.ones_like(eval_batch["trainable_mask"], dtype=torch.bool),
    )
    masked_metrics = compute_link_prediction_metrics(
        scripted_output.logits.detach().cpu().numpy(),
        eval_batch["labels"].detach().cpu().numpy(),
        eval_batch["trainable_mask"].detach().cpu().numpy(),
        ks=runtime.config.metrics.ks,
    )
    leaky_metrics = compute_link_prediction_metrics(
        scripted_output.logits.detach().cpu().numpy(),
        eval_batch["labels"].detach().cpu().numpy(),
        np.ones_like(eval_batch["trainable_mask"].detach().cpu().numpy(), dtype=bool),
        ks=runtime.config.metrics.ks,
    )
    eval_result = runtime.trainer.evaluate(
        runtime.datasets["val"],
        split_name="val",
        epoch=runtime.config.training_args.num_train_epochs,
    )

    assert float(scripted_output.loss) < float(leaky_loss)
    assert eval_result.loss == pytest.approx(float(scripted_output.loss))
    assert eval_result.metrics["f1"] == pytest.approx(masked_metrics["f1"])
    assert masked_metrics["f1"] == pytest.approx(1.0)
    assert leaky_metrics["f1"] < 1.0
