"""Config-driven CLI entrypoint for smoke and full pipeline runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import HGT_NODE_TYPE_IDS, RELATION_TYPE_IDS
from .dataset import AlarmGraphDataset
from .export import export_synthetic_splits
from .features import FEATURE_DIM
from .modeling import HGTForLinkPrediction
from .runtime_config import RuntimeConfig, RuntimeConfigError, load_runtime_config
from .trainer import LinkPredictionTrainer, build_compute_metrics

RUN_MODE_CHOICES = ("full", "smoke")
REQUIRED_TEST_METRIC_KEYS = ("precision", "recall", "f1")
SMOKE_MIN_F1 = 0.60
SMOKE_SUCCESS_PROMISE = "<promise>COMPLETE</promise>"


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    synthetic_output_dir: Path
    dataset_paths: dict[str, Path]
    checkpoints_dir: Path
    results_dir: Path


@dataclass
class RuntimeObjects:
    config: RuntimeConfig
    paths: ResolvedRuntimePaths
    datasets: dict[str, AlarmGraphDataset]
    model: HGTForLinkPrediction
    trainer: LinkPredictionTrainer


@dataclass(frozen=True)
class CheckpointArtifacts:
    last_checkpoint_path: Path
    best_checkpoint_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the alarm HGT pipeline")
    parser.add_argument("--config", required=True, help="Path to the runtime YAML config")
    parser.add_argument(
        "--run-mode",
        default="full",
        choices=RUN_MODE_CHOICES,
        help="Choose whether to use full split sizes or the smoke-sized split sizes",
    )
    return parser.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    return path.resolve()


def resolve_runtime_paths(config: RuntimeConfig) -> ResolvedRuntimePaths:
    return ResolvedRuntimePaths(
        synthetic_output_dir=_resolve_path(config.synthetic.output_dir),
        dataset_paths={
            "train": _resolve_path(config.dataset_paths.train),
            "val": _resolve_path(config.dataset_paths.val),
            "test": _resolve_path(config.dataset_paths.test),
        },
        checkpoints_dir=_resolve_path(config.outputs.checkpoints_dir),
        results_dir=_resolve_path(config.outputs.results_dir),
    )


def _select_split_sizes(config: RuntimeConfig, run_mode: str) -> dict[str, int]:
    section = config.synthetic.smoke_split_sizes if run_mode == "smoke" else config.synthetic.split_sizes
    return section.as_dict()


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_runtime_environment(paths: ResolvedRuntimePaths) -> None:
    paths.synthetic_output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_path in paths.dataset_paths.values():
        _ensure_parent(dataset_path)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)


def _log(stage: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    message = f"stage={stage}"
    if details:
        message = f"{message} {details}"
    print(message, flush=True)


def export_runtime_data(config: RuntimeConfig, paths: ResolvedRuntimePaths, run_mode: str) -> dict[str, Path]:
    split_sizes = _select_split_sizes(config, run_mode)
    return export_synthetic_splits(
        output_dir=paths.synthetic_output_dir,
        split_sizes=split_sizes,
        config=config.synthetic.to_generation_config(),
        seed=config.synthetic.seed,
        output_paths=paths.dataset_paths,
        representative_smoke=run_mode == "smoke",
    )


def _validate_runtime_design_compliance(
    config: RuntimeConfig,
    datasets: dict[str, AlarmGraphDataset],
) -> None:
    expected_model_values = {
        "model.in_dim": FEATURE_DIM,
        "model.num_types": len(HGT_NODE_TYPE_IDS),
        "model.num_relations": len(RELATION_TYPE_IDS),
    }
    observed_model_values = {
        "model.in_dim": config.model.in_dim,
        "model.num_types": config.model.num_types,
        "model.num_relations": config.model.num_relations,
    }
    drift_messages = [
        f"expected {field}={expected}, got {observed_model_values[field]}"
        for field, expected in expected_model_values.items()
        if observed_model_values[field] != expected
    ]
    if drift_messages:
        raise ValueError(f"model design drift: {'; '.join(drift_messages)}")

    expected_node_types = set(HGT_NODE_TYPE_IDS.values())
    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            continue

        sample = dataset[0]
        feature_width = int(sample["node_features"].shape[1])
        if feature_width != FEATURE_DIM:
            raise ValueError(
                f"model design drift: expected feature width {FEATURE_DIM}, "
                f"got {feature_width} in {split_name} sample {sample['sample_id']}"
            )

        observed_node_types = set(sample["node_type"].tolist())
        if observed_node_types != expected_node_types:
            raise ValueError(
                f"model design drift: expected HGT node types {sorted(expected_node_types)}, "
                f"got {sorted(observed_node_types)} in {split_name} sample {sample['sample_id']}"
            )

        leaked_alarm_ids = [
            alarm_entity_id
            for alarm_entity_id, is_trainable in zip(
                sample["alarm_entity_ids"],
                sample["trainable_mask"].tolist(),
                strict=False,
            )
            if is_trainable and not alarm_entity_id.startswith("ne_is_disconnected;")
        ]
        if leaked_alarm_ids:
            raise ValueError(
                "model design drift: non-trainable alarms leaked into trainable_mask "
                f"in {split_name} sample {sample['sample_id']}: {leaked_alarm_ids[:3]}"
            )


def build_runtime_objects(config: RuntimeConfig, paths: ResolvedRuntimePaths) -> RuntimeObjects:
    datasets = {split: AlarmGraphDataset(path) for split, path in paths.dataset_paths.items()}
    _validate_runtime_design_compliance(config, datasets)
    model = HGTForLinkPrediction(config.to_model_config())
    trainer = LinkPredictionTrainer(
        model=model,
        args=config.to_trainer_args(),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=build_compute_metrics(config.metrics.ks),
    )
    return RuntimeObjects(
        config=config,
        paths=paths,
        datasets=datasets,
        model=model,
        trainer=trainer,
    )


def _checkpoint_payload(
    *,
    run_mode: str,
    model_state_dict: dict[str, Any],
    epoch: int | None = None,
    val_loss: float | None = None,
) -> dict[str, Any]:
    return {
        "run_mode": run_mode,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model_state_dict,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def save_checkpoints(
    runtime: RuntimeObjects,
    *,
    run_mode: str,
    best_epoch: int | None,
    best_val_loss: float | None,
    best_model_state: dict[str, Any] | None,
) -> CheckpointArtifacts:
    if best_model_state is None:
        raise ValueError("training did not produce a best checkpoint state")

    last_checkpoint_path = runtime.paths.checkpoints_dir / f"{run_mode}-last.pt"
    best_checkpoint_path = runtime.paths.checkpoints_dir / f"{run_mode}-best.pt"
    torch.save(
        _checkpoint_payload(
            run_mode=run_mode,
            model_state_dict=runtime.model.state_dict(),
        ),
        last_checkpoint_path,
    )
    torch.save(
        _checkpoint_payload(
            run_mode=run_mode,
            model_state_dict=best_model_state,
            epoch=best_epoch,
            val_loss=best_val_loss,
        ),
        best_checkpoint_path,
    )
    return CheckpointArtifacts(
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
    )


def _evaluate_checkpoint_split(
    runtime: RuntimeObjects,
    *,
    checkpoint_path: Path,
    split_name: str,
    decision_threshold: float = 0.5,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    runtime.model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint.get("epoch")
    epoch = int(checkpoint_epoch) if checkpoint_epoch is not None else runtime.config.training_args.num_train_epochs
    return runtime.trainer.evaluate(
        runtime.datasets[split_name],
        split_name=split_name,
        epoch=epoch,
        decision_threshold=decision_threshold,
    )


def evaluate_test_split(
    runtime: RuntimeObjects,
    *,
    checkpoint_path: Path,
    decision_threshold: float = 0.5,
):
    return _evaluate_checkpoint_split(
        runtime,
        checkpoint_path=checkpoint_path,
        split_name="test",
        decision_threshold=decision_threshold,
    )


def _calibrate_validation_threshold(
    runtime: RuntimeObjects,
    *,
    checkpoint_path: Path,
) -> float:
    validation_result = _evaluate_checkpoint_split(
        runtime,
        checkpoint_path=checkpoint_path,
        split_name="val",
    )
    threshold = float(validation_result.metrics.get("edge_best_threshold", 0.5))
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"validation threshold must be within [0.0, 1.0], got {threshold}")
    return threshold


def _count_improving_transitions(
    train_history: list[dict[str, float | int]],
    val_history: list[dict[str, float | int]],
) -> int:
    if len(train_history) != len(val_history):
        raise ValueError("train and validation history lengths must match for smoke acceptance")

    improving_transitions = 0
    for index in range(1, len(train_history)):
        train_improved = float(train_history[index]["train_loss"]) < float(train_history[index - 1]["train_loss"])
        val_improved = float(val_history[index]["val_loss"]) < float(val_history[index - 1]["val_loss"])
        improving_transitions += int(train_improved or val_improved)
    return improving_transitions


def _enforce_smoke_acceptance(
    train_history: list[dict[str, float | int]],
    val_history: list[dict[str, float | int]],
    *,
    test_metrics: dict[str, float],
) -> None:
    transitions = len(train_history) - 1
    if transitions < 1:
        raise ValueError("smoke acceptance failed: requires at least two epochs")

    improving_transitions = _count_improving_transitions(
        train_history,
        val_history,
    )
    if improving_transitions <= transitions / 2:
        raise ValueError(
            "smoke acceptance failed: "
            f"{improving_transitions} of {transitions} epoch transitions improved train or validation loss"
        )

    f1 = float(test_metrics["f1"])
    if f1 < SMOKE_MIN_F1:
        raise ValueError(
            "smoke acceptance failed: "
            f"test f1 {f1:.3f} is below required threshold {SMOKE_MIN_F1:.2f}"
        )


def write_test_metrics(path: Path, test_metrics: dict[str, float]) -> Path:
    payload = dict(test_metrics)
    missing_or_invalid = [
        key
        for key in REQUIRED_TEST_METRIC_KEYS
        if key not in payload or not math.isfinite(float(payload[key]))
    ]
    if missing_or_invalid:
        missing = ", ".join(missing_or_invalid)
        raise ValueError(f"missing required test metrics: {missing}")
    return _write_json(path, payload)


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"{label} not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {json_path}")
    return payload


def _summary_path(summary_payload: dict[str, Any], *, key: str, label: str) -> Path:
    value = summary_payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run summary missing {key}")
    return Path(value)


def _load_loss_history(
    path: str | Path,
    *,
    split: str,
    metric_key: str,
) -> list[dict[str, float | int]]:
    payload = _load_json_object(path, label=f"{split} loss history")
    if payload.get("split") != split:
        raise ValueError(f"{split} loss history split mismatch: {path}")
    if payload.get("metric") != metric_key:
        raise ValueError(f"{split} loss history metric mismatch: {path}")

    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{split} loss history is empty: {path}")

    normalized_history: list[dict[str, float | int]] = []
    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{split} loss history entry {index} is not an object: {path}")
        if "epoch" not in entry or metric_key not in entry:
            raise ValueError(f"{split} loss history entry {index} is missing required keys: {path}")

        epoch = int(entry["epoch"])
        loss_value = float(entry[metric_key])
        if not math.isfinite(loss_value):
            raise ValueError(f"{split} loss history entry {index} is non-finite: {path}")
        normalized_history.append({"epoch": epoch, metric_key: loss_value})
    return normalized_history


def _load_test_metrics(path: str | Path) -> dict[str, float]:
    payload = _load_json_object(path, label="test metrics")
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"test metric {key} must be numeric: {path}") from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"test metric {key} is non-finite: {path}")
        metrics[key] = numeric_value

    missing_or_invalid = [key for key in REQUIRED_TEST_METRIC_KEYS if key not in metrics]
    if missing_or_invalid:
        missing = ", ".join(missing_or_invalid)
        raise ValueError(f"missing required test metrics: {missing}")
    return metrics


def _validate_checkpoint_artifact(path: str | Path, *, label: str) -> None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{label} checkpoint payload must be a mapping: {checkpoint_path}")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError(f"{label} checkpoint missing model_state_dict: {checkpoint_path}")


def verify_completion_artifacts(summary_path: str | Path, *, run_mode: str) -> dict[str, Path]:
    summary_path = Path(summary_path).resolve()
    summary = _load_json_object(summary_path, label="run summary")

    summary_run_mode = summary.get("run_mode")
    if summary_run_mode != run_mode:
        raise ValueError(f"run summary mode mismatch: expected {run_mode}, got {summary_run_mode}")

    result_paths = {
        "summary": summary_path,
        "train_history": _summary_path(summary, key="train_history_path", label="train history"),
        "val_history": _summary_path(summary, key="val_history_path", label="validation history"),
        "test_metrics": _summary_path(summary, key="test_metrics_path", label="test metrics"),
    }
    checkpoint_paths = {
        "checkpoint": _summary_path(summary, key="checkpoint_path", label="checkpoint"),
        "best_checkpoint": _summary_path(summary, key="best_checkpoint_path", label="best checkpoint"),
    }

    results_dir = summary_path.parent
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    checkpoints_dir = checkpoint_paths["best_checkpoint"].resolve().parent
    if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"checkpoints directory not found: {checkpoints_dir}")
    if not any(checkpoints_dir.iterdir()):
        raise ValueError(f"checkpoints directory is empty: {checkpoints_dir}")

    for label, path in result_paths.items():
        if path.resolve().parent != results_dir:
            raise ValueError(f"{label} artifact is outside the results directory: {path}")
    for label, path in checkpoint_paths.items():
        if path.resolve().parent != checkpoints_dir:
            raise ValueError(f"{label} artifact is outside the checkpoints directory: {path}")

    _validate_checkpoint_artifact(checkpoint_paths["checkpoint"], label="last")
    _validate_checkpoint_artifact(checkpoint_paths["best_checkpoint"], label="best")

    train_history = _load_loss_history(
        result_paths["train_history"],
        split="train",
        metric_key="train_loss",
    )
    val_history = _load_loss_history(
        result_paths["val_history"],
        split="val",
        metric_key="val_loss",
    )
    test_metrics = _load_test_metrics(result_paths["test_metrics"])

    summary_test_metrics = summary.get("test_metrics")
    if not isinstance(summary_test_metrics, dict):
        raise ValueError("run summary missing test_metrics")
    for key in REQUIRED_TEST_METRIC_KEYS:
        try:
            summary_metric_value = float(summary_test_metrics[key])
        except KeyError as exc:
            raise ValueError(f"run summary missing test metric {key}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"run summary test metric {key} must be numeric") from exc
        if not math.isclose(summary_metric_value, test_metrics[key], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"run summary test metric {key} does not match saved test metrics")

    if run_mode == "smoke":
        _enforce_smoke_acceptance(
            train_history,
            val_history,
            test_metrics=test_metrics,
        )

    return {
        **result_paths,
        **checkpoint_paths,
    }


def save_run_artifacts(
    runtime: RuntimeObjects,
    run_mode: str,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float],
    checkpoint_artifacts: CheckpointArtifacts,
    test_metrics_path: Path,
    train_history_path: Path | None = None,
    val_history_path: Path | None = None,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
) -> dict[str, Path]:
    summary_path = runtime.paths.results_dir / f"{run_mode}-summary.json"

    _write_json(
        summary_path,
        {
            "best_checkpoint_path": str(checkpoint_artifacts.best_checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(checkpoint_artifacts.last_checkpoint_path),
            "config_path": str(runtime.config.source_path.resolve()),
            "dataset_paths": {
                split: str(path) for split, path in runtime.paths.dataset_paths.items()
            },
            "run_mode": run_mode,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "test_metrics_path": str(test_metrics_path),
            "train_history_path": (
                str(train_history_path) if train_history_path is not None else None
            ),
            "train_loss": train_loss,
            "val_history_path": str(val_history_path) if val_history_path is not None else None,
            "validation_loss": val_loss,
        },
    )
    return {
        "best_checkpoint": checkpoint_artifacts.best_checkpoint_path,
        "checkpoint": checkpoint_artifacts.last_checkpoint_path,
        "summary": summary_path,
        "test_metrics": test_metrics_path,
    }


def run_pipeline(config_path: str | Path, run_mode: str) -> dict[str, Path]:
    config = load_runtime_config(config_path)
    paths = resolve_runtime_paths(config)

    _seed_runtime(config.training_args.seed)
    prepare_runtime_environment(paths)

    _log("export", run_mode=run_mode, output_dir=paths.synthetic_output_dir)
    export_runtime_data(config, paths, run_mode)

    runtime = build_runtime_objects(config, paths)

    _log("train", epochs=runtime.config.training_args.num_train_epochs)
    training_result = runtime.trainer.train_and_validate(
        num_epochs=runtime.config.training_args.num_train_epochs,
        learning_rate=runtime.config.training_args.learning_rate,
        weight_decay=runtime.config.training_args.weight_decay,
        results_dir=runtime.paths.results_dir,
    )
    train_loss = float(training_result.train_history[-1]["train_loss"])

    _log("validation")
    val_loss = float(training_result.val_history[-1]["val_loss"])

    checkpoint_artifacts = save_checkpoints(
        runtime,
        run_mode=run_mode,
        best_epoch=training_result.best_epoch,
        best_val_loss=training_result.best_val_loss,
        best_model_state=training_result.best_model_state,
    )
    decision_threshold = _calibrate_validation_threshold(
        runtime,
        checkpoint_path=checkpoint_artifacts.best_checkpoint_path,
    )
    _log(
        "test",
        checkpoint=checkpoint_artifacts.best_checkpoint_path,
        decision_threshold=decision_threshold,
    )
    test_result = evaluate_test_split(
        runtime,
        checkpoint_path=checkpoint_artifacts.best_checkpoint_path,
        decision_threshold=decision_threshold,
    )
    test_metrics_path = write_test_metrics(
        runtime.paths.results_dir / "test_metrics.json",
        test_result.metrics,
    )
    if run_mode == "smoke":
        _enforce_smoke_acceptance(
            training_result.train_history,
            training_result.val_history,
            test_metrics=test_result.metrics,
        )

    _log("artifact_save", checkpoints_dir=paths.checkpoints_dir, results_dir=paths.results_dir)
    artifacts = save_run_artifacts(
        runtime,
        run_mode,
        train_loss,
        val_loss,
        test_result.loss,
        test_result.metrics,
        checkpoint_artifacts,
        test_metrics_path,
        train_history_path=training_result.train_history_path,
        val_history_path=training_result.val_history_path,
        best_epoch=training_result.best_epoch,
        best_val_loss=training_result.best_val_loss,
    )
    _log("verify", summary=artifacts["summary"])
    verify_completion_artifacts(artifacts["summary"], run_mode=run_mode)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        artifacts = run_pipeline(args.config, args.run_mode)
    except (OSError, RuntimeConfigError, ValueError) as exc:
        print(f"error={exc}", file=sys.stderr, flush=True)
        return 1

    _log("finished", status="ok", summary=artifacts["summary"])
    print(SMOKE_SUCCESS_PROMISE, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
