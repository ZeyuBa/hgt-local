"""Checkpoint-backed inference helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from src.training.config import REQUIRED_TEST_METRIC_KEYS, SMOKE_MIN_F1


class TestPredictor:
    """Load a saved checkpoint and evaluate validation/test splits."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise ValueError(f"checkpoint missing model_state_dict: {checkpoint_path}")
        self.runtime.model.load_state_dict(model_state_dict)
        return checkpoint

    def evaluate(self, split_name: str, *, checkpoint_path: str | Path, decision_threshold: float = 0.5):
        checkpoint = self.load_checkpoint(checkpoint_path)
        epoch = checkpoint.get("epoch")
        metric_key_prefix = "eval" if split_name == "val" else "test"
        metrics = self.runtime.trainer.evaluate_with_threshold(
            self.runtime.datasets[split_name],
            metric_key_prefix=metric_key_prefix,
            decision_threshold=decision_threshold,
        )
        return {
            "epoch": int(epoch) if epoch is not None else self.runtime.config.training_args.num_train_epochs,
            "loss": float(metrics[f"{metric_key_prefix}_loss"]),
            "metrics": {
                key[len(f"{metric_key_prefix}_") :]: float(value)
                for key, value in metrics.items()
                if key.startswith(f"{metric_key_prefix}_") and key != f"{metric_key_prefix}_loss"
            },
        }


def count_improving_transitions(
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


def enforce_smoke_acceptance(
    train_history: list[dict[str, float | int]],
    val_history: list[dict[str, float | int]],
    *,
    test_metrics: dict[str, float],
) -> None:
    transitions = len(train_history) - 1
    if transitions < 1:
        raise ValueError("smoke acceptance failed: requires at least two epochs")

    improving_transitions = count_improving_transitions(train_history, val_history)
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
        raise ValueError(f"missing required test metrics: {', '.join(missing_or_invalid)}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"{label} not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {json_path}")
    return payload


def _summary_path(summary_payload: dict[str, Any], *, key: str) -> Path:
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
        raise ValueError(f"missing required test metrics: {', '.join(missing_or_invalid)}")
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
        "train_history": _summary_path(summary, key="train_history_path"),
        "val_history": _summary_path(summary, key="val_history_path"),
        "test_metrics": _summary_path(summary, key="test_metrics_path"),
    }
    checkpoint_paths = {
        "checkpoint": _summary_path(summary, key="checkpoint_path"),
        "best_checkpoint": _summary_path(summary, key="best_checkpoint_path"),
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

    train_history = _load_loss_history(result_paths["train_history"], split="train", metric_key="train_loss")
    val_history = _load_loss_history(result_paths["val_history"], split="val", metric_key="val_loss")
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
        enforce_smoke_acceptance(train_history, val_history, test_metrics=test_metrics)

    return {
        **result_paths,
        **checkpoint_paths,
    }
