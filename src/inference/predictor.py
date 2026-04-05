"""Checkpoint-backed inference helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.training.config import REQUIRED_TEST_METRIC_KEYS


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
