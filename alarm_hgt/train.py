"""Config-driven CLI entrypoint for smoke and full pipeline runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import EvalPrediction

from .dataset import AlarmGraphDataset
from .export import export_synthetic_splits
from .modeling import HGTForLinkPrediction
from .runtime_config import RuntimeConfig, RuntimeConfigError, load_runtime_config
from .trainer import LinkPredictionTrainer

RUN_MODE_CHOICES = ("full", "smoke")


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
    )


def build_runtime_objects(config: RuntimeConfig, paths: ResolvedRuntimePaths) -> RuntimeObjects:
    datasets = {split: AlarmGraphDataset(path) for split, path in paths.dataset_paths.items()}
    model = HGTForLinkPrediction(config.to_model_config())
    trainer = LinkPredictionTrainer(
        model=model,
        args=config.to_trainer_args(),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
    )
    return RuntimeObjects(
        config=config,
        paths=paths,
        datasets=datasets,
        model=model,
        trainer=trainer,
    )


def _move_batch_to_cpu(batch: dict[str, Any]) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.cpu()
        else:
            moved[key] = value
    return moved


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _pad_batch_arrays(arrays: list[np.ndarray], *, fill_value: float | bool) -> np.ndarray:
    max_width = max(array.shape[1] for array in arrays)
    padded: list[np.ndarray] = []
    for array in arrays:
        pad_width = max_width - array.shape[1]
        if pad_width:
            array = np.pad(array, ((0, 0), (0, pad_width)), constant_values=fill_value)
        padded.append(array)
    return np.concatenate(padded, axis=0)


def run_train_stage(runtime: RuntimeObjects) -> float:
    optimizer = torch.optim.AdamW(
        runtime.model.parameters(),
        lr=runtime.config.training_args.learning_rate,
        weight_decay=runtime.config.training_args.weight_decay,
    )
    runtime.model.train()
    losses: list[float] = []
    dataloader = runtime.trainer.get_train_dataloader()
    for _ in range(runtime.config.training_args.num_train_epochs):
        for batch in dataloader:
            batch = _move_batch_to_cpu(batch)
            optimizer.zero_grad(set_to_none=True)
            output = runtime.model(**batch)
            if output.loss is None:
                raise ValueError("Training batch did not produce a loss")
            output.loss.backward()
            optimizer.step()
            losses.append(float(output.loss.detach().cpu().item()))
    return _mean(losses)


def run_eval_stage(runtime: RuntimeObjects, split_name: str) -> tuple[float, dict[str, float]]:
    runtime.model.eval()
    if split_name == "val":
        dataloader = runtime.trainer.get_eval_dataloader()
    elif split_name == "test":
        dataloader = runtime.trainer.get_test_dataloader(runtime.datasets["test"])
    else:
        raise ValueError(f"Unsupported eval split: {split_name}")

    losses: list[float] = []
    logits_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    mask_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_cpu(batch)
            output = runtime.model(**batch)
            if output.loss is not None:
                losses.append(float(output.loss.detach().cpu().item()))
            logits_batches.append(output.logits.detach().cpu().numpy())
            labels_batches.append(batch["labels"].detach().cpu().numpy())
            mask_batches.append(batch["trainable_mask"].detach().cpu().numpy())

    if not logits_batches:
        return _mean(losses), {}

    logits = _pad_batch_arrays(logits_batches, fill_value=0.0)
    labels = _pad_batch_arrays(labels_batches, fill_value=0.0)
    trainable_mask = _pad_batch_arrays(mask_batches, fill_value=False).astype(bool)
    metrics = runtime.trainer.compute_metrics(
        EvalPrediction(predictions=logits, label_ids=(labels, trainable_mask))
    )
    return _mean(losses), metrics


def save_run_artifacts(
    runtime: RuntimeObjects,
    run_mode: str,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float],
) -> dict[str, Path]:
    checkpoint_path = runtime.paths.checkpoints_dir / f"{run_mode}-last.pt"
    summary_path = runtime.paths.results_dir / f"{run_mode}-summary.json"

    torch.save(
        {
            "run_mode": run_mode,
            "model_state_dict": runtime.model.state_dict(),
        },
        checkpoint_path,
    )
    summary_path.write_text(
        json.dumps(
            {
                "config_path": str(runtime.config.source_path.resolve()),
                "run_mode": run_mode,
                "dataset_paths": {
                    split: str(path) for split, path in runtime.paths.dataset_paths.items()
                },
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "checkpoint_path": str(checkpoint_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint_path,
        "summary": summary_path,
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
    train_loss = run_train_stage(runtime)

    _log("validation")
    val_loss, _ = run_eval_stage(runtime, "val")

    _log("test")
    test_loss, test_metrics = run_eval_stage(runtime, "test")

    _log("artifact_save", checkpoints_dir=paths.checkpoints_dir, results_dir=paths.results_dir)
    return save_run_artifacts(runtime, run_mode, train_loss, val_loss, test_loss, test_metrics)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        artifacts = run_pipeline(args.config, args.run_mode)
    except (OSError, RuntimeConfigError, ValueError) as exc:
        print(f"error={exc}", file=sys.stderr, flush=True)
        return 1

    _log("finished", status="ok", summary=artifacts["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
