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

from .dataset import AlarmGraphDataset
from .export import export_synthetic_splits
from .modeling import HGTForLinkPrediction
from .runtime_config import RuntimeConfig, RuntimeConfigError, load_runtime_config
from .trainer import LinkPredictionTrainer, build_compute_metrics

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
        compute_metrics=build_compute_metrics(config.metrics.ks),
    )
    return RuntimeObjects(
        config=config,
        paths=paths,
        datasets=datasets,
        model=model,
        trainer=trainer,
    )


def save_run_artifacts(
    runtime: RuntimeObjects,
    run_mode: str,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float],
    train_history_path: Path | None = None,
    val_history_path: Path | None = None,
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
                "train_history_path": (
                    str(train_history_path) if train_history_path is not None else None
                ),
                "val_history_path": str(val_history_path) if val_history_path is not None else None,
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
    training_result = runtime.trainer.train_and_validate(
        num_epochs=runtime.config.training_args.num_train_epochs,
        learning_rate=runtime.config.training_args.learning_rate,
        weight_decay=runtime.config.training_args.weight_decay,
        results_dir=runtime.paths.results_dir,
    )
    train_loss = float(training_result.train_history[-1]["train_loss"])

    _log("validation")
    val_loss = float(training_result.val_history[-1]["val_loss"])

    _log("test")
    test_result = runtime.trainer.evaluate(
        runtime.datasets["test"],
        split_name="test",
        epoch=runtime.config.training_args.num_train_epochs,
    )

    _log("artifact_save", checkpoints_dir=paths.checkpoints_dir, results_dir=paths.results_dir)
    return save_run_artifacts(
        runtime,
        run_mode,
        train_loss,
        val_loss,
        test_result.loss,
        test_result.metrics,
        train_history_path=training_result.train_history_path,
        val_history_path=training_result.val_history_path,
    )


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
