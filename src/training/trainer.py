"""Transformers Trainer integration and runtime orchestration."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule
from transformers.trainer_utils import get_last_checkpoint

from src.dataset.bucket_sampler import BucketBatchSampler
from src.dataset.builder import build_datasets
from src.dataset.collate import padding_collate_fn
from src.dataset.hgt_dataset import HGTDataset
from src.constants import HGT_NODE_TYPE_IDS, RELATION_TYPE_IDS
from src.graph.feature_extraction import FEATURE_DIM
from src.inference.predictor import (
    TestPredictor,
    write_test_metrics,
)
from src.models.hgt_for_link_prediction import HGTForLinkPrediction
from training_data.topo_complete import export_complete_splits

from .config import (
    DATA_SPLITS,
    TEST_METRICS_FILENAME,
    TRAIN_HISTORY_FILENAME,
    VAL_HISTORY_FILENAME,
    HGTConfig,
    RuntimeConfig,
    checkpoint_filename,
    load_runtime_config,
    summary_filename,
)
from .metrics import (
    build_compute_metrics,
    compute_link_prediction_metrics,
    eval_prediction_to_metrics_input,
)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])


def build_training_arguments(
    *,
    output_dir: str | Path,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    logging_steps: int,
    seed: int,
    dataloader_drop_last: bool = False,
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: bool = False,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=float(num_train_epochs),
        logging_steps=logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels", "trainable_mask"],
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
        max_grad_norm=0.0,
        dataloader_drop_last=dataloader_drop_last,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        seed=seed,
    )


def write_loss_history(
    path: str | Path,
    *,
    split: str,
    metric_key: str,
    history: list[dict[str, float | int]],
) -> Path:
    payload = {
        "split": split,
        "metric": metric_key,
        "history": history,
    }
    history_path = Path(path)
    history_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return history_path


class LinkPredictionTrainer(Trainer):
    """Real Transformers Trainer with bucketed dataloaders for HGT graphs."""

    def __init__(
        self,
        *args,
        metric_ks: tuple[int, ...] = (5, 10, 20, 50),
        **kwargs,
    ) -> None:
        kwargs.setdefault("data_collator", padding_collate_fn)
        kwargs.setdefault("compute_metrics", build_compute_metrics(metric_ks))
        preserve_rng_state = bool(kwargs.get("model") is not None or args)
        rng_state = _capture_rng_state() if preserve_rng_state else None
        super().__init__(*args, **kwargs)
        if rng_state is not None:
            _restore_rng_state(rng_state)
        self.metric_ks = metric_ks
        self.label_names = ["labels", "trainable_mask"]
        self._graph_size_cache: dict[int, list[int]] = {}

    def _graph_sizes(self, dataset) -> list[int]:
        dataset_id = id(dataset)
        if dataset_id not in self._graph_size_cache:
            self._graph_size_cache[dataset_id] = [
                dataset[index]["node_features"].shape[0] for index in range(len(dataset))
            ]
        return self._graph_size_cache[dataset_id]

    def _build_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        sampler = BucketBatchSampler(
            sizes=self._graph_sizes(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=self.args.dataloader_drop_last,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("LinkPredictionTrainer requires a train_dataset")
        return self._build_dataloader(self.train_dataset, self.args.train_batch_size, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("LinkPredictionTrainer requires an eval_dataset")
        return self._build_dataloader(dataset, self.args.eval_batch_size, shuffle=False)

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        return self._build_dataloader(test_dataset, self.args.eval_batch_size, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(**inputs)
        if outputs.loss is None:
            raise ValueError("model did not return loss")
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {float(loss.detach().cpu())}")
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        # Preserve the legacy training semantics while still using Trainer orchestration.
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        del num_training_steps
        if self.lr_scheduler is None:
            self.lr_scheduler = get_constant_schedule(optimizer or self.optimizer)
        return self.lr_scheduler

    def evaluate_with_threshold(
        self,
        eval_dataset=None,
        *,
        metric_key_prefix: str = "eval",
        decision_threshold: float = 0.5,
    ) -> dict[str, float]:
        original_compute_metrics = self.compute_metrics
        if original_compute_metrics is None:
            raise ValueError("compute_metrics is required for thresholded evaluation")
        thresholded_compute_metrics = build_compute_metrics(self.metric_ks)

        def thresholded(eval_prediction: EvalPrediction) -> dict[str, float]:
            return thresholded_compute_metrics(
                eval_prediction,
                decision_threshold=decision_threshold,
            )

        self.compute_metrics = thresholded
        try:
            return super().evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
        finally:
            self.compute_metrics = original_compute_metrics

    def history_rows(self, metric_key: str) -> list[dict[str, float | int]]:
        prefix = f"{metric_key}_"
        deduped_history: dict[int, dict[str, float | int]] = {}
        for entry in self.state.log_history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            value = entry.get(metric_key)
            if value is None:
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"non-finite {metric_key} history value: {numeric_value}")
            epoch_number = int(round(float(epoch)))
            deduped_history[epoch_number] = {"epoch": epoch_number, prefix[:-1]: numeric_value}
        return [deduped_history[epoch] for epoch in sorted(deduped_history)]

    def write_histories(
        self, results_dir: str | Path
    ) -> tuple[Path, list[dict], Path, list[dict]]:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        train_history = [
            {"epoch": row["epoch"], "train_loss": float(row["loss"])}
            for row in self.history_rows("loss")
        ]
        val_history = [
            {"epoch": row["epoch"], "val_loss": float(row["eval_loss"])}
            for row in self.history_rows("eval_loss")
        ]
        train_path = write_loss_history(
            results_dir / TRAIN_HISTORY_FILENAME,
            split="train",
            metric_key="train_loss",
            history=train_history,
        )
        val_path = write_loss_history(
            results_dir / VAL_HISTORY_FILENAME,
            split="val",
            metric_key="val_loss",
            history=val_history,
        )
        return train_path, train_history, val_path, val_history


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    synthetic_output_dir: Path
    dataset_paths: dict[str, Path]
    checkpoints_dir: Path
    results_dir: Path
    hf_output_dir: Path


@dataclass
class RuntimeObjects:
    config: RuntimeConfig
    paths: ResolvedRuntimePaths
    datasets: dict[str, HGTDataset]
    model: HGTForLinkPrediction
    trainer: LinkPredictionTrainer


@dataclass(frozen=True)
class CheckpointArtifacts:
    last_checkpoint_path: Path
    best_checkpoint_path: Path


def _artifact_paths(
    checkpoint_artifacts: CheckpointArtifacts,
    summary_path: Path,
    test_metrics_path: Path,
) -> dict[str, Path]:
    return {
        "best_checkpoint": checkpoint_artifacts.best_checkpoint_path,
        "checkpoint": checkpoint_artifacts.last_checkpoint_path,
        "summary": summary_path,
        "test_metrics": test_metrics_path,
    }


def _resolve_path(path: Path) -> Path:
    return path.resolve()


def resolve_runtime_paths(config: RuntimeConfig) -> ResolvedRuntimePaths:
    checkpoints_dir = _resolve_path(config.outputs.checkpoints_dir)
    return ResolvedRuntimePaths(
        synthetic_output_dir=_resolve_path(config.synthetic.output_dir),
        dataset_paths={
            split_name: _resolve_path(getattr(config.dataset_paths, split_name))
            for split_name in DATA_SPLITS
        },
        checkpoints_dir=checkpoints_dir,
        results_dir=_resolve_path(config.outputs.results_dir),
        hf_output_dir=checkpoints_dir / "hf",
    )


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_runtime_environment(paths: ResolvedRuntimePaths) -> None:
    paths.synthetic_output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_path in paths.dataset_paths.values():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.hf_output_dir.mkdir(parents=True, exist_ok=True)


def log_stage(stage: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    message = f"stage={stage}"
    if details:
        message = f"{message} {details}"
    print(message, flush=True)


def export_runtime_data(config: RuntimeConfig, paths: ResolvedRuntimePaths) -> dict[str, Path]:
    return export_complete_splits(
        output_dir=paths.synthetic_output_dir,
        split_sizes=config.synthetic.split_sizes.as_dict(),
        config=config.synthetic.to_generation_config(),
        seed=config.synthetic.seed,
        output_paths=paths.dataset_paths,
    )


def _validate_runtime_design_compliance(
    config: RuntimeConfig,
    datasets: dict[str, HGTDataset],
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
    datasets = build_datasets(paths.dataset_paths)
    _validate_runtime_design_compliance(config, datasets)
    model = HGTForLinkPrediction(config.to_model_config())
    trainer = LinkPredictionTrainer(
        model=model,
        args=config.to_training_arguments(paths.hf_output_dir),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        metric_ks=config.metrics.ks,
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
    model_state_dict: dict[str, Any],
    epoch: int | None = None,
    val_loss: float | None = None,
) -> dict[str, Any]:
    return {
        "run_mode": "full",
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model_state_dict,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _best_validation_entry(val_history: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    if not val_history:
        return None
    return min(val_history, key=lambda entry: float(entry["val_loss"]))


def _summary_payload(
    runtime: RuntimeObjects,
    *,
    checkpoint_artifacts: CheckpointArtifacts,
    test_loss: float,
    test_metrics: dict[str, float],
    test_metrics_path: Path,
    train_history_path: Path,
    val_history_path: Path,
    train_loss: float | None,
    val_loss: float | None,
    best_epoch: int | None,
    best_val_loss: float | None,
) -> dict[str, Any]:
    return {
        "best_checkpoint_path": str(checkpoint_artifacts.best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_artifacts.last_checkpoint_path),
        "config_path": str(runtime.config.source_path.resolve()),
        "dataset_paths": {split_name: str(path) for split_name, path in runtime.paths.dataset_paths.items()},
        "run_mode": "full",
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_metrics_path": str(test_metrics_path),
        "train_history_path": str(train_history_path),
        "train_loss": train_loss,
        "val_history_path": str(val_history_path),
        "validation_loss": val_loss,
    }


def _load_hf_model_state(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not weights_path.exists():
        # transformers >= 5 saves as safetensors by default
        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            return load_file(str(safetensors_path), device="cpu")
        raise FileNotFoundError(f"trainer checkpoint weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"invalid trainer checkpoint payload: {weights_path}")
    return state_dict


def save_checkpoints(
    runtime: RuntimeObjects,
    *,
    train_result,
    val_history: list[dict[str, float | int]],
) -> CheckpointArtifacts:
    del train_result
    last_checkpoint_dir = get_last_checkpoint(str(runtime.paths.hf_output_dir))
    if last_checkpoint_dir is None:
        raise ValueError("transformers trainer did not produce a last checkpoint")
    best_checkpoint_dir = runtime.trainer.state.best_model_checkpoint
    if best_checkpoint_dir is None:
        raise ValueError("transformers trainer did not produce a best checkpoint")

    best_entry = _best_validation_entry(val_history)
    best_epoch = None if best_entry is None else int(best_entry["epoch"])
    best_val_loss = None if best_entry is None else float(best_entry["val_loss"])

    last_checkpoint_path = runtime.paths.checkpoints_dir / checkpoint_filename(kind="last")
    best_checkpoint_path = runtime.paths.checkpoints_dir / checkpoint_filename(kind="best")
    torch.save(
        _checkpoint_payload(
            model_state_dict=_load_hf_model_state(last_checkpoint_dir),
        ),
        last_checkpoint_path,
    )
    torch.save(
        _checkpoint_payload(
            model_state_dict=_load_hf_model_state(best_checkpoint_dir),
            epoch=best_epoch,
            val_loss=best_val_loss,
        ),
        best_checkpoint_path,
    )
    return CheckpointArtifacts(
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
    )


def _calibrate_validation_threshold(
    runtime: RuntimeObjects,
    *,
    checkpoint_path: Path,
) -> float:
    predictor = TestPredictor(runtime)
    validation_result = predictor.evaluate("val", checkpoint_path=checkpoint_path)
    threshold = float(validation_result["metrics"].get("edge_best_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"validation threshold must be within [0.0, 1.0], got {threshold}")
    return threshold


def save_run_artifacts(
    runtime: RuntimeObjects,
    *,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float],
    checkpoint_artifacts: CheckpointArtifacts,
    test_metrics_path: Path,
    train_history_path: Path,
    val_history_path: Path,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
) -> dict[str, Path]:
    summary_path = runtime.paths.results_dir / summary_filename()
    _write_json(
        summary_path,
        _summary_payload(
            runtime,
            checkpoint_artifacts=checkpoint_artifacts,
            test_loss=test_loss,
            test_metrics=test_metrics,
            test_metrics_path=test_metrics_path,
            train_history_path=train_history_path,
            val_history_path=val_history_path,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        ),
    )
    return _artifact_paths(checkpoint_artifacts, summary_path, test_metrics_path)


def run_training_pipeline(runtime: RuntimeObjects) -> dict[str, Path]:
    if runtime.paths.hf_output_dir.exists():
        shutil.rmtree(runtime.paths.hf_output_dir)
    runtime.paths.hf_output_dir.mkdir(parents=True, exist_ok=True)
    train_result = runtime.trainer.train()
    train_history_path, train_history, val_history_path, val_history = runtime.trainer.write_histories(runtime.paths.results_dir)

    train_loss = float(train_history[-1]["train_loss"])
    val_loss = float(val_history[-1]["val_loss"])

    checkpoint_artifacts = save_checkpoints(
        runtime,
        train_result=train_result,
        val_history=val_history,
    )
    decision_threshold = _calibrate_validation_threshold(
        runtime,
        checkpoint_path=checkpoint_artifacts.best_checkpoint_path,
    )
    predictor = TestPredictor(runtime)
    test_result = predictor.evaluate(
        "test",
        checkpoint_path=checkpoint_artifacts.best_checkpoint_path,
        decision_threshold=decision_threshold,
    )
    test_metrics_path = write_test_metrics(
        runtime.paths.results_dir / TEST_METRICS_FILENAME,
        test_result["metrics"],
    )

    best_entry = _best_validation_entry(val_history)
    return save_run_artifacts(
        runtime,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=float(test_result["loss"]),
        test_metrics=test_result["metrics"],
        checkpoint_artifacts=checkpoint_artifacts,
        test_metrics_path=test_metrics_path,
        train_history_path=train_history_path,
        val_history_path=val_history_path,
        best_epoch=None if best_entry is None else int(best_entry["epoch"]),
        best_val_loss=None if best_entry is None else float(best_entry["val_loss"]),
    )


def run_inference_pipeline(
    runtime: RuntimeObjects,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Path]:
    predictor = TestPredictor(runtime)
    decision_threshold = _calibrate_validation_threshold(runtime, checkpoint_path=Path(checkpoint_path))
    test_result = predictor.evaluate(
        "test",
        checkpoint_path=checkpoint_path,
        decision_threshold=decision_threshold,
    )
    test_metrics_path = write_test_metrics(
        runtime.paths.results_dir / TEST_METRICS_FILENAME,
        test_result["metrics"],
    )
    train_history_path = runtime.paths.results_dir / TRAIN_HISTORY_FILENAME
    val_history_path = runtime.paths.results_dir / VAL_HISTORY_FILENAME
    if not train_history_path.exists() or not val_history_path.exists():
        raise FileNotFoundError("inference mode requires existing train/val history artifacts")
    checkpoint_artifacts = CheckpointArtifacts(
        last_checkpoint_path=Path(checkpoint_path),
        best_checkpoint_path=Path(checkpoint_path),
    )
    summary_path = runtime.paths.results_dir / summary_filename()
    _write_json(
        summary_path,
        _summary_payload(
            runtime,
            checkpoint_artifacts=checkpoint_artifacts,
            test_loss=float(test_result["loss"]),
            test_metrics=test_result["metrics"],
            test_metrics_path=test_metrics_path,
            train_history_path=train_history_path,
            val_history_path=val_history_path,
            train_loss=None,
            val_loss=None,
            best_epoch=test_result["epoch"],
            best_val_loss=None,
        ),
    )
    return _artifact_paths(checkpoint_artifacts, summary_path, test_metrics_path)


def run_pipeline(
    config_path: str | Path,
    *,
    mode: str,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Path]:
    config = load_runtime_config(config_path)
    paths = resolve_runtime_paths(config)

    _seed_runtime(config.training_args.seed)
    prepare_runtime_environment(paths)

    log_stage("export", output_dir=paths.synthetic_output_dir)
    export_runtime_data(config, paths)

    runtime = build_runtime_objects(config, paths)

    if mode == "train":
        log_stage("train", epochs=runtime.config.training_args.num_train_epochs)
        artifacts = run_training_pipeline(runtime)
    else:
        if checkpoint_path is None:
            checkpoint_path = paths.checkpoints_dir / checkpoint_filename(kind="best")
        log_stage("inference", checkpoint=checkpoint_path)
        artifacts = run_inference_pipeline(runtime, checkpoint_path=checkpoint_path)

    log_stage("finished", status="ok", summary=artifacts["summary"])
    return artifacts
