"""Trainer helpers for alarm HGT link prediction."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EvalPrediction

from .batching import BucketBatchSampler, padding_collate_fn
from .metrics import compute_link_prediction_metrics


def _extract_logits(predictions: Any) -> np.ndarray:
    if isinstance(predictions, dict):
        predictions = predictions["logits"]
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return np.asarray(predictions, dtype=np.float32)


def eval_prediction_to_metrics_input(eval_prediction: EvalPrediction) -> dict[str, np.ndarray]:
    """Convert HuggingFace EvalPrediction payloads into metric arrays."""

    logits = _extract_logits(eval_prediction.predictions)
    label_ids = eval_prediction.label_ids

    if isinstance(label_ids, dict):
        labels = label_ids["labels"]
        trainable_mask = label_ids["trainable_mask"]
    elif isinstance(label_ids, (tuple, list)) and len(label_ids) >= 2:
        labels, trainable_mask = label_ids[:2]
    else:
        raise TypeError("label_ids must provide both labels and trainable_mask")

    return {
        "logits": logits,
        "labels": np.asarray(labels, dtype=np.float32),
        "trainable_mask": np.asarray(trainable_mask, dtype=bool),
    }


def build_compute_metrics(ks: tuple[int, ...] = (5, 10, 20, 50)):
    """Build a Trainer-compatible compute_metrics callback."""

    def compute_metrics(
        eval_prediction: EvalPrediction,
        *,
        decision_threshold: float | None = None,
    ) -> dict[str, float]:
        metric_inputs = eval_prediction_to_metrics_input(eval_prediction)
        return compute_link_prediction_metrics(
            **metric_inputs,
            ks=ks,
            decision_threshold=0.5 if decision_threshold is None else decision_threshold,
        )

    return compute_metrics


@dataclass
class LinkPredictionTrainerArgs:
    """Minimal Trainer-like arguments for dataloader construction."""

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False

    @property
    def train_batch_size(self) -> int:
        return self.per_device_train_batch_size

    @property
    def eval_batch_size(self) -> int:
        return self.per_device_eval_batch_size


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingLoopResult:
    train_history: list[dict[str, float | int]]
    val_history: list[dict[str, float | int]]
    val_metrics: dict[str, float]
    train_history_path: Path | None = None
    val_history_path: Path | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    best_model_state: dict[str, Any] | None = None


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


class LinkPredictionTrainer:
    """Lightweight trainer utility with bucketed batching and metric hooks."""

    def __init__(
        self,
        model,
        args: LinkPredictionTrainerArgs,
        train_dataset=None,
        eval_dataset=None,
        data_collator=padding_collate_fn,
        compute_metrics=None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics or build_compute_metrics()
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

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_batch_to_device(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _require_trainable_positions(
        self,
        batch: dict[str, Any],
        *,
        split_name: str,
        epoch: int,
        batch_index: int,
    ) -> None:
        mask = batch.get("trainable_mask")
        if mask is None or not bool(mask.any().item()):
            raise ValueError(
                f"{split_name} epoch {epoch} batch {batch_index} has no trainable positions"
            )

    def _validate_loss(
        self,
        loss_value: float,
        *,
        split_name: str,
        epoch: int,
        batch_index: int,
    ) -> None:
        if not math.isfinite(loss_value):
            raise ValueError(
                f"{split_name} epoch {epoch} batch {batch_index} produced non-finite loss: "
                f"{loss_value}"
            )

    def _finalize_epoch_loss(self, losses: list[float], *, split_name: str, epoch: int) -> float:
        if not losses:
            raise ValueError(f"{split_name} epoch {epoch} did not process any batches")
        epoch_loss = float(sum(losses) / len(losses))
        if not math.isfinite(epoch_loss):
            raise ValueError(f"{split_name} epoch {epoch} produced non-finite loss: {epoch_loss}")
        return epoch_loss

    def _pad_batch_arrays(self, arrays: list[np.ndarray], *, fill_value: float | bool) -> np.ndarray:
        max_width = max(array.shape[1] for array in arrays)
        padded: list[np.ndarray] = []
        for array in arrays:
            pad_width = max_width - array.shape[1]
            if pad_width:
                array = np.pad(array, ((0, 0), (0, pad_width)), constant_values=fill_value)
            padded.append(array)
        return np.concatenate(padded, axis=0)

    def _snapshot_model_state(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key, value in self.model.state_dict().items():
            if isinstance(value, torch.Tensor):
                snapshot[key] = value.detach().cpu().clone()
            else:
                snapshot[key] = copy.deepcopy(value)
        return snapshot

    def _run_epoch(
        self,
        dataloader: DataLoader,
        *,
        split_name: str,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        collect_metrics: bool = False,
        decision_threshold: float = 0.5,
    ) -> EvaluationResult:
        training = optimizer is not None
        device = self._model_device()
        losses: list[float] = []
        logits_batches: list[np.ndarray] = []
        labels_batches: list[np.ndarray] = []
        mask_batches: list[np.ndarray] = []

        self.model.train(training)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(dataloader, start=1):
                batch = self._move_batch_to_device(batch, device)
                self._require_trainable_positions(
                    batch,
                    split_name=split_name,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                output = self.model(**batch)
                if output.loss is None:
                    raise ValueError(
                        f"{split_name} epoch {epoch} batch {batch_index} did not produce a loss"
                    )
                loss_value = float(output.loss.detach().cpu().item())
                self._validate_loss(
                    loss_value,
                    split_name=split_name,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                if training:
                    output.loss.backward()
                    optimizer.step()
                losses.append(loss_value)

                if collect_metrics:
                    logits_batches.append(output.logits.detach().cpu().numpy())
                    labels_batches.append(batch["labels"].detach().cpu().numpy())
                    mask_batches.append(batch["trainable_mask"].detach().cpu().numpy())

        epoch_loss = self._finalize_epoch_loss(losses, split_name=split_name, epoch=epoch)
        metrics: dict[str, float] = {}
        if collect_metrics and logits_batches:
            logits = self._pad_batch_arrays(logits_batches, fill_value=0.0)
            labels = self._pad_batch_arrays(labels_batches, fill_value=0.0)
            trainable_mask = self._pad_batch_arrays(mask_batches, fill_value=False).astype(bool)
            metrics = self.compute_metrics(
                EvalPrediction(predictions=logits, label_ids=(labels, trainable_mask)),
                decision_threshold=decision_threshold,
            )
        return EvaluationResult(loss=epoch_loss, metrics=metrics)

    def evaluate(
        self,
        eval_dataset=None,
        *,
        split_name: str = "val",
        epoch: int = 1,
        decision_threshold: float = 0.5,
    ) -> EvaluationResult:
        if split_name == "val":
            dataloader = self.get_eval_dataloader(eval_dataset)
        elif split_name == "test":
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if dataset is None:
                raise ValueError("LinkPredictionTrainer requires a test dataset")
            dataloader = self.get_test_dataloader(dataset)
        else:
            raise ValueError(f"Unsupported eval split: {split_name}")
        return self._run_epoch(
            dataloader,
            split_name=split_name,
            epoch=epoch,
            optimizer=None,
            collect_metrics=True,
            decision_threshold=decision_threshold,
        )

    def train_and_validate(
        self,
        *,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        results_dir: str | Path | None = None,
    ) -> TrainingLoopResult:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        train_history: list[dict[str, float | int]] = []
        val_history: list[dict[str, float | int]] = []
        latest_val_metrics: dict[str, float] = {}
        best_epoch: int | None = None
        best_val_loss: float | None = None
        best_model_state: dict[str, Any] | None = None

        train_dataloader = self.get_train_dataloader()
        for epoch in range(1, num_epochs + 1):
            train_result = self._run_epoch(
                train_dataloader,
                split_name="train",
                epoch=epoch,
                optimizer=optimizer,
                collect_metrics=False,
            )
            val_result = self.evaluate(split_name="val", epoch=epoch)
            train_history.append({"epoch": epoch, "train_loss": train_result.loss})
            val_history.append({"epoch": epoch, "val_loss": val_result.loss})
            latest_val_metrics = val_result.metrics
            if best_val_loss is None or val_result.loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_result.loss
                best_model_state = self._snapshot_model_state()

        train_history_path: Path | None = None
        val_history_path: Path | None = None
        if results_dir is not None:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            train_history_path = write_loss_history(
                results_path / "train_history.json",
                split="train",
                metric_key="train_loss",
                history=train_history,
            )
            val_history_path = write_loss_history(
                results_path / "val_history.json",
                split="val",
                metric_key="val_loss",
                history=val_history,
            )

        return TrainingLoopResult(
            train_history=train_history,
            val_history=val_history,
            val_metrics=latest_val_metrics,
            train_history_path=train_history_path,
            val_history_path=val_history_path,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_model_state=best_model_state,
        )
