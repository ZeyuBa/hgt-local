"""Trainer helpers for alarm HGT link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
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

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        metric_inputs = eval_prediction_to_metrics_input(eval_prediction)
        return compute_link_prediction_metrics(**metric_inputs, ks=ks)

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

    def _graph_sizes(self, dataset) -> list[int]:
        return [dataset[index]["node_features"].shape[0] for index in range(len(dataset))]

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
