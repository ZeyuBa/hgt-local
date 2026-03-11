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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import EvalPrediction, Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule
from transformers.trainer_utils import get_last_checkpoint

from src.dataset.bucket_sampler import BucketBatchSampler
from src.dataset.builder import build_datasets
from src.dataset.collate import padding_collate_fn
from src.dataset.hgt_dataset import HGTDataset
from src.graph.feature_extraction import FEATURE_DIM, HGT_NODE_TYPE_IDS, RELATION_TYPE_IDS
from src.inference.predictor import (
    TestPredictor,
    enforce_smoke_acceptance,
    verify_completion_artifacts,
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


def _extract_logits(predictions: Any) -> np.ndarray:
    if isinstance(predictions, dict):
        predictions = predictions["logits"]
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return np.asarray(predictions, dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _to_numpy(values: np.ndarray | Iterable) -> np.ndarray:
    return np.asarray(values)


def _flatten_masked(
    logits: np.ndarray,
    labels: np.ndarray,
    trainable_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = trainable_mask.astype(bool)
    return logits[mask], labels[mask]


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.0
    return float(roc_auc_score(labels, probabilities))


def _safe_ap(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, probabilities))


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[float, float, float]:
    predictions = probabilities >= threshold
    return (
        float(precision_score(labels, predictions, zero_division=0)),
        float(recall_score(labels, predictions, zero_division=0)),
        float(f1_score(labels, predictions, zero_division=0)),
    )


def _best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_f1 = 0.0
    best_threshold = 0.0
    for candidate_threshold in np.linspace(0.0, 1.0, 101):
        _, _, f1 = _classification_metrics(
            labels,
            probabilities,
            threshold=float(candidate_threshold),
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(candidate_threshold)
    return best_f1, best_threshold


def _precision_at_k(sorted_labels: np.ndarray, k: int) -> float:
    top_k = sorted_labels[: min(k, sorted_labels.size)]
    if top_k.size == 0:
        return 0.0
    return float(top_k.mean())


def _recall_at_k(sorted_labels: np.ndarray, k: int, positive_count: int) -> float:
    if positive_count == 0:
        return 0.0
    return float(sorted_labels[: min(k, sorted_labels.size)].sum() / positive_count)


def _ndcg_at_k(sorted_labels: np.ndarray, k: int) -> float:
    top_k = sorted_labels[: min(k, sorted_labels.size)]
    if top_k.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, top_k.size + 2))
    dcg = float(np.sum(top_k * discounts))
    ideal = np.sort(sorted_labels)[::-1][: top_k.size]
    ideal_dcg = float(np.sum(ideal * discounts))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _mrr(sorted_labels: np.ndarray) -> float:
    positives = np.flatnonzero(sorted_labels > 0)
    if positives.size == 0:
        return 0.0
    return float(1.0 / (positives[0] + 1))


def _graph_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    trainable_mask: np.ndarray,
) -> tuple[float, float]:
    graph_correct = 0
    graph_perfect_or_one_fp = 0
    eligible_graphs = 0

    for graph_labels, graph_probs, graph_mask in zip(labels, probabilities, trainable_mask, strict=False):
        masked_labels = graph_labels[graph_mask]
        masked_probs = graph_probs[graph_mask]
        if masked_labels.size == 0:
            continue

        eligible_graphs += 1
        predictions = masked_probs >= 0.5
        expected = masked_labels.astype(bool)

        if np.array_equal(predictions, expected):
            graph_correct += 1

        false_positives = np.logical_and(predictions, ~expected).sum()
        false_negatives = np.logical_and(~predictions, expected).sum()
        if false_negatives == 0 and false_positives <= 1:
            graph_perfect_or_one_fp += 1

    if eligible_graphs == 0:
        return 0.0, 0.0

    return (
        graph_correct / eligible_graphs,
        graph_perfect_or_one_fp / eligible_graphs,
    )


def _empty_metrics(ks: tuple[int, ...], *, threshold: float) -> dict[str, float]:
    metrics = {
        "edge_auc": 0.0,
        "edge_ap": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "decision_threshold": threshold,
        "edge_precision_at_0_5": 0.0,
        "edge_recall_at_0_5": 0.0,
        "edge_f1_at_0_5": 0.0,
        "edge_best_f1": 0.0,
        "edge_best_threshold": 0.0,
        "edge_mrr": 0.0,
        "graph_accuracy": 0.0,
        "graph_perfect_or_one_fp": 0.0,
    }
    for k in ks:
        metrics[f"edge_precision_at_{k}"] = 0.0
        metrics[f"edge_recall_at_{k}"] = 0.0
        metrics[f"edge_ndcg_at_{k}"] = 0.0
    return metrics


def compute_link_prediction_metrics(
    logits: np.ndarray | Iterable,
    labels: np.ndarray | Iterable,
    trainable_mask: np.ndarray | Iterable,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute masked edge-level and graph-level ranking metrics."""

    logits_array = _to_numpy(logits).astype(np.float32)
    labels_array = _to_numpy(labels).astype(np.float32)
    mask_array = _to_numpy(trainable_mask).astype(bool)
    threshold = float(decision_threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"decision_threshold must be within [0.0, 1.0], got {threshold}")

    masked_logits, masked_labels = _flatten_masked(logits_array, labels_array, mask_array)
    masked_probabilities = _sigmoid(masked_logits)
    masked_binary_labels = masked_labels.astype(np.int32)
    if masked_binary_labels.size == 0:
        return _empty_metrics(ks, threshold=threshold)

    order = np.argsort(masked_probabilities)[::-1]
    sorted_labels = masked_binary_labels[order]
    positive_count = int(sorted_labels.sum())

    decision_precision, decision_recall, decision_f1 = _classification_metrics(
        masked_binary_labels,
        masked_probabilities,
        threshold=threshold,
    )
    fixed_precision, fixed_recall, fixed_f1 = _classification_metrics(
        masked_binary_labels,
        masked_probabilities,
        threshold=0.5,
    )
    best_f1, best_threshold = _best_f1_threshold(
        masked_binary_labels,
        masked_probabilities,
    )

    probabilities = _sigmoid(logits_array)
    graph_accuracy, graph_perfect_or_one_fp = _graph_metrics(labels_array, probabilities, mask_array)

    metrics = {
        "edge_auc": _safe_auc(masked_binary_labels, masked_probabilities),
        "edge_ap": _safe_ap(masked_binary_labels, masked_probabilities),
        "precision": decision_precision,
        "recall": decision_recall,
        "f1": decision_f1,
        "decision_threshold": threshold,
        "edge_precision_at_0_5": fixed_precision,
        "edge_recall_at_0_5": fixed_recall,
        "edge_f1_at_0_5": fixed_f1,
        "edge_best_f1": best_f1,
        "edge_best_threshold": best_threshold,
        "edge_mrr": _mrr(sorted_labels),
        "graph_accuracy": float(graph_accuracy),
        "graph_perfect_or_one_fp": float(graph_perfect_or_one_fp),
    }
    for k in ks:
        metrics[f"edge_precision_at_{k}"] = _precision_at_k(sorted_labels, k)
        metrics[f"edge_recall_at_{k}"] = _recall_at_k(sorted_labels, k, positive_count)
        metrics[f"edge_ndcg_at_{k}"] = _ndcg_at_k(sorted_labels, k)
    return metrics


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


def build_compute_metrics(
    ks: tuple[int, ...] = (5, 10, 20, 50),
):
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
        overwrite_output_dir=True,
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
        save_safetensors=False,
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


def _select_split_sizes(config: RuntimeConfig, run_mode: str) -> dict[str, int]:
    section = config.synthetic.smoke_split_sizes if run_mode == "smoke" else config.synthetic.split_sizes
    return section.as_dict()


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


def export_runtime_data(config: RuntimeConfig, paths: ResolvedRuntimePaths, run_mode: str) -> dict[str, Path]:
    return export_complete_splits(
        output_dir=paths.synthetic_output_dir,
        split_sizes=_select_split_sizes(config, run_mode),
        config=config.synthetic.to_generation_config(),
        seed=config.synthetic.seed,
        output_paths=paths.dataset_paths,
        representative_smoke=run_mode == "smoke",
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
    run_mode: str,
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
        "run_mode": run_mode,
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
        raise FileNotFoundError(f"trainer checkpoint weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"invalid trainer checkpoint payload: {weights_path}")
    return state_dict


def save_checkpoints(
    runtime: RuntimeObjects,
    *,
    run_mode: str,
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

    last_checkpoint_path = runtime.paths.checkpoints_dir / checkpoint_filename(run_mode, kind="last")
    best_checkpoint_path = runtime.paths.checkpoints_dir / checkpoint_filename(run_mode, kind="best")
    torch.save(
        _checkpoint_payload(
            run_mode=run_mode,
            model_state_dict=_load_hf_model_state(last_checkpoint_dir),
        ),
        last_checkpoint_path,
    )
    torch.save(
        _checkpoint_payload(
            run_mode=run_mode,
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
    run_mode: str,
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
    summary_path = runtime.paths.results_dir / summary_filename(run_mode)
    _write_json(
        summary_path,
        _summary_payload(
            runtime,
            checkpoint_artifacts=checkpoint_artifacts,
            run_mode=run_mode,
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


def run_training_pipeline(runtime: RuntimeObjects, *, run_mode: str) -> dict[str, Path]:
    if runtime.paths.hf_output_dir.exists():
        shutil.rmtree(runtime.paths.hf_output_dir)
    runtime.paths.hf_output_dir.mkdir(parents=True, exist_ok=True)
    train_result = runtime.trainer.train()
    train_history_path, train_history, val_history_path, val_history = runtime.trainer.write_histories(runtime.paths.results_dir)

    train_loss = float(train_history[-1]["train_loss"])
    val_loss = float(val_history[-1]["val_loss"])

    checkpoint_artifacts = save_checkpoints(
        runtime,
        run_mode=run_mode,
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
    if run_mode == "smoke":
        enforce_smoke_acceptance(train_history, val_history, test_metrics=test_result["metrics"])

    best_entry = _best_validation_entry(val_history)
    artifacts = save_run_artifacts(
        runtime,
        run_mode=run_mode,
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
    verify_completion_artifacts(artifacts["summary"], run_mode=run_mode)
    return artifacts


def run_inference_pipeline(
    runtime: RuntimeObjects,
    *,
    run_mode: str,
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
    summary_path = runtime.paths.results_dir / summary_filename(run_mode)
    _write_json(
        summary_path,
        _summary_payload(
            runtime,
            checkpoint_artifacts=checkpoint_artifacts,
            run_mode=run_mode,
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
    run_mode: str,
    *,
    mode: str,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Path]:
    config = load_runtime_config(config_path)
    paths = resolve_runtime_paths(config)

    _seed_runtime(config.training_args.seed)
    prepare_runtime_environment(paths)

    log_stage("export", run_mode=run_mode, output_dir=paths.synthetic_output_dir)
    export_runtime_data(config, paths, run_mode)

    runtime = build_runtime_objects(config, paths)

    if mode == "train":
        log_stage("train", epochs=runtime.config.training_args.num_train_epochs)
        artifacts = run_training_pipeline(runtime, run_mode=run_mode)
    else:
        if checkpoint_path is None:
            checkpoint_path = paths.checkpoints_dir / checkpoint_filename(run_mode, kind="best")
        log_stage("inference", checkpoint=checkpoint_path)
        artifacts = run_inference_pipeline(runtime, run_mode=run_mode, checkpoint_path=checkpoint_path)

    log_stage("finished", status="ok", summary=artifacts["summary"])
    return artifacts
