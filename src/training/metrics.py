"""Link prediction metrics: edge-level, graph-level, and ranking."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import EvalPrediction


def _extract_logits(predictions: Any) -> np.ndarray:
    if isinstance(predictions, dict):
        predictions = predictions["logits"]
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return np.asarray(predictions, dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-values))


def _flatten_masked(
    logits: np.ndarray,
    labels: np.ndarray,
    trainable_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = trainable_mask.astype(bool)
    mask = mask & (labels != -100.0)
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
        valid = graph_mask & (graph_labels != -100.0)
        masked_labels = graph_labels[valid]
        masked_probs = graph_probs[valid]
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

    logits_array = np.asarray(logits).astype(np.float32)
    labels_array = np.asarray(labels).astype(np.float32)
    mask_array = np.asarray(trainable_mask).astype(bool)
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
