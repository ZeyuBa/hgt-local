"""Metrics for masked alarm link prediction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


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


def compute_link_prediction_metrics(
    logits: np.ndarray | Iterable,
    labels: np.ndarray | Iterable,
    trainable_mask: np.ndarray | Iterable,
    ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[str, float]:
    """Compute masked edge-level and graph-level ranking metrics."""

    logits_array = _to_numpy(logits).astype(np.float32)
    labels_array = _to_numpy(labels).astype(np.float32)
    mask_array = _to_numpy(trainable_mask).astype(bool)

    masked_logits, masked_labels = _flatten_masked(logits_array, labels_array, mask_array)
    masked_probabilities = _sigmoid(masked_logits)
    masked_binary_labels = masked_labels.astype(np.int32)
    if masked_binary_labels.size == 0:
        metrics = {
            "edge_auc": 0.0,
            "edge_ap": 0.0,
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

    order = np.argsort(masked_probabilities)[::-1]
    sorted_labels = masked_binary_labels[order]
    positive_count = int(sorted_labels.sum())

    fixed_threshold_predictions = masked_probabilities >= 0.5
    fixed_f1 = float(f1_score(masked_binary_labels, fixed_threshold_predictions, zero_division=0))

    best_f1 = 0.0
    best_threshold = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        score = float(
            f1_score(masked_binary_labels, masked_probabilities >= threshold, zero_division=0)
        )
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    probabilities = _sigmoid(logits_array)
    graph_accuracy, graph_perfect_or_one_fp = _graph_metrics(labels_array, probabilities, mask_array)

    metrics = {
        "edge_auc": _safe_auc(masked_binary_labels, masked_probabilities),
        "edge_ap": _safe_ap(masked_binary_labels, masked_probabilities),
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
