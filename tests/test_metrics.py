import math

import numpy as np
import pytest

from src.training.trainer import compute_link_prediction_metrics


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def test_edge_level_metrics_ignore_masked_positions():
    logits = np.array(
        [
            [_logit(0.95), _logit(0.05), _logit(1e-4)],
            [_logit(0.10), _logit(0.90), _logit(1.0 - 1e-4)],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    trainable_mask = np.array(
        [
            [True, True, False],
            [True, True, False],
        ],
        dtype=bool,
    )

    metrics = compute_link_prediction_metrics(logits, labels, trainable_mask, ks=(1, 2))

    assert metrics["edge_auc"] == pytest.approx(1.0)
    assert metrics["edge_ap"] == pytest.approx(1.0)
    assert metrics["edge_f1_at_0_5"] == pytest.approx(1.0)


def test_best_f1_threshold_scan_finds_better_threshold_than_point_five():
    logits = np.array(
        [[_logit(0.90), _logit(0.80), _logit(0.70), _logit(0.40)]],
        dtype=np.float32,
    )
    labels = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    trainable_mask = np.array([[True, True, True, True]], dtype=bool)

    metrics = compute_link_prediction_metrics(logits, labels, trainable_mask, ks=(1,))

    assert metrics["edge_f1_at_0_5"] == pytest.approx(0.5)
    assert metrics["edge_best_f1"] == pytest.approx(1.0)
    assert 0.80 <= metrics["edge_best_threshold"] <= 0.90


def test_graph_level_metrics_are_computed_per_graph():
    logits = np.array(
        [
            [_logit(0.90), _logit(0.10), _logit(0.20)],
            [_logit(0.90), _logit(0.70), _logit(0.20)],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    trainable_mask = np.array(
        [
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )

    metrics = compute_link_prediction_metrics(logits, labels, trainable_mask, ks=(1,))

    assert metrics["graph_accuracy"] == pytest.approx(0.5)
    assert metrics["graph_perfect_or_one_fp"] == pytest.approx(1.0)


def test_metrics_return_zeros_when_no_positions_are_trainable():
    logits = np.array([[0.1, -0.2]], dtype=np.float32)
    labels = np.array([[1.0, 0.0]], dtype=np.float32)
    trainable_mask = np.array([[False, False]], dtype=bool)

    metrics = compute_link_prediction_metrics(logits, labels, trainable_mask, ks=(1,))

    assert metrics["edge_auc"] == pytest.approx(0.0)
    assert metrics["edge_ap"] == pytest.approx(0.0)
    assert metrics["edge_f1_at_0_5"] == pytest.approx(0.0)
    assert metrics["graph_accuracy"] == pytest.approx(0.0)
