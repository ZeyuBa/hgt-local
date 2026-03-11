import math
import warnings

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


def test_metrics_tolerate_hf_padding_sentinel_in_labels():
    """HuggingFace Trainer pads label arrays with -100 when concatenating
    batches of different sequence lengths.  Boolean masks get padded with
    True (default for bool), so the sentinel labels leak through unless
    _flatten_masked explicitly filters them out."""
    logits = np.array(
        [
            [_logit(0.95), _logit(0.05), _logit(0.50), _logit(0.50)],
            [_logit(0.10), _logit(0.90), _logit(0.50), _logit(0.50)],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [1.0, 0.0, -100.0, -100.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    trainable_mask = np.array(
        [
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=bool,
    )

    metrics = compute_link_prediction_metrics(logits, labels, trainable_mask, ks=(1,))

    assert 0.0 <= metrics["edge_auc"] <= 1.0
    assert 0.0 <= metrics["edge_f1_at_0_5"] <= 1.0
    assert 0.0 <= metrics["graph_accuracy"] <= 1.0
    assert 0.0 <= metrics["graph_perfect_or_one_fp"] <= 1.0


def test_metrics_tolerate_hf_padding_sentinel_with_real_mask():
    """When the mask correctly marks sentinels as False, the result should
    be identical to a clean array without sentinels."""
    logits_clean = np.array(
        [
            [_logit(0.95), _logit(0.05)],
            [_logit(0.10), _logit(0.90)],
        ],
        dtype=np.float32,
    )
    labels_clean = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mask_clean = np.array([[True, True], [True, True]], dtype=bool)

    logits_padded = np.array(
        [
            [_logit(0.95), _logit(0.05), _logit(0.50)],
            [_logit(0.10), _logit(0.90), _logit(0.50)],
        ],
        dtype=np.float32,
    )
    labels_padded = np.array(
        [[1.0, 0.0, -100.0], [0.0, 1.0, -100.0]],
        dtype=np.float32,
    )
    mask_padded = np.array(
        [[True, True, False], [True, True, False]],
        dtype=bool,
    )

    metrics_clean = compute_link_prediction_metrics(
        logits_clean, labels_clean, mask_clean, ks=(1,)
    )
    metrics_padded = compute_link_prediction_metrics(
        logits_padded, labels_padded, mask_padded, ks=(1,)
    )

    for key in ("edge_auc", "edge_ap", "edge_f1_at_0_5", "graph_accuracy"):
        assert metrics_clean[key] == pytest.approx(metrics_padded[key]), (
            f"Mismatch on {key}: clean={metrics_clean[key]}, padded={metrics_padded[key]}"
        )


def test_metrics_handle_ragged_batch_simulation():
    """Simulate what happens when Trainer concatenates two batches of
    different sequence lengths: shorter batch gets -100 padding in labels
    and True padding in mask."""
    logits_cat = np.array(
        [
            [_logit(0.9), _logit(0.1), _logit(0.8), 0.0, 0.0],
            [_logit(0.9), _logit(0.1), _logit(0.8), _logit(0.2), _logit(0.7)],
        ],
        dtype=np.float32,
    )
    labels_cat = np.array(
        [
            [1.0, 0.0, 1.0, -100.0, -100.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    mask_cat = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        dtype=bool,
    )

    metrics = compute_link_prediction_metrics(logits_cat, labels_cat, mask_cat, ks=(1,))
    assert 0.0 <= metrics["edge_auc"] <= 1.0
    assert 0.0 <= metrics["edge_f1_at_0_5"] <= 1.0
    assert 0.0 <= metrics["graph_accuracy"] <= 1.0


def test_graph_metrics_exclude_sentinel_positions():
    """_graph_metrics must exclude -100 sentinel labels from per-graph
    accuracy computation, even when the mask says True."""
    logits = np.array(
        [
            [_logit(0.90), _logit(0.10), _logit(0.50)],
        ],
        dtype=np.float32,
    )
    labels = np.array([[1.0, 0.0, -100.0]], dtype=np.float32)
    mask = np.array([[True, True, True]], dtype=bool)

    metrics = compute_link_prediction_metrics(logits, labels, mask, ks=(1,))

    assert metrics["graph_accuracy"] == pytest.approx(1.0)


def test_sigmoid_does_not_overflow_on_extreme_logits():
    """Large-magnitude logits previously caused RuntimeWarning: overflow
    encountered in exp.  The clipped sigmoid should handle them silently."""
    logits = np.array(
        [[1000.0, -1000.0, 500.0, -500.0]],
        dtype=np.float32,
    )
    labels = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    mask = np.array([[True, True, True, True]], dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = compute_link_prediction_metrics(logits, labels, mask, ks=(1,))

    assert metrics["edge_auc"] == pytest.approx(1.0)
    assert metrics["edge_f1_at_0_5"] == pytest.approx(1.0)
