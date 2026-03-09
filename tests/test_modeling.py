import json
import math

import torch

from alarm_hgt.batching import padding_collate_fn
from alarm_hgt.config import AlarmHGTConfig
from alarm_hgt.dataset import AlarmGraphDataset
from alarm_hgt.modeling import EdgePredictor, HGTForLinkPrediction, masked_bce_loss
from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample


def _write_samples(tmp_path, samples):
    path = tmp_path / "model.json"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return path


def _build_batch(tmp_path):
    config = SyntheticGraphConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    sample_a = generate_sample(seed=50, config=config)
    sample_b = generate_sample(seed=51, config=config)
    dataset = AlarmGraphDataset(_write_samples(tmp_path, [sample_a, sample_b]))
    return padding_collate_fn([dataset[0], dataset[1]])


def test_config_defaults_match_design():
    config = AlarmHGTConfig()

    assert config.in_dim == 32
    assert config.n_hid == 64
    assert config.num_layers == 4
    assert config.n_heads == 4
    assert config.dropout == 0.2
    assert config.num_types == 3
    assert config.num_relations == 9


def test_edge_predictor_is_bilinear_over_normalized_embeddings():
    predictor = EdgePredictor(n_hid=3)
    with torch.no_grad():
        predictor.weight.copy_(torch.eye(3))

    left = torch.tensor([[3.0, 0.0, 0.0]])
    right = torch.tensor([[4.0, 0.0, 0.0]])
    score = predictor(left, right)

    assert score.shape == (1,)
    assert math.isclose(score.item(), 1.0, rel_tol=1e-5)


def test_masked_bce_loss_only_uses_trainable_positions():
    logits = torch.tensor([[2.0, -2.0]])
    labels = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[True, False]])

    loss = masked_bce_loss(logits, labels, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
    )
    assert torch.isclose(loss, expected)


def test_model_forward_returns_batched_logits_and_loss(tmp_path):
    batch = _build_batch(tmp_path)
    model = HGTForLinkPrediction(
        AlarmHGTConfig(
            n_hid=16,
            num_layers=2,
            n_heads=4,
            dropout=0.0,
            use_rte=False,
        )
    )

    output = model(**batch)

    assert output.logits.shape == batch["labels"].shape
    assert output.loss is not None
    assert torch.isfinite(output.loss)
