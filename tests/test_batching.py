import json

from alarm_hgt.batching import BucketBatchSampler, padding_collate_fn
from alarm_hgt.dataset import AlarmGraphDataset
from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample


def _write_samples(tmp_path, samples):
    path = tmp_path / "batch.json"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return path


def test_bucket_batch_sampler_groups_similar_sizes():
    sampler = BucketBatchSampler(
        sizes=[11, 3, 7, 4, 12],
        batch_size=2,
        shuffle=False,
    )

    batches = list(sampler)
    assert batches == [[1, 3], [2, 0], [4]]


def test_padding_collate_adds_isolated_padding_nodes_only_after_collation(tmp_path):
    sample_small = generate_sample(
        seed=40,
        config=SyntheticGraphConfig(
            num_sites=3,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
    )
    sample_large = generate_sample(
        seed=41,
        config=SyntheticGraphConfig(
            num_sites=5,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
    )
    dataset = AlarmGraphDataset(_write_samples(tmp_path, [sample_small, sample_large]))
    items = [dataset[0], dataset[1]]

    assert not any(item["owner_is_padding"].any().item() for item in items)

    batch = padding_collate_fn(items)

    assert batch["labels"].shape[0] == 2
    assert batch["labels"].shape[1] == max(item["labels"].numel() for item in items)
    assert batch["node_is_padding"].any().item() is True

    padding_indices = batch["node_is_padding"].nonzero(as_tuple=False).view(-1).tolist()
    edge_index = batch["edge_index"].t().tolist()
    for node_index in padding_indices:
        incident_edges = [edge for edge in edge_index if node_index in edge]
        assert incident_edges
        assert all(source == target == node_index for source, target in incident_edges)


def test_padded_positions_are_mask_disabled(tmp_path):
    sample_small = generate_sample(
        seed=42,
        config=SyntheticGraphConfig(
            num_sites=3,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
    )
    sample_large = generate_sample(
        seed=43,
        config=SyntheticGraphConfig(
            num_sites=5,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
    )
    dataset = AlarmGraphDataset(_write_samples(tmp_path, [sample_small, sample_large]))
    batch = padding_collate_fn([dataset[0], dataset[1]])

    padded_positions = batch["owner_is_padding"].nonzero(as_tuple=False)
    assert padded_positions.numel() > 0
    for batch_idx, ae_idx in padded_positions.tolist():
        assert bool(batch["trainable_mask"][batch_idx, ae_idx]) is False
