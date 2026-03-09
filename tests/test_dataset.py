import json

from alarm_hgt.constants import RELATION_TYPE_IDS
from alarm_hgt.dataset import AlarmGraphDataset
from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample


def _write_sample(tmp_path, sample):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return path


def test_dataset_builds_32_dim_features_and_broadcasts_site_flags(tmp_path):
    sample = generate_sample(
        seed=30,
        config=SyntheticGraphConfig(
            num_sites=4,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_001"],
        forced_fault_modes={"site_001": "mains_failure"},
        forced_noise_sites=[],
    )
    dataset = AlarmGraphDataset(_write_sample(tmp_path, sample))
    item = dataset[0]

    assert item["node_features"].shape[1] == 32

    fault_idx = item["alarm_entity_ids"].index("ne_is_disconnected;wl_station:site_001:0")
    an_idx = item["alarm_entity_ids"].index("ne_is_disconnected;wl_station:site_000:0")
    assert bool(item["owner_is_fault_or_risk_anchor"][fault_idx]) is True
    assert bool(item["owner_is_an"][fault_idx]) is False
    assert bool(item["owner_is_an"][an_idx]) is True
    assert bool(item["owner_is_padding"][fault_idx]) is False


def test_dataset_synthesizes_reverse_edges_and_self_loops(tmp_path):
    sample = generate_sample(seed=31)
    dataset = AlarmGraphDataset(_write_sample(tmp_path, sample))
    item = dataset[0]

    edge_types = item["edge_type"].tolist()
    assert edge_types.count(RELATION_TYPE_IDS["self"]) == item["node_features"].shape[0]
    assert RELATION_TYPE_IDS["ne_alarm_entity"] in edge_types
    assert RELATION_TYPE_IDS["rev_ne_alarm_entity"] in edge_types
    assert RELATION_TYPE_IDS["alarm_entity_alarm"] in edge_types
    assert RELATION_TYPE_IDS["rev_alarm_entity_alarm"] in edge_types
    assert RELATION_TYPE_IDS["co_site_ne_ne"] in edge_types
    assert RELATION_TYPE_IDS["rev_co_site_ne_ne"] in edge_types


def test_trainable_mask_excludes_anchor_and_an_sites_but_keeps_regular_targets(tmp_path):
    sample = generate_sample(
        seed=32,
        config=SyntheticGraphConfig(
            num_sites=4,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_001"],
        forced_fault_modes={"site_001": "mains_failure"},
        forced_noise_sites=[],
    )
    dataset = AlarmGraphDataset(_write_sample(tmp_path, sample))
    item = dataset[0]

    anchor_idx = item["alarm_entity_ids"].index("ne_is_disconnected;wl_station:site_001:0")
    an_idx = item["alarm_entity_ids"].index("ne_is_disconnected;wl_station:site_000:0")
    regular_idx = item["alarm_entity_ids"].index("ne_is_disconnected;wl_station:site_002:0")
    router_alarm_idx = item["alarm_entity_ids"].index("device_powered_off;router:site_001")

    assert bool(item["trainable_mask"][anchor_idx]) is False
    assert bool(item["trainable_mask"][an_idx]) is False
    assert bool(item["trainable_mask"][regular_idx]) is True
    assert bool(item["trainable_mask"][router_alarm_idx]) is False
