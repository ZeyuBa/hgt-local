from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample


def _alarm_labels(sample):
    return {entity["id"]: entity["label"] for entity in sample["alarm_entities"]}


def test_mains_failure_labels_local_and_downstream_disconnections():
    config = SyntheticGraphConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    sample = generate_sample(
        seed=11,
        config=config,
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_001"],
        forced_fault_modes={"site_001": "mains_failure"},
        forced_noise_sites=[],
    )
    labels = _alarm_labels(sample)

    assert labels["mains_failure;phy_site:site_001"] == 1
    assert labels["device_powered_off;router:site_001"] == 1
    assert labels["ne_is_disconnected;wl_station:site_001:0"] == 1
    assert labels["ne_is_disconnected;wl_station:site_002:0"] == 1
    assert labels["ne_is_disconnected;wl_station:site_003:0"] == 1
    assert labels["ne_is_disconnected;wl_station:site_000:0"] == 0


def test_link_down_labels_both_link_endpoints_and_propagates_to_downstream_sites():
    config = SyntheticGraphConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    sample = generate_sample(
        seed=12,
        config=config,
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_002"],
        forced_fault_modes={"site_002": "link_down"},
        forced_noise_sites=[],
    )
    labels = _alarm_labels(sample)

    assert labels["link_down;router:site_001"] == 1
    assert labels["link_down;router:site_002"] == 1
    assert labels["ne_is_disconnected;wl_station:site_002:0"] == 1
    assert labels["ne_is_disconnected;wl_station:site_003:0"] == 1
    assert labels["ne_is_disconnected;wl_station:site_001:0"] == 0
    assert any(
        edge["relation"] == "cross_site_ne_ne"
        and {edge["source"], edge["target"]} == {"router:site_001", "router:site_002"}
        for edge in sample["edges"]
    )


def test_noise_mains_failure_does_not_create_false_downstream_outages():
    config = SyntheticGraphConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(0, 0),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    sample = generate_sample(
        seed=13,
        config=config,
        forced_an_sites=["site_000"],
        forced_fault_sites=[],
        forced_fault_modes={},
        forced_noise_sites=["site_002"],
    )
    labels = _alarm_labels(sample)

    assert labels["mains_failure;phy_site:site_002"] == 1
    assert labels["ne_is_disconnected;wl_station:site_002:0"] == 0
    assert labels["ne_is_disconnected;wl_station:site_003:0"] == 0


def test_site_level_flags_are_broadcast_to_alarm_entities():
    config = SyntheticGraphConfig(
        num_sites=3,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    sample = generate_sample(
        seed=14,
        config=config,
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_001"],
        forced_fault_modes={"site_001": "mains_failure"},
        forced_noise_sites=[],
    )

    ae = next(entity for entity in sample["alarm_entities"] if entity["id"] == "ne_is_disconnected;wl_station:site_001:0")
    assert ae["owner_is_fault_or_risk_anchor"] is True
    assert ae["owner_is_an"] is False
    assert ae["owner_is_padding"] is False
