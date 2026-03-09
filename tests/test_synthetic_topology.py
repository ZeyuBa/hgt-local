from collections import defaultdict

from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample


def _ne_adjacency(sample):
    adjacency = defaultdict(set)
    nodes_by_id = {node["id"]: node for node in sample["nodes"]}
    for edge in sample["edges"]:
        if edge["relation"] not in {"co_site_ne_ne", "cross_site_ne_ne"}:
            continue
        source = nodes_by_id[edge["source"]]
        target = nodes_by_id[edge["target"]]
        if source["type"] in {"alarm", "alarm_entity"} or target["type"] in {"alarm", "alarm_entity"}:
            continue
        adjacency[source["id"]].add(target["id"])
        adjacency[target["id"]].add(source["id"])
    return adjacency


def test_generated_topology_is_connected_before_fault_simulation():
    config = SyntheticGraphConfig(
        num_sites=6,
        wl_stations_per_site=(2, 2),
        fault_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
    )
    sample = generate_sample(seed=7, config=config)
    adjacency = _ne_adjacency(sample)

    start = next(iter(adjacency))
    stack = [start]
    seen = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(adjacency[node_id] - seen)

    ne_ids = {
        node["id"]
        for node in sample["nodes"]
        if node["type"] in {"wl_station", "phy_site", "router"}
    }
    assert seen == ne_ids


def test_each_site_has_one_router_one_phy_site_and_at_least_one_station():
    sample = generate_sample(seed=13)
    grouped = defaultdict(lambda: defaultdict(int))
    for node in sample["nodes"]:
        if node["type"] in {"wl_station", "phy_site", "router"}:
            grouped[node["site_id"]][node["type"]] += 1

    assert grouped
    for counts in grouped.values():
        assert counts["phy_site"] == 1
        assert counts["router"] == 1
        assert counts["wl_station"] >= 1


def test_cross_site_edges_only_connect_routers():
    sample = generate_sample(seed=21)
    nodes_by_id = {node["id"]: node for node in sample["nodes"]}

    for edge in sample["edges"]:
        if edge["relation"] != "cross_site_ne_ne":
            continue
        assert nodes_by_id[edge["source"]]["type"] == "router"
        assert nodes_by_id[edge["target"]]["type"] == "router"


def test_fault_simulation_keeps_complete_exported_graph():
    config = SyntheticGraphConfig(
        num_sites=5,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
    )
    sample = generate_sample(seed=5, config=config)
    nodes_by_id = {node["id"]: node for node in sample["nodes"]}

    fault_site = sample["fault_or_risk_sites"][0]
    router_id = f"router:{fault_site}"
    assert router_id in nodes_by_id

    cross_site_edges = [
        edge
        for edge in sample["edges"]
        if edge["relation"] == "cross_site_ne_ne" and router_id in {edge["source"], edge["target"]}
    ]
    assert cross_site_edges, "faulty router must remain in the exported graph with its original edges"
