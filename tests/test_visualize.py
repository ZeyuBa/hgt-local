import json

from pyvis.network import Network

from training_data.topo_complete import export_synthetic_splits, generate_complete_sample
from training_data.topo_generator import SyntheticGraphConfig
from training_data.visualize import (
    build_pyvis_graph,
    compute_sample_stats,
    compute_split_stats,
    load_split_samples,
)

_SMALL_CONFIG = SyntheticGraphConfig(
    num_sites=4,
    fault_site_count=(1, 1),
    an_site_count=(1, 1),
    backup_link_probability=0.0,
    noise_probability=0.0,
    topology_mode="chain",
)


def _make_sample(seed: int = 42, **kwargs) -> dict:
    return generate_complete_sample(seed=seed, config=_SMALL_CONFIG, **kwargs)


def test_build_pyvis_graph_returns_network():
    sample = _make_sample()
    net = build_pyvis_graph(sample)
    assert isinstance(net, Network)


def test_build_pyvis_graph_has_correct_node_count():
    sample = _make_sample()
    net = build_pyvis_graph(sample)
    ne_count = len(sample["nodes"])
    assert len(net.nodes) == ne_count


def test_build_pyvis_graph_has_edges():
    sample = _make_sample()
    net = build_pyvis_graph(sample)
    ne_edges = [
        e for e in sample["edges"]
        if e["relation"] in {"co_site_ne_ne", "cross_site_ne_ne"}
    ]
    assert len(net.edges) == len(ne_edges)


def test_build_pyvis_graph_outage_nodes_have_red_border():
    sample = _make_sample()
    net = build_pyvis_graph(sample)
    outage_ids = {n["id"] for n in sample["nodes"] if n.get("is_outage")}
    assert outage_ids, "test sample should have at least one outage node"
    for vis_node in net.nodes:
        if vis_node["id"] in outage_ids:
            assert vis_node["color"]["border"] == "#E74C3C"


def test_build_pyvis_graph_blocked_edges_highlighted():
    sample = _make_sample(
        seed=22,
        forced_an_sites=["site_000"],
        forced_fault_sites=["site_002"],
        forced_fault_modes={"site_002": "link_down"},
        forced_noise_sites=[],
    )
    blocked = sample["logical_failures"]["blocked_edge_pairs"]
    assert blocked, "link_down sample should have blocked edge pairs"
    net = build_pyvis_graph(sample)
    red_edges = [e for e in net.edges if e.get("color") == "#E74C3C"]
    assert len(red_edges) >= len(blocked)


def test_build_pyvis_graph_generates_valid_html():
    sample = _make_sample()
    net = build_pyvis_graph(sample)
    html = net.generate_html()
    assert "<html>" in html.lower() or "vis-network" in html.lower()


def test_compute_sample_stats_keys():
    sample = _make_sample()
    stats = compute_sample_stats(sample)
    expected_keys = {
        "sample_id", "seed", "num_sites", "num_nodes", "node_type_counts",
        "num_edges", "edge_relation_counts", "fault_sites", "fault_modes",
        "an_sites", "noise_sites", "inactive_ne_ids", "blocked_edge_pairs",
        "total_alarm_entities", "positive_alarm_entities", "positive_ratio",
        "trainable_alarm_entities", "trainable_positive", "trainable_positive_ratio",
    }
    assert expected_keys.issubset(stats.keys())


def test_compute_sample_stats_counts_match():
    sample = _make_sample()
    stats = compute_sample_stats(sample)
    assert stats["num_nodes"] == len(sample["nodes"])
    assert stats["num_edges"] == len(sample["edges"])
    assert stats["total_alarm_entities"] == len(sample.get("alarm_entities", []))


def test_compute_split_stats_aggregation():
    samples = [_make_sample(seed=i) for i in range(5)]
    stats = compute_split_stats(samples)
    assert stats["total_samples"] == 5
    assert "mean" in stats["nodes"]
    assert "mean" in stats["positive_ratio"]
    assert isinstance(stats["fault_mode_distribution"], dict)


def test_compute_split_stats_empty():
    stats = compute_split_stats([])
    assert stats["total_samples"] == 0


def test_load_split_samples_reads_jsonl(tmp_path):
    paths = export_synthetic_splits(
        output_dir=tmp_path,
        split_sizes={"train": 3, "val": 0, "test": 0},
        config=_SMALL_CONFIG,
        seed=100,
    )
    samples = load_split_samples(paths["train"])
    assert len(samples) == 3
    assert all("sample_id" in s for s in samples)
    assert all("nodes" in s for s in samples)


def test_load_split_samples_nonexistent_path(tmp_path):
    samples = load_split_samples(tmp_path / "does_not_exist.json")
    assert samples == []
