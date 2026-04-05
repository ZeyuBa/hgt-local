"""Visualization helpers for generated topology samples using pyvis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyvis.network import Network


NODE_TYPE_COLORS: dict[str, str] = {
    "phy_site": "#4A90D9",
    "router": "#1ABC9C",
    "wl_station": "#F39C12",
}

NODE_TYPE_SHAPES: dict[str, str] = {
    "phy_site": "dot",
    "router": "square",
    "wl_station": "triangle",
}

NODE_TYPE_SIZES: dict[str, int] = {
    "phy_site": 18,
    "router": 30,
    "wl_station": 15,
}

EDGE_RELATION_COLORS: dict[str, str] = {
    "co_site_ne_ne": "#CCCCCC",
    "cross_site_ne_ne": "#2C3E50",
}

# Special role colors (override node type color when active)
FAULT_ANCHOR_COLOR = "#E74C3C"   # red — fault/risk injection point
AN_SITE_COLOR = "#9B59B6"        # purple — aggregation node
OUTAGE_BORDER_COLOR = "#E74C3C"  # red border — node is down


def build_pyvis_graph(
    sample: dict,
    height: str = "600px",
    width: str = "100%",
    directed: bool = False,
) -> Network:
    """Build an interactive pyvis Network from a completed topology sample."""

    net = Network(height=height, width=width, directed=directed, notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

    nodes_by_id: dict[str, dict] = {n["id"]: n for n in sample["nodes"]}
    logical = sample.get("logical_failures", {})
    inactive_ne_ids = set(logical.get("inactive_ne_ids", []))
    blocked_pairs = {
        frozenset(pair) for pair in logical.get("blocked_edge_pairs", [])
    }

    for node in sample["nodes"]:
        nid = node["id"]
        ntype = node["type"]
        color = NODE_TYPE_COLORS.get(ntype, "#CCCCCC")
        shape = NODE_TYPE_SHAPES.get(ntype, "dot")

        # Special roles: keep type shape, only change color
        if node.get("is_fault_or_risk_anchor"):
            color = FAULT_ANCHOR_COLOR
        if node.get("is_an"):
            color = AN_SITE_COLOR

        border_color = color
        border_width = 1
        if node.get("is_outage") or nid in inactive_ne_ids:
            border_color = "#E74C3C"
            border_width = 3

        title_lines = [
            f"<b>{nid}</b>",
            f"type: {ntype}",
            f"site: {node['site_id']}",
        ]
        if node.get("is_an"):
            title_lines.append("AN site")
        if node.get("is_fault_or_risk_anchor"):
            title_lines.append("Fault/Risk anchor")
        if node.get("is_outage"):
            title_lines.append("OUTAGE")
        title = "<br>".join(title_lines)

        net.add_node(
            nid,
            label=nid.split(":")[-1] if ":" in nid else nid,
            title=title,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#E74C3C"},
            },
            shape=shape,
            borderWidth=border_width,
            size=NODE_TYPE_SIZES.get(ntype, 20),
        )

    for edge in sample["edges"]:
        rel = edge["relation"]
        if rel not in {"co_site_ne_ne", "cross_site_ne_ne"}:
            continue

        src, tgt = edge["source"], edge["target"]
        pair = frozenset({src, tgt})

        if pair in blocked_pairs:
            edge_color = "#E74C3C"
            dashes = True
            edge_title = f"{rel} (BLOCKED)"
            edge_width = 3
        elif rel == "cross_site_ne_ne":
            edge_color = EDGE_RELATION_COLORS[rel]
            dashes = False
            edge_title = f"{rel} (backbone)"
            edge_width = 3
        else:
            edge_color = EDGE_RELATION_COLORS.get(rel, "#CCCCCC")
            dashes = False
            edge_title = rel
            edge_width = 1

        net.add_edge(
            src,
            tgt,
            color=edge_color,
            dashes=dashes,
            title=edge_title,
            width=edge_width,
        )

    return net


def compute_sample_stats(sample: dict) -> dict[str, Any]:
    """Compute summary statistics for a single topology sample."""

    nodes = sample["nodes"]
    edges = sample["edges"]
    alarm_entities = sample.get("alarm_entities", [])
    logical = sample.get("logical_failures", {})

    type_counts: dict[str, int] = {}
    for node in nodes:
        ntype = node["type"]
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    rel_counts: dict[str, int] = {}
    for edge in edges:
        rel = edge["relation"]
        rel_counts[rel] = rel_counts.get(rel, 0) + 1

    positive_labels = sum(1 for ae in alarm_entities if ae.get("label", 0) == 1)
    trainable = [ae for ae in alarm_entities if ae.get("is_trainable_alarm")]
    trainable_positive = sum(1 for ae in trainable if ae.get("label", 0) == 1)

    site_ids = sorted({n["site_id"] for n in nodes})

    return {
        "sample_id": sample.get("sample_id", ""),
        "seed": sample.get("seed", ""),
        "num_sites": len(site_ids),
        "num_nodes": len(nodes),
        "node_type_counts": type_counts,
        "num_edges": len(edges),
        "edge_relation_counts": rel_counts,
        "fault_sites": sample.get("fault_or_risk_sites", []),
        "fault_modes": sample.get("fault_modes", {}),
        "an_sites": sample.get("an_sites", []),
        "noise_sites": logical.get("noise_sites", []),
        "inactive_ne_ids": logical.get("inactive_ne_ids", []),
        "blocked_edge_pairs": logical.get("blocked_edge_pairs", []),
        "total_alarm_entities": len(alarm_entities),
        "positive_alarm_entities": positive_labels,
        "positive_ratio": positive_labels / len(alarm_entities) if alarm_entities else 0.0,
        "trainable_alarm_entities": len(trainable),
        "trainable_positive": trainable_positive,
        "trainable_positive_ratio": trainable_positive / len(trainable) if trainable else 0.0,
    }


def compute_split_stats(samples: list[dict]) -> dict[str, Any]:
    """Aggregate statistics across all samples in a split."""

    if not samples:
        return {"total_samples": 0}

    per_sample = [compute_sample_stats(s) for s in samples]

    site_counts = [s["num_sites"] for s in per_sample]
    node_counts = [s["num_nodes"] for s in per_sample]
    edge_counts = [s["num_edges"] for s in per_sample]
    pos_ratios = [s["positive_ratio"] for s in per_sample]
    trainable_ratios = [s["trainable_positive_ratio"] for s in per_sample]

    fault_mode_dist: dict[str, int] = {}
    for s in per_sample:
        for mode in s["fault_modes"].values():
            fault_mode_dist[mode] = fault_mode_dist.get(mode, 0) + 1

    def _stats(values: list[float]) -> dict[str, float]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return {
            "mean": round(mean, 3),
            "std": round(variance**0.5, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "total_samples": len(samples),
        "sites": _stats([float(x) for x in site_counts]),
        "nodes": _stats([float(x) for x in node_counts]),
        "edges": _stats([float(x) for x in edge_counts]),
        "positive_ratio": _stats(pos_ratios),
        "trainable_positive_ratio": _stats(trainable_ratios),
        "fault_mode_distribution": fault_mode_dist,
    }


def load_split_samples(jsonl_path: str | Path) -> list[dict]:
    """Load all samples from a transformed JSONL split file."""

    path = Path(jsonl_path)
    if not path.exists():
        return []
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples
