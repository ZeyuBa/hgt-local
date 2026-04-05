"""Build NetworkX graph views over topology samples."""

from __future__ import annotations

import networkx as nx


def build_ne_topology(sample: dict) -> nx.Graph:
    graph = nx.Graph()
    for node in sample["nodes"]:
        graph.add_node(node["id"], site_id=node["site_id"], node_type=node["type"])
    for edge in sample["edges"]:
        if edge["relation"] in {"co_site_ne_ne", "cross_site_ne_ne"}:
            graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
    return graph

