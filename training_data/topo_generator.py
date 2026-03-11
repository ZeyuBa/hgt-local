"""Stage 1: generate base network topologies before alarm expansion."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import networkx as nx

from src.graph.feature_extraction import ALARM_DEFINITIONS, EdgeRecord, NodeRecord


@dataclass(frozen=True)
class TopologyGenerationConfig:
    """Controls synthetic topology generation."""

    num_sites: int | tuple[int, int] = (8, 16)
    wl_stations_per_site: tuple[int, int] = (2, 4)
    fault_site_count: tuple[int, int] = (1, 3)
    an_site_count: tuple[int, int] = (1, 2)
    backup_link_probability: float = 0.18
    noise_probability: float = 0.08
    topology_mode: str = "random_tree"


SyntheticGraphConfig = TopologyGenerationConfig


def _serialize_rng_state(state) -> list | int | None:
    if isinstance(state, tuple):
        return [_serialize_rng_state(item) for item in state]
    return state


def deserialize_rng_state(state):
    if isinstance(state, list):
        return tuple(deserialize_rng_state(item) for item in state)
    return state


def _pick_int(value: int | tuple[int, int], rng: random.Random) -> int:
    if isinstance(value, int):
        return value
    low, high = value
    return rng.randint(low, high)


def _site_id(index: int) -> str:
    return f"site_{index:03d}"


def _ne_nodes_for_site(site_id: str, is_an: bool, is_anchor: bool, station_count: int) -> list[NodeRecord]:
    nodes: list[NodeRecord] = [
        {
            "id": f"phy_site:{site_id}",
            "type": "phy_site",
            "site_id": site_id,
            "is_an": is_an,
            "is_fault_or_risk_anchor": is_anchor,
            "is_padding": False,
            "is_outage": False,
        },
        {
            "id": f"router:{site_id}",
            "type": "router",
            "site_id": site_id,
            "is_an": is_an,
            "is_fault_or_risk_anchor": is_anchor,
            "is_padding": False,
            "is_outage": False,
        },
    ]
    for station_index in range(station_count):
        nodes.append(
            {
                "id": f"wl_station:{site_id}:{station_index}",
                "type": "wl_station",
                "site_id": site_id,
                "is_an": is_an,
                "is_fault_or_risk_anchor": is_anchor,
                "is_padding": False,
                "is_outage": False,
            }
        )
    return nodes


def _site_edges(site_id: str, station_count: int) -> list[EdgeRecord]:
    router_id = f"router:{site_id}"
    edges: list[EdgeRecord] = [
        {
            "source": f"phy_site:{site_id}",
            "target": router_id,
            "relation": "co_site_ne_ne",
        }
    ]
    for station_index in range(station_count):
        edges.append(
            {
                "source": router_id,
                "target": f"wl_station:{site_id}:{station_index}",
                "relation": "co_site_ne_ne",
            }
        )
    return edges


def _backbone_edges(
    site_ids: list[str],
    rng: random.Random,
    topology_mode: str,
) -> tuple[list[tuple[str, str]], dict[str, str | None]]:
    edges: list[tuple[str, str]] = []
    primary_upstream_by_site: dict[str, str | None] = {site_ids[0]: None}
    if topology_mode == "chain":
        for index in range(1, len(site_ids)):
            site_id = site_ids[index]
            parent = site_ids[index - 1]
            edges.append((site_id, parent))
            primary_upstream_by_site[site_id] = parent
        return edges, primary_upstream_by_site
    if topology_mode != "random_tree":
        raise ValueError(f"unsupported topology_mode: {topology_mode}")
    for index in range(1, len(site_ids)):
        site_id = site_ids[index]
        parent = site_ids[rng.randrange(index)]
        edges.append((site_id, parent))
        primary_upstream_by_site[site_id] = parent
    return edges, primary_upstream_by_site


def _with_backup_edges(
    site_ids: list[str],
    backbone_edges: Iterable[tuple[str, str]],
    probability: float,
    rng: random.Random,
) -> list[tuple[str, str]]:
    existing = {tuple(sorted(edge)) for edge in backbone_edges}
    all_edges = list(existing)
    for source_index, source_site in enumerate(site_ids):
        for target_site in site_ids[source_index + 1 :]:
            pair = tuple(sorted((source_site, target_site)))
            if pair in existing:
                continue
            if rng.random() < probability:
                all_edges.append(pair)
                existing.add(pair)
    return all_edges


def _ensure_known_sites(site_ids: set[str], values: Sequence[str], label: str) -> None:
    unknown = sorted(set(values) - site_ids)
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")


def build_ne_graph(nodes: list[NodeRecord], edges: list[EdgeRecord]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], site_id=node["site_id"], node_type=node["type"])
    for edge in edges:
        if edge["relation"] not in {"co_site_ne_ne", "cross_site_ne_ne"}:
            continue
        graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
    return graph


def active_graph(
    graph: nx.Graph,
    inactive_ne_ids: set[str],
    blocked_edge_pairs: set[frozenset[str]],
) -> nx.Graph:
    active = graph.copy()
    active.remove_nodes_from(inactive_ne_ids)
    for edge in list(active.edges()):
        if frozenset(edge) in blocked_edge_pairs:
            active.remove_edge(*edge)
    return active


def site_nodes(nodes: list[NodeRecord], site_id: str, node_type: str | None = None) -> list[NodeRecord]:
    return [
        node
        for node in nodes
        if node["site_id"] == site_id and (node_type is None or node["type"] == node_type)
    ]


def select_noise_sites(
    site_ids: list[str],
    an_sites: set[str],
    fault_or_risk_sites: list[str],
    active_topology: nx.Graph,
    nodes: list[NodeRecord],
    rng: random.Random,
    probability: float,
    forced_noise_sites: Sequence[str] | None,
) -> list[str]:
    connected_sites = set()
    an_roots = [node["id"] for node in nodes if node["site_id"] in an_sites and active_topology.has_node(node["id"])]
    for node in nodes:
        if node["type"] != "wl_station" or not active_topology.has_node(node["id"]):
            continue
        if any(nx.has_path(active_topology, node["id"], root) for root in an_roots):
            connected_sites.add(node["site_id"])

    candidate_sites = sorted(connected_sites - set(fault_or_risk_sites) - an_sites)
    if forced_noise_sites is not None:
        _ensure_known_sites(set(site_ids), forced_noise_sites, "noise sites")
        invalid = sorted(set(forced_noise_sites) - set(candidate_sites))
        if invalid:
            raise ValueError(f"invalid noise sites: {invalid}")
        return sorted(forced_noise_sites)
    return [site_id for site_id in candidate_sites if rng.random() < probability]


def build_alarm_entities(
    nodes: list[NodeRecord],
    labels: dict[str, int],
) -> list[dict]:
    entities: list[dict] = []
    for node in nodes:
        for alarm_name, definition in ALARM_DEFINITIONS.items():
            if definition["node_type"] != node["type"]:
                continue
            entity_id = f"{alarm_name};{node['id']}"
            entities.append(
                {
                    "id": entity_id,
                    "ne_id": node["id"],
                    "site_id": node["site_id"],
                    "alarm_name": alarm_name,
                    "alarm_id": definition["alarm_id"],
                    "domain": definition["domain"],
                    "label": int(labels.get(entity_id, 0)),
                    "is_trainable_alarm": bool(definition["trainable"]),
                    "owner_is_an": bool(node["is_an"]),
                    "owner_is_fault_or_risk_anchor": bool(node["is_fault_or_risk_anchor"]),
                    "owner_is_padding": bool(node["is_padding"]),
                }
            )
    return entities


def generate_topology_sample(
    seed: int,
    config: TopologyGenerationConfig | None = None,
    forced_an_sites: Sequence[str] | None = None,
    forced_fault_sites: Sequence[str] | None = None,
    forced_fault_modes: dict[str, str] | None = None,
) -> dict:
    """Generate one base topology sample without alarm-entity expansion."""

    config = config or TopologyGenerationConfig()
    rng = random.Random(seed)

    num_sites = _pick_int(config.num_sites, rng)
    site_ids = [_site_id(index) for index in range(num_sites)]

    backbone_edges, primary_upstream_by_site = _backbone_edges(
        site_ids,
        rng,
        config.topology_mode,
    )
    site_level_edges = _with_backup_edges(
        site_ids=site_ids,
        backbone_edges=backbone_edges,
        probability=config.backup_link_probability,
        rng=rng,
    )

    if forced_an_sites is not None:
        _ensure_known_sites(set(site_ids), forced_an_sites, "AN sites")
        an_sites = set(forced_an_sites)
    else:
        an_site_count = min(_pick_int(config.an_site_count, rng), len(site_ids))
        an_sites = set(rng.sample(site_ids, k=an_site_count))

    candidate_fault_sites = [site_id for site_id in site_ids if primary_upstream_by_site[site_id] is not None]
    if not candidate_fault_sites:
        candidate_fault_sites = site_ids[:]
    if forced_fault_sites is not None:
        _ensure_known_sites(set(site_ids), forced_fault_sites, "fault or risk sites")
        invalid_fault_sites = sorted(set(forced_fault_sites) - set(candidate_fault_sites) - {site_ids[0]})
        if invalid_fault_sites:
            raise ValueError(f"fault sites must have a primary upstream: {invalid_fault_sites}")
        fault_or_risk_sites = sorted(forced_fault_sites)
    else:
        fault_site_count = min(_pick_int(config.fault_site_count, rng), len(candidate_fault_sites))
        fault_or_risk_sites = sorted(rng.sample(candidate_fault_sites, k=fault_site_count))

    fault_modes: dict[str, str] = {}
    for site_id in fault_or_risk_sites:
        if forced_fault_modes and site_id in forced_fault_modes:
            fault_modes[site_id] = forced_fault_modes[site_id]
        elif primary_upstream_by_site[site_id] is None:
            fault_modes[site_id] = "mains_failure"
        else:
            fault_modes[site_id] = rng.choice(("mains_failure", "link_down"))

    wl_station_counts = {
        site_id: rng.randint(*config.wl_stations_per_site)
        for site_id in site_ids
    }

    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    for site_id in site_ids:
        nodes.extend(
            _ne_nodes_for_site(
                site_id=site_id,
                is_an=site_id in an_sites,
                is_anchor=site_id in fault_or_risk_sites,
                station_count=wl_station_counts[site_id],
            )
        )
        edges.extend(_site_edges(site_id, wl_station_counts[site_id]))

    for source_site, target_site in site_level_edges:
        edges.append(
            {
                "source": f"router:{source_site}",
                "target": f"router:{target_site}",
                "relation": "cross_site_ne_ne",
            }
        )

    return {
        "sample_id": f"sample-{seed}",
        "seed": seed,
        "fault_or_risk_sites": fault_or_risk_sites,
        "fault_modes": fault_modes,
        "an_sites": sorted(an_sites),
        "primary_upstream_by_site": primary_upstream_by_site,
        "nodes": nodes,
        "edges": edges,
        "noise_probability": config.noise_probability,
        "_annotation_rng_state": _serialize_rng_state(rng.getstate()),
    }
