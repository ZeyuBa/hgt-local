"""Feature construction for alarm HGT samples."""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import numpy as np

from .constants import ALARM_DEFINITIONS, ALARM_IDS, NE_SUBTYPE_IDS, ROLE_IDS


FEATURE_DIM = 32
MAX_DOMAIN = max(int(definition["domain"]) for definition in ALARM_DEFINITIONS.values())


def bucketize_distance(distance: int | None, is_na: bool = False) -> np.ndarray:
    bucket = np.zeros(8, dtype=np.float32)
    if is_na:
        bucket[0] = 1.0
        return bucket
    if distance is None:
        bucket[7] = 1.0
        return bucket
    if distance == 0:
        bucket[1] = 1.0
        return bucket
    if distance == 1:
        bucket[2] = 1.0
        return bucket
    if distance == 2:
        bucket[3] = 1.0
        return bucket
    if distance == 3:
        bucket[4] = 1.0
        return bucket
    if distance == 4:
        bucket[5] = 1.0
        return bucket
    bucket[6] = 1.0
    return bucket


def build_ne_topology(sample: dict) -> nx.Graph:
    graph = nx.Graph()
    for node in sample["nodes"]:
        graph.add_node(node["id"], site_id=node["site_id"], node_type=node["type"])
    for edge in sample["edges"]:
        if edge["relation"] not in {"co_site_ne_ne", "cross_site_ne_ne"}:
            continue
        graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
    return graph


def _distance_map(graph: nx.Graph, source_ids: Iterable[str]) -> dict[str, int]:
    sources = [source_id for source_id in source_ids if source_id in graph]
    if not sources:
        return {}
    return nx.multi_source_dijkstra_path_length(graph, sources)


def _degree_maps(sample: dict) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    co_degree: dict[str, int] = {}
    cross_degree: dict[str, int] = {}
    total_degree: dict[str, int] = {}
    for edge in sample["edges"]:
        if edge["relation"] not in {"co_site_ne_ne", "cross_site_ne_ne"}:
            continue
        buckets = [co_degree] if edge["relation"] == "co_site_ne_ne" else [cross_degree]
        for degree_map in buckets:
            degree_map[edge["source"]] = degree_map.get(edge["source"], 0) + 1
            degree_map[edge["target"]] = degree_map.get(edge["target"], 0) + 1
        total_degree[edge["source"]] = total_degree.get(edge["source"], 0) + 1
        total_degree[edge["target"]] = total_degree.get(edge["target"], 0) + 1
    return co_degree, cross_degree, total_degree


def _normalize_degree(values: dict[str, int], node_id: str) -> float:
    max_value = max(values.values(), default=0)
    if max_value == 0:
        return 0.0
    return float(values.get(node_id, 0)) / float(max_value)


def _base_ne_feature(
    node: dict,
    fault_distance_map: dict[str, int],
    an_distance_map: dict[str, int],
    co_degree: dict[str, int],
    cross_degree: dict[str, int],
    total_degree: dict[str, int],
) -> np.ndarray:
    feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    feature[0] = float(bool(node["is_an"]))
    feature[1 + NE_SUBTYPE_IDS[node["type"]]] = 1.0
    feature[4 + ROLE_IDS["ne"]] = 1.0
    feature[12:20] = bucketize_distance(fault_distance_map.get(node["id"]))
    feature[20:28] = bucketize_distance(an_distance_map.get(node["id"]))
    feature[28] = _normalize_degree(co_degree, node["id"])
    feature[29] = _normalize_degree(cross_degree, node["id"])
    feature[30] = _normalize_degree(total_degree, node["id"])
    feature[31] = float(bool(node["is_fault_or_risk_anchor"]))
    return feature


def _alarm_feature(alarm_name: str) -> np.ndarray:
    definition = ALARM_DEFINITIONS[alarm_name]
    feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    feature[1 + NE_SUBTYPE_IDS[str(definition["node_type"])]] = 1.0
    feature[4 + ROLE_IDS["alarm"]] = 1.0
    feature[7 + ALARM_IDS[alarm_name]] = 1.0
    feature[11] = math.log1p(int(definition["domain"])) / math.log1p(MAX_DOMAIN)
    feature[12:20] = bucketize_distance(None, is_na=True)
    feature[20:28] = bucketize_distance(None, is_na=True)
    return feature


def _alarm_entity_feature(entity: dict, ne_feature: np.ndarray) -> np.ndarray:
    feature = np.array(ne_feature, copy=True)
    feature[4:7] = 0.0
    feature[4 + ROLE_IDS["alarm_entity"]] = 1.0
    feature[7:11] = 0.0
    feature[7 + int(entity["alarm_id"])] = 1.0
    feature[11] = math.log1p(int(entity["domain"])) / math.log1p(MAX_DOMAIN)
    return feature


def build_feature_bundle(sample: dict) -> dict:
    """Create node-level features and metadata for one sample."""

    topology = build_ne_topology(sample)
    fault_distance_map = _distance_map(
        topology,
        [node["id"] for node in sample["nodes"] if node["is_fault_or_risk_anchor"]],
    )
    an_distance_map = _distance_map(
        topology,
        [node["id"] for node in sample["nodes"] if node["is_an"]],
    )
    co_degree, cross_degree, total_degree = _degree_maps(sample)

    ne_nodes = [dict(node) for node in sample["nodes"]]
    alarm_entities = [dict(entity) for entity in sample["alarm_entities"]]
    alarm_nodes = [
        {
            "id": f"alarm:{alarm_name}",
            "alarm_name": alarm_name,
        }
        for alarm_name in ALARM_IDS
    ]

    node_ids: list[str] = []
    node_features: list[np.ndarray] = []
    node_type: list[int] = []
    ne_id_to_index: dict[str, int] = {}

    for node in ne_nodes:
        index = len(node_ids)
        node_ids.append(node["id"])
        node_features.append(
            _base_ne_feature(
                node=node,
                fault_distance_map=fault_distance_map,
                an_distance_map=an_distance_map,
                co_degree=co_degree,
                cross_degree=cross_degree,
                total_degree=total_degree,
            )
        )
        node_type.append(0)
        ne_id_to_index[node["id"]] = index

    alarm_entity_ids: list[str] = []
    ae_node_indices: list[int] = []
    ae_owner_ne_indices: list[int] = []
    labels: list[float] = []
    owner_is_an: list[bool] = []
    owner_is_fault_or_risk_anchor: list[bool] = []
    owner_is_padding: list[bool] = []
    trainable_mask: list[bool] = []

    for entity in alarm_entities:
        index = len(node_ids)
        owner_index = ne_id_to_index[entity["ne_id"]]
        node_ids.append(entity["id"])
        node_features.append(_alarm_entity_feature(entity, node_features[owner_index]))
        node_type.append(1)
        alarm_entity_ids.append(entity["id"])
        ae_node_indices.append(index)
        ae_owner_ne_indices.append(owner_index)
        labels.append(float(entity["label"]))
        owner_is_an.append(bool(entity["owner_is_an"]))
        owner_is_fault_or_risk_anchor.append(bool(entity["owner_is_fault_or_risk_anchor"]))
        owner_is_padding.append(bool(entity["owner_is_padding"]))
        trainable_mask.append(
            bool(entity["is_trainable_alarm"])
            and not bool(entity["owner_is_fault_or_risk_anchor"])
            and not bool(entity["owner_is_an"])
            and not bool(entity["owner_is_padding"])
        )

    alarm_name_to_index: dict[str, int] = {}
    for alarm_node in alarm_nodes:
        index = len(node_ids)
        node_ids.append(alarm_node["id"])
        node_features.append(_alarm_feature(alarm_node["alarm_name"]))
        node_type.append(2)
        alarm_name_to_index[alarm_node["alarm_name"]] = index

    return {
        "node_ids": node_ids,
        "node_features": np.stack(node_features).astype(np.float32),
        "node_type": np.asarray(node_type, dtype=np.int64),
        "ne_id_to_index": ne_id_to_index,
        "alarm_entity_ids": alarm_entity_ids,
        "ae_node_indices": ae_node_indices,
        "ae_owner_ne_indices": ae_owner_ne_indices,
        "labels": labels,
        "owner_is_an": owner_is_an,
        "owner_is_fault_or_risk_anchor": owner_is_fault_or_risk_anchor,
        "owner_is_padding": owner_is_padding,
        "trainable_mask": trainable_mask,
        "alarm_name_to_index": alarm_name_to_index,
    }
