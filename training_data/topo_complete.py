"""Stage 3: expand topologies into complete heterogeneous graph samples."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Sequence

import networkx as nx

from src.training.config import DATA_SPLITS, transformed_split_path

from .topo_combiner import export_topology_splits
from .topo_generator import (
    TopologyGenerationConfig,
    active_graph,
    build_alarm_entities,
    build_ne_graph,
    deserialize_rng_state,
    generate_topology_sample,
    select_noise_sites,
    site_nodes,
)


MAX_REPRESENTATIVE_SAMPLE_ATTEMPTS = 10_000
REPRESENTATIVE_SMOKE_VAL_POSITIVE_OFFSET = 1
REPRESENTATIVE_SMOKE_TEST_POSITIVE_OFFSET = 5


def _annotate_labels(
    sample: dict,
    site_ids: list[str],
    rng: random.Random,
    forced_noise_sites: Sequence[str] | None,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    nodes = [dict(node) for node in sample["nodes"]]
    nodes_by_id = {node["id"]: node for node in nodes}
    graph = build_ne_graph(nodes, sample["edges"])

    labels: dict[str, int] = {}
    inactive_ne_ids: set[str] = set()
    blocked_edge_pairs: set[frozenset[str]] = set()

    for site_id in sample["fault_or_risk_sites"]:
        mode = sample["fault_modes"][site_id]
        if mode == "mains_failure":
            labels[f"mains_failure;phy_site:{site_id}"] = 1
            labels[f"device_powered_off;router:{site_id}"] = 1
            inactive_ne_ids.add(f"router:{site_id}")
            nodes_by_id[f"phy_site:{site_id}"]["is_outage"] = True
            nodes_by_id[f"router:{site_id}"]["is_outage"] = True
            for station in site_nodes(nodes, site_id, "wl_station"):
                labels[f"ne_is_disconnected;{station['id']}"] = 1
                station["is_outage"] = True
        elif mode == "link_down":
            upstream = sample["primary_upstream_by_site"][site_id]
            if upstream is None:
                raise ValueError(f"site {site_id} has no upstream for link_down")
            labels[f"link_down;router:{site_id}"] = 1
            labels[f"link_down;router:{upstream}"] = 1
            blocked_edge_pairs.add(frozenset({f"router:{site_id}", f"router:{upstream}"}))
            for station in site_nodes(nodes, site_id, "wl_station"):
                labels[f"ne_is_disconnected;{station['id']}"] = 1
                station["is_outage"] = True
        else:
            raise ValueError(f"unsupported fault mode: {mode}")

    active_topology = active_graph(graph, inactive_ne_ids, blocked_edge_pairs)
    noise_sites = select_noise_sites(
        site_ids=site_ids,
        an_sites=set(sample["an_sites"]),
        fault_or_risk_sites=sample["fault_or_risk_sites"],
        active_topology=active_topology,
        nodes=nodes,
        rng=rng,
        probability=float(sample.get("noise_probability", 0.0)),
        forced_noise_sites=forced_noise_sites,
    )
    an_roots = [node["id"] for node in nodes if node["site_id"] in sample["an_sites"] and active_topology.has_node(node["id"])]
    for node in nodes:
        if node["type"] != "wl_station":
            continue
        if labels.get(f"ne_is_disconnected;{node['id']}", 0) == 1:
            continue
        if not active_topology.has_node(node["id"]):
            labels[f"ne_is_disconnected;{node['id']}"] = 1
            node["is_outage"] = True
            continue
        if not any(nx.has_path(active_topology, node["id"], root) for root in an_roots):
            labels[f"ne_is_disconnected;{node['id']}"] = 1
            node["is_outage"] = True

    for site_id in noise_sites:
        labels[f"mains_failure;phy_site:{site_id}"] = 1

    alarm_entities = build_alarm_entities(nodes, labels)
    logical_failures = {
        "inactive_ne_ids": sorted(inactive_ne_ids),
        "blocked_edge_pairs": sorted(sorted(pair) for pair in blocked_edge_pairs),
        "noise_sites": sorted(noise_sites),
    }
    return nodes, alarm_entities, logical_failures


def complete_topology_sample(
    sample: dict,
    *,
    forced_noise_sites: Sequence[str] | None = None,
) -> dict:
    """Expand a base topology sample into a full heterogeneous graph sample."""

    completed = dict(sample)
    annotation_rng_state = completed.pop("_annotation_rng_state", None)
    rng = random.Random()
    if annotation_rng_state is None:
        rng.seed(int(completed["seed"]))
    else:
        rng.setstate(deserialize_rng_state(annotation_rng_state))
    site_ids = sorted({node["site_id"] for node in completed["nodes"]})
    nodes, alarm_entities, logical_failures = _annotate_labels(
        completed,
        site_ids=site_ids,
        rng=rng,
        forced_noise_sites=forced_noise_sites,
    )
    completed["nodes"] = nodes
    completed["alarm_entities"] = alarm_entities
    completed["logical_failures"] = logical_failures
    completed["noise_sites"] = logical_failures["noise_sites"]
    return completed


def generate_complete_sample(
    seed: int,
    config: TopologyGenerationConfig | None = None,
    forced_an_sites: Sequence[str] | None = None,
    forced_fault_sites: Sequence[str] | None = None,
    forced_fault_modes: dict[str, str] | None = None,
    forced_noise_sites: Sequence[str] | None = None,
) -> dict:
    """One-shot helper for tests and scripts."""

    base_sample = generate_topology_sample(
        seed=seed,
        config=config,
        forced_an_sites=forced_an_sites,
        forced_fault_sites=forced_fault_sites,
        forced_fault_modes=forced_fault_modes,
    )
    return complete_topology_sample(base_sample, forced_noise_sites=forced_noise_sites)


def _trainable_positive_count(sample: dict) -> int:
    return sum(
        int(entity["label"])
        for entity in sample["alarm_entities"]
        if entity["is_trainable_alarm"]
        and not entity["owner_is_fault_or_risk_anchor"]
        and not entity["owner_is_an"]
        and not entity["owner_is_padding"]
    )


def _representative_positive_targets(split_sizes: dict[str, int]) -> dict[str, int]:
    train_size = int(split_sizes.get("train", 0))
    return {
        "train": min(train_size, max(1, train_size // 2)) if train_size > 0 else 0,
        "val": 1 if int(split_sizes.get("val", 0)) > 0 else 0,
        "test": 1 if int(split_sizes.get("test", 0)) > 0 else 0,
    }


def _collect_representative_samples(
    split_sizes: dict[str, int],
    *,
    config: TopologyGenerationConfig,
    seed: int,
) -> dict[str, list[dict]]:
    positive_targets = _representative_positive_targets(split_sizes)
    train_positive_end = positive_targets["train"]
    evaluation_offsets: dict[str, int] = {}
    if positive_targets["val"] > 0:
        evaluation_offsets["val"] = REPRESENTATIVE_SMOKE_VAL_POSITIVE_OFFSET
    if positive_targets["test"] > 0:
        evaluation_offsets["test"] = REPRESENTATIVE_SMOKE_TEST_POSITIVE_OFFSET
    required_positive_count = train_positive_end
    for split_name, offset in evaluation_offsets.items():
        required_positive_count = max(
            required_positive_count,
            train_positive_end + offset + positive_targets[split_name],
        )
    total_samples = sum(int(split_sizes.get(split_name, 0)) for split_name in DATA_SPLITS)
    required_negative_count = total_samples - sum(positive_targets.values())

    positives: list[dict] = []
    negatives: list[dict] = []
    current_seed = seed

    while len(positives) < required_positive_count or len(negatives) < required_negative_count:
        if current_seed - seed >= MAX_REPRESENTATIVE_SAMPLE_ATTEMPTS:
            raise ValueError(
                "unable to build representative smoke splits within "
                f"{MAX_REPRESENTATIVE_SAMPLE_ATTEMPTS} generated samples"
            )
        sample = complete_topology_sample(generate_topology_sample(seed=current_seed, config=config))
        if _trainable_positive_count(sample) > 0:
            positives.append(sample)
        else:
            negatives.append(sample)
        current_seed += 1

    selected_samples: dict[str, list[dict]] = {}
    negative_index = 0
    for split_name in DATA_SPLITS:
        split_sample_count = int(split_sizes.get(split_name, 0))
        chosen: list[dict] = []
        if split_name == "train":
            chosen.extend(positives[:train_positive_end])
        else:
            for index in range(positive_targets[split_name]):
                positive_index = train_positive_end + evaluation_offsets[split_name] + index
                chosen.append(positives[positive_index])
        while len(chosen) < split_sample_count:
            chosen.append(negatives[negative_index])
            negative_index += 1
        selected_samples[split_name] = chosen
    return selected_samples


def complete_topology_file(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(json.dumps(complete_topology_sample(json.loads(line)), ensure_ascii=True) + "\n")
    return output_path


def export_complete_splits(
    output_dir: str | Path,
    split_sizes: dict[str, int],
    config: TopologyGenerationConfig | None = None,
    seed: int = 0,
    output_paths: dict[str, str | Path] | None = None,
    representative_smoke: bool = False,
) -> dict[str, Path]:
    """Write completed train/val/test JSONL samples."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or TopologyGenerationConfig()

    exported_paths = {
        split_name: Path(output_paths[split_name]) if output_paths and split_name in output_paths else transformed_split_path(output_dir, split_name)
        for split_name in DATA_SPLITS
    }
    for path in exported_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if representative_smoke:
        selected_samples = _collect_representative_samples(split_sizes, config=config, seed=seed)
        for split_name, path in exported_paths.items():
            with path.open("w", encoding="utf-8") as handle:
                for sample in selected_samples[split_name]:
                    handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
        return exported_paths

    base_paths = export_topology_splits(
        output_dir=output_dir,
        split_sizes=split_sizes,
        config=config,
        seed=seed,
    )
    for split_name in DATA_SPLITS:
        complete_topology_file(base_paths[split_name], exported_paths[split_name])
    return exported_paths


export_synthetic_splits = export_complete_splits
