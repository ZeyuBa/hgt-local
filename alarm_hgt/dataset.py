"""Dataset loading and tensorization for alarm HGT samples."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .constants import FORWARD_TO_REVERSE_RELATION, RELATION_TYPE_IDS
from .features import build_feature_bundle


class AlarmGraphDataset(Dataset):
    """Loads JSONL graph samples and turns them into HGT-ready tensors."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        bundle = build_feature_bundle(sample)

        edge_index: list[list[int]] = []
        edge_type: list[int] = []

        def add_edge(source_idx: int, target_idx: int, relation_name: str) -> None:
            edge_index.append([source_idx, target_idx])
            edge_type.append(RELATION_TYPE_IDS[relation_name])

        for edge in sample["edges"]:
            source_idx = bundle["ne_id_to_index"][edge["source"]]
            target_idx = bundle["ne_id_to_index"][edge["target"]]
            add_edge(source_idx, target_idx, edge["relation"])
            add_edge(target_idx, source_idx, FORWARD_TO_REVERSE_RELATION[edge["relation"]])

        for entity in sample["alarm_entities"]:
            owner_idx = bundle["ne_id_to_index"][entity["ne_id"]]
            ae_pos = bundle["alarm_entity_ids"].index(entity["id"])
            ae_idx = bundle["ae_node_indices"][ae_pos]
            alarm_idx = bundle["alarm_name_to_index"][entity["alarm_name"]]
            add_edge(owner_idx, ae_idx, "ne_alarm_entity")
            add_edge(ae_idx, owner_idx, "rev_ne_alarm_entity")
            add_edge(ae_idx, alarm_idx, "alarm_entity_alarm")
            add_edge(alarm_idx, ae_idx, "rev_alarm_entity_alarm")

        for node_index in range(len(bundle["node_ids"])):
            add_edge(node_index, node_index, "self")

        return {
            "sample_id": sample["sample_id"],
            "node_ids": bundle["node_ids"],
            "node_features": torch.tensor(bundle["node_features"], dtype=torch.float32),
            "node_type": torch.tensor(bundle["node_type"], dtype=torch.long),
            "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            "edge_type": torch.tensor(edge_type, dtype=torch.long),
            "edge_time": torch.zeros(len(edge_type), dtype=torch.long),
            "alarm_entity_ids": bundle["alarm_entity_ids"],
            "ae_node_indices": torch.tensor(bundle["ae_node_indices"], dtype=torch.long),
            "ae_owner_ne_indices": torch.tensor(bundle["ae_owner_ne_indices"], dtype=torch.long),
            "labels": torch.tensor(bundle["labels"], dtype=torch.float32),
            "owner_is_an": torch.tensor(bundle["owner_is_an"], dtype=torch.bool),
            "owner_is_fault_or_risk_anchor": torch.tensor(
                bundle["owner_is_fault_or_risk_anchor"], dtype=torch.bool
            ),
            "owner_is_padding": torch.tensor(bundle["owner_is_padding"], dtype=torch.bool),
            "trainable_mask": torch.tensor(bundle["trainable_mask"], dtype=torch.bool),
        }
