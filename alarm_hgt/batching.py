"""Mini-batch sampling and padding utilities."""

from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler

from .constants import HGT_NODE_TYPE_IDS, RELATION_TYPE_IDS


class BucketBatchSampler(Sampler[list[int]]):
    """Groups similarly sized graphs into the same mini-batch."""

    def __init__(
        self,
        sizes: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.sizes = list(sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        ordered = sorted(range(len(self.sizes)), key=lambda index: self.sizes[index])
        batches = [
            ordered[start : start + self.batch_size]
            for start in range(0, len(ordered), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        batch_count, remainder = divmod(len(self.sizes), self.batch_size)
        if remainder and not self.drop_last:
            return batch_count + 1
        return batch_count


def padding_collate_fn(items: list[dict]) -> dict:
    """Collate variable-sized graphs using isolated padding nodes."""

    if not items:
        raise ValueError("padding_collate_fn requires at least one item")

    feature_dim = items[0]["node_features"].shape[1]
    max_ae = max(item["labels"].numel() for item in items)

    batched_node_features: list[torch.Tensor] = []
    batched_node_type: list[torch.Tensor] = []
    batched_node_is_padding: list[torch.Tensor] = []
    batched_edge_index: list[torch.Tensor] = []
    batched_edge_type: list[torch.Tensor] = []
    batched_edge_time: list[torch.Tensor] = []
    batched_ae_node_indices: list[torch.Tensor] = []
    batched_ae_owner_ne_indices: list[torch.Tensor] = []
    batched_labels: list[torch.Tensor] = []
    batched_owner_is_an: list[torch.Tensor] = []
    batched_owner_is_fault_anchor: list[torch.Tensor] = []
    batched_owner_is_padding: list[torch.Tensor] = []
    batched_trainable_mask: list[torch.Tensor] = []
    sample_ids: list[str] = []

    node_offset = 0
    for item in items:
        sample_ids.append(item["sample_id"])

        local_features = item["node_features"]
        local_types = item["node_type"]
        local_padding_mask = torch.zeros(local_features.shape[0], dtype=torch.bool)
        local_edge_index = item["edge_index"]
        local_edge_type = item["edge_type"]
        local_edge_time = item["edge_time"]

        pad_ae_count = max_ae - item["labels"].numel()
        pad_ne_local_index: int | None = None
        pad_ae_local_indices = torch.empty(0, dtype=torch.long)

        if pad_ae_count > 0:
            pad_ne_local_index = local_features.shape[0]
            pad_ne_feature = torch.zeros((1, feature_dim), dtype=local_features.dtype)
            pad_ne_type = torch.tensor([HGT_NODE_TYPE_IDS["ne"]], dtype=torch.long)
            local_features = torch.cat([local_features, pad_ne_feature], dim=0)
            local_types = torch.cat([local_types, pad_ne_type], dim=0)
            local_padding_mask = torch.cat([local_padding_mask, torch.tensor([True])], dim=0)

            pad_ae_local_indices = torch.arange(
                start=local_features.shape[0],
                end=local_features.shape[0] + pad_ae_count,
                dtype=torch.long,
            )
            pad_ae_features = torch.zeros((pad_ae_count, feature_dim), dtype=local_features.dtype)
            pad_ae_types = torch.full(
                (pad_ae_count,),
                HGT_NODE_TYPE_IDS["alarm_entity"],
                dtype=torch.long,
            )
            local_features = torch.cat([local_features, pad_ae_features], dim=0)
            local_types = torch.cat([local_types, pad_ae_types], dim=0)
            local_padding_mask = torch.cat(
                [local_padding_mask, torch.ones(pad_ae_count, dtype=torch.bool)], dim=0
            )

            pad_self_indices = torch.arange(
                start=pad_ne_local_index,
                end=local_features.shape[0],
                dtype=torch.long,
            )
            pad_self_edges = torch.stack([pad_self_indices, pad_self_indices], dim=0)
            local_edge_index = torch.cat([local_edge_index, pad_self_edges], dim=1)
            local_edge_type = torch.cat(
                [
                    local_edge_type,
                    torch.full(
                        (pad_self_indices.numel(),),
                        RELATION_TYPE_IDS["self"],
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            local_edge_time = torch.cat(
                [local_edge_time, torch.zeros(pad_self_indices.numel(), dtype=torch.long)],
                dim=0,
            )

        batched_node_features.append(local_features)
        batched_node_type.append(local_types)
        batched_node_is_padding.append(local_padding_mask)
        batched_edge_index.append(local_edge_index + node_offset)
        batched_edge_type.append(local_edge_type)
        batched_edge_time.append(local_edge_time)

        if pad_ae_count > 0:
            ae_node_indices = torch.cat([item["ae_node_indices"], pad_ae_local_indices], dim=0)
            owner_indices = torch.cat(
                [
                    item["ae_owner_ne_indices"],
                    torch.full((pad_ae_count,), pad_ne_local_index, dtype=torch.long),
                ],
                dim=0,
            )
            labels = torch.cat([item["labels"], torch.zeros(pad_ae_count, dtype=torch.float32)], dim=0)
            owner_is_an = torch.cat(
                [item["owner_is_an"], torch.zeros(pad_ae_count, dtype=torch.bool)], dim=0
            )
            owner_is_fault_anchor = torch.cat(
                [
                    item["owner_is_fault_or_risk_anchor"],
                    torch.zeros(pad_ae_count, dtype=torch.bool),
                ],
                dim=0,
            )
            owner_is_padding = torch.cat(
                [item["owner_is_padding"], torch.ones(pad_ae_count, dtype=torch.bool)], dim=0
            )
            trainable_mask = torch.cat(
                [item["trainable_mask"], torch.zeros(pad_ae_count, dtype=torch.bool)], dim=0
            )
        else:
            ae_node_indices = item["ae_node_indices"]
            owner_indices = item["ae_owner_ne_indices"]
            labels = item["labels"]
            owner_is_an = item["owner_is_an"]
            owner_is_fault_anchor = item["owner_is_fault_or_risk_anchor"]
            owner_is_padding = item["owner_is_padding"]
            trainable_mask = item["trainable_mask"]

        batched_ae_node_indices.append(ae_node_indices + node_offset)
        batched_ae_owner_ne_indices.append(owner_indices + node_offset)
        batched_labels.append(labels)
        batched_owner_is_an.append(owner_is_an)
        batched_owner_is_fault_anchor.append(owner_is_fault_anchor)
        batched_owner_is_padding.append(owner_is_padding)
        batched_trainable_mask.append(trainable_mask)

        node_offset += local_features.shape[0]

    return {
        "sample_ids": sample_ids,
        "node_features": torch.cat(batched_node_features, dim=0),
        "node_type": torch.cat(batched_node_type, dim=0),
        "node_is_padding": torch.cat(batched_node_is_padding, dim=0),
        "edge_index": torch.cat(batched_edge_index, dim=1),
        "edge_type": torch.cat(batched_edge_type, dim=0),
        "edge_time": torch.cat(batched_edge_time, dim=0),
        "ae_node_indices": torch.stack(batched_ae_node_indices, dim=0),
        "ae_owner_ne_indices": torch.stack(batched_ae_owner_ne_indices, dim=0),
        "labels": torch.stack(batched_labels, dim=0),
        "owner_is_an": torch.stack(batched_owner_is_an, dim=0),
        "owner_is_fault_or_risk_anchor": torch.stack(batched_owner_is_fault_anchor, dim=0),
        "owner_is_padding": torch.stack(batched_owner_is_padding, dim=0),
        "trainable_mask": torch.stack(batched_trainable_mask, dim=0),
    }
