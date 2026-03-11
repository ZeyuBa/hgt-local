"""Dataset builder helpers."""

from __future__ import annotations

from pathlib import Path

from .hgt_dataset import HGTDataset


def build_datasets(dataset_paths: dict[str, str | Path]) -> dict[str, HGTDataset]:
    return {split_name: HGTDataset(path) for split_name, path in dataset_paths.items()}
