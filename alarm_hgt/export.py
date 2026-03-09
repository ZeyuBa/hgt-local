"""Synthetic dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .synthetic import SyntheticGraphConfig, generate_sample


def export_synthetic_splits(
    output_dir: str | Path,
    split_sizes: dict[str, int],
    config: SyntheticGraphConfig | None = None,
    seed: int = 0,
    output_paths: dict[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Export train/val/test splits as JSONL data in `.json` files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config or SyntheticGraphConfig()
    current_seed = seed
    exported_paths: dict[str, Path] = {}

    for split_name in ("train", "val", "test"):
        sample_count = split_sizes.get(split_name, 0)
        if output_paths and split_name in output_paths:
            path = Path(output_paths[split_name])
        else:
            path = output_dir / f"transformed_{split_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        exported_paths[split_name] = path
        with path.open("w", encoding="utf-8") as handle:
            for _ in range(sample_count):
                sample = generate_sample(seed=current_seed, config=config)
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                current_seed += 1
    return exported_paths
