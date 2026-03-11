"""Stage 2: create split files of base topology samples."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.config import DATA_SPLITS

from .topo_generator import TopologyGenerationConfig, generate_topology_sample


def base_split_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / f"topology_{split_name}.json"


def export_topology_splits(
    output_dir: str | Path,
    split_sizes: dict[str, int],
    config: TopologyGenerationConfig | None = None,
    seed: int = 0,
    output_paths: dict[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Write base topology JSONL splits."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or TopologyGenerationConfig()

    current_seed = seed
    exported_paths: dict[str, Path] = {}
    for split_name in DATA_SPLITS:
        path = Path(output_paths[split_name]) if output_paths and split_name in output_paths else base_split_path(output_dir, split_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        exported_paths[split_name] = path
        with path.open("w", encoding="utf-8") as handle:
            for _ in range(int(split_sizes.get(split_name, 0))):
                sample = generate_topology_sample(seed=current_seed, config=config)
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                current_seed += 1
    return exported_paths
