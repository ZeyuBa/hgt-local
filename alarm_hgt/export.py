"""Synthetic dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .synthetic import SyntheticGraphConfig, generate_sample


MAX_REPRESENTATIVE_SAMPLE_ATTEMPTS = 10_000
REPRESENTATIVE_SMOKE_VAL_POSITIVE_OFFSET = 1
REPRESENTATIVE_SMOKE_TEST_POSITIVE_OFFSET = 5


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
    targets = {
        "train": min(train_size, max(1, train_size // 2)) if train_size > 0 else 0,
        "val": 1 if int(split_sizes.get("val", 0)) > 0 else 0,
        "test": 1 if int(split_sizes.get("test", 0)) > 0 else 0,
    }
    return targets


def _collect_representative_samples(
    split_sizes: dict[str, int],
    *,
    config: SyntheticGraphConfig,
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
    total_samples = sum(int(split_sizes.get(split_name, 0)) for split_name in ("train", "val", "test"))
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
        sample = generate_sample(seed=current_seed, config=config)
        if _trainable_positive_count(sample) > 0:
            positives.append(sample)
        else:
            negatives.append(sample)
        current_seed += 1

    selected_samples: dict[str, list[dict]] = {}
    negative_index = 0
    for split_name in ("train", "val", "test"):
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


def export_synthetic_splits(
    output_dir: str | Path,
    split_sizes: dict[str, int],
    config: SyntheticGraphConfig | None = None,
    seed: int = 0,
    output_paths: dict[str, str | Path] | None = None,
    representative_smoke: bool = False,
) -> dict[str, Path]:
    """Export train/val/test splits as JSONL data in `.json` files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config or SyntheticGraphConfig()
    exported_paths: dict[str, Path] = {}
    selected_samples: dict[str, list[dict]] | None = None
    if representative_smoke:
        selected_samples = _collect_representative_samples(
            split_sizes,
            config=config,
            seed=seed,
        )
    current_seed = seed

    for split_name in ("train", "val", "test"):
        sample_count = split_sizes.get(split_name, 0)
        if output_paths and split_name in output_paths:
            path = Path(output_paths[split_name])
        else:
            path = output_dir / f"transformed_{split_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        exported_paths[split_name] = path
        with path.open("w", encoding="utf-8") as handle:
            if selected_samples is not None:
                for sample in selected_samples[split_name]:
                    handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
            else:
                for _ in range(sample_count):
                    sample = generate_sample(seed=current_seed, config=config)
                    handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                    current_seed += 1
    return exported_paths
