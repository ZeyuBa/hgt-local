"""Configuration for the HGT model and runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from transformers import PretrainedConfig

from training_data.topo_generator import TopologyGenerationConfig


DATA_SPLITS = ("train", "val", "test")
RUN_MODE_CHOICES = ("full", "smoke")
EXECUTION_MODE_CHOICES = ("train", "inference")
REQUIRED_TEST_METRIC_KEYS = ("precision", "recall", "f1")
SMOKE_MIN_F1 = 0.60
SMOKE_SUCCESS_PROMISE = "<promise>COMPLETE</promise>"

TRAIN_HISTORY_FILENAME = "train_history.json"
VAL_HISTORY_FILENAME = "val_history.json"
TEST_METRICS_FILENAME = "test_metrics.json"


def checkpoint_filename(run_mode: str, *, kind: str) -> str:
    return f"{run_mode}-{kind}.pt"


def summary_filename(run_mode: str) -> str:
    return f"{run_mode}-summary.json"


def transformed_split_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / f"transformed_{split_name}.json"


class HGTConfig(PretrainedConfig):
    """Configuration wrapper for the pyHGT-based link prediction model."""

    model_type = "alarm-hgt"

    def __init__(
        self,
        in_dim: int = 32,
        n_hid: int = 64,
        num_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.2,
        num_types: int = 3,
        num_relations: int = 9,
        conv_name: str = "hgt",
        use_rte: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_types = num_types
        self.num_relations = num_relations
        self.conv_name = conv_name
        self.use_rte = use_rte


class RuntimeConfigError(ValueError):
    """Raised when the runtime YAML contract is missing required fields."""


IntOrRange = int | tuple[int, int]


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"Invalid config section at {path}: expected a mapping")
    return value


def _require_field(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise RuntimeConfigError(f"Missing required config field: {path}.{key}")
    return mapping[key]


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeConfigError(f"Invalid config value at {path}: expected an integer")
    if minimum is not None and value < minimum:
        raise RuntimeConfigError(
            f"Invalid config value at {path}: expected an integer >= {minimum}"
        )
    return value


def _require_float(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigError(f"Invalid config value at {path}: expected a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise RuntimeConfigError(f"Invalid config value at {path}: expected a number >= {minimum}")
    if maximum is not None and number > maximum:
        raise RuntimeConfigError(f"Invalid config value at {path}: expected a number <= {maximum}")
    return number


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeConfigError(f"Invalid config value at {path}: expected a boolean")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"Invalid config value at {path}: expected a non-empty string")
    return value


def _require_path(value: Any, path: str) -> Path:
    return Path(_require_string(value, path))


def _require_int_list(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeConfigError(f"Invalid config value at {path}: expected a non-empty list")
    return tuple(_require_int(item, f"{path}[{index}]", minimum=1) for index, item in enumerate(value))


def _require_int_or_range(value: Any, path: str, *, minimum: int = 0) -> IntOrRange:
    if isinstance(value, list):
        if len(value) != 2:
            raise RuntimeConfigError(f"Invalid config value at {path}: expected a 2-item integer range")
        low = _require_int(value[0], f"{path}[0]", minimum=minimum)
        high = _require_int(value[1], f"{path}[1]", minimum=minimum)
        if low > high:
            raise RuntimeConfigError(f"Invalid config value at {path}: expected low <= high")
        return (low, high)
    return _require_int(value, path, minimum=minimum)


@dataclass(frozen=True)
class SplitSizes:
    train: int
    val: int
    test: int

    def as_dict(self) -> dict[str, int]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class SyntheticRuntimeSection:
    output_dir: Path
    seed: int
    split_sizes: SplitSizes
    smoke_split_sizes: SplitSizes
    num_sites: IntOrRange
    wl_stations_per_site: IntOrRange
    fault_site_count: IntOrRange
    an_site_count: IntOrRange
    backup_link_probability: float
    noise_probability: float
    topology_mode: str

    def to_generation_config(self) -> TopologyGenerationConfig:
        return TopologyGenerationConfig(
            num_sites=self.num_sites,
            wl_stations_per_site=self.wl_stations_per_site,
            fault_site_count=self.fault_site_count,
            an_site_count=self.an_site_count,
            backup_link_probability=self.backup_link_probability,
            noise_probability=self.noise_probability,
            topology_mode=self.topology_mode,
        )


@dataclass(frozen=True)
class DatasetPathsSection:
    train: Path
    val: Path
    test: Path


@dataclass(frozen=True)
class BatchingSection:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    dataloader_drop_last: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool


@dataclass(frozen=True)
class ModelSection:
    in_dim: int
    n_hid: int
    num_layers: int
    n_heads: int
    dropout: float
    num_types: int
    num_relations: int
    conv_name: str
    use_rte: bool

    def to_model_config(self) -> HGTConfig:
        return HGTConfig(
            in_dim=self.in_dim,
            n_hid=self.n_hid,
            num_layers=self.num_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            num_types=self.num_types,
            num_relations=self.num_relations,
            conv_name=self.conv_name,
            use_rte=self.use_rte,
        )


@dataclass(frozen=True)
class MetricsSection:
    ks: tuple[int, ...]


@dataclass(frozen=True)
class TrainingArgsSection:
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    seed: int


@dataclass(frozen=True)
class OutputsSection:
    checkpoints_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    source_path: Path
    synthetic: SyntheticRuntimeSection
    dataset_paths: DatasetPathsSection
    batching: BatchingSection
    model: ModelSection
    metrics: MetricsSection
    training_args: TrainingArgsSection
    outputs: OutputsSection

    def to_model_config(self) -> HGTConfig:
        return self.model.to_model_config()

    def to_training_arguments(self, output_dir: str | Path | None = None):
        from .trainer import build_training_arguments

        resolved_output_dir = self.outputs.checkpoints_dir / "hf" if output_dir is None else output_dir
        return build_training_arguments(
            output_dir=resolved_output_dir,
            per_device_train_batch_size=self.batching.per_device_train_batch_size,
            per_device_eval_batch_size=self.batching.per_device_eval_batch_size,
            num_train_epochs=self.training_args.num_train_epochs,
            learning_rate=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
            warmup_ratio=self.training_args.warmup_ratio,
            logging_steps=self.training_args.logging_steps,
            seed=self.training_args.seed,
            dataloader_drop_last=self.batching.dataloader_drop_last,
            dataloader_num_workers=self.batching.dataloader_num_workers,
            dataloader_pin_memory=self.batching.dataloader_pin_memory,
        )

    def to_trainer_args(self):
        return self.to_training_arguments()


def _load_split_sizes(section: dict[str, Any], path: str) -> SplitSizes:
    return SplitSizes(
        train=_require_int(_require_field(section, "train", path), f"{path}.train", minimum=0),
        val=_require_int(_require_field(section, "val", path), f"{path}.val", minimum=0),
        test=_require_int(_require_field(section, "test", path), f"{path}.test", minimum=0),
    )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load and validate the runtime YAML config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    top_level = _require_mapping(raw_config, "root")

    synthetic_raw = _require_mapping(
        _require_field(top_level, "synthetic", "root"),
        "synthetic",
    )
    dataset_paths_raw = _require_mapping(
        _require_field(top_level, "dataset_paths", "root"),
        "dataset_paths",
    )
    batching_raw = _require_mapping(
        _require_field(top_level, "batching", "root"),
        "batching",
    )
    model_raw = _require_mapping(
        _require_field(top_level, "model", "root"),
        "model",
    )
    metrics_raw = _require_mapping(
        _require_field(top_level, "metrics", "root"),
        "metrics",
    )
    training_args_raw = _require_mapping(
        _require_field(top_level, "training_args", "root"),
        "training_args",
    )
    outputs_raw = _require_mapping(
        _require_field(top_level, "outputs", "root"),
        "outputs",
    )

    split_sizes_raw = _require_mapping(
        _require_field(synthetic_raw, "split_sizes", "synthetic"),
        "synthetic.split_sizes",
    )
    smoke_split_sizes_raw = _require_mapping(
        _require_field(synthetic_raw, "smoke_split_sizes", "synthetic"),
        "synthetic.smoke_split_sizes",
    )

    return RuntimeConfig(
        source_path=config_path,
        synthetic=SyntheticRuntimeSection(
            output_dir=_require_path(
                _require_field(synthetic_raw, "output_dir", "synthetic"),
                "synthetic.output_dir",
            ),
            seed=_require_int(
                _require_field(synthetic_raw, "seed", "synthetic"),
                "synthetic.seed",
            ),
            split_sizes=_load_split_sizes(split_sizes_raw, "synthetic.split_sizes"),
            smoke_split_sizes=_load_split_sizes(
                smoke_split_sizes_raw,
                "synthetic.smoke_split_sizes",
            ),
            num_sites=_require_int_or_range(
                _require_field(synthetic_raw, "num_sites", "synthetic"),
                "synthetic.num_sites",
                minimum=1,
            ),
            wl_stations_per_site=_require_int_or_range(
                _require_field(synthetic_raw, "wl_stations_per_site", "synthetic"),
                "synthetic.wl_stations_per_site",
                minimum=1,
            ),
            fault_site_count=_require_int_or_range(
                _require_field(synthetic_raw, "fault_site_count", "synthetic"),
                "synthetic.fault_site_count",
                minimum=0,
            ),
            an_site_count=_require_int_or_range(
                _require_field(synthetic_raw, "an_site_count", "synthetic"),
                "synthetic.an_site_count",
                minimum=0,
            ),
            backup_link_probability=_require_float(
                _require_field(synthetic_raw, "backup_link_probability", "synthetic"),
                "synthetic.backup_link_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            noise_probability=_require_float(
                _require_field(synthetic_raw, "noise_probability", "synthetic"),
                "synthetic.noise_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            topology_mode=_require_string(
                _require_field(synthetic_raw, "topology_mode", "synthetic"),
                "synthetic.topology_mode",
            ),
        ),
        dataset_paths=DatasetPathsSection(
            train=_require_path(
                _require_field(dataset_paths_raw, "train", "dataset_paths"),
                "dataset_paths.train",
            ),
            val=_require_path(
                _require_field(dataset_paths_raw, "val", "dataset_paths"),
                "dataset_paths.val",
            ),
            test=_require_path(
                _require_field(dataset_paths_raw, "test", "dataset_paths"),
                "dataset_paths.test",
            ),
        ),
        batching=BatchingSection(
            per_device_train_batch_size=_require_int(
                _require_field(batching_raw, "per_device_train_batch_size", "batching"),
                "batching.per_device_train_batch_size",
                minimum=1,
            ),
            per_device_eval_batch_size=_require_int(
                _require_field(batching_raw, "per_device_eval_batch_size", "batching"),
                "batching.per_device_eval_batch_size",
                minimum=1,
            ),
            dataloader_drop_last=_require_bool(
                _require_field(batching_raw, "dataloader_drop_last", "batching"),
                "batching.dataloader_drop_last",
            ),
            dataloader_num_workers=_require_int(
                _require_field(batching_raw, "dataloader_num_workers", "batching"),
                "batching.dataloader_num_workers",
                minimum=0,
            ),
            dataloader_pin_memory=_require_bool(
                _require_field(batching_raw, "dataloader_pin_memory", "batching"),
                "batching.dataloader_pin_memory",
            ),
        ),
        model=ModelSection(
            in_dim=_require_int(_require_field(model_raw, "in_dim", "model"), "model.in_dim", minimum=1),
            n_hid=_require_int(_require_field(model_raw, "n_hid", "model"), "model.n_hid", minimum=1),
            num_layers=_require_int(
                _require_field(model_raw, "num_layers", "model"),
                "model.num_layers",
                minimum=1,
            ),
            n_heads=_require_int(_require_field(model_raw, "n_heads", "model"), "model.n_heads", minimum=1),
            dropout=_require_float(
                _require_field(model_raw, "dropout", "model"),
                "model.dropout",
                minimum=0.0,
                maximum=1.0,
            ),
            num_types=_require_int(
                _require_field(model_raw, "num_types", "model"),
                "model.num_types",
                minimum=1,
            ),
            num_relations=_require_int(
                _require_field(model_raw, "num_relations", "model"),
                "model.num_relations",
                minimum=1,
            ),
            conv_name=_require_string(
                _require_field(model_raw, "conv_name", "model"),
                "model.conv_name",
            ),
            use_rte=_require_bool(
                _require_field(model_raw, "use_rte", "model"),
                "model.use_rte",
            ),
        ),
        metrics=MetricsSection(
            ks=_require_int_list(_require_field(metrics_raw, "ks", "metrics"), "metrics.ks"),
        ),
        training_args=TrainingArgsSection(
            num_train_epochs=_require_int(
                _require_field(training_args_raw, "num_train_epochs", "training_args"),
                "training_args.num_train_epochs",
                minimum=1,
            ),
            learning_rate=_require_float(
                _require_field(training_args_raw, "learning_rate", "training_args"),
                "training_args.learning_rate",
                minimum=0.0,
            ),
            weight_decay=_require_float(
                _require_field(training_args_raw, "weight_decay", "training_args"),
                "training_args.weight_decay",
                minimum=0.0,
            ),
            warmup_ratio=_require_float(
                _require_field(training_args_raw, "warmup_ratio", "training_args"),
                "training_args.warmup_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            logging_steps=_require_int(
                _require_field(training_args_raw, "logging_steps", "training_args"),
                "training_args.logging_steps",
                minimum=1,
            ),
            seed=_require_int(
                _require_field(training_args_raw, "seed", "training_args"),
                "training_args.seed",
            ),
        ),
        outputs=OutputsSection(
            checkpoints_dir=_require_path(
                _require_field(outputs_raw, "checkpoints_dir", "outputs"),
                "outputs.checkpoints_dir",
            ),
            results_dir=_require_path(
                _require_field(outputs_raw, "results_dir", "outputs"),
                "outputs.results_dir",
            ),
        ),
    )
