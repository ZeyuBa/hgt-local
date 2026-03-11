from pathlib import Path

import pytest

from src.training.config import RuntimeConfigError, load_runtime_config


VALID_RUNTIME_CONFIG_YAML = """
synthetic:
  output_dir: data/synthetic
  seed: 7
  split_sizes:
    train: 24
    val: 8
    test: 8
  smoke_split_sizes:
    train: 6
    val: 2
    test: 2
  num_sites: [8, 16]
  wl_stations_per_site: [2, 4]
  fault_site_count: [1, 3]
  an_site_count: [1, 2]
  backup_link_probability: 0.18
  noise_probability: 0.08
  topology_mode: random_tree
dataset_paths:
  train: data/datasets/transformed_train.json
  val: data/datasets/transformed_val.json
  test: data/datasets/transformed_test.json
batching:
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  dataloader_drop_last: false
  dataloader_num_workers: 0
  dataloader_pin_memory: false
model:
  in_dim: 32
  n_hid: 64
  num_layers: 4
  n_heads: 4
  dropout: 0.2
  num_types: 3
  num_relations: 9
  conv_name: hgt
  use_rte: false
metrics:
  ks: [5, 10, 20, 50]
training_args:
  num_train_epochs: 3
  learning_rate: 0.0005
  weight_decay: 0.01
  warmup_ratio: 0.1
  logging_steps: 1
  seed: 13
outputs:
  checkpoints_dir: outputs/checkpoints
  results_dir: outputs/results
"""


def write_runtime_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content.strip() + "\n", encoding="utf-8")
    return config_path


def test_load_runtime_config_populates_typed_sections_without_hidden_overrides(tmp_path):
    config_path = write_runtime_config(tmp_path, VALID_RUNTIME_CONFIG_YAML)

    runtime_config = load_runtime_config(config_path)

    assert runtime_config.model.n_hid == 64
    assert runtime_config.model.num_layers == 4
    assert runtime_config.batching.per_device_train_batch_size == 4
    assert runtime_config.outputs.checkpoints_dir == Path("outputs/checkpoints")
    assert runtime_config.outputs.results_dir == Path("outputs/results")
    assert runtime_config.synthetic.smoke_split_sizes.train == 6
    assert runtime_config.metrics.ks == (5, 10, 20, 50)
    assert runtime_config.to_model_config().n_hid == 64
    assert runtime_config.to_trainer_args().per_device_train_batch_size == 4


@pytest.mark.parametrize(
    ("missing_snippet", "error_path"),
    [
        ("  results_dir: outputs/results\n", "outputs.results_dir"),
        ("  num_relations: 9\n", "model.num_relations"),
        ("  smoke_split_sizes:\n    train: 6\n    val: 2\n    test: 2\n", "synthetic.smoke_split_sizes"),
    ],
)
def test_load_runtime_config_rejects_missing_required_fields_with_readable_error(
    tmp_path,
    missing_snippet,
    error_path,
):
    config_path = write_runtime_config(
        tmp_path,
        VALID_RUNTIME_CONFIG_YAML.replace(missing_snippet, ""),
    )

    with pytest.raises(RuntimeConfigError, match=error_path):
        load_runtime_config(config_path)
