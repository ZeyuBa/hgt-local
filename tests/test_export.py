import json

from alarm_hgt.dataset import AlarmGraphDataset
from alarm_hgt.export import export_synthetic_splits
from alarm_hgt.synthetic import SyntheticGraphConfig


def _positive_trainable_graph_count(dataset: AlarmGraphDataset) -> int:
    positive_graphs = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = sample["labels"][sample["trainable_mask"]]
        if int(labels.sum().item()) > 0:
            positive_graphs += 1
    return positive_graphs


def test_export_writes_train_val_test_json_files(tmp_path):
    paths = export_synthetic_splits(
        output_dir=tmp_path,
        split_sizes={"train": 3, "val": 2, "test": 1},
        config=SyntheticGraphConfig(
            num_sites=4,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
        seed=100,
    )

    assert paths["train"].name == "transformed_train.json"
    assert paths["val"].name == "transformed_val.json"
    assert paths["test"].name == "transformed_test.json"
    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["test"].exists()


def test_export_writes_one_sample_per_line_with_required_keys(tmp_path):
    paths = export_synthetic_splits(
        output_dir=tmp_path,
        split_sizes={"train": 2, "val": 0, "test": 0},
        config=SyntheticGraphConfig(
            num_sites=3,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
        seed=200,
    )

    lines = paths["train"].read_text().strip().splitlines()
    assert len(lines) == 2

    sample = json.loads(lines[0])
    assert {
        "sample_id",
        "seed",
        "fault_or_risk_sites",
        "nodes",
        "edges",
        "alarm_entities",
    }.issubset(sample)


def test_export_can_build_representative_smoke_splits_with_positive_coverage(tmp_path):
    paths = export_synthetic_splits(
        output_dir=tmp_path,
        split_sizes={"train": 12, "val": 4, "test": 4},
        config=SyntheticGraphConfig(),
        seed=7,
        representative_smoke=True,
    )

    train_dataset = AlarmGraphDataset(paths["train"])
    val_dataset = AlarmGraphDataset(paths["val"])
    test_dataset = AlarmGraphDataset(paths["test"])

    assert _positive_trainable_graph_count(train_dataset) >= 6
    assert _positive_trainable_graph_count(val_dataset) >= 1
    assert _positive_trainable_graph_count(test_dataset) >= 1
