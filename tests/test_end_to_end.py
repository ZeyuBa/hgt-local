import torch

from alarm_hgt.batching import padding_collate_fn
from alarm_hgt.config import AlarmHGTConfig
from alarm_hgt.dataset import AlarmGraphDataset
from alarm_hgt.export import export_synthetic_splits
from alarm_hgt.modeling import HGTForLinkPrediction
from alarm_hgt.synthetic import SyntheticGraphConfig


def test_end_to_end_export_load_collate_and_forward(tmp_path):
    paths = export_synthetic_splits(
        output_dir=tmp_path,
        split_sizes={"train": 2, "val": 0, "test": 0},
        config=SyntheticGraphConfig(
            num_sites=4,
            wl_stations_per_site=(1, 1),
            fault_site_count=(1, 1),
            an_site_count=(1, 1),
            backup_link_probability=0.0,
            noise_probability=0.0,
            topology_mode="chain",
        ),
        seed=300,
    )
    dataset = AlarmGraphDataset(paths["train"])
    batch = padding_collate_fn([dataset[0], dataset[1]])
    model = HGTForLinkPrediction(
        AlarmHGTConfig(n_hid=16, num_layers=2, n_heads=4, dropout=0.0, use_rte=False)
    )

    output = model(**batch)

    assert output.logits.shape == batch["labels"].shape
    assert output.loss is not None
    assert torch.isfinite(output.loss)
