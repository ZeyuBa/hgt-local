import json

import numpy as np
from transformers import EvalPrediction

from alarm_hgt.batching import BucketBatchSampler, padding_collate_fn
from alarm_hgt.config import AlarmHGTConfig
from alarm_hgt.dataset import AlarmGraphDataset
from alarm_hgt.modeling import HGTForLinkPrediction
from alarm_hgt.synthetic import SyntheticGraphConfig, generate_sample
from alarm_hgt.trainer import (
    LinkPredictionTrainer,
    LinkPredictionTrainerArgs,
    eval_prediction_to_metrics_input,
)


def _write_samples(tmp_path, samples):
    path = tmp_path / "trainer.json"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return path


def _build_dataset(tmp_path):
    config = SyntheticGraphConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    samples = [generate_sample(seed=60, config=config), generate_sample(seed=61, config=config)]
    return AlarmGraphDataset(_write_samples(tmp_path, samples))


def test_trainer_uses_bucket_sampler_and_padding_collator(tmp_path):
    dataset = _build_dataset(tmp_path)
    trainer = LinkPredictionTrainer(
        model=HGTForLinkPrediction(
            AlarmHGTConfig(n_hid=16, num_layers=2, n_heads=4, dropout=0.0, use_rte=False)
        ),
        args=LinkPredictionTrainerArgs(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    dataloader = trainer.get_train_dataloader()

    assert isinstance(dataloader.batch_sampler, BucketBatchSampler)
    assert dataloader.collate_fn is padding_collate_fn


def test_eval_prediction_adapter_extracts_metric_inputs():
    logits = np.array([[1.5, -1.5], [0.2, 0.8]], dtype=np.float32)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    trainable_mask = np.array([[True, False], [True, True]], dtype=bool)

    converted = eval_prediction_to_metrics_input(
        EvalPrediction(predictions=logits, label_ids=(labels, trainable_mask))
    )

    assert converted["logits"].shape == (2, 2)
    assert converted["labels"].shape == (2, 2)
    assert converted["trainable_mask"].shape == (2, 2)
    assert converted["trainable_mask"].dtype == np.bool_
    assert bool(converted["trainable_mask"][0, 1]) is False
