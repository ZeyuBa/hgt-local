import json
import math

import numpy as np
import pytest
import torch
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


class _StaticMaskDataset:
    def __init__(self, dataset, *, trainable_value: bool):
        self.dataset = dataset
        self.trainable_value = trainable_value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["trainable_mask"] = torch.full_like(item["trainable_mask"], self.trainable_value)
        return item


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


def test_trainer_train_and_validate_records_finite_epoch_histories_and_serializes_json(tmp_path):
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

    result = trainer.train_and_validate(
        num_epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        results_dir=tmp_path,
    )

    assert [entry["epoch"] for entry in result.train_history] == [1, 2]
    assert [entry["epoch"] for entry in result.val_history] == [1, 2]
    assert all(math.isfinite(entry["train_loss"]) for entry in result.train_history)
    assert all(math.isfinite(entry["val_loss"]) for entry in result.val_history)
    assert result.train_history_path == tmp_path / "train_history.json"
    assert result.val_history_path == tmp_path / "val_history.json"

    train_payload = json.loads(result.train_history_path.read_text(encoding="utf-8"))
    val_payload = json.loads(result.val_history_path.read_text(encoding="utf-8"))

    assert train_payload["split"] == "train"
    assert train_payload["metric"] == "train_loss"
    assert [entry["epoch"] for entry in train_payload["history"]] == [1, 2]
    assert all(math.isfinite(entry["train_loss"]) for entry in train_payload["history"])

    assert val_payload["split"] == "val"
    assert val_payload["metric"] == "val_loss"
    assert [entry["epoch"] for entry in val_payload["history"]] == [1, 2]
    assert all(math.isfinite(entry["val_loss"]) for entry in val_payload["history"])


def test_trainer_train_and_validate_tracks_best_validation_checkpoint_state(tmp_path):
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

    result = trainer.train_and_validate(
        num_epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        results_dir=tmp_path,
    )

    expected_best = min(result.val_history, key=lambda entry: entry["val_loss"])

    assert result.best_epoch == expected_best["epoch"]
    assert result.best_val_loss == pytest.approx(expected_best["val_loss"])
    assert result.best_model_state is not None
    assert result.best_model_state
    assert all(isinstance(value, torch.Tensor) for value in result.best_model_state.values())
    assert all(value.device.type == "cpu" for value in result.best_model_state.values())


def test_trainer_train_and_validate_fails_when_batch_has_no_trainable_positions(tmp_path):
    dataset = _StaticMaskDataset(_build_dataset(tmp_path), trainable_value=False)
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

    with pytest.raises(ValueError, match="no trainable positions"):
        trainer.train_and_validate(
            num_epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            results_dir=tmp_path,
        )


def test_trainer_train_and_validate_fails_on_non_finite_loss(tmp_path, monkeypatch):
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
    original_forward = trainer.model.forward

    def nan_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        return output.__class__(
            loss=torch.tensor(float("nan"), dtype=torch.float32, requires_grad=True),
            logits=output.logits,
            node_embeddings=output.node_embeddings,
        )

    monkeypatch.setattr(trainer.model, "forward", nan_forward)

    with pytest.raises(ValueError, match="non-finite loss"):
        trainer.train_and_validate(
            num_epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            results_dir=tmp_path,
        )
