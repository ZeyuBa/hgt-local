import json
import random

import numpy as np
import torch

from transformers import Trainer

from src.dataset.hgt_dataset import HGTDataset
from src.dataset.collate import padding_collate_fn
from src.dataset.bucket_sampler import BucketBatchSampler
from src.models.hgt_for_link_prediction import HGTForLinkPrediction
from src.training.config import HGTConfig
from src.training.trainer import LinkPredictionTrainer, build_training_arguments
from training_data.topo_complete import complete_topology_sample
from training_data.topo_generator import TopologyGenerationConfig, generate_topology_sample


def _write_samples(tmp_path, samples):
    path = tmp_path / "trainer.json"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return path


def test_link_prediction_trainer_is_a_real_transformers_trainer_and_uses_custom_batching(tmp_path):
    config = TopologyGenerationConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    samples = [
        complete_topology_sample(generate_topology_sample(seed=60, config=config)),
        complete_topology_sample(generate_topology_sample(seed=61, config=config)),
    ]
    dataset = HGTDataset(_write_samples(tmp_path, samples))

    trainer = LinkPredictionTrainer(
        model=HGTForLinkPrediction(
            HGTConfig(n_hid=16, num_layers=2, n_heads=4, dropout=0.0, use_rte=False)
        ),
        args=build_training_arguments(
            output_dir=tmp_path / "trainer-output",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            warmup_ratio=0.0,
            logging_steps=1,
            seed=13,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        metric_ks=(1, 2),
    )

    dataloader = trainer.get_train_dataloader()

    assert isinstance(trainer, Trainer)
    assert isinstance(dataloader.batch_sampler, BucketBatchSampler)
    assert dataloader.collate_fn is padding_collate_fn


def test_link_prediction_trainer_preserves_existing_rng_state(tmp_path):
    config = TopologyGenerationConfig(
        num_sites=4,
        wl_stations_per_site=(1, 1),
        fault_site_count=(1, 1),
        an_site_count=(1, 1),
        backup_link_probability=0.0,
        noise_probability=0.0,
        topology_mode="chain",
    )
    samples = [
        complete_topology_sample(generate_topology_sample(seed=62, config=config)),
        complete_topology_sample(generate_topology_sample(seed=63, config=config)),
    ]
    dataset = HGTDataset(_write_samples(tmp_path, samples))

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    model = HGTForLinkPrediction(HGTConfig(n_hid=16, num_layers=2, n_heads=4, dropout=0.0, use_rte=False))

    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    LinkPredictionTrainer(
        model=model,
        args=build_training_arguments(
            output_dir=tmp_path / "trainer-output",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            warmup_ratio=0.0,
            logging_steps=1,
            seed=13,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        metric_ks=(1, 2),
    )

    restored_numpy_state = np.random.get_state()

    assert random.getstate() == random_state
    assert restored_numpy_state[0] == numpy_state[0]
    assert restored_numpy_state[2:] == numpy_state[2:]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
