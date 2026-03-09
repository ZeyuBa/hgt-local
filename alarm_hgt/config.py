"""Configuration for the alarm HGT link prediction model."""

from __future__ import annotations

from transformers import PretrainedConfig


class AlarmHGTConfig(PretrainedConfig):
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
