"""HGT encoder wrapper."""

from __future__ import annotations

from torch import nn

from pyHGT.model import GNN


class HGTEncoder(nn.Module):
    """Thin wrapper around pyHGT's heterogeneous encoder."""

    def __init__(
        self,
        *,
        in_dim: int,
        n_hid: int,
        num_types: int,
        num_relations: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        conv_name: str,
        use_rte: bool,
    ) -> None:
        super().__init__()
        self.gnn = GNN(
            in_dim=in_dim,
            n_hid=n_hid,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=n_heads,
            n_layers=num_layers,
            dropout=dropout,
            conv_name=conv_name,
            use_RTE=use_rte,
        )

    def forward(self, **kwargs):
        return self.gnn(**kwargs)

