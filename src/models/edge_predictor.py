"""Bilinear edge predictor."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    """Bilinear edge scorer over normalized embeddings."""

    def __init__(self, n_hid: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_hid, n_hid))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = F.normalize(left, p=2, dim=-1)
        right = F.normalize(right, p=2, dim=-1)
        projected = torch.matmul(left, self.weight)
        return (projected * right).sum(dim=-1)

