"""HuggingFace-style wrapper for pyHGT link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.training.config import HGTConfig

from .edge_predictor import EdgePredictor
from .hgt import HGTEncoder


def focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for better handling of hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This down-weights easy examples and focuses on hard ones.
    """
    active_logits = logits[mask]
    active_labels = labels[mask]
    if active_labels.numel() == 0:
        return logits.sum() * 0.0

    # Compute probabilities
    probs = torch.sigmoid(active_logits)
    p_t = probs * active_labels + (1 - probs) * (1 - active_labels)

    # Compute focal weight
    focal_weight = (1 - p_t) ** gamma

    # Compute BCE with class balancing
    positive_count = int(active_labels.sum().item())
    negative_count = int(active_labels.numel() - positive_count)
    if positive_count > 0 and negative_count > 0:
        pos_weight = active_logits.new_tensor(float(negative_count / positive_count))
        bce = F.binary_cross_entropy_with_logits(
            active_logits,
            active_labels,
            pos_weight=pos_weight,
            reduction="none",
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            active_logits,
            active_labels,
            reduction="none",
        )

    # Apply focal weight
    return (focal_weight * bce).mean()


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy averaged only over masked positions."""

    active_logits = logits[mask]
    active_labels = labels[mask]
    if active_labels.numel() == 0:
        return logits.sum() * 0.0

    positive_count = int(active_labels.sum().item())
    negative_count = int(active_labels.numel() - positive_count)
    if positive_count > 0 and negative_count > 0:
        pos_weight = active_logits.new_tensor(float(negative_count / positive_count))
        return F.binary_cross_entropy_with_logits(
            active_logits,
            active_labels,
            pos_weight=pos_weight,
        )
    return F.binary_cross_entropy_with_logits(active_logits, active_labels)


@dataclass
class LinkPredictionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    node_embeddings: Optional[torch.Tensor] = None


class HGTForLinkPrediction(PreTrainedModel):
    """pyHGT encoder plus bilinear NE-AE predictor."""

    config_class = HGTConfig
    main_input_name = "node_features"

    def __init__(self, config: HGTConfig) -> None:
        super().__init__(config)
        self.encoder = HGTEncoder(
            in_dim=config.in_dim,
            n_hid=config.n_hid,
            num_types=config.num_types,
            num_relations=config.num_relations,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            conv_name=config.conv_name,
            use_rte=config.use_rte,
        )
        self.edge_predictor = EdgePredictor(config.n_hid)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, EdgePredictor):
            module.reset_parameters()

    def forward(
        self,
        node_features: torch.Tensor,
        node_type: torch.Tensor,
        edge_time: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        ae_node_indices: torch.Tensor,
        ae_owner_ne_indices: torch.Tensor,
        labels: torch.Tensor | None = None,
        trainable_mask: torch.Tensor | None = None,
        owner_is_an: torch.Tensor | None = None,
        owner_is_fault_or_risk_anchor: torch.Tensor | None = None,
        owner_is_padding: torch.Tensor | None = None,
        node_is_padding: torch.Tensor | None = None,
        sample_ids: list[str] | None = None,
        return_dict: bool | None = None,
    ) -> LinkPredictionOutput | tuple[torch.Tensor | None, torch.Tensor]:
        del owner_is_an, owner_is_fault_or_risk_anchor, owner_is_padding, node_is_padding, sample_ids

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        node_embeddings = self.encoder(
            node_feature=node_features,
            node_type=node_type,
            edge_time=edge_time,
            edge_index=edge_index,
            edge_type=edge_type,
        )
        ae_embeddings = node_embeddings[ae_node_indices]
        owner_embeddings = node_embeddings[ae_owner_ne_indices]
        logits = self.edge_predictor(owner_embeddings, ae_embeddings)

        loss = None
        if labels is not None and trainable_mask is not None:
            loss = focal_bce_loss(logits, labels, trainable_mask, gamma=2.0)

        if not return_dict:
            return loss, logits
        return LinkPredictionOutput(
            loss=loss,
            logits=logits,
            node_embeddings=node_embeddings,
        )

