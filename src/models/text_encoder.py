from __future__ import annotations

import torch.nn as nn

from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    ModernBERT encoder.

    Input
    -----
    input_ids
    attention_mask

    Output
    ------
    768-dimensional CLS embedding
    """

    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        dropout=0.30,
    ):

        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
        )

        # Reduce VRAM usage during training
        self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(
            dropout,
        )

    @property
    def output_dim(self):

        return self.backbone.config.hidden_size

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.backbone(

            input_ids=input_ids,

            attention_mask=attention_mask,

        )

        ####################################################
        # Masked Mean Pooling
        ####################################################

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        summed = (token_embeddings * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)

        pooled = summed / counts

        pooled = self.dropout(
            pooled,
        )

        return pooled