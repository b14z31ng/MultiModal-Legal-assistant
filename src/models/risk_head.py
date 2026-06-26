from __future__ import annotations

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """
    Predicts overall legal risk.

    Input:
        1024-dimensional fused embedding

    Output:
        Risk score
    """

    def __init__(

        self,

        input_dim=1024,

        hidden_dim=512,

        dropout=0.30,

    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 128),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(128, 1),

            nn.Sigmoid(),

        )

    def forward(

        self,

        fused_features,

    ):

        return self.network(
            fused_features,
        )