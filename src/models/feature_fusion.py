from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Multimodal Feature Fusion

    Vision Features : 1024

    Text Features : 768

    Output : Shared Embedding
    """

    def __init__(

        self,

        vision_dim=1024,

        text_dim=768,

        hidden_dim=1024,

        dropout=0.30,

    ):

        super().__init__()

        ####################################################
        # Projection
        ####################################################

        self.vision_projection = nn.Sequential(

            nn.Linear(
                vision_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

        )

        self.text_projection = nn.Sequential(

            nn.Linear(
                text_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

        )

        ####################################################
        # Fusion
        ####################################################

        self.fusion = nn.Sequential(

            nn.Linear(

                hidden_dim * 2,

                hidden_dim,

            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(

                hidden_dim,

                hidden_dim,

            ),

        )

    def forward(

        self,

        vision_features,

        text_features,

    ):

        vision = self.vision_projection(

            vision_features,

        )

        text = self.text_projection(

            text_features,

        )

        fused = torch.cat(

            [

                vision,

                text,

            ],

            dim=1,

        )

        fused = self.fusion(

            fused,

        )

        return fused