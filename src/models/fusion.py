import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion block.

    Text attends to vision features.
    """

    def __init__(
        self,
        vision_dim=1024,
        text_dim=1024,
        hidden_dim=1024,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        ####################################################
        # Projection
        ####################################################

        self.vision_projection = nn.Linear(
            vision_dim,
            hidden_dim,
        )

        self.text_projection = nn.Linear(
            text_dim,
            hidden_dim,
        )

        ####################################################
        # Cross Attention
        ####################################################

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        ####################################################
        # Transformer Encoder
        ####################################################

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        ####################################################
        # Fusion Head
        ####################################################

        self.fusion_head = nn.Sequential(

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

        )

    @property
    def output_dim(self):

        return 1024

    def forward(
        self,
        vision_features,
        text_features,
    ):

        ####################################################
        # Projection
        ####################################################

        vision = self.vision_projection(
            vision_features,
        )

        text = self.text_projection(
            text_features,
        )

        ####################################################
        # Sequence
        ####################################################

        vision = vision.unsqueeze(1)

        text = text.unsqueeze(1)

        ####################################################
        # Cross Attention
        ####################################################

        attended, _ = self.cross_attention(

            query=text,

            key=vision,

            value=vision,

        )

        ####################################################
        # Transformer
        ####################################################

        fused = torch.cat(

            [

                attended,

                vision,

            ],

            dim=1,

        )

        fused = self.transformer(
            fused,
        )

        ####################################################
        # Pool
        ####################################################

        fused = fused.mean(
            dim=1,
        )

        ####################################################
        # Final Head
        ####################################################

        fused = self.fusion_head(
            fused,
        )

        return fused