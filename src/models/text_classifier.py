from __future__ import annotations

import torch.nn as nn

from src.models.text_encoder import TextEncoder


class TextClassifier(nn.Module):
    """
    ModernBERT + MLP classifier for CUAD.
    """

    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        num_labels=41,
        dropout=0.30,
    ):

        super().__init__()

        self.encoder = TextEncoder(
            model_name=model_name,
            dropout=dropout,
        )

        hidden = self.encoder.output_dim

        self.classifier = nn.Sequential(

            nn.LayerNorm(hidden),

            nn.Linear(hidden, hidden),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden, 512),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(512, num_labels),

        )

    def forward(

        self,

        input_ids,

        attention_mask,

    ):

        features = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask,

        )

        logits = self.classifier(
            features,
        )

        return logits