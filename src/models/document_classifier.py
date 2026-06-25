import torch.nn as nn

from src.models.document_encoder import DocumentEncoder


class DocumentClassifier(nn.Module):

    def __init__(
        self,
        backbone="convnext_base.fb_in22k_ft_in1k",
        pretrained=True,
        num_classes=6,
        hidden_dim=1024,
        dropout=0.30,
    ):

        super().__init__()

        self.encoder = DocumentEncoder(
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )

        feature_dim = self.encoder.output_dim

        self.classifier = nn.Sequential(

            nn.LayerNorm(feature_dim),

            nn.Linear(feature_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),

        )

        self._init_weights()

    def _init_weights(self):

        for m in self.classifier.modules():

            if isinstance(m, nn.Linear):

                nn.init.trunc_normal_(m.weight, std=0.02)

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_values,
        return_features=False,
    ):

        features = self.encoder(pixel_values)

        logits = self.classifier(features)

        if return_features:

            return logits, features

        return logits