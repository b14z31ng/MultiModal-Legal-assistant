import torch
import torch.nn as nn

from src.models.document_encoder import DocumentEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import CrossAttentionFusion


class RiskAuditorFusionModel(nn.Module):
    """
    Production Multimodal Legal Risk Auditor

        Image
            │
      ConvNeXt Base
            │
            ▼
      1024 Vision Features
            │
            │
            ▼
    Cross Attention Fusion
            ▲
            │
      1024 Text Features
            ▲
     DeBERTa-v3-Large
            ▲
            │
      Contract Text

            │
            ▼

      Multi-label Classifier

            │
            ▼

         41 Risks
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        print("=" * 60)
        print("Building Multimodal Risk Auditor")
        print("=" * 60)

        ####################################################
        # Vision Encoder
        ####################################################

        self.vision_encoder = DocumentEncoder(

            backbone=config.vision.backbone,

            pretrained=config.vision.pretrained,

            dropout=config.classifier.dropout,

        )

        ####################################################
        # Text Encoder
        ####################################################

        self.text_encoder = TextEncoder(

            model_name=config.text.encoder,

            dropout=config.classifier.dropout,

        )

        ####################################################
        # Fusion
        ####################################################

        self.fusion = CrossAttentionFusion(

            vision_dim=self.vision_encoder.output_dim,

            text_dim=self.text_encoder.output_dim,

            hidden_dim=config.fusion.hidden_dim,

            num_heads=config.fusion.num_heads,

            dropout=config.classifier.dropout,

        )

        ####################################################
        # Multi-label Classifier
        ####################################################

        hidden = config.fusion.hidden_dim

        self.classifier = nn.Sequential(

            nn.LayerNorm(hidden),

            nn.Linear(
                hidden,
                hidden,
            ),

            nn.GELU(),

            nn.Dropout(
                config.classifier.dropout,
            ),

            nn.Linear(
                hidden,
                hidden // 2,
            ),

            nn.GELU(),

            nn.Dropout(
                config.classifier.dropout,
            ),

            nn.Linear(
                hidden // 2,
                41,
            ),

        )

    def forward(

        self,

        pixel_values,

        input_ids,

        attention_mask,

        return_features=False,

    ):

        ####################################################
        # Vision
        ####################################################

        vision_features = self.vision_encoder(

            pixel_values,

        )

        ####################################################
        # Text
        ####################################################

        text_features = self.text_encoder(

            input_ids,

            attention_mask,

        )

        ####################################################
        # Fusion
        ####################################################

        fused = self.fusion(

            vision_features,

            text_features,

        )

        ####################################################
        # Prediction
        ####################################################

        logits = self.classifier(

            fused,

        )

        if return_features:

            return (

                logits,

                fused,

            )

        return logits