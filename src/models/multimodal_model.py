from __future__ import annotations

import torch
import torch.nn as nn
from src.models.risk_head import RiskHead
from src.models.document_encoder import DocumentEncoder
from src.models.text_encoder import TextEncoder
from src.models.feature_fusion import FeatureFusion


class MultiModalModel(nn.Module):
    """
    Stage-1 Multimodal Model

    Uses:
        Vision Encoder
        ModernBERT Encoder

    Produces:
        Shared embedding
        Document logits
        Clause embedding
    """

    def __init__(

        self,

        vision_backbone="convnext_base.fb_in22k_ft_in1k",

        text_backbone="answerdotai/ModernBERT-base",

        document_classes=16,

        fusion_dim=1024,

        dropout=0.30,

    ):

        super().__init__()

        ####################################################
        # Encoders
        ####################################################

        self.vision_encoder = DocumentEncoder(

            backbone=vision_backbone,

            pretrained=False,

            dropout=dropout,

        )

        self.text_encoder = TextEncoder(

            model_name=text_backbone,

            dropout=dropout,

        )

        ####################################################
        # Fusion
        ####################################################

        self.fusion = FeatureFusion(

            vision_dim=self.vision_encoder.output_dim,

            text_dim=self.text_encoder.output_dim,

            hidden_dim=fusion_dim,

            dropout=dropout,

        )

        ####################################################
        # Document Classification Head
        ####################################################

        self.document_head = nn.Sequential(

            nn.LayerNorm(fusion_dim),

            nn.Linear(
                fusion_dim,
                fusion_dim // 2,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                fusion_dim // 2,
                document_classes,
            ),

        )
    ####################################################
        # Clause Head
        ####################################################

        self.clause_head = nn.Sequential(

            nn.LayerNorm(fusion_dim),

            nn.Linear(
                fusion_dim,
                512,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                41,
            ),

        )

        ####################################################
        # Risk Head
        ####################################################

        self.risk_head = RiskHead(

            input_dim=fusion_dim,

            hidden_dim=512,

            dropout=dropout,

        )
    def forward(

        self,

        image,

        input_ids,

        attention_mask,

    ):

        ####################################################
        # Feature Extraction
        ####################################################

        vision_features = self.vision_encoder(

            image,

        )

        text_features = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask,

        )

        ####################################################
        # Fusion
        ####################################################

        fused = self.fusion(

            vision_features,

            text_features,

        )

        ####################################################
        # Heads
        ####################################################

        document_logits = self.document_head(

            fused,

        )
        clause_logits = self.clause_head(
            fused,
        )

        risk_score = self.risk_head(
            fused,
        )
        return {

    "vision_features": vision_features,

    "text_features": text_features,

    "fused_features": fused,

    "document_logits": document_logits,

    "clause_logits": clause_logits,

    "risk_score": risk_score,

}