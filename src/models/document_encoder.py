import timm
import torch.nn as nn


class DocumentEncoder(nn.Module):
    """
    Generic Vision Encoder.

    Supports every TIMM backbone.

    Examples

    - convnext_base.fb_in22k_ft_in1k
    - convnext_large.fb_in22k_ft_in1k
    - swin_large_patch4_window12_384
    - efficientnetv2_l
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        dropout: float = 0.30,
    ):

        super().__init__()

        print("=" * 70)
        print("Building Vision Encoder")
        print("=" * 70)
        print(f"Backbone   : {backbone}")
        print(f"Pretrained : {pretrained}")

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_dim = self.backbone.num_features

        self.dropout = nn.Dropout(dropout)

        print(f"Feature Dimension : {self.feature_dim}")
        print("=" * 70)

    def forward(self, pixel_values):

        features = self.backbone(pixel_values)

        features = self.dropout(features)

        return features

    @property
    def output_dim(self):

        return self.feature_dim