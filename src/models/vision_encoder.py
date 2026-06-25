from typing import Tuple

import torch.nn as nn

from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


def build_vision_encoder(
    model_name: str,
) -> Tuple[nn.Module, int]:

    if model_name == "resnet18":

        model = resnet18(
            weights=(
                ResNet18_Weights.DEFAULT
            )
        )

        model.fc = nn.Identity()

        return model, 512

    elif model_name == "resnet50":

        model = resnet50(
            weights=(
                ResNet50_Weights.DEFAULT
            )
        )

        model.fc = nn.Identity()

        return model, 2048

    raise ValueError(
        f"Unsupported vision model: "
        f"{model_name}"
    )