from __future__ import annotations

from pathlib import Path

import torch

from src.models.multimodal_model import MultiModalModel


def load_multimodal_model(

    device="cuda",

):

    model = MultiModalModel()

    ####################################################
    # Load Vision Encoder
    ####################################################

    vision_path = Path(
        "artifacts/models/final/document_encoder.pt"
    )

    if vision_path.exists():

        state = torch.load(
            vision_path,
            map_location="cpu",
        )

        model.vision_encoder.load_state_dict(
            state,
        )

        print(
            "✓ Loaded Vision Encoder"
        )

    else:

        print(
            "Vision encoder not found."
        )

    ####################################################
    # Load Text Encoder
    ####################################################

    text_path = Path(
        "artifacts/models/final/text_encoder.pt"
    )

    if text_path.exists():

        state = torch.load(
            text_path,
            map_location="cpu",
        )

        model.text_encoder.load_state_dict(
            state,
        )

        print(
            "✓ Loaded Text Encoder"
        )

    else:

        print(
            "Text encoder not found."
        )

    ####################################################
    # Freeze Encoders
    ####################################################

    for parameter in model.vision_encoder.parameters():

        parameter.requires_grad = False

    for parameter in model.text_encoder.parameters():

        parameter.requires_grad = False

    model.eval()

    model.to(device)

    return model