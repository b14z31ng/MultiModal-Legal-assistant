from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device,
):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (
        model,
        optimizer,
        checkpoint["epoch"],
    )