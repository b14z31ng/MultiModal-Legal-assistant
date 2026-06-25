from pathlib import Path

import torch


############################################################
# Early Stopping
############################################################


class EarlyStopping:
    """
    Early stopping based on validation loss.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.stop = False

    def step(
        self,
        loss,
    ):

        if loss < self.best_loss - self.min_delta:

            self.best_loss = loss

            self.counter = 0

            return True

        self.counter += 1

        if self.counter >= self.patience:

            self.stop = True

        return False


############################################################
# Checkpoint Helpers
############################################################


def save_checkpoint(

    model,

    optimizer,

    scheduler,

    epoch,

    loss,

    path,

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    torch.save(

        {

            "epoch": epoch,

            "loss": loss,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict()
                if scheduler is not None
                else None,

        },

        path,

    )


def load_checkpoint(

    model,

    optimizer,

    scheduler,

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

    if (

        scheduler is not None

        and

        checkpoint["scheduler_state_dict"] is not None

    ):

        scheduler.load_state_dict(

            checkpoint["scheduler_state_dict"]

        )

    return checkpoint["epoch"]


############################################################
# Export Helpers
############################################################


def export_model(

    model,

    path,

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    torch.save(

        model.state_dict(),

        path,

    )