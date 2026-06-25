from pathlib import Path

import torch


############################################################
# Generic Checkpoint
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


############################################################
# Load Checkpoint
############################################################

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

    if optimizer is not None:

        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

    if (

        scheduler is not None

        and

        checkpoint["scheduler_state_dict"]

        is not None

    ):

        scheduler.load_state_dict(

            checkpoint["scheduler_state_dict"]

        )

    return checkpoint["epoch"]


############################################################
# Export Encoder
############################################################

def save_encoder(

    model,

    path,

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    torch.save(

        model.encoder.state_dict(),

        path,

    )


############################################################
# Export Classifier
############################################################

def save_classifier(

    model,

    path,

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    torch.save(

        model.classifier.state_dict(),

        path,

    )


############################################################
# Export Complete Vision Model
############################################################

def save_full_model(

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


############################################################
# Load Encoder
############################################################

def load_pretrained_encoder(

    encoder,

    checkpoint_path,

    device="cpu",

):

    state_dict = torch.load(

        checkpoint_path,

        map_location=device,

    )

    encoder.load_state_dict(

        state_dict,

        strict=False,

    )

    return encoder