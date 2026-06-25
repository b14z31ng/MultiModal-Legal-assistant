from pathlib import Path

import torch


############################################################
# Generic Checkpoint
############################################################

class CheckpointManager:

    """
    Generic checkpoint manager.
    """

    @staticmethod
    def save(

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

    ########################################################

    @staticmethod
    def load(

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

        return checkpoint


############################################################
# Vision Encoder
############################################################

def save_document_encoder(

    encoder,

    path,

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    torch.save(

        encoder.state_dict(),

        path,

    )


############################################################
# Vision Classifier
############################################################

def save_document_classifier(

    classifier,

    path,

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    torch.save(

        classifier.state_dict(),

        path,

    )


############################################################
# Multimodal Model
############################################################

def save_risk_auditor(

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
# Load Pretrained Vision Encoder
############################################################

def load_pretrained_encoder(

    encoder,

    checkpoint_path,

    device="cpu",

):

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

    )

    encoder.load_state_dict(

        checkpoint,

        strict=False,

    )

    return encoder