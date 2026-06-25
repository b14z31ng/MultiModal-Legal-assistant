import torch.nn as nn


def build_loss(task="multilabel"):
    """
    Build the appropriate loss function.

    Parameters
    ----------
    task : str

        "vision"
            Single-label document classification.

        "multilabel"
            Multi-label legal risk classification.
    """

    if task == "vision":

        return nn.CrossEntropyLoss()

    if task == "multilabel":

        return nn.BCEWithLogitsLoss()

    raise ValueError(
        f"Unknown task: {task}"
    )