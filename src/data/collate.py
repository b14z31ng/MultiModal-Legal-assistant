from torch.utils.data._utils.collate import default_collate


def multimodal_collate_fn(batch):
    """
    Generic collate function.

    Supports

    - Multimodal training
    - Vision training
    - Future inference

    Automatically handles optional images.
    """

    pixel_values = [
        sample["pixel_values"]
        for sample in batch
    ]

    ####################################################
    # Images
    ####################################################

    if pixel_values[0] is None:

        pixel_values = None

    else:

        pixel_values = default_collate(
            pixel_values
        )

    ####################################################
    # Text
    ####################################################

    input_ids = default_collate(

        [

            sample["input_ids"]

            for sample in batch

        ]

    )

    attention_mask = default_collate(

        [

            sample["attention_mask"]

            for sample in batch

        ]

    )

    ####################################################
    # Labels
    ####################################################

    labels = default_collate(

        [

            sample["labels"]

            for sample in batch

        ]

    )

    return {

        "pixel_values":

            pixel_values,

        "input_ids":

            input_ids,

        "attention_mask":

            attention_mask,

        "labels":

            labels,

    }