import torch


def get_device(device_name: str = "cuda"):

    if (
        device_name == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_autocast_enabled(
    mixed_precision: bool
) -> bool:
    return (
        mixed_precision
        and torch.cuda.is_available()
    )


def clear_cuda_cache():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_grad_scaler(
    mixed_precision: bool
):
    if (
        mixed_precision
        and torch.cuda.is_available()
    ):
        return torch.amp.GradScaler("cuda")

    return None