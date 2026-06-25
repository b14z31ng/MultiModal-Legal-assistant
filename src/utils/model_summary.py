import torch


def model_summary(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print(f"Frozen Parameters    : {total-trainable:,}")
    print("=" * 70)