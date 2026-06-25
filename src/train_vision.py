from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src.utils.model_summary import model_summary

from torch.utils.data import DataLoader

from transformers import (
    get_cosine_schedule_with_warmup,
)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device

from src.data.vision_dataset import (
    VisionDataset,
)

from src.models.document_classifier import (
    DocumentClassifier,
)

from src.training.vision_trainer import (
    VisionTrainer,
)


############################################################
# Build DataLoader
############################################################

def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    config,
):

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if (
        torch.cuda.is_available()
        and getattr(
            config.training,
            "persistent_workers",
            False,
        )
    ):
        loader_kwargs["persistent_workers"] = True

    if (
        torch.cuda.is_available()
        and hasattr(
            config.training,
            "prefetch_factor",
        )
    ):
        loader_kwargs["prefetch_factor"] = (
            config.training.prefetch_factor
        )

    return DataLoader(
        **loader_kwargs,
    )


############################################################
# Main
############################################################

def main():

    ########################################################
    # CLI
    ########################################################

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/modal.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    ########################################################
    # Config
    ########################################################

    config = load_config(
        args.config,
    )

    ########################################################
    # Seed
    ########################################################

    set_seed(
        config.seed,
    )

    ########################################################
    # Device
    ########################################################

    device = get_device()

    print("=" * 70)
    print("Device :", device)
    print("=" * 70)
    print()

    ########################################################
    # Datasets
    ########################################################
    from pathlib import Path

    print("=" * 60)
    print("Checking dataset")
    print("=" * 60)

    root = Path(config.data.image_root)

    print("image_root:", root)
    print("exists:", root.exists())

    sample = root / "test/invoice/00921976.tif"

    print("sample:", sample)
    print("sample exists:", sample.exists())

    print("=" * 60)

    print("=" * 80)
    print("TRAIN CSV:", config.data.train_csv)
    print("=" * 80)
    train_dataset = VisionDataset(

        csv_path=config.data.train_csv,

        image_root=config.data.image_root,

        image_size=config.vision.image_size,

        train=True,

)
    print("VAL CSV:", config.data.val_csv)

    val_dataset = VisionDataset(

    csv_path=config.data.val_csv,

    image_root=config.data.image_root,

    image_size=config.vision.image_size,

    train=False,

)

    ########################################################
    # DataLoaders
    ########################################################

    train_loader = build_dataloader(

        dataset=train_dataset,

        batch_size=config.training.batch_size,

        shuffle=True,

        config=config,

    )

    val_loader = build_dataloader(

        dataset=val_dataset,

        batch_size=config.training.batch_size,

        shuffle=False,

        config=config,

    )

    ########################################################
    # Model
    ########################################################

    model = DocumentClassifier(

        backbone=config.vision.backbone,

        pretrained=config.vision.pretrained,

        num_classes=config.data.num_classes,

        hidden_dim=config.vision.hidden_dim,

        dropout=config.vision.dropout,

    )

    model_summary(model)
    

    print("=" * 60)
    print("OUTPUT CLASSES:")
    print(model.classifier[-1].out_features)
    print("=" * 60)

    print("=" * 80)
    print("MODEL TYPE:", type(model))
    print("IS MODULE:", isinstance(model, nn.Module))
    print(model)
    print("=" * 80)


    model.to(device)


    ########################################################
    # Loss
    ########################################################

    criterion = nn.CrossEntropyLoss()

    ########################################################
    # Optimizer
    ########################################################

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.training.learning_rate,

        weight_decay=config.training.weight_decay,

    )

    ########################################################
    # Scheduler
    ########################################################

    total_steps = (
        len(train_loader)
        * config.training.epochs
    )

    warmup_steps = int(
        total_steps
        * config.training.warmup_ratio
    )

    scheduler = get_cosine_schedule_with_warmup(

        optimizer,

        num_warmup_steps=warmup_steps,

        num_training_steps=total_steps,

    )

    ########################################################
    # Trainer
    ########################################################
    print("=" * 80)
    print("MODEL BEFORE TRAINER")
    print(type(model))
    print(model)
    print("FORWARD:", model.forward)
    print("=" * 80)

    trainer = VisionTrainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        config=config,

        device=device,

    )

    ########################################################
    # Train
    ########################################################

    trainer.fit()


############################################################
# Entry Point
############################################################

if __name__ == "__main__":

    main()