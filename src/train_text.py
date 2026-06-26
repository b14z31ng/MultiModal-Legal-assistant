from __future__ import annotations

import argparse
from src.utils.class_weights import compute_pos_weights
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import (
    get_cosine_schedule_with_warmup,
)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.model_summary import model_summary

from src.data.text_dataset import (
    CUADTextDataset,
)

from src.models.text_classifier import (
    TextClassifier,
)

from src.training.text_trainer import (
    TextTrainer,
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

        default="configs/text.yaml",

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
    # Dataset
    ########################################################

    print("=" * 80)
    print("TRAIN CSV:", config.data.train_csv)
    print("=" * 80)

    train_dataset = CUADTextDataset(

        csv_file=config.data.train_csv,

        tokenizer_name=config.text.backbone,

        max_length=config.text.max_length,

    )

    print("=" * 80)
    print("VAL CSV:", config.data.val_csv)
    print("=" * 80)

    val_dataset = CUADTextDataset(

        csv_file=config.data.val_csv,

        tokenizer_name=config.text.backbone,

        max_length=config.text.max_length,

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

    model = TextClassifier(

        model_name=config.text.backbone,

        num_labels=config.text.num_labels,

        dropout=config.text.dropout,

    )

    model_summary(model)

    print("=" * 70)
    print("OUTPUT LABELS :", config.text.num_labels)
    print("=" * 70)
    print()

    print("=" * 70)
    print("MODEL")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print()

    model.to(device)

    ########################################################
    # Loss
    ########################################################

    weights = compute_pos_weights(
    config.data.train_csv,
)

    weights = weights.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=weights,
    )
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

    trainer = TextTrainer(

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