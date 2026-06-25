from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device

from src.data.dataset import MultimodalRiskDataset
from src.data.collate import multimodal_collate_fn

from src.models.risk_auditor import RiskAuditorFusionModel

from src.training.losses import build_loss
from src.training.trainer import Trainer


def main():

    ####################################################
    # Config
    ####################################################

    CONFIG_PATH = "configs/modal.yaml"

    config = load_config(
        CONFIG_PATH
    )

    ####################################################
    # Seed
    ####################################################

    set_seed(
        config.seed
    )

    ####################################################
    # Device
    ####################################################

    device = get_device()

    print("=" * 60)
    print(f"Device : {device}")
    print("=" * 60)

    ####################################################
    # Dataset
    ####################################################

    train_dataset = MultimodalRiskDataset(

        csv_path="data/processed/train.csv",

        image_root="data/images/invoice",

        tokenizer_name=config.text.encoder,

        image_size=config.vision.image_size,

        max_length=config.text.max_length,

        vision_enabled=True,

    )

    val_dataset = MultimodalRiskDataset(

        csv_path="data/processed/val.csv",

        image_root="data/images/invoice",

        tokenizer_name=config.text.encoder,

        image_size=config.vision.image_size,

        max_length=config.text.max_length,

        vision_enabled=True,

    )

    ####################################################
    # DataLoader
    ####################################################

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.training.batch_size,

        shuffle=True,

        num_workers=config.training.num_workers,

        pin_memory=torch.cuda.is_available(),

        collate_fn=multimodal_collate_fn,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=config.training.batch_size,

        shuffle=False,

        num_workers=config.training.num_workers,

        pin_memory=torch.cuda.is_available(),

        collate_fn=multimodal_collate_fn,

    )

    ####################################################
    # Model
    ####################################################

    model = RiskAuditorFusionModel(
        config
    ).to(device)

    ####################################################
    # Optimizer
    ####################################################

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.training.learning_rate,

        weight_decay=config.training.weight_decay,

    )

    ####################################################
    # Scheduler
    ####################################################

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

    ####################################################
    # Loss
    ####################################################

    criterion = build_loss(
        task="multilabel"
    )

    ####################################################
    # Trainer
    ####################################################

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        config=config,

        device=device,

    )

    ####################################################
    # Train
    ####################################################

    trainer.fit()


if __name__ == "__main__":

    main()