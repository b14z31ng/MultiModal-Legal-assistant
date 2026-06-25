from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlflow
import mlflow.pytorch

import numpy as np

import torch

from torch import nn
from tqdm import tqdm

from src.training.vision.early_stopping import (
    EarlyStopping,
)

from src.training.vision.metrics import (
    compute_metrics,
)

from src.training.vision.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_encoder,
    save_classifier,
    save_full_model,
)


class VisionTrainer:
    """
    Production Vision Trainer

    Stage-1

    ConvNeXt Document Classification
    """

    ############################################################
    # Initialization
    ############################################################

    def __init__(

        self,

        model,

        train_loader,

        val_loader,

        criterion,

        optimizer,

        scheduler,

        config,

        device,

    ):


########################################################
# Core
########################################################

        self.model = model
        print("=" * 80)
        print("TRAINER MODEL TYPE:", type(self.model))
        print("IS MODULE:", isinstance(self.model, nn.Module))
        print(self.model)
        print("=" * 80)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.config = config

        self.device = device

        ########################################################
        # CUDA Optimizations
        ########################################################

        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = True

            torch.backends.cudnn.allow_tf32 = True

            torch.backends.cudnn.benchmark = True

        ########################################################
        # Move Model
        ########################################################

        self.model.to(device)

        ########################################################
        # Channels Last (ConvNeXt Optimization)
        ########################################################

        if torch.cuda.is_available():

            self.model = self.model.to(
                memory_format=torch.channels_last
            )

        

        ########################################################
        # Mixed Precision
        ########################################################

        self.use_amp = torch.cuda.is_available()

        precision = getattr(

            config.hardware,

            "mixed_precision",

            "fp16",

        )

        if precision is None:

            precision = "fp16"

        if not isinstance(precision, str):

            raise ValueError(

                f"mixed_precision must be a string, got {type(precision).__name__}"

            )

        precision_lower = precision.lower()

        if precision_lower not in ("fp16", "bf16"):

            raise ValueError(

                f"Invalid mixed_precision value: '{precision}'. Allowed values are 'fp16' or 'bf16'."

            )

        if precision_lower == "bf16":

            self.autocast_dtype = torch.bfloat16

        else:

            self.autocast_dtype = torch.float16

        self.scaler = torch.amp.GradScaler(

            "cuda",

            enabled=(

                self.use_amp

                and

                self.autocast_dtype == torch.float16

            ),

        )

        ########################################################
        # Gradient Accumulation
        ########################################################

        self.grad_accum = (

            config.training.gradient_accumulation_steps

        )

        ########################################################
        # Early Stopping
        ########################################################

        self.early_stopping = EarlyStopping(

            patience=config.training.early_stopping_patience,

        )

        ########################################################
        # Checkpoints
        ########################################################

        self.best_checkpoint = (

            Path(

                config.paths.best_checkpoint_dir

            )

            /

            "vision_best.pt"

        )

        self.last_checkpoint = (

            Path(

                config.paths.last_checkpoint_dir

            )

            /

            "vision_last.pt"

        )

        ########################################################

        self.best_loss = float("inf")

    ############################################################
    # Forward
    ############################################################

    def _forward(

        self,

        batch,

    ):

        images = batch["pixel_values"].to(

            self.device,

            non_blocking=True,

        )

        labels = batch["label"].to(

            self.device,

            non_blocking=True,

        )

        with torch.amp.autocast(

            device_type="cuda",

            enabled=self.use_amp,

            dtype=self.autocast_dtype,

        ):

            logits = self.model(

                images,

            )

            loss = self.criterion(

                logits,

                labels,

            )

        return (

            logits,

            loss,

            labels,

        )

    ############################################################
    # Metric Cleanup
    ############################################################

    @staticmethod
    def _clean_metrics(

        metrics,

    ):

        cleaned = {}

        for key, value in metrics.items():

            if key in (

                "confusion_matrix",

                "classification_report",

            ):

                continue

            cleaned[key] = float(value)

        return cleaned
    
    ############################################################
    # Train One Epoch
    ############################################################

    def train_one_epoch(
        self,
        epoch: int,
   ) -> Dict[str, float]:

        print("INSIDE TRAIN_ONE_EPOCH")
        print(type(self.model))
        print(self.model)

        print("=" * 80)
        print("INSIDE train_one_epoch")
        print("type(self.model):", type(self.model))
        print("repr(self.model):", repr(self.model))
        print("id(self.model):", id(self.model))
        print("has train:", hasattr(self.model, "train"))
        print("has forward:", hasattr(self.model, "forward"))

        if callable(self.model):
            print("callable:", self.model)

        print("=" * 80)

        self.model.train()

        running_loss = 0.0

        all_logits = []

        all_labels = []

        progress = tqdm(

            enumerate(self.train_loader),

            total=len(self.train_loader),

            desc=f"Train {epoch + 1}",

            dynamic_ncols=True,

            leave=False,

        )

        ########################################################

        self.optimizer.zero_grad(

            set_to_none=True,

        )

        ########################################################

        for step, batch in progress:

            logits, loss, labels = self._forward(

                batch,

            )

            ####################################################
            # Gradient Accumulation
            ####################################################

            loss = (

                loss

                / self.grad_accum

            )

            if self.scaler.is_enabled():

                self.scaler.scale(

                    loss

                ).backward()

            else:

                loss.backward()

            ####################################################
            # Optimizer Step
            ####################################################

            should_step = (

                ((step + 1) % self.grad_accum == 0)

                or

                ((step + 1) == len(self.train_loader))

            )

            if should_step:

                ############################################

                if self.scaler.is_enabled():

                    self.scaler.unscale_(

                        self.optimizer,

                    )

                ############################################

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    max_norm=1.0,

                )

                ############################################

                if self.scaler.is_enabled():

                    self.scaler.step(

                        self.optimizer,

                    )

                    self.scaler.update()

                else:

                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()

                ############################################

                self.optimizer.zero_grad(

                    set_to_none=True,

                )

            ####################################################
            # Statistics
            ####################################################

            running_loss += (

                loss.item()

                * self.grad_accum

            )

            all_logits.append(

                logits.detach()

                .float()

                .cpu()

                .numpy()

            )

            all_labels.append(

                labels.detach()

                .cpu()

                .numpy()

            )

            ####################################################
            # GPU Memory
            ####################################################

            if torch.cuda.is_available():

                gpu_memory = (

                    torch.cuda.memory_allocated()

                    /

                    1024**3

                )

            else:

                gpu_memory = 0.0

            ####################################################
            # Progress
            ####################################################

            progress.set_postfix(

                {

                    "loss":

                        f"{running_loss/(step+1):.4f}",

                    "lr":

                        f"{self.optimizer.param_groups[0]['lr']:.2e}",

                    "gpu":

                        f"{gpu_memory:.2f}GB",

                }

            )

        ########################################################
        # Scheduler
        ########################################################

        ########################################################
        # Metrics
        ########################################################

        logits = np.concatenate(

            all_logits,

            axis=0,

        )

        labels = np.concatenate(

            all_labels,

            axis=0,

        )

        metrics = compute_metrics(

            labels,

            logits,

        )

        metrics["loss"] = (

            running_loss

            /

            len(self.train_loader)

        )

        ########################################################
        # Console
        ########################################################

        print()

        print("=" * 70)

        print("Training")

        print("=" * 70)

        print(

            f"Loss        : {metrics['loss']:.4f}"

        )

        print(

            f"Accuracy    : {metrics['accuracy']:.4f}"

        )

        print(

            f"Top-3       : {metrics['top3_accuracy']:.4f}"

        )

        print(

            f"Top-5       : {metrics['top5_accuracy']:.4f}"

        )

        print(

            f"Precision   : {metrics['precision']:.4f}"

        )

        print(

            f"Recall      : {metrics['recall']:.4f}"

        )

        print(

            f"F1          : {metrics['f1']:.4f}"

        )

        ########################################################

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        return metrics

    ############################################################
    # Validation
    ############################################################

    @torch.no_grad()
    def validate(

        self,

        epoch: int,

    ) -> Dict[str, float]:

        self.model.eval()

        running_loss = 0.0

        all_logits = []

        all_labels = []

        progress = tqdm(

            self.val_loader,

            total=len(self.val_loader),

            desc=f"Validation {epoch + 1}",

            dynamic_ncols=True,

            leave=False,

        )

        ########################################################

        for batch in progress:

            logits, loss, labels = self._forward(

                batch,

            )

            running_loss += loss.item()

            all_logits.append(

                logits.detach()

                .float()

                .cpu()

                .numpy()

            )

            all_labels.append(

                labels.detach()

                .cpu()

                .numpy()

            )

            progress.set_postfix(

                {

                    "loss":

                        f"{running_loss/len(all_logits):.4f}"

                }

            )

        ########################################################

        logits = np.concatenate(

            all_logits,

            axis=0,

        )

        labels = np.concatenate(

            all_labels,

            axis=0,

        )

        metrics = compute_metrics(

            labels,

            logits,

        )

        metrics["loss"] = (

            running_loss

            /

            len(self.val_loader)

        )

        ########################################################
        # Best Model
        ########################################################

        if metrics["loss"] < self.best_loss:

            self.best_loss = metrics["loss"]

            save_checkpoint(

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch,

                loss=self.best_loss,

                path=self.best_checkpoint,

            )

            print()

            print("=" * 70)

            print("✓ New Best Vision Model")

            print(

                f"Validation Loss : {self.best_loss:.4f}"

            )

            print("=" * 70)

        ########################################################

        self.early_stopping.step(

            metrics["loss"]

        )

        ########################################################

        print()

        print("=" * 70)

        print("Validation")

        print("=" * 70)

        print(

            f"Loss        : {metrics['loss']:.4f}"

        )

        print(

            f"Accuracy    : {metrics['accuracy']:.4f}"

        )

        print(

            f"Top-3       : {metrics['top3_accuracy']:.4f}"

        )

        print(

            f"Top-5       : {metrics['top5_accuracy']:.4f}"

        )

        print(

            f"Precision   : {metrics['precision']:.4f}"

        )

        print(

            f"Recall      : {metrics['recall']:.4f}"

        )

        print(

            f"F1          : {metrics['f1']:.4f}"

        )

        ########################################################

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        return metrics

    ############################################################
    # Resume Training
    ############################################################

    def resume(

        self,

    ) -> int:

        if not self.last_checkpoint.exists():

            print()

            print("=" * 70)

            print("Starting Fresh Training")

            print("=" * 70)

            print()

            return 0

        print()

        print("=" * 70)

        print("Loading Previous Checkpoint")

        print("=" * 70)

        checkpoint = load_checkpoint(

            model=self.model,

            optimizer=self.optimizer,

            scheduler=self.scheduler,

            path=self.last_checkpoint,

            device=self.device,

        )

        if isinstance(checkpoint, dict):

            start_epoch = checkpoint["epoch"]

            self.best_loss = checkpoint.get(

                "loss",

                float("inf"),

            )

        else:

            start_epoch = checkpoint

        print(

            f"Resuming from Epoch {start_epoch}"

        )

        print("=" * 70)

        print()

        return start_epoch + 1

    ############################################################
    # Training Loop
    ############################################################

    def fit(

        self,

    ):

        ########################################################
        # Resume
        ########################################################

        start_epoch = self.resume()

        ########################################################
        # MLflow
        ########################################################
        from pathlib import Path

        from src.training.vision.mlflow_utils import initialize_mlflow

        initialize_mlflow(
            self.config.project.experiment_name
        )
        
        with mlflow.start_run():

            ####################################################
            # Hyperparameters
            ####################################################

            mlflow.log_params({

                "model":
                    self.config.vision.backbone,

                "image_size":
                    self.config.vision.image_size,

                "epochs":
                    self.config.training.epochs,

                "batch_size":
                    self.config.training.batch_size,

                "learning_rate":
                    self.config.training.learning_rate,

                "weight_decay":
                    self.config.training.weight_decay,

                "gradient_accumulation":
                    self.grad_accum,

                "mixed_precision":
                    self.config.hardware.mixed_precision,

            })

            ####################################################
            # Epoch Loop
            ####################################################

            for epoch in range(

                start_epoch,

                self.config.training.epochs,

            ):

                print()

                print("=" * 70)

                print(

                    f"Epoch {epoch + 1}"

                    f"/"

                    f"{self.config.training.epochs}"

                )

                print("=" * 70)

                ################################################
                # Train
                ################################################

                print("=" * 80)
                print("BEFORE TRAIN_ONE_EPOCH")
                print(type(self.model))
                print(self.model)
                print("=" * 80)

                train_metrics = self.train_one_epoch(

                    epoch,

                )

                ################################################
                # Validation
                ################################################

                val_metrics = self.validate(

                    epoch,

                )

                ################################################
                # Save Last Checkpoint
                ################################################

                save_checkpoint(

                    model=self.model,

                    optimizer=self.optimizer,

                    scheduler=self.scheduler,

                    epoch=epoch,

                    loss=val_metrics["loss"],

                    path=self.last_checkpoint,

                )

                ################################################
                # MLflow Metrics
                ################################################

                metrics = {

                    "train_loss":
                        train_metrics["loss"],

                    "train_accuracy":
                        train_metrics["accuracy"],

                    "train_top3":
                        train_metrics["top3_accuracy"],

                    "train_top5":
                        train_metrics["top5_accuracy"],

                    "train_precision":
                        train_metrics["precision"],

                    "train_recall":
                        train_metrics["recall"],

                    "train_f1":
                        train_metrics["f1"],

                    "val_loss":
                        val_metrics["loss"],

                    "val_accuracy":
                        val_metrics["accuracy"],

                    "val_top3":
                        val_metrics["top3_accuracy"],

                    "val_top5":
                        val_metrics["top5_accuracy"],

                    "val_precision":
                        val_metrics["precision"],

                    "val_recall":
                        val_metrics["recall"],

                    "val_f1":
                        val_metrics["f1"],

                }

                mlflow.log_metrics(

                    self._clean_metrics(

                        metrics,

                    ),

                    step=epoch,

                )

                ################################################
                # Epoch Summary
                ################################################

                print()

                print("=" * 70)
                print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
                print("=" * 70)

                print()
                print("Training")
                print("-" * 70)
                print(f"Loss        : {train_metrics['loss']:.4f}")
                print(f"Accuracy    : {train_metrics['accuracy']:.4f}")
                print(f"Top-3       : {train_metrics['top3_accuracy']:.4f}")
                print(f"Top-5       : {train_metrics['top5_accuracy']:.4f}")
                print(f"Precision   : {train_metrics['precision']:.4f}")
                print(f"Recall      : {train_metrics['recall']:.4f}")
                print(f"F1 Score    : {train_metrics['f1']:.4f}")

                print()
                print("Validation")
                print("-" * 70)
                print(f"Loss        : {val_metrics['loss']:.4f}")
                print(f"Accuracy    : {val_metrics['accuracy']:.4f}")
                print(f"Top-3       : {val_metrics['top3_accuracy']:.4f}")
                print(f"Top-5       : {val_metrics['top5_accuracy']:.4f}")
                print(f"Precision   : {val_metrics['precision']:.4f}")
                print(f"Recall      : {val_metrics['recall']:.4f}")
                print(f"F1 Score    : {val_metrics['f1']:.4f}")

                print()
                print("-" * 70)
                print(f"Learning Rate      : {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"Best Val Loss      : {self.early_stopping.best_loss:.4f}")
                print(f"EarlyStop Counter  : {self.early_stopping.counter}/{self.early_stopping.patience}")

                if val_metrics["loss"] <= self.early_stopping.best_loss:
                    print("Best Model         : ✓ Updated")
                else:
                    print("Best Model         : No Improvement")

                print("=" * 70)

                ################################################
                # Early Stopping
                ################################################

                if self.early_stopping.should_stop:

                    print()

                    print("=" * 70)

                    print("Early stopping triggered.")

                    print("=" * 70)

                    break
        ########################################################
        # Training Finished
        ########################################################

        print()

        print("=" * 70)

        print("Vision Training Finished")

        print("=" * 70)

        ########################################################
        # Export Directory
        ########################################################

        model_dir = Path(

            self.config.paths.model_dir,

        )

        model_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        ########################################################
        # Save Full Vision Model
        ########################################################

        full_model_path = (

            model_dir

            /

            "vision_model.pt"

        )

        save_full_model(

            self.model,

            full_model_path,

        )

        print(

            f"✓ Saved Full Model : {full_model_path}"

        )

        ########################################################
        # Export Encoder
        ########################################################

        try:

            encoder_path = (

                model_dir

                /

                "document_encoder.pt"

            )

            save_encoder(

                self.model,

                encoder_path,

            )

            print(

                f"✓ Saved Encoder : {encoder_path}"

            )

        except Exception as e:

            print()

            print(

                "Encoder export skipped."

            )

            print(e)

        ########################################################
        # Export Classifier
        ########################################################

        try:

            classifier_path = (

                model_dir

                /

                "document_classifier.pt"

            )

            save_classifier(

                self.model,

                classifier_path,

            )

            print(

                f"✓ Saved Classifier : {classifier_path}"

            )

        except Exception as e:

            print()

            print(

                "Classifier export skipped."

            )

            print(e)

        ########################################################
        # MLflow Model
        ########################################################

        try:

            mlflow.pytorch.log_model(

                pytorch_model=self.model,

                artifact_path="vision_model",

            )

            print()

            print(

                "✓ MLflow model logged."

            )

        except Exception as e:

            print()

            print(

                "MLflow model logging skipped."

            )

            print(e)

        ########################################################
        # CUDA Cleanup
        ########################################################

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        ########################################################
        # Final Summary
        ########################################################

        print()

        print("=" * 70)

        print("Vision Training Complete")

        print("=" * 70)

        print(f"Best Checkpoint : {self.best_checkpoint}")

        print(f"Last Checkpoint : {self.last_checkpoint}")

        print(f"Export Directory : {model_dir}")

        print("=" * 70)