from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlflow
import mlflow.pytorch

import numpy as np

import torch

from tqdm import tqdm

from src.training.callbacks import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
)

from src.training.metrics import (
    compute_multilabel_metrics,
)


class Trainer:
    """
    Production Multimodal Trainer

    Features
    --------
    ✓ Modal Ready
    ✓ Local GPU Ready
    ✓ BF16
    ✓ FP16
    ✓ Automatic Mixed Precision
    ✓ Gradient Accumulation
    ✓ Gradient Clipping
    ✓ MLflow
    ✓ Resume Training
    ✓ Best/Last Checkpoints
    ✓ tqdm Progress Bar
    """

    ########################################################
    # Initialization
    ########################################################

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

        ####################################################
        # Core Components
        ####################################################

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.config = config

        self.device = device

        self.model.to(device)

        ####################################################
        # Precision
        ####################################################

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

        ####################################################
        # GradScaler
        #
        # Only required for FP16.
        # BF16 does not require scaling.
        ####################################################

        self.scaler = torch.amp.GradScaler(

            "cuda",

            enabled=(

                self.use_amp

                and

                self.autocast_dtype == torch.float16

            ),

        )

        ####################################################
        # Gradient Accumulation
        ####################################################

        self.grad_accum = (

            config.training.gradient_accumulation_steps

        )

        ####################################################
        # Early Stopping
        ####################################################

        self.early_stopping = EarlyStopping(

            patience=config.training.early_stopping_patience,

        )

        ####################################################
        # Checkpoints
        ####################################################

        self.best_checkpoint = (

            Path(

                config.paths.best_checkpoint_dir

            )

            /

            "best_model.pt"

        )

        self.last_checkpoint = (

            Path(

                config.paths.last_checkpoint_dir

            )

            /

            "last_model.pt"

        )

        self.best_loss = float("inf")

    ########################################################
    # Move Batch To Device
    ########################################################

    def _move_batch_to_device(

        self,

        batch,

    ):

        pixel_values = batch["pixel_values"]

        if pixel_values is not None:

            pixel_values = pixel_values.to(

                self.device,

                non_blocking=True,

            )

        input_ids = batch["input_ids"].to(

            self.device,

            non_blocking=True,

        )

        attention_mask = batch["attention_mask"].to(

            self.device,

            non_blocking=True,

        )

        labels = batch["labels"].to(

            self.device,

            non_blocking=True,

        )

        return (

            pixel_values,

            input_ids,

            attention_mask,

            labels,

        )

    ########################################################
    # Forward Pass
    ########################################################

    def _forward(

        self,

        batch,

    ):

        (

            pixel_values,

            input_ids,

            attention_mask,

            labels,

        ) = self._move_batch_to_device(

            batch,

        )

        with torch.amp.autocast(

            device_type="cuda",

            enabled=self.use_amp,

            dtype=self.autocast_dtype,

        ):

            logits = self.model(

                pixel_values=pixel_values,

                input_ids=input_ids,

                attention_mask=attention_mask,

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

    ########################################################
    # Metric Cleanup
    ########################################################

    @staticmethod
    def _clean_metrics(

        metrics,

    ):

        cleaned = {}

        for key, value in metrics.items():

            if value is None:

                continue

            if isinstance(value, float):

                if np.isnan(value):

                    continue

            cleaned[key] = float(value)

        return cleaned
    
    ########################################################
    # Train One Epoch
    ########################################################

    def train_one_epoch(

        self,

        epoch: int,

    ) -> Dict[str, float]:

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

        self.optimizer.zero_grad(

            set_to_none=True,

        )

        ####################################################
        # Iterate over batches
        ####################################################

        for step, batch in progress:

            ################################################
            # Forward
            ################################################

            logits, loss, labels = self._forward(

                batch,

            )

            ################################################
            # Gradient Accumulation
            ################################################

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

            ################################################
            # Optimizer Step
            ################################################

            should_step = (

                (step + 1)

                %

                self.grad_accum

                == 0

            ) or (

                step + 1

                ==

                len(self.train_loader)

            )

            if should_step:

                ############################################
                # Gradient Clipping
                ############################################

                if self.scaler.is_enabled():

                    self.scaler.unscale_(

                        self.optimizer

                    )

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    max_norm=1.0,

                )

                ############################################
                # Optimizer
                ############################################

                if self.scaler.is_enabled():

                    self.scaler.step(

                        self.optimizer,

                    )

                    self.scaler.update()

                else:

                    self.optimizer.step()

                self.optimizer.zero_grad(

                    set_to_none=True,

                )

            ################################################
            # Statistics
            ################################################

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

            ################################################
            # GPU Memory
            ################################################

            if torch.cuda.is_available():

                gpu_memory = (

                    torch.cuda.memory_allocated()

                    /

                    1024**3

                )

            else:

                gpu_memory = 0.0

            ################################################
            # Progress Bar
            ################################################

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

        ####################################################
        # Scheduler
        ####################################################

        if self.scheduler is not None:

            self.scheduler.step()

        ####################################################
        # Epoch Metrics
        ####################################################

        epoch_logits = np.concatenate(

            all_logits,

            axis=0,

        )

        epoch_labels = np.concatenate(

            all_labels,

            axis=0,

        )

        metrics = compute_multilabel_metrics(

            epoch_labels,

            epoch_logits,

            threshold=self.config.evaluation.threshold,

        )

        metrics["loss"] = (

            running_loss

            /

            len(self.train_loader)

        )

        ####################################################
        # Console Summary
        ####################################################

        print()

        print("=" * 60)

        print(

            f"Train Loss : {metrics['loss']:.4f}"

        )

        print(

            f"Subset Accuracy : "

            f"{metrics['subset_accuracy']:.4f}"

        )

        print(

            f"Label Accuracy : "

            f"{metrics['label_accuracy']:.4f}"

        )

        print(

            f"F1 Score : "

            f"{metrics['f1']:.4f}"

        )

        print("=" * 60)

        ####################################################
        # Cleanup
        ####################################################

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        return metrics
    
    ########################################################
    # Validation
    ########################################################

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

        ####################################################
        # Validation Loop
        ####################################################

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

        ####################################################
        # Metrics
        ####################################################

        epoch_logits = np.concatenate(

            all_logits,

            axis=0,

        )

        epoch_labels = np.concatenate(

            all_labels,

            axis=0,

        )

        metrics = compute_multilabel_metrics(

            epoch_labels,

            epoch_logits,

            threshold=self.config.evaluation.threshold,

        )

        metrics["loss"] = (

            running_loss

            /

            len(self.val_loader)

        )

        ####################################################
        # Best Model
        ####################################################

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

            print("=" * 60)

            print("✓ Best Model Updated")

            print(

                f"Validation Loss : {self.best_loss:.4f}"

            )

            print("=" * 60)

        ####################################################
        # Early Stopping
        ####################################################

        self.early_stopping.step(

            metrics["loss"]

        )

        ####################################################
        # Console
        ####################################################

        print()

        print("=" * 60)

        print(

            f"Validation Loss : {metrics['loss']:.4f}"

        )

        print(

            f"Subset Accuracy : "

            f"{metrics['subset_accuracy']:.4f}"

        )

        print(

            f"Label Accuracy : "

            f"{metrics['label_accuracy']:.4f}"

        )

        print(

            f"F1 Score : "

            f"{metrics['f1']:.4f}"

        )

        print("=" * 60)

        ####################################################
        # Cleanup
        ####################################################

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        return metrics

    ########################################################
    # Resume Training
    ########################################################

    def resume(

        self,

    ) -> int:

        if not self.last_checkpoint.exists():

            print()

            print("=" * 60)

            print(

                "Starting New Training"

            )

            print("=" * 60)

            print()

            return 0

        print()

        print("=" * 60)

        print(

            "Resuming Training"

        )

        print("=" * 60)

        start_epoch = load_checkpoint(

            model=self.model,

            optimizer=self.optimizer,

            scheduler=self.scheduler,

            path=self.last_checkpoint,

            device=self.device,

        )

        print(

            f"Checkpoint Loaded (Epoch {start_epoch})"

        )

        print("=" * 60)

        print()

        return start_epoch + 1
    ########################################################
    # Training Loop
    ########################################################

    def fit(

        self,

    ):

        ####################################################
        # Resume
        ####################################################

        start_epoch = self.resume()

        ####################################################
        # MLflow
        ####################################################

        mlflow.set_experiment(

            self.config.project.experiment_name,

        )

        with mlflow.start_run():

            ################################################
            # Hyperparameters
            ################################################

            mlflow.log_params({

                "vision_backbone":
                    self.config.vision.backbone,

                "text_encoder":
                    self.config.text.encoder,

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

                "max_length":
                    self.config.text.max_length,

                "image_size":
                    self.config.vision.image_size,

                "fusion_hidden":
                    self.config.fusion.hidden_dim,

            })

            ################################################
            # Epoch Loop
            ################################################

            for epoch in range(

                start_epoch,

                self.config.training.epochs,

            ):

                print()

                print("=" * 70)

                print(

                    f"Epoch "

                    f"{epoch+1}"

                    f"/"

                    f"{self.config.training.epochs}"

                )

                print("=" * 70)

                ################################################
                # Training
                ################################################

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
                # Merge Metrics
                ################################################

                metrics = {

                    "train_loss":

                        train_metrics["loss"],

                    "train_subset_accuracy":

                        train_metrics["subset_accuracy"],

                    "train_label_accuracy":

                        train_metrics["label_accuracy"],

                    "train_precision":

                        train_metrics["precision"],

                    "train_recall":

                        train_metrics["recall"],

                    "train_f1":

                        train_metrics["f1"],

                    "train_hamming":

                        train_metrics["hamming_loss"],

                    "train_jaccard":

                        train_metrics["jaccard"],

                    "train_auc":

                        train_metrics["roc_auc"],

                    "train_pr_auc":

                        train_metrics["pr_auc"],

                    "val_loss":

                        val_metrics["loss"],

                    "val_subset_accuracy":

                        val_metrics["subset_accuracy"],

                    "val_label_accuracy":

                        val_metrics["label_accuracy"],

                    "val_precision":

                        val_metrics["precision"],

                    "val_recall":

                        val_metrics["recall"],

                    "val_f1":

                        val_metrics["f1"],

                    "val_hamming":

                        val_metrics["hamming_loss"],

                    "val_jaccard":

                        val_metrics["jaccard"],

                    "val_auc":

                        val_metrics["roc_auc"],

                    "val_pr_auc":

                        val_metrics["pr_auc"],

                }

                ################################################
                # Remove NaN
                ################################################

                metrics = self._clean_metrics(

                    metrics,

                )

                ################################################
                # MLflow
                ################################################

                mlflow.log_metrics(

                    metrics,

                    step=epoch,

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
                # Console Summary
                ################################################

                print()

                print("=" * 70)

                print("Training Metrics")

                print("=" * 70)

                for k, v in train_metrics.items():

                    print(

                        f"{k:<24}: {v}"

                    )

                print()

                print("=" * 70)

                print("Validation Metrics")

                print("=" * 70)

                for k, v in val_metrics.items():

                    print(

                        f"{k:<24}: {v}"

                    )

                ################################################
                # Early Stopping
                ################################################

                if self.early_stopping.stop:

                    print()

                    print("=" * 60)

                    print(

                        "Early stopping activated."

                    )

                    print("=" * 60)

                    break
        ####################################################
        # Training Finished
        ####################################################

        print()

        print("=" * 70)

        print("Training Finished")

        print("=" * 70)

        ####################################################
        # Export Directory
        ####################################################

        model_dir = Path(

            self.config.paths.model_dir

        )

        model_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        ####################################################
        # Save Full Model
        ####################################################

        full_model_path = (

            model_dir

            /

            "risk_auditor.pt"

        )

        torch.save(

            self.model.state_dict(),

            full_model_path,

        )

        print()

        print(

            f"✓ Saved : {full_model_path}"

        )

        ####################################################
        # Save Vision Encoder
        ####################################################

        try:

            encoder_path = (

                model_dir

                /

                "document_encoder.pt"

            )

            torch.save(

                self.model.vision_encoder.state_dict(),

                encoder_path,

            )

            print(

                f"✓ Saved : {encoder_path}"

            )

        except Exception as e:

            print()

            print(

                "Vision encoder export skipped."

            )

            print(e)

        ####################################################
        # Save Fusion Module
        ####################################################

        try:

            fusion_path = (

                model_dir

                /

                "fusion_module.pt"

            )

            torch.save(

                self.model.fusion.state_dict(),

                fusion_path,

            )

            print(

                f"✓ Saved : {fusion_path}"

            )

        except Exception as e:

            print()

            print(

                "Fusion export skipped."

            )

            print(e)

        ####################################################
        # Save Classifier
        ####################################################

        try:

            classifier_path = (

                model_dir

                /

                "risk_classifier.pt"

            )

            torch.save(

                self.model.classifier.state_dict(),

                classifier_path,

            )

            print(

                f"✓ Saved : {classifier_path}"

            )

        except Exception as e:

            print()

            print(

                "Classifier export skipped."

            )

            print(e)

        ####################################################
        # MLflow Model
        ####################################################

        try:

            mlflow.pytorch.log_model(

                pytorch_model=self.model,

                artifact_path="risk_auditor",

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

        ####################################################
        # CUDA Cleanup
        ####################################################

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        ####################################################
        # Finished
        ####################################################

        print()

        print("=" * 70)

        print("Training Complete")

        print("=" * 70)