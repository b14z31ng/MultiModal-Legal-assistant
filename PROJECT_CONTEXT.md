# PROJECT_CONTEXT.md

## 1. Project Overview

### Project Name
**Multimodal Legal Risk Auditor** (internally configured as `multimodal-risk-auditor` or `multimodal-risk-auditor-modal`).

### Project Purpose
The Multimodal Legal Risk Auditor is a dual-stage machine learning system designed to analyze legal contracts and documents. The core function is to detect and flag contract clauses associated with high legal and financial risk (e.g., *Termination For Convenience*, *Non-Compete*, *Most Favored Nation*). It achieves this by combining textual information from contracts with visual/structural features of document page scans.

### Business & Research Objective
Automate the review of commercial agreements to dramatically reduce the manual effort of risk assessment for legal operations, contract management teams, and risk auditors. From a research standpoint, the project showcases multimodal fusion (cross-attention between transformer-based text models and convolutional/vision encoders) applied to specialized legal domain tasks.

### Current Development Status
- **Core ML Layer (Complete)**: Model architectures, dataset classes, collate functions, trainers, loss functions, custom metrics, and callbacks are fully implemented.
- **Data Engineering (Complete)**: Preprocessing scripts for both raw contract texts (CUAD JSON dataset) and document page scans (OCR/layout categorization datasets) are written and operational.
- **Inference & Serving (Skeleton)**: Serving scripts ([serve.py](file:///home/reshad/Projects/multi_new/src/serve.py)), batch predictors ([predict.py](file:///home/reshad/Projects/multi_new/src/predict.py)), evaluation entry points ([evaluate.py](file:///home/reshad/Projects/multi_new/src/evaluate.py)), API structures ([src/api/](file:///home/reshad/Projects/multi_new/src/api/)), and post-processing filters are empty template files ready for implementation.
- **Test Suite (Skeleton)**: Testing folders and files exist under `tests/` but are currently empty placeholders containing no functional test suites.
- **Documentation (Skeleton)**: Dedicated documents in `docs/` (such as `api.md`, `architecture.md`, etc.) are empty placeholders.

### Primary Use Cases
1. **Stage-1: Document Type Classification (Vision-Only)**
   - Scan raw document pages and classify them into standard visual layouts: `invoice`, `form`, `letter`, `memo`, `specification`, or `budget`.
2. **Stage-2: Multimodal Legal Risk Auditing (Text + Vision)**
   - Audit contracts to detect the presence or absence of 41 legal risk classes based on both the textual semantics (contract text paragraphs) and layout structures (visual page maps).

### High-Level Architecture
The system is divided into two primary training pipelines:

```text
               ┌──────────────────────────────────────────────┐
               │    Stage-1: Document Classification (Vision)  │
               │   - Backbone: ConvNeXt / ResNet              │
               │   - Goal: Classify page layout (6 classes)   │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼ (Saves DocumentEncoder Weights)
┌─────────────────────────────────────┴─────────────────────────────────────┐
│              Stage-2: Multimodal Risk Auditor (Vision + Text)             │
│                                                                           │
│   ┌───────────────┐     ┌──────────────────────┐     ┌────────────────┐   │
│   │ Contract Text │     │   Cross-Attention    │     │ Document Image │   │
│   │       │       │     │        Fusion        │     │       │        │   │
│   │ DeBERTa Large ├────►│ Text attends to image├◄────┤ ConvNeXt Base  │   │
│   │       │       │     │  Transformer Encoder │     │       │        │   │
│   └───────────────┘     └──────────┬───────────┘     └────────────────┘   │
│                                    │                                      │
│                                    ▼                                      │
│                        ┌───────────────────────┐                          │
│                        │ Multi-label Head (41) │                          │
│                        └───────────────────────┘                          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```text
/home/reshad/Projects/multi_new/
├── configs/                     # YAML configuration files for different runtime environments
│   ├── development.yaml         # Empty placeholder for development config
│   ├── modal.yaml               # Settings optimized for Cloud GPU resources (Modal)
│   ├── production.yaml          # Empty placeholder for production config
│   ├── rtx2050.yaml             # Settings constrained for local low-VRAM GPUs (RTX 2050)
│   └── rtx3090.yaml             # Empty placeholder for mid-range local GPU config
├── data/                        # Dataset directories (Raw, interim, and processed files)
│   ├── processed/               # Data splits (.csv) and class maps (.json) ready for training
│   └── raw/                     # Original external source datasets (CUAD v1 JSON, raw images)
├── docs/                        # Project documentation (Skeletons / Empty)
│   ├── api.md
│   ├── architecture.md
│   ├── deployment-guide.md
│   └── training-guide.md
├── scripts/                     # Standalone preprocessing and setup scripts
│   ├── convert_vision_csv_to_relative.py  # Fixes raw image paths in datasets to be relative
│   ├── evaluate.py              # Empty placeholder script for metric evaluation
│   ├── export_model.py          # Empty placeholder script for model export
│   ├── infer.py                 # Empty placeholder script for standalone prediction
│   ├── prep_cuad.py             # Parses CUAD JSON to build multi-label text targets
│   ├── prepare_dataset.py       # Empty placeholder script
│   ├── prepare_vision_dataset.py # Scans raw documents, runs validation checks, splits data
│   └── train.py                 # Empty placeholder script
├── src/                         # Principal Python codebase
│   ├── api/                     # FastAPI endpoint definitions (Skeletons / Empty)
│   ├── data/                    # Dataset loaders, data transforms, and collators
│   ├── inference/               # Production prediction submodules (Skeletons / Empty)
│   ├── models/                  # Neural network model definitions (encoders, heads, fusion)
│   ├── training/                # Training pipelines, loaders, and logging classes
│   │   └── vision/              # Dedicated Stage-1 vision training helper files
│   ├── utils/                   # Device helper functions, seeds, loggers, configuration loader
│   ├── constants.py             # Global constants, target classes, supported backbones
│   ├── evaluate.py              # Empty placeholder
│   ├── predict.py               # Empty placeholder
│   ├── serve.py                 # Empty placeholder
│   ├── train.py                 # Stage-2 Multimodal Training Entrypoint
│   └── train_vision.py          # Stage-1 Document Classification Training Entrypoint
├── tests/                       # Automated testing directory (Skeletons / Empty)
├── Dockerfile                   # Docker build definition (Placeholder / Commented)
├── docker-compose.yml           # Compose services definition (Placeholder / Commented)
├── mlflow.db                    # SQLite database containing MLflow experiment tracking logs
└── mlruns/                      # Default local MLflow storage fallback folder
```

### Detailed Folder Responsibilities & Important Files

#### `configs/`
- **Purpose**: Defines model dimensions, training epochs, data paths, hardware profiles, optimizer, and scheduler hyperparameters.
- **Important Files**:
  - [configs/modal.yaml](file:///home/reshad/Projects/multi_new/configs/modal.yaml): High-throughput configuration for large encoders and big batch sizes.
  - [configs/rtx2050.yaml](file:///home/reshad/Projects/multi_new/configs/rtx2050.yaml): VRAM-optimized config for resource-limited systems (VRAM <= 4GB).

#### `src/data/`
- **Purpose**: Implements PyTorch custom datasets, image augmentation transforms, and custom batch collators.
- **Important Files**:
  - [src/data/dataset.py](file:///home/reshad/Projects/multi_new/src/data/dataset.py): Implements `MultimodalRiskDataset` which loads text columns, performs tokenization, grabs related image page files, and generates multi-label float targets.
  - [src/data/vision_dataset.py](file:///home/reshad/Projects/multi_new/src/data/vision_dataset.py): Implements `VisionDataset` for layout classification, applying randomized visual augmentations during training.
  - [src/data/collate.py](file:///home/reshad/Projects/multi_new/src/data/collate.py): Implements `multimodal_collate_fn` to handle optional image paths.
  - [src/data/transforms.py](file:///home/reshad/Projects/multi_new/src/data/transforms.py): Implements generic image standardizations.

#### `src/models/`
- **Purpose**: Neural architecture models.
- **Important Files**:
  - [src/models/risk_auditor.py](file:///home/reshad/Projects/multi_new/src/models/risk_auditor.py): Implements `RiskAuditorFusionModel` containing text/vision encoders, cross-attention fusion block, and the 41-label linear classifier.
  - [src/models/fusion.py](file:///home/reshad/Projects/multi_new/src/models/fusion.py): Implements `CrossAttentionFusion` mapping inputs to query/key/values.
  - [src/models/document_encoder.py](file:///home/reshad/Projects/multi_new/src/models/document_encoder.py): Integrates `timm` backbones.
  - [src/models/text_encoder.py](file:///home/reshad/Projects/multi_new/src/models/text_encoder.py): Handles transformer model embeddings.

#### `src/training/`
- **Purpose**: Defines training steps, validation loops, backpropagation checkpoints, metrics calculations, and logging.
- **Important Files**:
  - [src/training/trainer.py](file:///home/reshad/Projects/multi_new/src/training/trainer.py): Multimodal trainer (precision autocasting, grad scaling, early stopping, checkpoints).
  - [src/training/vision_trainer.py](file:///home/reshad/Projects/multi_new/src/training/vision_trainer.py): Stage-1 trainer (includes `channels_last` performance boosts for ConvNeXt).
  - [src/training/metrics.py](file:///home/reshad/Projects/multi_new/src/training/metrics.py): Exact match subset accuracy, Hamming loss, Macro F1, Precision, Recall, AUC.

---

## 3. Configuration System

### All Config Files
1. `configs/modal.yaml`: Core target profile for running heavy GPU jobs.
2. `configs/rtx2050.yaml`: Constrained local profile for testing.
3. `configs/development.yaml` *(Empty template)*
4. `configs/production.yaml` *(Empty template)*
5. `configs/rtx3090.yaml` *(Empty template)*

### Config Hierarchy and Loading
Configs are loaded via [src/utils/config.py](file:///home/reshad/Projects/multi_new/src/utils/config.py) using the utility function `load_config(config_path)`. It reads YAML structure and wraps it recursively inside a `Config` class helper:
```python
class Config:
    def __init__(self, config_dict):
        self._config = config_dict
    def __getattr__(self, item):
        value = self._config.get(item)
        if isinstance(value, dict):
            return Config(value)
        return value
```
This enables dot-notation lookup of nested parameters (e.g., `config.training.batch_size`).

### Settings Explained

| YAML Path | Type | Default (`modal`) | Default (`rtx2050`) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `seed` | int | `42` | `42` | Global seed value for reproducibility (PyTorch, Numpy, Python Random). |
| `project.name` | str | `multimodal-risk-auditor` | `multimodal-risk-auditor` | Project key identifier. |
| `project.experiment_name` | str | `multimodal-risk-auditor-modal` | `multimodal-risk-auditor` | MLflow experiment tag identifier. |
| `hardware.device` | str | `cuda` | `cuda` | Hardware accelerator backend. |
| `hardware.mixed_precision` | str | `bf16` | `fp16` | Target autocast precision: `bf16` (preferred for newer GPUs) or `fp16`. |
| `hardware.compile` | bool | `false` | *Not Specified* | If true, applies `torch.compile` optimization. |
| `vision.backbone` | str | `convnext_base.fb_in22k_ft_in1k` | `resnet18` (via `vision.encoder`) | Timm model ID for vision feature extractor backbone. |
| `vision.pretrained` | bool | `true` | *Not Specified* | Whether to load pretrained ImageNet weights. |
| `vision.image_size` | int | `384` | `224` | Resolution size of input tensors for document scans. |
| `vision.enabled` | bool | *Not Specified* | `false` | Toggles processing of image inputs in datasets. |
| `text.encoder` | str | `microsoft/deberta-v3-large` | `distilbert-base-uncased` | Hugging Face model key for text representations. |
| `text.max_length` | int | `512` | `256` | Max sequence token length limit for contract texts. |
| `fusion.hidden_dim` | int | `1024` | *Not Specified* | Projected dimensions inside multimodal attention. |
| `fusion.num_heads` | int | `8` | *Not Specified* | Attention head count for the fusion transformer block. |
| `classifier.hidden_dim` | int | `1024` | `512` (via `classifier.hidden_dim`) | Hidden layer size of the classifier MLP head. |
| `classifier.dropout` | float | `0.3` | `0.3` | Dropout regularizer probability. |
| `training.epochs` | int | `1` | `5` | Epoch target limit. |
| `training.batch_size` | int | `64` | `2` | Batch size limit. |
| `training.gradient_accumulation_steps` | int | `1` | `4` | How many steps to accumulate gradients over. |
| `training.learning_rate` | float | `2e-5` | `2e-5` | Optimization base learning rate. |
| `training.weight_decay` | float | `0.01` | `0.01` | L2 weight regularizer. |
| `training.warmup_ratio` | float | `0.1` | `0.1` | Ratio of training steps used for warmup schedules. |
| `training.num_workers` | int | `8` | `2` | Data loading subprocess workers. |
| `freeze.text_encoder` | bool | *Not Specified* | `true` | Locks text parameters to optimize VRAM. |
| `freeze.vision_encoder` | bool | *Not Specified* | `true` | Locks vision parameters to optimize VRAM. |

---

## 4. Data Pipeline

The project supports two separate pipelines depending on target task.

### Pipeline A: Stage-1 Document Classifier (Vision Only)
- **Source**: Raw image files located under `data/raw/test/` partitioned by folder structure representing class name label (Target: 6 categories).
- **Label Mapping**: Target classes mapped to index indices:
  ```json
  {
      "budget": 0,
      "form": 1,
      "invoice": 2,
      "letter": 3,
      "memo": 4,
      "specification": 5
  }
  ```
- **Train/Val/Test Split**: Stratified split generated by [scripts/prepare_vision_dataset.py](file:///home/reshad/Projects/multi_new/scripts/prepare_vision_dataset.py).
  - Train: 80%
  - Val: 10%
  - Test: 10%
- **Augmentation Pipeline (Training)**:
  - Resize (`image_size` x `image_size`)
  - Random Horizontal Flip (`p=0.5`)
  - Random Rotation (`degrees=5`)
  - Random Affine (`translate=(0.02, 0.02)`, `scale=(0.95, 1.05)`)
  - Color Jitter (`brightness=0.10`, `contrast=0.10`)
  - To Tensor
  - Normalize (ImageNet statistics: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
- **Augmentation Pipeline (Validation/Inference)**:
  - Resize (`image_size` x `image_size`), To Tensor, Normalize (ImageNet stats).

### Pipeline B: Stage-2 Multimodal Risk Auditor (Multimodal)
- **Source**: CUAD v1 JSON raw file (`data/raw/cuad-main/data/CUADv1.json`) parsed using [scripts/prep_cuad.py](file:///home/reshad/Projects/multi_new/scripts/prep_cuad.py).
- **Format**:
  - `title`: Key matching document contract name.
  - `contract_text`: The string text corpus extracted.
  - Labels: 41 fields prefixing `label::<clause_type>` (values 0 or 1).
- **Train/Val/Test Split**: Stratified by random seed splitter (72% Train, 8% Val, 20% Test).
- **Data Loading Flow**:
  1. Datasets read data frames from `data/processed/train.csv` / `val.csv`.
  2. Text strings are padded or truncated up to `max_length` using Hugging Face AutoTokenizer (`AutoTokenizer.from_pretrained(tokenizer_name)`).
  3. Image loads matching contract image title: `image_root / f"{title}.png"`. If it doesn't exist, it generates a fallback white canvas.
  4. Yields dictionary representation containing `pixel_values`, `input_ids`, `attention_mask`, and float label tensors.
  5. The collator function `multimodal_collate_fn` handles stacking into batched dictionary structures.

### Potential Issues / Code Reference
- **Image Fallback Strategy**:
  - Located in [src/data/dataset.py:L129-L145](file:///home/reshad/Projects/multi_new/src/data/dataset.py#L129-L145) and [src/data/vision_dataset.py:L221-L225](file:///home/reshad/Projects/multi_new/src/data/vision_dataset.py#L221-L225).
  - If document image is missing or corrupted, a placeholder white image tensor is returned silently instead of failing fast. This can obscure path issues during training.
- **Transforms kwargs Bug**:
  - In [src/data/dataset.py:L63-L66](file:///home/reshad/Projects/multi_new/src/data/dataset.py#L63-L66), the initialization executes:
    `self.transform = build_image_transform(image_size=image_size, train=True)`.
  - However, [src/data/transforms.py:L4-L6](file:///home/reshad/Projects/multi_new/src/data/transforms.py#L4-L6) has signature:
    `def build_image_transform(image_size: int):`.
  - Since it does not accept the `train` parameter, invoking this crashes with a `TypeError`.
- **Validation Transforms Duplication**:
  - Because `build_image_transform` does not implement conditional validation settings, the training preprocessing normalization is reused for validation tensors.

---

## 5. Model Architecture

This repository contains components for both vision-only and multimodal architectures.

### DocumentEncoder
- **File Location**: [src/models/document_encoder.py](file:///home/reshad/Projects/multi_new/src/models/document_encoder.py)
- **Purpose**: Generates visual representations of document layout features.
- **Inputs**: Shape `(B, 3, H, W)` representing normalized document page scans.
- **Outputs**: Feature vector tensor of shape `(B, output_dim)`.
- **Architecture**:
  - Timm model backbone (configurable, default is `convnext_base.fb_in22k_ft_in1k`).
  - Average pooling (`global_pool="avg"`) to extract flat representation.
  - Dropout layer standardizer (uses classifier dropout rate).

### TextEncoder
- **File Location**: [src/models/text_encoder.py](file:///home/reshad/Projects/multi_new/src/models/text_encoder.py)
- **Purpose**: Processes contract text embeddings.
- **Inputs**: Token IDs `input_ids` shape `(B, L)` and mask tensors `attention_mask` shape `(B, L)`.
- **Outputs**: CLS vector embeddings shape `(B, hidden_size)`.
- **Architecture**:
  - Hugging Face model backbone (configurable, default is `microsoft/deberta-v3-large`).
  - Grabs the sequence representation coordinate `[:, 0]` (the `[CLS]` token) of the last hidden states output.
  - Dropout layer standardizer.

### CrossAttentionFusion
- **File Location**: [src/models/fusion.py](file:///home/reshad/Projects/multi_new/src/models/fusion.py)
- **Purpose**: Performs cross-modal merging where textual semantics query visual page layout highlights.
- **Inputs**:
  - `vision_features`: Shape `(B, vision_dim)` (e.g. 1024)
  - `text_features`: Shape `(B, text_dim)` (e.g. 1024)
- **Outputs**: Joint embedding representation of shape `(B, 1024)`.
- **Architecture**:
  - Linear layers project text and vision features to `hidden_dim`.
  - Unsqueezes dimensions to add a sequence token axis.
  - `nn.MultiheadAttention`: Cross-attention calculation where query = Text, key/value = Vision.
  - Concatenates the cross-attended representation with original vision representation along the sequence axis.
  - Transformer encoder block (2 layers, default 8 attention heads, 4x feedforward projection).
  - Global average pooling across the sequence dimension.
  - Layer normalization and dropout.

### RiskAuditorFusionModel (Multimodal Risk Auditor)
- **File Location**: [src/models/risk_auditor.py](file:///home/reshad/Projects/multi_new/src/models/risk_auditor.py)
- **Purpose**: Core Stage-2 classification model.
- **Inputs**:
  - `pixel_values` (images): Shape `(B, 3, H, W)`
  - `input_ids` (tokens): Shape `(B, L)`
  - `attention_mask`: Shape `(B, L)`
- **Outputs**: Output logits tensor of shape `(B, 41)`.
- **Architecture**:
  - `self.vision_encoder` -> Extract vision features.
  - `self.text_encoder` -> Extract text features.
  - `self.fusion` -> Perform Cross-Attention fusion.
  - Classifier MLP Head:
    `LayerNorm` -> `Linear(1024, 1024)` -> `GELU` -> `Dropout` -> `Linear(1024, 512)` -> `GELU` -> `Dropout` -> `Linear(512, 41)`.

### DocumentClassifier (Vision-only)
- **File Location**: [src/models/document_classifier.py](file:///home/reshad/Projects/multi_new/src/models/document_classifier.py)
- **Purpose**: Core Stage-1 visual document type classifier.
- **Inputs**: Shape `(B, 3, H, W)` images.
- **Outputs**: Output logits tensor of shape `(B, 16)`.
- **Architecture**:
  - `self.encoder` -> Swaps timm backbones.
  - Deep classifier head:
    `LayerNorm` -> `Linear(F, hidden_dim)` -> `GELU` -> `Dropout` -> `Linear(hidden_dim, hidden_dim/2)` -> `GELU` -> `Dropout` -> `Linear(hidden_dim/2, 16)`.

---

## 6. Training Pipeline

### End-to-End Training Flow

```text
train_vision.py (Stage-1 Entrypoint)         train.py (Stage-2 Entrypoint)
    │                                             │
    ▼                                             ▼
VisionDataset                                 MultimodalRiskDataset
    │                                             │
    ▼                                             ▼
DocumentClassifier                            RiskAuditorFusionModel
    │                                             │
    ▼                                             ▼
VisionTrainer                                 Trainer
    │                                             │
    ├── Autocast Mixed Precision (AMP)            ├── Autocast Mixed Precision (AMP)
    ├── GradScaler (FP16 only)                    ├── GradScaler (FP16 only)
    ├── Gradient Accumulation                     ├── Gradient Accumulation
    └── Channels Last Memory Format               └── CheckpointManager
    │                                             │
    ▼                                             ▼
Metrics (Accuracy, Top-k F1)                  Metrics (Subset Accuracy, AUC)
    │                                             │
    ▼                                             ▼
Checkpoint (vision_best.pt)                   Checkpoint (best_model.pt)
```

### Trainer Mechanisms
- **Optimizer**: AdamW optimizer applied with weight decay parameter defaults of `0.01` to stabilize model params.
- **Scheduler**: Cosine scheduler with warmup. Warmup ratio defaults to `0.1`.
- **Mixed Precision**: Uses `torch.amp.autocast("cuda")`. Defaults to `bf16` on Cloud GPU profile (`modal.yaml`) or `fp16` on local GPU profile (`rtx2050.yaml`).
- **Gradient Scaling**: Configured dynamically. Scaler is enabled ONLY when precision is set to `fp16` (bfloat16 precision does not require scaling).
- **Gradient Accumulation**: Divides computed loss tensors by `gradient_accumulation_steps` before executing backwards pass, stepping optimizer only once target accumulation count is matched.
- **Early Stopping**: Validation loss checker halts training if validation loss fails to improve for `early_stopping_patience` epochs.
- **Checkpointing**:
  - Saves model state, optimizer state, scheduler state, current epoch index, and validation loss to checkpoint files.
  - VisionTrainer outputs `vision_best.pt` and `vision_last.pt` under `artifacts/checkpoints`.
  - Multimodal trainer outputs `best_model.pt` and `last_model.pt`.
  - Models are exported under `artifacts/models/final/` upon completion.

---

## 7. Evaluation System

Metrics are calculated using scikit-learn helper functions.

### Vision Metrics
- **Location**: [src/training/vision/metrics.py:L22-L28](file:///home/reshad/Projects/multi_new/src/training/vision/metrics.py#L22-L28)
- **Metrics Calculated**:
  - `accuracy`: Accuracy classification score.
  - `top3_accuracy` / `top5_accuracy`: Top-k accuracy check.
  - `precision` / `recall` / `f1`: Macro average configurations with `zero_division=0`.
  - `confusion_matrix`: Confusion matrix grid.
  - `classification_report`: Nested dict output.

### Multimodal Metrics
- **Location**: [src/training/metrics.py:L159-L167](file:///home/reshad/Projects/multi_new/src/training/metrics.py#L159-L167)
- **Conversion**: Converts outputs through a Sigmoid threshold operation (`probabilities >= threshold` where threshold defaults to `0.5`).
- **Metrics Calculated**:
  - `subset_accuracy`: Exact match ratio (every single class index prediction must exactly match label tensor coordinates).
  - `label_accuracy`: Mean label classification accuracy across all 41 indices.
  - `precision` / `recall` / `f1`: Macro averages.
  - `hamming_loss`: Fraction of label coordinates that are incorrectly predicted.
  - `jaccard`: Jaccard similarity coefficient score.
  - `roc_auc` / `pr_auc`: Macro averages of Receiver Operating Characteristic and Precision-Recall area under the curve.

---

## 8. MLflow Integration

- **Backend Configuration**: SQLite database tracking storage configured via [src/training/vision/mlflow_utils.py](file:///home/reshad/Projects/multi_new/src/training/vision/mlflow_utils.py).
- **Tracking URI**: `sqlite:///artifacts/mlflow/mlflow.db` (falls back to local `mlruns/` directories if setup is skipped).
- **Logged Properties**:
  - Hyperparameters: `model`, `max_length`, `image_size`, `learning_rate`, `epochs`, `batch_size`, `mixed_precision`, `gradient_accumulation`.
  - Metrics: All loss and score metrics logged per epoch.
  - Artifacts: Active PyTorch models logged to tracking storage via `mlflow.pytorch.log_model`.
- **How to Launch UI**:
  ```bash
  mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5000
  ```

---

## 9. Dependency Graph

The execution dependencies are structured as follows:

```text
src/train_vision.py
 ├── src/utils/config.py ── load_config
 ├── src/utils/seed.py ── set_seed
 ├── src/utils/device.py ── get_device
 ├── src/data/vision_dataset.py ── VisionDataset
 │    └── PIL.Image
 ├── src/models/document_classifier.py ── DocumentClassifier
 │    └── src/models/document_encoder.py ── DocumentEncoder
 │         └── timm
 └── src/training/vision_trainer.py ── VisionTrainer
      ├── src/training/vision/early_stopping.py ── EarlyStopping
      ├── src/training/vision/metrics.py ── compute_metrics
      │    └── sklearn.metrics
      └── src/training/vision/checkpoint.py ── checkpoint_helpers

src/train.py
 ├── src/utils/config.py ── load_config
 ├── src/utils/seed.py ── set_seed
 ├── src/utils/device.py ── get_device
 ├── src/data/dataset.py ── MultimodalRiskDataset
 │    └── src/data/transforms.py ── build_image_transform
 ├── src/data/collate.py ── multimodal_collate_fn
 ├── src/models/risk_auditor.py ── RiskAuditorFusionModel
 │    ├── src/models/document_encoder.py ── DocumentEncoder
 │    ├── src/models/text_encoder.py ── TextEncoder
 │    │    └── transformers.AutoModel
 │    └── src/models/fusion.py ── CrossAttentionFusion
 ├── src/training/losses.py ── build_loss
 └── src/training/trainer.py ── Trainer
      ├── src/training/callbacks.py ── EarlyStopping, save_checkpoint
      └── src/training/metrics.py ── compute_multilabel_metrics
```

---

## 10. Runtime Flow

### Training Flow (Stage-2 Multimodal Example)
1. Invoke execution of [src/train.py](file:///home/reshad/Projects/multi_new/src/train.py).
2. `load_config` reads hyperparameter options from `configs/modal.yaml`.
3. `set_seed` configures random seed states across Numpy, Python, and PyTorch.
4. `MultimodalRiskDataset` initializes training and validation datasets.
5. `DataLoader` instances partition the datasets into batches, using `multimodal_collate_fn` to collate tensors.
6. Instantiates `RiskAuditorFusionModel` and transfers parameters to the device.
7. Optimizer (AdamW) and schedule variables are initialized.
8. Initializes `Trainer` and invokes `.fit()`.
9. `.fit()` triggers the training loop:
   - For each epoch, runs `.train_one_epoch()` processing batches.
   - Computes model outputs and calculates gradients with mixed precision (AMP).
   - If gradient accumulation steps match, applies gradient clipping and updates model parameters.
   - Tracks metrics, then runs `.validate()`.
   - Compares validation loss to update best checkpoints.
   - Logs metrics to MLflow.
   - Evaluates early stopping checks.
10. Saves final serialized model states.

### Validation / Evaluation Flow
1. Model parameters are frozen (`model.eval()`) and gradient calculation is disabled (`torch.no_grad()`).
2. Iterates over validation loader batches.
3. Computes metrics (Subset accuracy, precision, recall, confusion matrix, AUC).
4. Saves metric dictionary representations and reports to `outputs/reports/`.
5. Exports plots to `outputs/confusion_matrices/` and `outputs/roc_curves/`.

---

## 11. Known Issues

### 1. `transforms.py` Argument Mismatch
- **Root Cause**: [src/data/dataset.py:L63](file:///home/reshad/Projects/multi_new/src/data/dataset.py#L63) invokes `build_image_transform(image_size=image_size, train=True)`. However, the definition in [src/data/transforms.py:L4](file:///home/reshad/Projects/multi_new/src/data/transforms.py#L4) only accepts `image_size: int` and has no `train` keyword parameter.
- **Current Status**: Active bug. Running the multimodal dataset loader causes a crash.
- **Recommended Fix**: Update the signature of `build_image_transform` in `transforms.py` to accept the `train` parameter, and implement conditional transform logic to bypass training augmentations during validation.

### 2. Epoch-Level Scheduler Stepping
- **Root Cause**: The Cosine schedule warmup step counts (`total_steps`) are calculated in `train.py` and `train_vision.py` as `len(train_loader) * epochs` (batch-level steps). However, the trainer loops step the scheduler once per *epoch* rather than once per *batch*:
  - `trainer.py` [L603-L605](file:///home/reshad/Projects/multi_new/src/training/trainer.py#L603-L605): `if self.scheduler is not None: self.scheduler.step()`.
- **Current Status**: Active bug. Warmup logic is computed incorrectly.
- **Recommended Fix**: Relocate `scheduler.step()` inside the batch iteration loops after the optimizer parameters are updated.

### 3. Classification Layout Output Mismatch
- **Root Cause**: `prepare_vision_dataset.py` configures classification targets based on 6 classes. However, the default constructor of `DocumentClassifier` sets `num_classes=16`:
  - `document_classifier.py` [L23](file:///home/reshad/Projects/multi_new/src/models/document_classifier.py#L23): `num_classes=16`.
- **Current Status**: Mismatch bug. Running vision training with default parameters causes shape mismatches against layout class labels.
- **Recommended Fix**: Pass the correct number of classes (`num_classes=6`) to `DocumentClassifier` during initialization in `train_vision.py`, or update the default value in the model definition.

### 4. MLflow Missing Tracker URIs in Multimodal Loop
- **Root Cause**: `Trainer.fit()` logs metrics directly using `mlflow` without first setting the SQLite tracking URI via `initialize_mlflow()`. Only the vision trainer imports and calls this tracking configuration.
- **Current Status**: Logs are split across different storage formats (SQLite database vs local folders).
- **Recommended Fix**: Import and run `initialize_mlflow` inside `src/train.py` before starting training.

---

## 12. Performance Notes

### Memory & VRAM Optimizations (RTX 2050 Constraints)
- Local environments running on constrained GPUs (e.g., RTX 2050 with 4GB VRAM) must run with features constrained:
  - Freeze the text encoder backbone: `freeze.text_encoder: true`
  - Freeze the vision encoder backbone: `freeze.vision_encoder: true`
  - Disable processing of image tensors to run in text-only mode: `vision.enabled: false`
  - Limit training configurations: `batch_size: 2` combined with `gradient_accumulation_steps: 4` (effective batch size of 8).
  - Use lower sequence limits: `max_length: 256` and lower image resolutions: `image_size: 224`.

### Cloud GPU Optimizations (Modal)
- Use bfloat16 mixed-precision autocasting (`mixed_precision: bf16`) to speed up training on newer architectures.
- VisionTrainer applies memory format optimization on CUDA GPUs to speed up convolutional operations:
  ```python
  self.model = self.model.to(memory_format=torch.channels_last)
  ```
- Dataloaders configure pinning memory and persistent data workers (`persistent_workers=True`) when execution runs on CUDA backends to maximize throughput.

---

## 13. Current Technical Debt

### Skeletons and Missing Functionality
- **API Server & Routing**: The FastAPI endpoints inside [src/api/app.py](file:///home/reshad/Projects/multi_new/src/api/app.py) and related files are empty placeholders.
- **Production Inference Codes**: Standalone prediction tools ([src/predict.py](file:///home/reshad/Projects/multi_new/src/predict.py), [src/serve.py](file:///home/reshad/Projects/multi_new/src/serve.py), and `src/inference/`) contain no serving logic.
- **Empty Documentation Guides**: Markdown files inside `docs/` contain only headers or empty space.
- **Empty Test Suite**: All unit, integration, and API test files inside `tests/` contain no executable test cases.

### Code Duplication
- **Duplicate Metrics Logic**:
  - `compute_metrics` in [src/training/vision/metrics.py](file:///home/reshad/Projects/multi_new/src/training/vision/metrics.py) contains duplicate calculations also defined in `compute_vision_metrics` inside [src/training/metrics.py](file:///home/reshad/Projects/multi_new/src/training/metrics.py).
- **Duplicate Checkpointing Blocks**:
  - `CheckpointManager` class inside [src/training/checkpoint.py](file:///home/reshad/Projects/multi_new/src/training/checkpoint.py) replicates save/load functions defined in [src/training/callbacks.py](file:///home/reshad/Projects/multi_new/src/training/callbacks.py) and `src/training/vision/checkpoint.py`.

### Unused Code
- **Unused ResNet Builder**:
  - `build_vision_encoder` in [src/models/vision_encoder.py](file:///home/reshad/Projects/multi_new/src/models/vision_encoder.py) is defined but never imported or called by the model definitions.
- **Unused Classifier Head Definition**:
  - `RiskClassifier` in [src/models/classifier.py](file:///home/reshad/Projects/multi_new/src/models/classifier.py) is unused; `RiskAuditorFusionModel` defines its classification head inline.

---

## 14. Development Workflow

### Setup Instructions
1. Clone the repository and configure a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set up environment variables from the template:
   ```bash
   cp .env.example .env
   ```

### Preprocessing and Dataset Preparation Commands
- **Generate Multimodal Text Splits (Stage-2)**:
  ```bash
  python scripts/prep_cuad.py
  ```
  *(Processes `data/raw/cuad-main/data/CUADv1.json` and outputs data splits to `data/processed/`)*
  
- **Generate Vision Classification splits (Stage-1)**:
  ```bash
  python scripts/prepare_vision_dataset.py
  ```
  *(Scans document page layout image folders and outputs splits to `data/processed/vision/`)*
  
- **Convert Absolute Image Paths to Relative**:
  ```bash
  python scripts/convert_vision_csv_to_relative.py
  ```

### Training Execution Commands
- **Run Stage-1 Vision Classifier Training**:
  ```bash
  python src/train_vision.py --config configs/rtx2050.yaml
  ```
- **Run Stage-2 Multimodal Auditor Training**:
  - *Note*: Hardcoded to use `configs/modal.yaml` by default. Update the path string variable inside `src/train.py` before executing:
  ```bash
  python src/train.py
  ```

### MLflow UI Server Command
```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db
```

---

## 15. Instructions For Future AI Agents

### Critical Architectural Constraints
- **Multimodal Feature Fusion Dimensions**:
  - The fusion layer `CrossAttentionFusion` expects text and vision representations to have an embed dimension of `1024`.
  - When swapping backbones (e.g., to lower dimension models like ResNet18 outputting 512 dimensions), you MUST update projection settings in configuration files so projection dimensions map correctly.

### Modifying Code Guidelines
- **Always Validate Augmentation Pipelines**:
  - Augmentation scripts must be checked for target keyword arguments matching the transforms module signatures.
- **Retain Exception Handling**:
  - Keep fallback protections in dataset load routines (e.g., returning white placeholders) to avoid crashing training jobs if single image assets are corrupted.
- **Ensure Scheduler Step Sync**:
  - If refactoring train loops, always ensure scheduler variables are updated per-batch to keep warmup schedules in sync.

### Coding Conventions
- **Explicit Parameter Configuration**:
  - Avoid parsing options from command line arguments inside utility files; instead, pass config parameters through explicit arguments.
- **Clean Namespace Boundaries**:
  - Do not cross-import components across `src/training/vision/` and `src/training/` unless abstracting shared classes to root utility files.
