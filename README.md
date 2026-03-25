# Skin Tone Classification in Dermatology

A fairness-aware machine learning pipeline that classifies dermatology images into all 6 individual Fitzpatrick skin tone types (I-VI) and quantifies performance disparities across each type. Built with PyTorch, trained on Google Colab, and deployed to Google Cloud Vertex AI.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Label Strategy](#label-strategy)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Evaluation & Fairness](#evaluation--fairness)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage Guide](#usage-guide)
- [Google Cloud Deployment](#google-cloud-deployment)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Technical Details](#technical-details)
- [Future Roadmap](#future-roadmap)
- [References](#references)

---

## Problem Statement

Dermatology AI systems frequently exhibit reduced diagnostic performance on darker skin tones. This bias stems from:

- **Underrepresentation** — Training datasets disproportionately contain lighter skin tones
- **Class imbalance** — Fitzpatrick Types V-VI are severely underrepresented
- **Shortcut learning** — Models may rely on lesion discoloration rather than morphological features
- **Poor metadata standardization** — Inconsistent skin tone labeling across datasets

This project builds a reproducible pipeline to:

1. **Train** a skin tone classifier on Fitzpatrick-labeled dermatology images
2. **Balance** underrepresented classes (types 3-6) via augmentation
3. **Measure** per-class performance across skin tone groups
4. **Quantify** fairness disparities using recall-based gap analysis
5. **Compare** custom fine-tuned models against an AutoML baseline
6. **Deploy** to Google Cloud Vertex AI as a cloud-native ML workflow

---

## Project Architecture

```
                                    Vertex AI
                                   Model Registry
                                        ^
                                        |
Fitzpatrick17k ──> Data Cleaning ──> Augmentation ──> Colab Training ──> Evaluation
    CSV + Images     (prepare.py)   (augment_       (trainer.py)     (metrics.py +
                          |          minority.py)        |            fairness.py)
                          v               |              v                 |
                     Cleaned CSVs    Balanced CSV    EfficientNetV2-S     Gap Report
                     Train/Val/Test  (types 3-6     ResNet50             Confusion
                                      augmented)    AutoML Baseline      Matrices
```

**Pipeline Flow:**

| Step | Artifact | What Happens |
|------|----------|-------------|
| 1 | `08_balance_and_automl.ipynb` | Download Fitzpatrick17k from GCS, augment types 3-6 via 5 transform buckets, upload balanced dataset, generate AutoML manifest |
| 2 | `scripts/train.py` | CLI training with EfficientNetV2-S or ResNet50 (two-phase transfer learning) |
| 3 | Vertex AI Console | Train AutoML model using the generated manifest CSV |

---

## Dataset

### Fitzpatrick17k

- **Source:** [github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
- **Paper:** Groh et al., "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology" (CVPR 2021 Workshop)
- **Size:** ~16,577 clinical dermatology images (~10,500 recoverable from GCS)
- **Labels:** Fitzpatrick skin types I-VI, disease condition labels, partition labels
- **Format:** CSV metadata with image URLs; images stored in `gs://fitzpatrick-dataset/all_images/`

**CSV Columns (upstream):**

| Upstream Column | Renamed To | Type | Description |
|----------------|-----------|------|-------------|
| `url_alphanum` | `hasher` | string | Unique image identifier (filename) |
| `fitzpatrick_scale` | `fitzpatrick` | int (1-6) | Fitzpatrick skin type label |
| `url` | `url` | string | Source URL for image download |
| `label` | `label` | string | Dermatological condition name |
| `md5hash` | `md5hash` | string | Image content hash |

### Data Cleaning Pipeline

Images pass through 5 sequential filters before training:

1. **Label validation** — Drop rows with missing or invalid Fitzpatrick values (not in 1-6)
2. **Image validation** — Remove files that don't exist on disk or fail PIL verification (corrupted)
3. **Human/skin filtering** — Remove non-human images using heuristics:
   - Minimum dimensions: 50x50 pixels
   - Must be RGB color mode (filters grayscale diagrams)
   - Color standard deviation > 10 (filters solid-color fills and text images)
4. **Deduplication** — Perceptual hash (pHash) via `imagehash` to remove visually identical images
5. **Class distribution report** — Log counts, percentages, and imbalance ratio

### Class Balancing (Augmentation)

Types 3-6 are augmented to a minimum of 2,500 images each using 5 transform buckets at equal 20% proportion:

| Bucket | Transform | Probability |
|--------|-----------|-------------|
| flip | HorizontalFlip | p=1.0 |
| rotate | Rotate ±30° | p=1.0 |
| crop | RandomResizedCrop (0.8-1.0) | p=1.0 |
| noise | GaussNoise (std 0.02-0.06) | p=1.0 |
| combined | All four together | varied |

**Forbidden transforms:** brightness, contrast, color jitter, blurring (per professor guidance — these corrupt the skin tone signal).

---

## Label Strategy

### 6-Class Direct Encoding

Each of the 6 Fitzpatrick skin types is treated as its own class:

| Fitzpatrick Type | Numeric Label | Description |
|-----------------|---------------|-------------|
| **I** | 0 | Very fair, always burns |
| **II** | 1 | Fair, usually burns |
| **III** | 2 | Medium, sometimes burns |
| **IV** | 3 | Olive, rarely burns |
| **V** | 4 | Brown, very rarely burns |
| **VI** | 5 | Dark brown/black, never burns |

**Why 6 individual classes?**
- Preserves the full granularity of the Fitzpatrick scale
- Enables per-type fairness analysis (detect disparities between any two types)
- More clinically meaningful — each type has distinct dermatological properties
- Class imbalance handled via inverse-frequency class weighting + augmentation

### Data Split

| Split | Ratio | Stratified |
|-------|-------|------------|
| Train | 70% | Yes, by skin tone label |
| Validation | 15% | Yes |
| Test | 15% | Yes |

Stratification ensures each split has proportional representation of all 6 classes.

---

## Model Architecture

### Custom Models

Two pretrained CNN backbones fine-tuned via transfer learning:

#### EfficientNetV2-S (Primary)

- **Pretrained on:** ImageNet-1K (`EfficientNet_V2_S_Weights.IMAGENET1K_V1`)
- **Feature dimension:** 1,280
- **Classification head:** `Dropout(0.3) -> Linear(1280, 6)`

#### ResNet50 (Comparison Baseline)

- **Pretrained on:** ImageNet-1K (`ResNet50_Weights.IMAGENET1K_V2`)
- **Feature dimension:** 2,048
- **Classification head:** `Dropout(0.3) -> Linear(2048, 6)`

### AutoML Baseline

- **Platform:** Vertex AI AutoML Image Classification
- **Configuration:** CLOUD model type, 8 node-hours budget
- **Purpose:** Zero-effort comparison

---

## Training Strategy

### Two-Phase Transfer Learning

**Phase 1 — Classifier Head Only (Epochs 1-5)**
- Backbone weights are **frozen** (no gradient computation)
- Only the classification head (Dropout + Linear) is trained

**Phase 2 — Fine-Tuning (Epochs 6-20)**
- Last 2 blocks of the backbone are **unfrozen**
- All trainable parameters updated with lower learning rate
- Optimizer, scheduler, and early stopping are re-initialized

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | `CrossEntropyLoss` with inverse-frequency class weights |
| Optimizer | `Adam` (lr=1e-4, weight_decay=1e-4) |
| Scheduler | `CosineAnnealingLR` (T_max=phase length) |
| Batch Size | 32 |
| Epochs | 20 (max) |
| Early Stopping | Patience=5 on validation loss |
| Input Size | 224x224 |
| Dropout | 0.3 |
| Random Seed | 42 |

### Data Augmentation

| Augmentation | Parameter |
|-------------|-----------|
| Random Horizontal Flip | p=0.5 |
| Random Rotation | ±30 degrees |
| Random Resized Crop | scale 0.8-1.0 |
| Gaussian Noise | std=0.03, p=0.2 |
| Normalization | ImageNet mean/std |

**Forbidden:** brightness, contrast, color jitter, blurring.

Validation and test sets receive only resize + normalization (no augmentation).

---

## Evaluation & Fairness

### Metrics

| Metric | Scope | Purpose |
|--------|-------|---------|
| Accuracy | Overall | General correctness |
| Macro F1 | Overall | Balanced performance across classes |
| ROC-AUC | Overall + per-class | Discrimination ability |
| Precision | Per-class (Fitz I-VI) | False positive rate per type |
| Recall | Per-class (Fitz I-VI) | False negative rate per type |
| Confusion Matrix | 6x6 | Misclassification patterns |

### Fairness Gap Analysis

```
Fairness Gap = max(recall across classes) - min(recall across classes)
```

- Gap < 5% — Fair
- Gap 5-15% — Moderate
- Gap > 15% — **Significant**

---

## Project Structure

```
NST_Class/
├── src/                           # Core Python package
│   ├── data/
│   │   ├── prepare.py             # Data cleaning pipeline (11 functions)
│   │   ├── dataset.py             # FitzpatrickDataset (PyTorch Dataset)
│   │   └── transforms.py          # Train/eval augmentation pipelines
│   ├── models/
│   │   ├── backbone.py            # EfficientNetV2-S & ResNet50 wrappers
│   │   └── classifier.py          # SkinToneClassifier (backbone + FC head)
│   ├── training/
│   │   ├── config.py              # TrainingConfig dataclass (YAML-loadable)
│   │   └── trainer.py             # Two-phase training loop + early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy, F1, precision, recall, ROC-AUC
│   │   ├── fairness.py            # Fairness gap analysis + cross-model comparison
│   │   └── confusion.py           # Confusion matrix & fairness bar chart plots
│   └── utils/
│       ├── gcs.py                 # GCS upload + AutoML manifest generation
│       └── logging.py             # Weights & Biases integration
├── notebooks/
│   └── 08_balance_and_automl.ipynb # Balance dataset + generate AutoML manifest
├── scripts/
│   ├── train.py                   # CLI training entrypoint
│   ├── augment_minority.py        # Albumentations augmentation (5 buckets)
│   ├── download_all_sources.py    # Multi-source image downloader
│   ├── add_gaussian_noise.py      # Gaussian noise augmentation
│   └── add_rotation.py            # Rotation augmentation
├── tests/                         # Unit + integration tests
│   ├── test_prepare.py            # Data pipeline tests
│   ├── test_augment_minority.py   # Augmentation pipeline tests
│   ├── test_gcs.py                # GCS utilities tests
│   ├── test_logging_utils.py      # W&B integration tests
│   ├── test_download_all_sources.py # Downloader tests
│   ├── test_add_gaussian_noise.py # Noise augmentation tests
│   ├── test_add_rotation.py       # Rotation augmentation tests
│   ├── test_backbone.py           # Backbone loading tests
│   ├── test_classifier.py         # Classifier tests
│   ├── test_dataset.py            # Dataset loading tests
│   ├── test_trainer.py            # Training loop tests
│   ├── test_metrics.py            # Evaluation metrics tests
│   ├── test_fairness.py           # Fairness gap tests
│   ├── test_config.py             # Config loading tests
│   ├── test_transforms.py         # Transform pipeline tests
│   └── test_integration.py        # End-to-end smoke test
├── configs/
│   └── default.yaml               # Training configuration
├── Dockerfile                     # Vertex AI custom training container
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # pytest config
└── docs/                          # Design documents & project log
```

---

## Setup & Installation

### Google Colab (Recommended)

Open `notebooks/08_balance_and_automl.ipynb` in Colab and run all cells. Cell 1 automatically clones the repo, installs dependencies, and downloads metadata.

### Local Setup

```bash
git clone https://github.com/AayushBaniya2006/NST_Class.git
cd NST_Class
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## Usage Guide

### Notebook: Balance + AutoML

Open `notebooks/08_balance_and_automl.ipynb` in Colab:

1. **Cells 1-3:** Setup, authenticate GCS, configure augmentation targets
2. **Cell 4:** Download all Fitzpatrick17k images from `gs://fitzpatrick-dataset/`
3. **Cell 5:** Validate downloaded images (parallel)
4. **Cell 6:** Augment types 3-6 using 5 transform buckets (20% each)
5. **Cells 7-9:** Visual spot-check, build balanced CSV, upload to GCS
6. **Cell 10:** Generate AutoML manifest (`ML_USE,GCS_PATH,LABEL` format)
7. Import `gs://augmentedbuckets/automl_manifest.csv` in Vertex AI to train

### CLI Training

```bash
python scripts/train.py --config configs/default.yaml --no-wandb
python scripts/train.py --backbone resnet50 --epochs 10 --lr 0.001
```

---

## Configuration Reference

All training parameters are defined in `configs/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| backbone | `efficientnet_v2_s` | Model backbone |
| num_classes | 6 | Output classes |
| dropout | 0.3 | Dropout before head |
| freeze_backbone | true | Freeze in phase 1 |
| unfreeze_after_epochs | 5 | When to start fine-tuning |
| unfreeze_n_blocks | 2 | Blocks to unfreeze |
| epochs | 20 | Max training epochs |
| batch_size | 32 | Batch size |
| learning_rate | 0.0001 | Adam learning rate |
| early_stopping_patience | 5 | Epochs before stopping |
| use_class_weights | true | Inverse-frequency weighting |
| image_size | 224 | Input resolution |

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Technical Details

### Class Weight Computation

```python
weight[i] = total_samples / (num_classes * class_count[i])
```

### Perceptual Deduplication

Uses `imagehash.phash()` — catches visually identical images even after re-encoding or minor crops.

### Backbone Unfreezing

- **ResNet50:** Unfreezes `layer4` and `layer3`
- **EfficientNetV2-S:** Unfreezes the last 2 feature blocks

---

## References

1. Groh, M. et al. (2021). "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology." CVPR 2021 ISIC Workshop.
2. Fitzpatrick, T.B. (1988). "The validity and practicality of sun-reactive skin types I through VI." Archives of Dermatology.
3. Daneshjou, R. et al. (2022). "Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set." Science Advances.
4. Tan, M. & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training." ICML 2021.
5. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.

---

## License

This project is for academic and research purposes. The Fitzpatrick17k dataset is subject to its own license terms — see [the dataset repository](https://github.com/mattgroh/fitzpatrick17k) for details.
