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
2. **Measure** per-class performance across skin tone groups
3. **Quantify** fairness disparities using recall-based gap analysis
4. **Compare** custom fine-tuned models against an AutoML baseline
5. **Deploy** to Google Cloud Vertex AI as a cloud-native ML workflow

---

## Project Architecture

```
                                    Vertex AI
                                   Model Registry
                                        ^
                                        |
Fitzpatrick17k ──> Data Cleaning ──> Colab Training ──> Evaluation ──> Fairness Analysis
    CSV + Images     (prepare.py)     (trainer.py)     (metrics.py)    (fairness.py)
                          |                |                |               |
                          v                v                v               v
                     Cleaned CSVs    EfficientNetV2-S   Per-Class      Gap Report
                     Train/Val/Test  ResNet50           Metrics        Bar Charts
                                     AutoML Baseline    Confusion      Comparison
                                                        Matrices       Table
```

**Pipeline Flow:**

| Step | Notebook | What Happens |
|------|----------|-------------|
| 1 | `01_data_exploration.ipynb` | Load Fitzpatrick17k, clean data, validate images, filter non-human images, deduplicate, encode labels, stratified split |
| 2 | `04_automl_baseline.ipynb` | Upload images to GCS, train Vertex AI AutoML model as zero-effort baseline |
| 3 | `02_training.ipynb` | Train EfficientNetV2-S and ResNet50 with two-phase transfer learning |
| 4 | `03_evaluation.ipynb` | Evaluate all 3 models, compute per-class metrics, fairness gap analysis, cross-model comparison |

---

## Dataset

### Fitzpatrick17k

- **Source:** [github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
- **Paper:** Groh et al., "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology" (CVPR 2021 Workshop)
- **Size:** ~16,577 clinical dermatology images
- **Labels:** Fitzpatrick skin types I-VI, disease condition labels, partition labels
- **Format:** CSV metadata with image URLs; images downloaded separately

**CSV Columns Used:**

| Column | Type | Description |
|--------|------|-------------|
| `hasher` | string | Unique image identifier (filename) |
| `url` | string | Source URL for image download |
| `fitzpatrick` | int (1-6) | Fitzpatrick skin type label |
| `label` | string | Dermatological condition name |
| `three_partition_label` | string | High-level condition category |
| `md5hash` | string | Image content hash |

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
- Class imbalance handled via inverse-frequency class weighting

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

- **Pretrained on:** ImageNet-1K (IMAGENET1K_V1 weights)
- **Feature dimension:** 1,280
- **Classification head:** Dropout(0.3) -> Linear(1280, 6)
- **Why:** Best accuracy-to-compute ratio among modern CNNs. Compound scaling (depth + width + resolution) produces strong features for fine-grained classification.

#### ResNet50 (Comparison Baseline)

- **Pretrained on:** ImageNet-1K (IMAGENET1K_V2 weights)
- **Feature dimension:** 2,048
- **Classification head:** Dropout(0.3) -> Linear(2048, 6)
- **Why:** Most widely studied backbone in transfer learning research. Strong, stable baseline with well-understood behavior.

### AutoML Baseline

- **Platform:** Vertex AI AutoML Image Classification
- **Configuration:** CLOUD model type, 8 node-hours budget
- **Purpose:** Zero-effort comparison — demonstrates the value of custom training choices (class weighting, two-phase training, augmentation)

---

## Training Strategy

### Two-Phase Transfer Learning

Training proceeds in two distinct phases:

**Phase 1 — Classifier Head Only (Epochs 1-5)**
- Backbone weights are **frozen** (no gradient computation)
- Only the classification head (Dropout + Linear) is trained
- Fast convergence on the new task without disrupting pretrained features

**Phase 2 — Fine-Tuning (Epochs 6-20)**
- Last 2 blocks of the backbone are **unfrozen**
- All trainable parameters updated with lower learning rate
- Optimizer and scheduler are re-initialized for the new parameter set
- Allows the backbone to adapt its high-level features to dermatology images

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss with inverse-frequency class weights |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch Size | 32 |
| Epochs | 20 (max) |
| Early Stopping | Patience=5 on validation loss |
| Input Size | 224x224 |

### Class Weighting

Inverse-frequency weighting compensates for class imbalance:

```
weight[class] = total_samples / (num_classes * class_count)
```

Minority classes (typically Fitz V and VI) receive higher loss weight, forcing the model to pay more attention to underrepresented skin tones.

### Data Augmentation (Training Only)

| Augmentation | Parameter |
|-------------|-----------|
| Random Horizontal Flip | p=0.5 |
| Random Rotation | +/- 15 degrees |
| Color Jitter: Brightness | +/- 0.2 |
| Color Jitter: Contrast | +/- 0.2 |
| Color Jitter: Saturation | +/- 0.2 |
| Color Jitter: Hue | +/- 0.1 |
| Normalization | ImageNet mean/std |

Validation and test sets receive only resize + normalization (no augmentation).

---

## Evaluation & Fairness

### Metrics

All metrics are computed on the held-out **test set** for each model:

| Metric | Scope | Purpose |
|--------|-------|---------|
| Accuracy | Overall | General correctness |
| Macro F1 | Overall | Balanced performance across classes |
| ROC-AUC | Overall + per-class (OVR) | Discrimination ability |
| Precision | Per-class (Fitz I-VI) | False positive rate per type |
| Recall | Per-class (Fitz I-VI) | False negative rate per type |
| F1 | Per-class (Fitz I-VI) | Harmonic mean per type |
| Confusion Matrix | 6x6 | Misclassification patterns |

### Fairness Gap Analysis

The fairness gap measures the **maximum disparity in recall** across all 6 Fitzpatrick types:

```
Fairness Gap = max(recall across classes) - min(recall across classes)
```

**Interpretation:**
- Gap < 5% — Fair: minimal performance disparity
- Gap 5-15% — Moderate: some disparity, may warrant investigation
- Gap > 15% — **Significant**: the model performs meaningfully worse on at least one skin tone type

**Example output:**

| Fitzpatrick Type | Recall |
|-----------------|--------|
| I | 0.85 |
| II | 0.82 |
| III | 0.80 |
| IV | 0.76 |
| V | 0.60 |
| VI | 0.50 |

Gap = 0.85 - 0.50 = **0.35 (35%)** — Significant fairness issue.

### Cross-Model Comparison

The evaluation notebook produces a side-by-side comparison:

| Model | Accuracy | Macro F1 | Fairness Gap | Significant? |
|-------|----------|----------|-------------|-------------|
| EfficientNetV2-S | — | — | — | — |
| ResNet50 | — | — | — | — |
| AutoML Baseline | — | — | — | — |

Key question answered: **"Did custom training reduce the fairness gap vs AutoML?"**

---

## Project Structure

```
skin_tone_classifier/
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
│       ├── gcs.py                 # GCS upload/download + AutoML manifest
│       └── logging.py             # Weights & Biases integration
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data cleaning & distribution analysis
│   ├── 02_training.ipynb          # Model training with W&B logging
│   ├── 03_evaluation.ipynb        # Evaluation & fairness comparison
│   └── 04_automl_baseline.ipynb   # Vertex AI AutoML baseline
├── scripts/
│   ├── train.py                   # CLI entrypoint (local or Vertex AI)
│   └── upload_to_vertex.py        # Upload model to Vertex AI Model Registry
├── tests/                         # 69 tests (unit + integration)
│   ├── test_prepare.py            # 43 tests for data pipeline
│   ├── test_backbone.py           # 5 tests for backbone loading & forward pass
│   ├── test_classifier.py         # 4 tests for classifier freeze/unfreeze
│   ├── test_dataset.py            # 4 tests for dataset loading
│   ├── test_metrics.py            # 3 tests for evaluation metrics
│   ├── test_fairness.py           # 4 tests for fairness gap
│   ├── test_trainer.py            # 5 tests for class weights & early stopping
│   └── test_integration.py        # 1 end-to-end smoke test
├── configs/
│   └── default.yaml               # Full experiment configuration
├── Dockerfile                     # Vertex AI custom training container
├── requirements.txt               # Python dependencies
└── docs/plans/                    # Design & implementation documents
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Google Cloud account with Vertex AI enabled
- Weights & Biases account (free for students)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/skin-tone-classifier.git
cd skin-tone-classifier

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest tests/ -v
```

### Google Colab Setup

```python
# In the first cell of any notebook:
!git clone https://github.com/YOUR_USERNAME/skin-tone-classifier.git
%cd skin-tone-classifier
!pip install -q -r requirements.txt
```

---

## Usage Guide

### Step 1: Obtain the Dataset

```bash
# Download fitzpatrick17k.csv from the official repository
# Place it at: data/fitzpatrick17k.csv

# Option A: Download images via the notebook (slow, uses URLs from CSV)
# Option B: Request bulk download from the dataset authors
# Place images at: data/images/
```

### Step 2: Data Exploration & Cleaning

Open `notebooks/01_data_exploration.ipynb` in Colab and run all cells. This will:
- Load and inspect the raw CSV
- Run the 5-step cleaning pipeline
- Display class distribution histograms
- Perform stratified train/val/test split
- Save cleaned CSVs to `data/cleaned/`

### Step 3: AutoML Baseline (Optional)

Open `notebooks/04_automl_baseline.ipynb`. Fill in your GCP project ID and bucket name:
```python
PROJECT_ID = "your-gcp-project-id"
BUCKET_NAME = "your-gcs-bucket"
```
Run all cells to train the Vertex AI AutoML baseline.

### Step 4: Custom Model Training

Open `notebooks/02_training.ipynb`:
1. Ensure runtime is set to **GPU** (Runtime -> Change runtime type -> T4)
2. Run all cells to train EfficientNetV2-S
3. Change `backbone="resnet50"` and re-run to train ResNet50
4. Monitor training at [wandb.ai](https://wandb.ai)

### Step 5: Evaluation & Fairness

Open `notebooks/03_evaluation.ipynb` and run all cells to:
- Load trained models and run inference on test set
- Compute all metrics and confusion matrices
- Perform fairness gap analysis
- Generate cross-model comparison chart

### Alternative: CLI Training

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml --no-wandb

# Train with overrides
python scripts/train.py --backbone resnet50 --epochs 10 --lr 0.001 --batch-size 16

# Train for Vertex AI (with W&B)
python scripts/train.py --config configs/default.yaml --backbone efficientnet_v2_s
```

---

## Google Cloud Deployment

### GCS Bucket Structure

```
gs://skin-tone-project/
├── images/              # All dermatology images
├── automl/
│   └── manifest.csv     # AutoML training manifest
└── models/
    └── efficientnet_v2_s/  # Model artifacts
```

### Vertex AI Custom Training Job

```bash
# Build and push Docker container
docker build -t gcr.io/YOUR_PROJECT/skin-tone-trainer .
docker push gcr.io/YOUR_PROJECT/skin-tone-trainer

# Submit training job (from GCP console or gcloud CLI)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=skin-tone-training \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT/skin-tone-trainer
```

### Upload to Model Registry

```bash
python scripts/upload_to_vertex.py \
  --model-path gs://skin-tone-project/models/efficientnet_v2_s/ \
  --display-name skin-tone-classifier-v1 \
  --project YOUR_PROJECT_ID
```

---

## Configuration Reference

All training parameters are defined in `configs/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| data | image_size | 224 | Input image resolution |
| data | num_classes | 6 | Number of output classes |
| data | split_ratios | 70/15/15 | Train/val/test split |
| training | backbone | efficientnet_v2_s | Model backbone |
| training | learning_rate | 0.0001 | Initial learning rate |
| training | batch_size | 32 | Training batch size |
| training | epochs | 20 | Maximum training epochs |
| training | freeze_backbone | true | Freeze backbone in phase 1 |
| training | unfreeze_after_epochs | 5 | When to start fine-tuning |
| training | early_stopping_patience | 5 | Epochs without improvement before stopping |
| training | use_class_weights | true | Enable inverse-frequency weighting |
| augmentation | rotation_degrees | 15 | Random rotation range |
| augmentation | brightness | 0.2 | Color jitter brightness |
| evaluation | fairness_gap_threshold | 0.15 | Significance threshold for fairness gap |

---

## Testing

```bash
# Run all 69 tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_prepare.py -v      # Data pipeline (43 tests)
python -m pytest tests/test_backbone.py -v     # Backbone models (5 tests)
python -m pytest tests/test_classifier.py -v   # Classifier (4 tests)
python -m pytest tests/test_integration.py -v  # End-to-end smoke test (1 test)
```

The integration test (`test_integration.py`) creates synthetic images, runs the full pipeline (data prep -> training -> evaluation -> fairness), and verifies correctness on CPU in ~10 seconds.

---

## Technical Details

### Class Weight Computation

```python
# Inverse-frequency weighting
weight[i] = total_samples / (num_classes * count[i])

# Example: 6000 train samples across 6 classes
# Fitz I: 2000, Fitz II: 1500, Fitz III: 1000, Fitz IV: 800, Fitz V: 500, Fitz VI: 200
# weight_I  = 6000 / (6 * 2000) = 0.50
# weight_VI = 6000 / (6 * 200)  = 5.00  <-- 10x more weight than Fitz I
```

### Perceptual Deduplication

Uses `imagehash.phash()` (perceptual hash) rather than MD5:
- MD5 catches exact byte-for-byte duplicates only
- pHash catches visually identical images even after re-encoding, minor crops, or compression changes

### ImageNet Normalization

All images are normalized using ImageNet statistics because the backbones were pretrained on ImageNet:
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Backbone Unfreezing Strategy

- **ResNet50:** Unfreezes `layer4` and `layer3` (the last 2 residual blocks)
- **EfficientNetV2-S:** Unfreezes the last 2 feature blocks from `model.features`

Only high-level feature blocks are unfrozen — early layers (edges, textures) transfer well and don't need adaptation.

---

## Future Roadmap

| Milestone | Description |
|-----------|-------------|
| 2 | Introduce lesion segmentation (U-Net) to mask non-skin regions |
| 3 | Convert RGB to Lab color space to disentangle luminance from color |
| 4 | Multi-task learning: predict skin tone + condition simultaneously |
| 5 | Compare Fitzpatrick scale vs Monk Skin Tone scale |

### Potential Bias Mitigation Techniques

- **Adversarial debiasing** — Train the classifier to be invariant to skin tone features
- **Focal loss** — Down-weight easy examples, focus on hard-to-classify samples
- **Mixup augmentation** — Interpolate between images of different skin tone groups
- **Calibration** — Post-hoc probability calibration per skin tone group

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Severe class imbalance (Fitz V-VI underrepresented) | Model ignores minority classes | Inverse-frequency class weighting, augmentation |
| Shortcut learning (model uses lesion redness as proxy) | Inflated accuracy, poor generalization | Future: segmentation masking, Lab color space |
| Overfitting on small dataset | Poor test performance | Early stopping, dropout, validation monitoring |
| AutoML outperforms custom models | Undermines custom approach narrative | Frame as "what does custom training add beyond AutoML?" |

---

## Deliverables Checklist

| Deliverable | Location |
|-------------|----------|
| Dataset summary | Notebook 01 output |
| Label strategy documentation | This README, Label Strategy section |
| Model architecture description | This README, Model Architecture section |
| Training configuration | `configs/default.yaml` |
| Per-class metrics table | Notebook 03 output |
| Confusion matrices | `results/cm_efficientnet.png`, `results/cm_resnet50.png` |
| Fairness gap analysis | Notebook 03 output |
| Cross-model comparison | Notebook 03 output |
| Vertex AI Model ID | `scripts/upload_to_vertex.py` output |
| W&B experiment dashboard | [wandb.ai](https://wandb.ai) project |

---

## References

1. Groh, M. et al. (2021). "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology." CVPR 2021 ISIC Workshop. [Paper](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf)
2. Fitzpatrick, T.B. (1988). "The validity and practicality of sun-reactive skin types I through VI." Archives of Dermatology.
3. Daneshjou, R. et al. (2022). "Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set." Science Advances.
4. Tan, M. & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training." ICML 2021.
5. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.

---

## License

This project is for academic and research purposes. The Fitzpatrick17k dataset is subject to its own license terms — see [the dataset repository](https://github.com/mattgroh/fitzpatrick17k) for details.
