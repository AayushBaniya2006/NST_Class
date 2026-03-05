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
| 1-3 | `skin_tone_pipeline.ipynb` | Full pipeline: data cleaning, model training (EfficientNetV2-S + ResNet50), evaluation, fairness analysis |
| 4 | `04_automl_baseline.ipynb` | Upload images to GCS, train Vertex AI AutoML model as zero-effort baseline |

---

## Dataset

### Fitzpatrick17k

- **Source:** [github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
- **Paper:** Groh et al., "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology" (CVPR 2021 Workshop)
- **Size:** ~16,577 clinical dermatology images
- **Labels:** Fitzpatrick skin types I-VI, disease condition labels, partition labels
- **Format:** CSV metadata with image URLs; images downloaded separately

**CSV Columns (upstream):**

The raw CSV from the Fitzpatrick17k repository uses column names that differ from what the pipeline expects. The `load_metadata()` function auto-renames them:

| Upstream Column | Renamed To | Type | Description |
|----------------|-----------|------|-------------|
| `url_alphanum` | `hasher` | string | Unique image identifier (filename) |
| `fitzpatrick_scale` | `fitzpatrick` | int (1-6) | Fitzpatrick skin type label |
| `url` | `url` | string | Source URL for image download |
| `label` | `label` | string | Dermatological condition name |
| `three_partition_label` | `three_partition_label` | string | High-level condition category |
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

- **Pretrained on:** ImageNet-1K (`EfficientNet_V2_S_Weights.IMAGENET1K_V1`)
- **Feature dimension:** 1,280
- **Classification head:** `Dropout(0.3) -> Linear(1280, 6)`
- **Why:** Best accuracy-to-compute ratio among modern CNNs. Compound scaling (depth + width + resolution) produces strong features for fine-grained classification.

#### ResNet50 (Comparison Baseline)

- **Pretrained on:** ImageNet-1K (`ResNet50_Weights.IMAGENET1K_V2`)
- **Feature dimension:** 2,048
- **Classification head:** `Dropout(0.3) -> Linear(2048, 6)`
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
- Optimizer, scheduler, and early stopping are re-initialized for the new parameter set
- Allows the backbone to adapt its high-level features to dermatology images

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

Validation and test sets receive only resize + center crop + normalization (no augmentation).

---

## Evaluation & Fairness

### Metrics

All metrics are computed on the held-out **test set** for each model:

| Metric | Scope | Purpose |
|--------|-------|---------|
| Accuracy | Overall | General correctness |
| Macro F1 | Overall | Balanced performance across classes |
| ROC-AUC | Overall (macro, OVR) | Discrimination ability |
| ROC-AUC | Per-class (binary OVR) | Per-type discrimination |
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

Default significance threshold: **0.15** (configurable in `configs/default.yaml`).

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
NST_Class/
├── src/                           # Core Python package
│   ├── data/
│   │   ├── prepare.py             # Data cleaning pipeline (12 functions)
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
│   ├── skin_tone_pipeline.ipynb   # Full pipeline: data → training → evaluation
│   └── 04_automl_baseline.ipynb   # Vertex AI AutoML baseline
├── scripts/
│   └── train.py                   # CLI entrypoint (local or Vertex AI)
├── tests/                         # 83 tests (unit + integration)
│   ├── test_prepare.py            # 47 tests for data pipeline
│   ├── test_backbone.py           # 7 tests for backbone loading & freeze/unfreeze
│   ├── test_classifier.py         # 4 tests for classifier
│   ├── test_dataset.py            # 4 tests for dataset loading
│   ├── test_metrics.py            # 3 tests for evaluation metrics
│   ├── test_fairness.py           # 4 tests for fairness gap
│   ├── test_trainer.py            # 7 tests for training utils & training loop
│   ├── test_integration.py        # 1 end-to-end smoke test
│   ├── test_config.py             # 3 tests for config loading
│   └── test_transforms.py         # 3 tests for transform pipelines
├── configs/
│   └── default.yaml               # Full experiment configuration
├── Dockerfile                     # Vertex AI custom training container
├── .dockerignore                  # Docker build exclusions
├── pyproject.toml                 # Python project metadata & pytest config
├── requirements.txt               # Python dependencies
└── docs/plans/                    # Design & implementation documents
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Google Cloud account with Vertex AI enabled (for notebooks 04 and deployment)
- Weights & Biases account (free for students, for experiment tracking)

### Google Colab (Recommended)

All notebooks are **fully self-contained**. The first cell of each notebook automatically:
1. Clones the repo from GitHub
2. Installs all dependencies
3. Downloads the Fitzpatrick17k CSV (notebook 01)
4. Sets up `sys.path`

Just open any notebook in Colab and run all cells.

### Local Setup

```bash
# Clone the repository
git clone https://github.com/AayushBaniya2006/NST_Class.git
cd NST_Class

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest tests/ -v
```

---

## Usage Guide

### Steps 1-3: Full Pipeline

Open `notebooks/skin_tone_pipeline.ipynb` in Colab and run all cells. This single notebook covers:
1. **Data Exploration & Cleaning** — Download Fitzpatrick17k, clean data, validate images, deduplicate, encode labels, stratified split
2. **Model Training** — Train both EfficientNetV2-S and ResNet50 with two-phase transfer learning
3. **Evaluation & Fairness** — Per-class metrics, confusion matrices, fairness gap analysis, cross-model comparison

Ensure runtime is set to **GPU** (Runtime -> Change runtime type -> T4). Google Drive backup/restore is built in to persist data across sessions.

### Step 4: AutoML Baseline (Optional)

Open `notebooks/04_automl_baseline.ipynb`. Fill in your GCP project ID and bucket name:
```python
PROJECT_ID = "your-gcp-project-id"
BUCKET_NAME = "your-gcs-bucket"
```
Run all cells to train the Vertex AI AutoML baseline.

### Alternative: CLI Training

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml --no-wandb

# Train with overrides
python scripts/train.py --backbone resnet50 --epochs 10 --lr 0.001 --batch-size 16

# Specify data and output directories
python scripts/train.py --data-dir data/cleaned --image-dir data/images --output-dir checkpoints
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/default.yaml` | Path to config YAML |
| `--backbone` | (from config) | Override backbone name |
| `--epochs` | (from config) | Override epochs |
| `--batch-size` | (from config) | Override batch size |
| `--lr` | (from config) | Override learning rate |
| `--data-dir` | `data/cleaned` | Directory with train/val CSVs |
| `--image-dir` | `data/images` | Directory with images |
| `--output-dir` | `checkpoints` | Where to save model |
| `--no-wandb` | `false` | Disable W&B logging |

---

## Google Cloud Deployment

### GCS Bucket Structure

```
gs://skin-tone-project/
├── images/              # All dermatology images
├── automl/
│   └── manifest.csv     # AutoML training manifest (ML_USE,GCS_PATH,LABEL)
└── models/
    └── efficientnet_v2_s/  # Model artifacts
```

### Vertex AI Custom Training Job

```bash
# Build and push Docker container
# Base image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
docker build -t gcr.io/YOUR_PROJECT/skin-tone-trainer .
docker push gcr.io/YOUR_PROJECT/skin-tone-trainer

# Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=skin-tone-training \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT/skin-tone-trainer
```

---

## Configuration Reference

All training parameters are defined in `configs/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| training | backbone | `efficientnet_v2_s` | Model backbone (`efficientnet_v2_s` or `resnet50`) |
| training | pretrained | true | Use ImageNet pretrained weights |
| training | num_classes | 6 | Number of output classes |
| training | dropout | 0.3 | Dropout before classification head |
| training | freeze_backbone | true | Freeze backbone in phase 1 |
| training | unfreeze_after_epochs | 5 | When to start fine-tuning |
| training | unfreeze_n_blocks | 2 | Number of backbone blocks to unfreeze |
| training | epochs | 20 | Maximum training epochs |
| training | batch_size | 32 | Training batch size |
| training | num_workers | 4 | DataLoader workers |
| training | learning_rate | 0.0001 | Adam learning rate |
| training | weight_decay | 0.0001 | Adam weight decay |
| training | early_stopping_patience | 5 | Epochs without improvement before stopping |
| training | use_class_weights | true | Enable inverse-frequency weighting |
| training | image_size | 224 | Input image resolution |
| training | random_seed | 42 | Random seed for reproducibility |
| logging | wandb_project | `skin-tone-classifier` | W&B project name |
| logging | wandb_entity | null | W&B team/user name |
| logging | log_every_n_steps | 10 | Logging frequency |
| logging | checkpoint_dir | `checkpoints` | Model checkpoint directory |

---

## Testing

```bash
# Run all 83 tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_prepare.py -v      # Data pipeline (47 tests)
python -m pytest tests/test_backbone.py -v     # Backbone models (7 tests)
python -m pytest tests/test_classifier.py -v   # Classifier (4 tests)
python -m pytest tests/test_dataset.py -v      # Dataset loading (4 tests)
python -m pytest tests/test_trainer.py -v      # Training utils + loop (7 tests)
python -m pytest tests/test_metrics.py -v      # Evaluation metrics (3 tests)
python -m pytest tests/test_fairness.py -v     # Fairness gap (4 tests)
python -m pytest tests/test_config.py -v       # Config loading (3 tests)
python -m pytest tests/test_transforms.py -v   # Transform pipelines (3 tests)
python -m pytest tests/test_integration.py -v  # End-to-end smoke test (1 test)
```

The integration test (`test_integration.py`) creates synthetic images, runs the full pipeline (data prep -> training -> evaluation -> fairness), and verifies correctness on CPU.

---

## Technical Details

### Class Weight Computation

```python
# Inverse-frequency weighting (src/training/trainer.py)
counts = np.bincount(labels, minlength=num_classes)
counts[counts == 0] = 1  # avoid division by zero
total = sum(counts)
weight[i] = total / (num_classes * counts[i])
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

### Per-Class ROC-AUC

Per-class AUC is computed via binary one-vs-rest (scikit-learn's `multi_class` parameter does not support `average=None`):
```python
for each class i:
    binary_true = (y_true == i)
    auc_i = roc_auc_score(binary_true, y_proba[:, i])
```

### Image File Discovery

The `_find_image_path()` function in `prepare.py` handles filenames with or without extensions:
1. First checks if the bare hasher path exists as a file (handles double-extension filenames like `abc.jpg.jpg`)
2. Then tries appending `.jpg`, `.jpeg`, `.png`, `.bmp` in order

This is the canonical implementation used by both `FitzpatrickDataset` and the data cleaning pipeline.

### CSV Column Auto-Rename

The upstream Fitzpatrick17k CSV uses different column names than the pipeline expects. `load_metadata()` auto-renames:
- `fitzpatrick_scale` -> `fitzpatrick`
- `url_alphanum` -> `hasher`

---

## Dependencies

```
torch>=2.6.0
torchvision>=0.21.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
imagehash>=4.3.0
requests>=2.31.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0
google-cloud-storage>=2.10.0
google-cloud-aiplatform>=1.30.0
pyyaml>=6.0
ipykernel>=6.25.0
tqdm>=4.65.0
pytest>=7.4.0
```

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
| Dead source URLs (dataset is from 2021) | Fewer training images | Request bulk download from dataset authors |
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
| Confusion matrices | Notebook 03 output |
| Fairness gap analysis | Notebook 03 output |
| Cross-model comparison | Notebook 03 output |
| Vertex AI Model ID | Vertex AI console (after deployment) |
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
