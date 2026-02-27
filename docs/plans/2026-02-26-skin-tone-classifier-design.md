# Skin Tone Classification Pipeline — Design Document

**Date:** 2026-02-26
**Status:** Approved

## Summary

Build a PyTorch-based ML pipeline to classify dermatology images into 3 Fitzpatrick skin tone groups (I-II, III-IV, V-VI) using the Fitzpatrick17k dataset. Train on Google Colab, deploy to Vertex AI Model Registry. Compare custom fine-tuned models against a Vertex AutoML baseline. Evaluate fairness across skin tone categories.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | PyTorch | Flexibility, research dominance, strong Colab support |
| Dataset | Fitzpatrick17k only | Sufficient for Milestone 1, clean Fitzpatrick labels |
| Training env | Colab → Vertex AI | Fast iteration in Colab, production artifacts in Vertex |
| Experiment tracking | Weights & Biases | Best-in-class, free for students |
| Architecture | Modular Python package + notebooks | Reusable, testable, Vertex-compatible |
| Label strategy | 3-class grouped (12, 34, 56) | Stable distribution, cleaner fairness comparison |
| Augmentation | Standard only (flips, rotation, brightness) | No synthetic generation |
| Baseline | Vertex AutoML Image Classification | Zero-effort comparison point |

## Project Structure

```
skin_tone_classifier/
├── src/
│   ├── data/
│   │   ├── dataset.py         # Fitzpatrick17k PyTorch Dataset class
│   │   ├── transforms.py      # Train/val/test augmentation pipelines
│   │   └── prepare.py         # Download, clean, encode labels, split
│   ├── models/
│   │   ├── backbone.py        # EfficientNetV2-S and ResNet50 wrappers
│   │   └── classifier.py      # 3-class classification head
│   ├── training/
│   │   ├── trainer.py         # Training loop with class weighting
│   │   └── config.py          # Hyperparameter dataclass
│   ├── evaluation/
│   │   ├── metrics.py         # Accuracy, F1, per-class precision/recall
│   │   ├── confusion.py       # Confusion matrix generation
│   │   └── fairness.py        # Fairness gap analysis
│   └── utils/
│       ├── gcs.py             # GCS upload/download helpers
│       └── logging.py         # W&B integration
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_automl_baseline.ipynb
├── scripts/
│   ├── train.py               # CLI entrypoint for Vertex AI custom job
│   └── upload_to_vertex.py    # Upload model to Vertex Model Registry
├── configs/
│   └── default.yaml           # Experiment config
├── requirements.txt
├── Dockerfile                 # For Vertex AI custom training container
└── docs/plans/
```

## Data Cleaning Pipeline

Order of operations in `src/data/prepare.py`:

1. **Load metadata CSV** — parse Fitzpatrick labels, image URLs, condition labels
2. **Validate images** — remove corrupted/unreadable files (PIL open test)
3. **Human/skin verification** — filter to only images containing human skin; remove diagrams, equipment photos, histology slides, non-human subjects
4. **Verify Fitzpatrick labels** — drop rows with missing or invalid skin tone labels
5. **Deduplicate** — perceptual hash-based duplicate detection
6. **Resize & normalize** — standardize to 224x224
7. **Generate cleaning report** — before/after counts per class, dropped image log
8. **Class distribution analysis** — histogram, imbalance ratio, confirm 3-class grouping
9. **Stratified split** — 70% train / 15% val / 15% test, stratified by class

## Model Training

### Custom Models

Two backbones trained for comparison:

| Parameter | EfficientNetV2-S | ResNet50 |
|---|---|---|
| Pretrained on | ImageNet | ImageNet |
| Input size | 224x224 | 224x224 |
| Classifier head | FC → 3 classes | FC → 3 classes |
| Loss | CrossEntropy + class weights | CrossEntropy + class weights |
| Optimizer | Adam, lr=1e-4 | Adam, lr=1e-4 |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Epochs | 20 | 20 |
| Batch size | 32 | 32 |
| Augmentation | HFlip, rotation(15), brightness, color jitter | Same |
| Early stopping | patience=5 on val loss | Same |

Training strategy:
- **Phase 1:** Freeze backbone, train classifier head only (5 epochs)
- **Phase 2:** Unfreeze last 2 blocks, fine-tune end-to-end (15 epochs)
- W&B logs: loss, accuracy, F1 per epoch, learning rate, confusion matrix

### AutoML Baseline

- Upload cleaned dataset to GCS in Vertex AutoML format (CSV manifest + image paths)
- Train Vertex AutoML Image Classification model
- Export metrics for comparison

## Evaluation & Fairness

Metrics computed for all 3 models (EfficientNetV2, ResNet50, AutoML):

| Metric | Scope |
|---|---|
| Accuracy | Overall |
| Macro F1 | Overall |
| Precision | Per-class (12, 34, 56) |
| Recall | Per-class (12, 34, 56) |
| Confusion Matrix | 3x3 |
| Fairness Gap | max(recall) - min(recall) |
| ROC-AUC | Per-class (one-vs-rest) |

Fairness analysis:
- Side-by-side comparison table across all 3 models
- Bar chart of per-class recall
- Fairness gap highlighted (>15% threshold = significant disparity)
- Narrative: "Did custom training reduce the fairness gap vs AutoML?"

## Pipeline Flow

1. `notebooks/01` → Explore Fitzpatrick17k, clean data, analyze distribution
2. `notebooks/04` → Train AutoML baseline on Vertex AI, record metrics
3. `notebooks/02` → Train EfficientNetV2-S and ResNet50 in Colab
4. `notebooks/03` → Evaluate all 3 models, fairness gap comparison
5. `scripts/upload_to_vertex.py` → Push best model to Vertex Model Registry
