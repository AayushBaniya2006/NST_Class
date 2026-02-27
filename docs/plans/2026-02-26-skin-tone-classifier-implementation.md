# Skin Tone Classifier — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a PyTorch pipeline to classify dermatology images into 3 Fitzpatrick skin tone groups, train in Colab, compare against Vertex AutoML baseline, and evaluate fairness across skin tone categories.

**Architecture:** Modular Python package (`src/`) with PyTorch components for data loading, model definition, training, and evaluation. Colab notebooks orchestrate the workflow. Scripts enable Vertex AI custom training. AutoML baseline trained directly via Vertex AI API.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, Weights & Biases, Google Cloud Storage, Vertex AI, Pillow, scikit-learn, matplotlib, seaborn, imagehash, pandas

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/data src/models src/training src/evaluation src/utils
mkdir -p notebooks scripts configs tests docs/plans
touch src/__init__.py src/data/__init__.py src/models/__init__.py
touch src/training/__init__.py src/evaluation/__init__.py src/utils/__init__.py
touch tests/__init__.py
```

**Step 2: Create requirements.txt**

```
# Core ML
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# Data
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
imagehash>=4.3.0
requests>=2.31.0

# Evaluation & Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Experiment Tracking
wandb>=0.15.0

# Google Cloud
google-cloud-storage>=2.10.0
google-cloud-aiplatform>=1.30.0

# Config
pyyaml>=6.0

# Notebook support
ipykernel>=6.25.0
tqdm>=4.65.0
```

**Step 3: Create configs/default.yaml**

```yaml
# Skin Tone Classifier Configuration

data:
  dataset_name: "fitzpatrick17k"
  csv_path: "data/fitzpatrick17k.csv"
  image_dir: "data/images"
  cleaned_csv_path: "data/fitzpatrick17k_cleaned.csv"
  image_size: 224
  num_classes: 3
  label_column: "fitzpatrick"
  # Grouped labels: 12 = Fitz I-II, 34 = Fitz III-IV, 56 = Fitz V-VI
  class_names: ["12", "34", "56"]
  split_ratios:
    train: 0.70
    val: 0.15
    test: 0.15
  random_seed: 42

training:
  backbone: "efficientnet_v2_s"  # or "resnet50"
  pretrained: true
  freeze_backbone: true
  unfreeze_after_epochs: 5
  unfreeze_layers: "last_2_blocks"
  epochs: 20
  batch_size: 32
  num_workers: 4
  optimizer: "adam"
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler: "cosine"
  early_stopping_patience: 5
  use_class_weights: true

augmentation:
  horizontal_flip: true
  rotation_degrees: 15
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1

logging:
  wandb_project: "skin-tone-classifier"
  wandb_entity: null  # set to your W&B username
  log_every_n_steps: 10
  save_checkpoints: true
  checkpoint_dir: "checkpoints"

gcs:
  bucket_name: "skin-tone-project"
  project_id: null  # set to your GCP project ID
  region: "us-central1"

evaluation:
  fairness_gap_threshold: 0.15
```

**Step 4: Commit**

```bash
git init
git add requirements.txt configs/default.yaml src/ tests/
git commit -m "feat: project scaffolding with directory structure and config"
```

---

## Task 2: Data Preparation — Download & Clean (`src/data/prepare.py`)

**Files:**
- Create: `src/data/prepare.py`
- Test: `tests/test_prepare.py`

**Step 1: Write tests for data preparation functions**

```python
# tests/test_prepare.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.prepare import (
    load_metadata,
    encode_grouped_labels,
    validate_fitzpatrick_labels,
    compute_class_distribution,
    generate_cleaning_report,
    stratified_split,
)


def make_sample_df(n=100):
    """Create a sample dataframe mimicking Fitzpatrick17k CSV."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "hasher": [f"img_{i}" for i in range(n)],
        "url": [f"http://example.com/img_{i}.jpg" for i in range(n)],
        "label": rng.choice(["acne", "melanoma", "eczema"], n),
        "fitzpatrick": rng.choice([1, 2, 3, 4, 5, 6], n),
        "three_partition_label": rng.choice(["benign", "malignant", "non-neoplastic"], n),
        "nine_partition_label": [f"cat_{i % 9}" for i in range(n)],
        "md5hash": [f"hash_{i}" for i in range(n)],
        "qc": [None] * n,
    })


class TestEncodeGroupedLabels:
    def test_maps_1_and_2_to_group_12(self):
        df = pd.DataFrame({"fitzpatrick": [1, 2]})
        result = encode_grouped_labels(df)
        assert list(result["skin_tone_group"]) == ["12", "12"]

    def test_maps_3_and_4_to_group_34(self):
        df = pd.DataFrame({"fitzpatrick": [3, 4]})
        result = encode_grouped_labels(df)
        assert list(result["skin_tone_group"]) == ["34", "34"]

    def test_maps_5_and_6_to_group_56(self):
        df = pd.DataFrame({"fitzpatrick": [5, 6]})
        result = encode_grouped_labels(df)
        assert list(result["skin_tone_group"]) == ["56", "56"]

    def test_adds_numeric_label_column(self):
        df = pd.DataFrame({"fitzpatrick": [1, 3, 5]})
        result = encode_grouped_labels(df)
        assert list(result["skin_tone_label"]) == [0, 1, 2]


class TestValidateFitzpatrickLabels:
    def test_drops_missing_labels(self):
        df = pd.DataFrame({"fitzpatrick": [1, None, 3, np.nan, 5]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 3

    def test_drops_invalid_labels(self):
        df = pd.DataFrame({"fitzpatrick": [1, 7, 0, -1, 3]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 2
        assert list(result["fitzpatrick"]) == [1, 3]


class TestComputeClassDistribution:
    def test_returns_counts_and_percentages(self):
        df = pd.DataFrame({"skin_tone_group": ["12"] * 50 + ["34"] * 30 + ["56"] * 20})
        dist = compute_class_distribution(df, "skin_tone_group")
        assert dist["12"]["count"] == 50
        assert dist["12"]["percentage"] == pytest.approx(50.0)
        assert dist["56"]["count"] == 20


class TestStratifiedSplit:
    def test_split_ratios(self):
        df = make_sample_df(300)
        df = encode_grouped_labels(df)
        train, val, test = stratified_split(df, "skin_tone_label", (0.7, 0.15, 0.15), seed=42)
        assert len(train) == pytest.approx(210, abs=10)
        assert len(val) == pytest.approx(45, abs=10)
        assert len(test) == pytest.approx(45, abs=10)

    def test_no_overlap(self):
        df = make_sample_df(300)
        df = encode_grouped_labels(df)
        train, val, test = stratified_split(df, "skin_tone_label", (0.7, 0.15, 0.15), seed=42)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
```

**Step 2: Run tests to verify they fail**

```bash
cd /Volumes/CS_Stuff/NST_Class
python -m pytest tests/test_prepare.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.prepare'`

**Step 3: Implement `src/data/prepare.py`**

```python
# src/data/prepare.py
"""Data preparation pipeline for Fitzpatrick17k dataset.

Handles downloading, cleaning, label encoding, and stratified splitting.
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

VALID_FITZPATRICK = {1, 2, 3, 4, 5, 6}
GROUP_MAP = {1: "12", 2: "12", 3: "34", 4: "34", 5: "56", 6: "56"}
GROUP_TO_LABEL = {"12": 0, "34": 1, "56": 2}


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load the Fitzpatrick17k metadata CSV."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def validate_fitzpatrick_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or invalid Fitzpatrick labels."""
    before = len(df)
    df = df.dropna(subset=["fitzpatrick"])
    df = df[df["fitzpatrick"].isin(VALID_FITZPATRICK)].copy()
    after = len(df)
    logger.info(f"Fitzpatrick validation: {before} → {after} ({before - after} dropped)")
    return df.reset_index(drop=True)


def encode_grouped_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Fitzpatrick 1-6 into 3-class grouped labels.

    Groups:
        12: Fitzpatrick I-II (label 0)
        34: Fitzpatrick III-IV (label 1)
        56: Fitzpatrick V-VI (label 2)
    """
    df = df.copy()
    df["skin_tone_group"] = df["fitzpatrick"].map(GROUP_MAP)
    df["skin_tone_label"] = df["skin_tone_group"].map(GROUP_TO_LABEL)
    return df


def validate_images(image_dir: str, df: pd.DataFrame, hasher_col: str = "hasher") -> pd.DataFrame:
    """Remove rows whose images are corrupted or unreadable."""
    image_path = Path(image_dir)
    valid_rows = []
    invalid_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        # Try common extensions
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_path / f"{row[hasher_col]}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            invalid_count += 1
            continue

        try:
            img = Image.open(img_file)
            img.verify()
            valid_rows.append(idx)
        except Exception:
            invalid_count += 1

    logger.info(f"Image validation: {invalid_count} corrupted/missing images removed")
    return df.loc[valid_rows].reset_index(drop=True)


def filter_human_images(image_dir: str, df: pd.DataFrame, hasher_col: str = "hasher") -> pd.DataFrame:
    """Filter to only images containing human skin.

    Uses a simple heuristic: checks image dimensions and color statistics
    to filter out diagrams, histology slides, and non-human images.
    For production use, replace with a pretrained skin detector.
    """
    image_path = Path(image_dir)
    valid_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering non-human images"):
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_path / f"{row[hasher_col]}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        try:
            img = Image.open(img_file).convert("RGB")
            img_array = np.array(img)

            # Filter out grayscale images (diagrams, text)
            if len(img_array.shape) < 3:
                continue

            # Filter out very small images (likely thumbnails/icons)
            h, w = img_array.shape[:2]
            if h < 50 or w < 50:
                continue

            # Filter out images with very low color variance (solid colors, diagrams)
            color_std = img_array.std(axis=(0, 1)).mean()
            if color_std < 10:
                continue

            valid_rows.append(idx)
        except Exception:
            continue

    removed = len(df) - len(valid_rows)
    logger.info(f"Human filter: {removed} non-human images removed")
    return df.loc[valid_rows].reset_index(drop=True)


def deduplicate_images(image_dir: str, df: pd.DataFrame, hasher_col: str = "hasher") -> pd.DataFrame:
    """Remove duplicate images using perceptual hashing."""
    image_path = Path(image_dir)
    seen_hashes = set()
    unique_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Deduplicating"):
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_path / f"{row[hasher_col]}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        try:
            img = Image.open(img_file)
            phash = str(imagehash.phash(img))
            if phash not in seen_hashes:
                seen_hashes.add(phash)
                unique_rows.append(idx)
        except Exception:
            continue

    removed = len(df) - len(unique_rows)
    logger.info(f"Deduplication: {removed} duplicates removed")
    return df.loc[unique_rows].reset_index(drop=True)


def compute_class_distribution(df: pd.DataFrame, column: str) -> dict:
    """Compute image count and percentage per class."""
    counts = df[column].value_counts()
    total = len(df)
    distribution = {}
    for cls, count in counts.items():
        distribution[cls] = {
            "count": int(count),
            "percentage": round(count / total * 100, 2),
        }
    return distribution


def generate_cleaning_report(
    original_count: int,
    cleaned_df: pd.DataFrame,
    column: str,
    dropped_reasons: dict,
) -> dict:
    """Generate a cleaning summary report."""
    distribution = compute_class_distribution(cleaned_df, column)
    return {
        "original_count": original_count,
        "cleaned_count": len(cleaned_df),
        "total_dropped": original_count - len(cleaned_df),
        "dropped_reasons": dropped_reasons,
        "class_distribution": distribution,
    }


def stratified_split(
    df: pd.DataFrame,
    label_column: str,
    ratios: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple:
    """Split dataset into train/val/test with stratification."""
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df[label_column],
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[label_column],
        random_state=seed,
    )

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def download_images(df: pd.DataFrame, output_dir: str, url_col: str = "url", hasher_col: str = "hasher") -> int:
    """Download images from URLs in the metadata CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        url = row[url_col]
        filename = row[hasher_col]

        # Check if already downloaded
        existing = list(output_path.glob(f"{filename}.*"))
        if existing:
            downloaded += 1
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Determine extension from content type
            content_type = response.headers.get("content-type", "image/jpeg")
            ext = ".jpg"
            if "png" in content_type:
                ext = ".png"
            elif "bmp" in content_type:
                ext = ".bmp"

            filepath = output_path / f"{filename}{ext}"
            filepath.write_bytes(response.content)
            downloaded += 1
        except Exception as e:
            failed += 1
            logger.debug(f"Failed to download {url}: {e}")

    logger.info(f"Downloaded: {downloaded}, Failed: {failed}")
    return downloaded


def run_full_pipeline(csv_path: str, image_dir: str, output_dir: str, seed: int = 42) -> dict:
    """Run the complete data preparation pipeline.

    Returns a dict with cleaned dataframes and report.
    """
    # Step 1: Load metadata
    df = load_metadata(csv_path)
    original_count = len(df)
    dropped_reasons = {}

    # Step 2: Validate Fitzpatrick labels
    before = len(df)
    df = validate_fitzpatrick_labels(df)
    dropped_reasons["invalid_fitzpatrick"] = before - len(df)

    # Step 3: Validate images
    before = len(df)
    df = validate_images(image_dir, df)
    dropped_reasons["corrupted_images"] = before - len(df)

    # Step 4: Filter to human-only images
    before = len(df)
    df = filter_human_images(image_dir, df)
    dropped_reasons["non_human_images"] = before - len(df)

    # Step 5: Deduplicate
    before = len(df)
    df = deduplicate_images(image_dir, df)
    dropped_reasons["duplicates"] = before - len(df)

    # Step 6: Encode grouped labels
    df = encode_grouped_labels(df)

    # Step 7: Generate report
    report = generate_cleaning_report(original_count, df, "skin_tone_group", dropped_reasons)

    # Step 8: Stratified split
    train_df, val_df, test_df = stratified_split(df, "skin_tone_label", seed=seed)

    # Step 9: Save cleaned data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "fitzpatrick17k_cleaned.csv", index=False)
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    logger.info(f"Cleaning report: {report}")

    return {
        "full": df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "report": report,
    }
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prepare.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/data/prepare.py tests/test_prepare.py
git commit -m "feat: data preparation pipeline with cleaning, label encoding, and splitting"
```

---

## Task 3: Augmentation Transforms (`src/data/transforms.py`)

**Files:**
- Create: `src/data/transforms.py`

**Step 1: Implement transforms**

```python
# src/data/transforms.py
"""Image augmentation pipelines for train, validation, and test sets."""
from torchvision import transforms


def get_train_transforms(image_size: int = 224, config: dict = None) -> transforms.Compose:
    """Training augmentation pipeline.

    Includes horizontal flip, rotation, color jitter, and normalization.
    """
    cfg = config or {}
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg.get("rotation_degrees", 15)),
        transforms.ColorJitter(
            brightness=cfg.get("brightness", 0.2),
            contrast=cfg.get("contrast", 0.2),
            saturation=cfg.get("saturation", 0.2),
            hue=cfg.get("hue", 0.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transform pipeline. No augmentation, just resize and normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
```

**Step 2: Commit**

```bash
git add src/data/transforms.py
git commit -m "feat: train and eval augmentation transforms"
```

---

## Task 4: PyTorch Dataset (`src/data/dataset.py`)

**Files:**
- Create: `src/data/dataset.py`
- Test: `tests/test_dataset.py`

**Step 1: Write tests**

```python
# tests/test_dataset.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile
import os

from src.data.dataset import FitzpatrickDataset


@pytest.fixture
def sample_data(tmp_path):
    """Create sample images and CSV for testing."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(image_dir / f"img_{i}.jpg")
        rows.append({
            "hasher": f"img_{i}",
            "fitzpatrick": (i % 6) + 1,
            "skin_tone_group": ["12", "12", "34", "34", "56", "56"][i % 6],
            "skin_tone_label": [0, 0, 1, 1, 2, 2][i % 6],
            "label": "acne",
        })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, str(image_dir), df


class TestFitzpatrickDataset:
    def test_length(self, sample_data):
        csv_path, image_dir, df = sample_data
        dataset = FitzpatrickDataset(df, image_dir)
        assert len(dataset) == 10

    def test_returns_image_and_label(self, sample_data):
        csv_path, image_dir, df = sample_data
        dataset = FitzpatrickDataset(df, image_dir)
        image, label = dataset[0]
        assert image is not None
        assert isinstance(label, int)
        assert label in [0, 1, 2]

    def test_with_transforms(self, sample_data):
        from src.data.transforms import get_eval_transforms
        csv_path, image_dir, df = sample_data
        transform = get_eval_transforms(224)
        dataset = FitzpatrickDataset(df, image_dir, transform=transform)
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: FAIL

**Step 3: Implement dataset**

```python
# src/data/dataset.py
"""PyTorch Dataset for Fitzpatrick17k skin tone classification."""
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FitzpatrickDataset(Dataset):
    """Dataset for loading Fitzpatrick17k images with skin tone labels.

    Args:
        df: DataFrame with 'hasher' and 'skin_tone_label' columns.
        image_dir: Directory containing the images.
        transform: Optional torchvision transforms to apply.
        label_column: Column name for the integer label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        label_column: str = "skin_tone_label",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, hasher: str) -> Optional[Path]:
        """Find image file by hasher name, trying common extensions."""
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            path = self.image_dir / f"{hasher}{ext}"
            if path.exists():
                return path
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hasher = row["hasher"]
        label = int(row[self.label_column])

        img_path = self._find_image(hasher)
        if img_path is None:
            raise FileNotFoundError(f"No image found for hasher: {hasher}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: Fitzpatrick17k PyTorch Dataset with image loading and transforms"
```

---

## Task 5: Model Backbones (`src/models/backbone.py`)

**Files:**
- Create: `src/models/backbone.py`
- Test: `tests/test_backbone.py`

**Step 1: Write tests**

```python
# tests/test_backbone.py
import pytest
import torch
from src.models.backbone import get_backbone


class TestGetBackbone:
    def test_efficientnet_returns_model_and_feature_dim(self):
        model, feature_dim = get_backbone("efficientnet_v2_s", pretrained=False)
        assert model is not None
        assert isinstance(feature_dim, int)
        assert feature_dim > 0

    def test_resnet50_returns_model_and_feature_dim(self):
        model, feature_dim = get_backbone("resnet50", pretrained=False)
        assert model is not None
        assert isinstance(feature_dim, int)
        assert feature_dim == 2048

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError):
            get_backbone("not_a_model", pretrained=False)

    def test_efficientnet_forward_pass(self):
        model, dim = get_backbone("efficientnet_v2_s", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, dim)

    def test_resnet50_forward_pass(self):
        model, dim = get_backbone("resnet50", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, dim)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_backbone.py -v
```

**Step 3: Implement backbones**

```python
# src/models/backbone.py
"""Pretrained backbone wrappers for feature extraction.

Supported backbones: EfficientNetV2-S, ResNet50.
Returns the backbone (with classification head removed) and its feature dimension.
"""
import torch.nn as nn
from torchvision import models


BACKBONE_REGISTRY = {
    "efficientnet_v2_s": {
        "constructor": models.efficientnet_v2_s,
        "weights": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    },
    "resnet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V2,
    },
}


def get_backbone(name: str, pretrained: bool = True) -> tuple:
    """Load a pretrained backbone with its classification head removed.

    Args:
        name: One of 'efficientnet_v2_s' or 'resnet50'.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Tuple of (backbone_model, feature_dimension).
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(BACKBONE_REGISTRY.keys())}")

    entry = BACKBONE_REGISTRY[name]
    weights = entry["weights"] if pretrained else None
    model = entry["constructor"](weights=weights)

    if name == "resnet50":
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_v2_s":
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

    return model, feature_dim


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model: nn.Module, name: str, n: int = 2) -> None:
    """Unfreeze the last n blocks of the backbone for fine-tuning."""
    if name == "resnet50":
        layers = [model.layer4, model.layer3][:n]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    elif name == "efficientnet_v2_s":
        # EfficientNet features are in model.features
        blocks = list(model.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_backbone.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/models/backbone.py tests/test_backbone.py
git commit -m "feat: backbone wrappers for EfficientNetV2-S and ResNet50"
```

---

## Task 6: Classifier Head (`src/models/classifier.py`)

**Files:**
- Create: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write tests**

```python
# tests/test_classifier.py
import pytest
import torch
from src.models.classifier import SkinToneClassifier


class TestSkinToneClassifier:
    def test_forward_shape(self):
        model = SkinToneClassifier("resnet50", num_classes=3, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 3)

    def test_efficientnet_forward_shape(self):
        model = SkinToneClassifier("efficientnet_v2_s", num_classes=3, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 3)

    def test_freeze_backbone(self):
        model = SkinToneClassifier("resnet50", num_classes=3, pretrained=False)
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        # Classifier head should still be trainable
        for param in model.head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self):
        model = SkinToneClassifier("resnet50", num_classes=3, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_last_blocks(n=2)
        # At least some backbone params should now require grad
        trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert trainable > 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classifier.py -v
```

**Step 3: Implement classifier**

```python
# src/models/classifier.py
"""Full classifier: pretrained backbone + classification head."""
import torch.nn as nn

from src.models.backbone import get_backbone, freeze_backbone, unfreeze_last_n_blocks


class SkinToneClassifier(nn.Module):
    """Skin tone classifier with pretrained backbone and FC head.

    Args:
        backbone_name: Name of the backbone ('resnet50' or 'efficientnet_v2_s').
        num_classes: Number of output classes (default 3 for grouped Fitzpatrick).
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout probability before the final FC layer.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = get_backbone(backbone_name, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        freeze_backbone(self.backbone)

    def unfreeze_last_blocks(self, n: int = 2):
        """Unfreeze the last n blocks for fine-tuning."""
        unfreeze_last_n_blocks(self.backbone, self.backbone_name, n=n)
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_classifier.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: SkinToneClassifier with backbone + FC head"
```

---

## Task 7: Training Config (`src/training/config.py`)

**Files:**
- Create: `src/training/config.py`

**Step 1: Implement config dataclass**

```python
# src/training/config.py
"""Training configuration as a dataclass, loadable from YAML."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    # Model
    backbone: str = "efficientnet_v2_s"
    num_classes: int = 3
    pretrained: bool = True
    dropout: float = 0.3

    # Training phases
    freeze_backbone: bool = True
    unfreeze_after_epochs: int = 5
    unfreeze_n_blocks: int = 2

    # Optimization
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 5
    use_class_weights: bool = True

    # Data
    image_size: int = 224
    num_workers: int = 4
    class_names: list = field(default_factory=lambda: ["12", "34", "56"])

    # Logging
    wandb_project: str = "skin-tone-classifier"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10

    # Paths
    checkpoint_dir: str = "checkpoints"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Flatten nested config
        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)
```

**Step 2: Commit**

```bash
git add src/training/config.py
git commit -m "feat: TrainingConfig dataclass with YAML loading"
```

---

## Task 8: Training Loop (`src/training/trainer.py`)

**Files:**
- Create: `src/training/trainer.py`
- Test: `tests/test_trainer.py`

**Step 1: Write tests**

```python
# tests/test_trainer.py
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.training.trainer import compute_class_weights, EarlyStopping


class TestComputeClassWeights:
    def test_balanced_classes(self):
        labels = [0] * 100 + [1] * 100 + [2] * 100
        weights = compute_class_weights(labels, num_classes=3)
        assert len(weights) == 3
        # Balanced classes should have roughly equal weights
        assert abs(weights[0] - weights[1]) < 0.1

    def test_imbalanced_classes(self):
        labels = [0] * 500 + [1] * 100 + [2] * 50
        weights = compute_class_weights(labels, num_classes=3)
        # Minority class should have highest weight
        assert weights[2] > weights[1] > weights[0]


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3)
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.8)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.7)
        assert es.step(0.8)  # 3 consecutive non-improvements → stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.4)  # improvement resets counter
        es.step(0.5)
        es.step(0.6)
        assert not es.step(0.7)  # only 2 non-improvements since reset
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_trainer.py -v
```

**Step 3: Implement trainer**

```python
# src/training/trainer.py
"""Training loop with class weighting, two-phase training, and early stopping."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def compute_class_weights(labels: list, num_classes: int) -> list:
    """Compute inverse-frequency class weights for imbalanced data.

    Args:
        labels: List of integer labels.
        num_classes: Total number of classes.

    Returns:
        List of float weights, one per class.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1  # avoid division by zero
    total = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    return weights


class EarlyStopping:
    """Stops training when validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Two-phase trainer: frozen backbone → fine-tuned backbone.

    Args:
        model: SkinToneClassifier instance.
        config: TrainingConfig.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_weights: Optional tensor of class weights.
        device: Torch device.
        wandb_run: Optional W&B run for logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
        wandb_run=None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run

        # Loss function
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # Optimizer — initially only head parameters if backbone is frozen
        self.optimizer = self._create_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> Adam:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return Adam(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro")

        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro")

        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = self.checkpoint_dir / f"best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, path)
        logger.info(f"Saved checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")

    def train(self) -> dict:
        """Run the full two-phase training loop.

        Phase 1: Frozen backbone (epochs 0 to unfreeze_after_epochs)
        Phase 2: Fine-tune last blocks (remaining epochs)

        Returns dict with training history.
        """
        history = {"train": [], "val": []}

        for epoch in range(self.config.epochs):
            # Phase transition: unfreeze backbone
            if epoch == self.config.unfreeze_after_epochs and self.config.freeze_backbone:
                logger.info(f"Epoch {epoch}: Unfreezing last {self.config.unfreeze_n_blocks} blocks")
                self.model.unfreeze_last_blocks(n=self.config.unfreeze_n_blocks)
                self.optimizer = self._create_optimizer()
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs - epoch,
                )

            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate()
            self.scheduler.step()

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
                f"LR: {lr:.6f}"
            )

            # W&B logging
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/f1": train_metrics["f1"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1": val_metrics["f1"],
                    "learning_rate": lr,
                })

            # Checkpointing
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"])

            # Early stopping
            if self.early_stopping.step(val_metrics["loss"]):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_trainer.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: two-phase training loop with class weighting and early stopping"
```

---

## Task 9: Evaluation Metrics (`src/evaluation/metrics.py`)

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write tests**

```python
# tests/test_metrics.py
import pytest
import numpy as np
from src.evaluation.metrics import compute_all_metrics


class TestComputeAllMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=["12", "34", "56"])
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_per_class_metrics_present(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        y_proba = np.random.rand(6, 3)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=["12", "34", "56"])
        assert "per_class" in metrics
        assert "12" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["12"]
        assert "recall" in metrics["per_class"]["12"]

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 0, 2]
        y_proba = np.random.rand(6, 3)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=["12", "34", "56"])
        assert metrics["confusion_matrix"].shape == (3, 3)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py -v
```

**Step 3: Implement metrics**

```python
# src/evaluation/metrics.py
"""Evaluation metrics for skin tone classification."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


def compute_all_metrics(
    y_true: list,
    y_pred: list,
    y_proba: np.ndarray,
    class_names: list = None,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (N x num_classes).
        class_names: Names for each class.

    Returns:
        Dict with accuracy, macro_f1, per_class metrics, confusion_matrix, roc_auc.
    """
    if class_names is None:
        class_names = ["12", "34", "56"]

    num_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(np.sum(y_true == i)),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        per_class_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
        for i, name in enumerate(class_names):
            per_class[name]["roc_auc"] = float(per_class_auc[i])
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: evaluation metrics with per-class precision, recall, F1, and ROC-AUC"
```

---

## Task 10: Fairness Analysis (`src/evaluation/fairness.py`)

**Files:**
- Create: `src/evaluation/fairness.py`
- Test: `tests/test_fairness.py`

**Step 1: Write tests**

```python
# tests/test_fairness.py
import pytest
from src.evaluation.fairness import compute_fairness_gap, compare_model_fairness


class TestFairnessGap:
    def test_zero_gap_when_equal(self):
        per_class = {
            "12": {"recall": 0.80},
            "34": {"recall": 0.80},
            "56": {"recall": 0.80},
        }
        result = compute_fairness_gap(per_class)
        assert result["gap"] == 0.0
        assert not result["significant"]

    def test_detects_significant_gap(self):
        per_class = {
            "12": {"recall": 0.90},
            "34": {"recall": 0.75},
            "56": {"recall": 0.50},
        }
        result = compute_fairness_gap(per_class, threshold=0.15)
        assert result["gap"] == pytest.approx(0.40)
        assert result["significant"]
        assert result["best_class"] == "12"
        assert result["worst_class"] == "56"

    def test_custom_threshold(self):
        per_class = {
            "12": {"recall": 0.80},
            "34": {"recall": 0.70},
            "56": {"recall": 0.65},
        }
        result = compute_fairness_gap(per_class, threshold=0.10)
        assert result["significant"]  # gap = 0.15 > 0.10


class TestCompareModelFairness:
    def test_comparison_table(self):
        models = {
            "EfficientNetV2": {
                "12": {"recall": 0.85, "precision": 0.80},
                "34": {"recall": 0.78, "precision": 0.75},
                "56": {"recall": 0.55, "precision": 0.60},
            },
            "AutoML": {
                "12": {"recall": 0.80, "precision": 0.78},
                "34": {"recall": 0.72, "precision": 0.70},
                "56": {"recall": 0.50, "precision": 0.55},
            },
        }
        table = compare_model_fairness(models)
        assert "EfficientNetV2" in table
        assert "AutoML" in table
        assert "gap" in table["EfficientNetV2"]
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_fairness.py -v
```

**Step 3: Implement fairness analysis**

```python
# src/evaluation/fairness.py
"""Fairness gap analysis across skin tone groups."""
import logging

logger = logging.getLogger(__name__)


def compute_fairness_gap(
    per_class_metrics: dict,
    metric: str = "recall",
    threshold: float = 0.15,
) -> dict:
    """Compute the fairness gap across skin tone groups.

    Gap = max(metric) - min(metric) across classes.
    Significant if gap > threshold.

    Args:
        per_class_metrics: Dict of {class_name: {metric: value}}.
        metric: Which metric to compare (default: recall).
        threshold: Gap threshold for significance.

    Returns:
        Dict with gap value, significance flag, best/worst classes.
    """
    values = {cls: m[metric] for cls, m in per_class_metrics.items()}
    max_class = max(values, key=values.get)
    min_class = min(values, key=values.get)
    gap = values[max_class] - values[min_class]

    result = {
        "metric": metric,
        "gap": round(gap, 4),
        "significant": gap > threshold,
        "threshold": threshold,
        "best_class": max_class,
        "best_value": round(values[max_class], 4),
        "worst_class": min_class,
        "worst_value": round(values[min_class], 4),
        "per_class_values": {k: round(v, 4) for k, v in values.items()},
    }

    if result["significant"]:
        logger.warning(
            f"SIGNIFICANT FAIRNESS GAP: {gap:.2%} ({metric}) — "
            f"best: {max_class} ({values[max_class]:.2%}), "
            f"worst: {min_class} ({values[min_class]:.2%})"
        )

    return result


def compare_model_fairness(
    model_per_class_metrics: dict,
    metric: str = "recall",
    threshold: float = 0.15,
) -> dict:
    """Compare fairness gaps across multiple models.

    Args:
        model_per_class_metrics: Dict of {model_name: {class: {metric: val}}}.
        metric: Metric to compare.
        threshold: Significance threshold.

    Returns:
        Dict of {model_name: fairness_gap_result}.
    """
    results = {}
    for model_name, per_class in model_per_class_metrics.items():
        results[model_name] = compute_fairness_gap(per_class, metric, threshold)

    # Rank by gap
    ranked = sorted(results.items(), key=lambda x: x[1]["gap"])
    logger.info("Fairness ranking (smallest gap = most fair):")
    for i, (name, result) in enumerate(ranked):
        logger.info(f"  {i+1}. {name}: gap={result['gap']:.2%}")

    return results
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_fairness.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/evaluation/fairness.py tests/test_fairness.py
git commit -m "feat: fairness gap analysis with cross-model comparison"
```

---

## Task 11: Confusion Matrix Visualization (`src/evaluation/confusion.py`)

**Files:**
- Create: `src/evaluation/confusion.py`

**Step 1: Implement confusion matrix plotting**

```python
# src/evaluation/confusion.py
"""Confusion matrix visualization."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array (num_classes x num_classes).
        class_names: Labels for each class.
        title: Plot title.
        save_path: Optional path to save the figure.
        normalize: Whether to show percentages.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = ["Fitz I-II", "Fitz III-IV", "Fitz V-VI"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fairness_comparison(
    fairness_results: dict,
    metric: str = "recall",
    save_path: str = None,
) -> plt.Figure:
    """Bar chart comparing per-class recall across models.

    Args:
        fairness_results: Output of compare_model_fairness().
        metric: Metric being compared.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    model_names = list(fairness_results.keys())
    class_names = list(fairness_results[model_names[0]]["per_class_values"].keys())

    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(model_names):
        values = [fairness_results[model]["per_class_values"][c] for c in class_names]
        gap = fairness_results[model]["gap"]
        bars = ax.bar(x + i * width, values, width, label=f"{model} (gap={gap:.2%})")

    ax.set_xlabel("Skin Tone Group")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Per-Class {metric.capitalize()} by Model — Fairness Comparison")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(["Fitz I-II", "Fitz III-IV", "Fitz V-VI"])
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add threshold line
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.3, label="Target")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
```

**Step 2: Commit**

```bash
git add src/evaluation/confusion.py
git commit -m "feat: confusion matrix and fairness comparison visualizations"
```

---

## Task 12: GCS Utilities (`src/utils/gcs.py`)

**Files:**
- Create: `src/utils/gcs.py`

**Step 1: Implement GCS helpers**

```python
# src/utils/gcs.py
"""Google Cloud Storage upload/download helpers."""
import logging
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)


def upload_directory_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> int:
    """Upload a local directory to GCS.

    Args:
        local_dir: Local directory path.
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix path in the bucket.

    Returns:
        Number of files uploaded.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)
    count = 0

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            blob_name = f"{gcs_prefix}/{file_path.relative_to(local_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            count += 1

    logger.info(f"Uploaded {count} files to gs://{bucket_name}/{gcs_prefix}/")
    return count


def upload_file_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """Upload a single file to GCS.

    Returns:
        GCS URI of the uploaded file.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Uploaded {local_path} to {uri}")
    return uri


def download_file_from_gcs(bucket_name: str, blob_name: str, local_path: str) -> str:
    """Download a single file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")
    return local_path


def generate_automl_csv(
    df,
    image_gcs_prefix: str,
    split_column: str = "split",
    label_column: str = "skin_tone_group",
    hasher_column: str = "hasher",
    output_path: str = "automl_manifest.csv",
) -> str:
    """Generate a CSV manifest for Vertex AI AutoML Image Classification.

    Format: ML_USE,GCS_FILE_PATH,LABEL

    Args:
        df: DataFrame with split, label, and hasher columns.
        image_gcs_prefix: GCS prefix where images are stored (e.g. gs://bucket/images).
        split_column: Column indicating train/val/test split.
        label_column: Column with the class label.
        hasher_column: Column with the image filename (without extension).
        output_path: Where to save the manifest CSV.

    Returns:
        Path to the saved manifest.
    """
    split_map = {"train": "TRAINING", "val": "VALIDATION", "test": "TEST"}

    rows = []
    for _, row in df.iterrows():
        ml_use = split_map.get(row[split_column], "TRAINING")
        gcs_path = f"{image_gcs_prefix}/{row[hasher_column]}.jpg"
        label = str(row[label_column])
        rows.append(f"{ml_use},{gcs_path},{label}")

    with open(output_path, "w") as f:
        f.write("\n".join(rows))

    logger.info(f"Generated AutoML manifest with {len(rows)} rows at {output_path}")
    return output_path
```

**Step 2: Commit**

```bash
git add src/utils/gcs.py
git commit -m "feat: GCS helpers and AutoML manifest generation"
```

---

## Task 13: W&B Logging Utilities (`src/utils/logging.py`)

**Files:**
- Create: `src/utils/logging.py`

**Step 1: Implement W&B helpers**

```python
# src/utils/logging.py
"""Weights & Biases logging integration."""
import logging
from typing import Optional

import wandb
import numpy as np

logger = logging.getLogger(__name__)


def init_wandb(
    project: str,
    config: dict,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: list = None,
):
    """Initialize a W&B run.

    Args:
        project: W&B project name.
        config: Hyperparameter dict to log.
        entity: W&B team/user name.
        run_name: Optional custom run name.
        tags: Optional list of tags.

    Returns:
        W&B run object.
    """
    run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=run_name,
        tags=tags or [],
    )
    logger.info(f"W&B run initialized: {run.url}")
    return run


def log_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    step: Optional[int] = None,
):
    """Log a confusion matrix to W&B."""
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )
    }, step=step)


def log_metrics_table(metrics: dict, model_name: str):
    """Log evaluation metrics as a W&B table."""
    table = wandb.Table(columns=["Model", "Accuracy", "Macro F1", "ROC-AUC", "Fairness Gap"])
    table.add_data(
        model_name,
        metrics.get("accuracy"),
        metrics.get("macro_f1"),
        metrics.get("roc_auc"),
        metrics.get("fairness_gap"),
    )
    wandb.log({"evaluation_summary": table})


def log_fairness_chart(fairness_results: dict):
    """Log fairness comparison as a W&B bar chart."""
    data = []
    for model_name, result in fairness_results.items():
        for cls, val in result["per_class_values"].items():
            data.append([model_name, cls, val])

    table = wandb.Table(data=data, columns=["Model", "Skin Tone Group", "Recall"])
    wandb.log({
        "fairness_comparison": wandb.plot.bar(
            table, "Skin Tone Group", "Recall", title="Per-Class Recall by Model"
        )
    })
```

**Step 2: Commit**

```bash
git add src/utils/logging.py
git commit -m "feat: W&B logging utilities for metrics, confusion matrices, and fairness"
```

---

## Task 14: Notebook 01 — Data Exploration & Cleaning

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`

**Step 1: Create the data exploration notebook**

This notebook is the entry point. It downloads data, runs cleaning, and produces the class distribution analysis.

```python
# Cell 1 — Setup
"""
# 01 — Data Exploration & Cleaning
This notebook downloads the Fitzpatrick17k dataset, cleans it, and analyzes the class distribution.
"""

# Cell 2 — Install dependencies (Colab)
# !pip install -q imagehash wandb google-cloud-storage google-cloud-aiplatform

# Cell 3 — Imports
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path (for Colab: mount drive or clone repo first)
# sys.path.insert(0, '/content/skin_tone_classifier')

from src.data.prepare import (
    load_metadata,
    validate_fitzpatrick_labels,
    encode_grouped_labels,
    validate_images,
    filter_human_images,
    deduplicate_images,
    compute_class_distribution,
    generate_cleaning_report,
    stratified_split,
    download_images,
)

# Cell 4 — Configuration
CSV_PATH = "data/fitzpatrick17k.csv"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/cleaned"
RANDOM_SEED = 42

# Cell 5 — Load metadata
df = load_metadata(CSV_PATH)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
df.head()

# Cell 6 — Explore raw Fitzpatrick distribution
print("\nRaw Fitzpatrick distribution:")
print(df["fitzpatrick"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(8, 5))
df["fitzpatrick"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Raw Fitzpatrick Skin Type Distribution")
ax.set_xlabel("Fitzpatrick Type")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# Cell 7 — Download images (if not already downloaded)
# Uncomment and run if you need to download images from URLs
# downloaded = download_images(df, IMAGE_DIR)
# print(f"Downloaded {downloaded} images")

# Cell 8 — Step 1: Validate Fitzpatrick labels
original_count = len(df)
df = validate_fitzpatrick_labels(df)
print(f"After label validation: {len(df)} ({original_count - len(df)} dropped)")

# Cell 9 — Step 2: Validate images
before = len(df)
df = validate_images(IMAGE_DIR, df)
print(f"After image validation: {len(df)} ({before - len(df)} dropped)")

# Cell 10 — Step 3: Filter to human-only images
before = len(df)
df = filter_human_images(IMAGE_DIR, df)
print(f"After human filter: {len(df)} ({before - len(df)} dropped)")

# Cell 11 — Step 4: Deduplicate
before = len(df)
df = deduplicate_images(IMAGE_DIR, df)
print(f"After deduplication: {len(df)} ({before - len(df)} dropped)")

# Cell 12 — Step 5: Encode grouped labels
df = encode_grouped_labels(df)
print("\nGrouped label distribution:")
print(df["skin_tone_group"].value_counts().sort_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6-class distribution
df["fitzpatrick"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Cleaned: 6-Class Distribution")
axes[0].set_xlabel("Fitzpatrick Type")
axes[0].set_ylabel("Count")

# 3-class distribution
df["skin_tone_group"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color="coral")
axes[1].set_title("Cleaned: 3-Class Grouped Distribution")
axes[1].set_xlabel("Skin Tone Group")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# Cell 13 — Class distribution report
dist = compute_class_distribution(df, "skin_tone_group")
print("\nClass Distribution Report:")
for cls, info in sorted(dist.items()):
    print(f"  {cls}: {info['count']} images ({info['percentage']:.1f}%)")

imbalance_ratio = max(d["count"] for d in dist.values()) / min(d["count"] for d in dist.values())
print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")

# Cell 14 — Stratified split
train_df, val_df, test_df = stratified_split(df, "skin_tone_label", (0.7, 0.15, 0.15), seed=RANDOM_SEED)

print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{split_name} distribution:")
    print(split_df["skin_tone_group"].value_counts().sort_index())

# Cell 15 — Save cleaned data
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
df.to_csv(f"{OUTPUT_DIR}/fitzpatrick17k_cleaned.csv", index=False)
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
print(f"\nSaved cleaned data to {OUTPUT_DIR}/")

# Cell 16 — Summary
print("\n" + "="*60)
print("DATA CLEANING SUMMARY")
print("="*60)
print(f"Original images:     {original_count}")
print(f"After cleaning:      {len(df)}")
print(f"Total dropped:       {original_count - len(df)}")
print(f"Imbalance ratio:     {imbalance_ratio:.2f}x")
print(f"Train/Val/Test:      {len(train_df)}/{len(val_df)}/{len(test_df)}")
print("="*60)
```

**Step 2: Commit**

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "feat: data exploration and cleaning notebook"
```

---

## Task 15: Notebook 02 — Training

**Files:**
- Create: `notebooks/02_training.ipynb`

**Step 1: Create the training notebook**

```python
# Cell 1 — Setup
"""
# 02 — Model Training
Train EfficientNetV2-S and ResNet50 on cleaned Fitzpatrick17k data.
Two-phase training: frozen backbone → fine-tuned.
"""

# Cell 2 — Install (Colab)
# !pip install -q wandb timm

# Cell 3 — Imports
import sys
import torch
import pandas as pd
import numpy as np
import wandb
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import SkinToneClassifier
from src.training.config import TrainingConfig
from src.training.trainer import Trainer, compute_class_weights
from src.utils.logging import init_wandb

# Cell 4 — Configuration
config = TrainingConfig(
    backbone="efficientnet_v2_s",
    num_classes=3,
    pretrained=True,
    freeze_backbone=True,
    unfreeze_after_epochs=5,
    epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    early_stopping_patience=5,
    use_class_weights=True,
    wandb_project="skin-tone-classifier",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

IMAGE_DIR = "data/images"
DATA_DIR = "data/cleaned"

# Cell 5 — Load data splits
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")

print(f"Train: {len(train_df)}, Val: {len(val_df)}")
print(f"Train distribution:\n{train_df['skin_tone_group'].value_counts().sort_index()}")

# Cell 6 — Create datasets and loaders
train_transform = get_train_transforms(config.image_size)
eval_transform = get_eval_transforms(config.image_size)

train_dataset = FitzpatrickDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset = FitzpatrickDataset(val_df, IMAGE_DIR, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
)

# Cell 7 — Compute class weights
labels = train_df["skin_tone_label"].tolist()
weights = compute_class_weights(labels, num_classes=3)
class_weights = torch.tensor(weights, dtype=torch.float32)
print(f"Class weights: {weights}")

# Cell 8 — Initialize model
model = SkinToneClassifier(
    backbone_name=config.backbone,
    num_classes=config.num_classes,
    pretrained=config.pretrained,
)
if config.freeze_backbone:
    model.freeze_backbone()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.1%})")

# Cell 9 — Initialize W&B
run = init_wandb(
    project=config.wandb_project,
    config=vars(config),
    run_name=f"{config.backbone}_lr{config.learning_rate}_bs{config.batch_size}",
    tags=["milestone1", config.backbone],
)

# Cell 10 — Train
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    class_weights=class_weights if config.use_class_weights else None,
    device=DEVICE,
    wandb_run=run,
)

history = trainer.train()

# Cell 11 — Training curves
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs = range(1, len(history["train"]) + 1)

# Loss
axes[0].plot(epochs, [m["loss"] for m in history["train"]], label="Train")
axes[0].plot(epochs, [m["loss"] for m in history["val"]], label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# Accuracy
axes[1].plot(epochs, [m["accuracy"] for m in history["train"]], label="Train")
axes[1].plot(epochs, [m["accuracy"] for m in history["val"]], label="Val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

# F1
axes[2].plot(epochs, [m["f1"] for m in history["train"]], label="Train")
axes[2].plot(epochs, [m["f1"] for m in history["val"]], label="Val")
axes[2].set_title("Macro F1")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()
plt.show()

# Cell 12 — Save model artifact
torch.save(model.state_dict(), f"checkpoints/{config.backbone}_final.pt")
wandb.save(f"checkpoints/{config.backbone}_final.pt")
print(f"Model saved to checkpoints/{config.backbone}_final.pt")

# Cell 13 — Finish W&B run
wandb.finish()
print("Training complete!")

# Cell 14 — Repeat for ResNet50
# Change config.backbone to "resnet50" and re-run cells 4-13
# config = TrainingConfig(backbone="resnet50", ...)
```

**Step 2: Commit**

```bash
git add notebooks/02_training.ipynb
git commit -m "feat: training notebook with two-phase fine-tuning and W&B logging"
```

---

## Task 16: Notebook 03 — Evaluation & Fairness

**Files:**
- Create: `notebooks/03_evaluation.ipynb`

**Step 1: Create the evaluation notebook**

```python
# Cell 1 — Setup
"""
# 03 — Evaluation & Fairness Analysis
Evaluate EfficientNetV2-S, ResNet50, and AutoML baseline.
Compute per-class metrics and fairness gap.
"""

# Cell 2 — Imports
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_eval_transforms
from src.models.classifier import SkinToneClassifier
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.fairness import compute_fairness_gap, compare_model_fairness
from src.evaluation.confusion import plot_confusion_matrix, plot_fairness_comparison

# Cell 3 — Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "data/images"
DATA_DIR = "data/cleaned"
CLASS_NAMES = ["12", "34", "56"]
DISPLAY_NAMES = ["Fitz I-II", "Fitz III-IV", "Fitz V-VI"]

# Cell 4 — Load test data
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Test set: {len(test_df)} images")
print(test_df["skin_tone_group"].value_counts().sort_index())

transform = get_eval_transforms(224)
test_dataset = FitzpatrickDataset(test_df, IMAGE_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Cell 5 — Helper: run inference
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Cell 6 — Evaluate EfficientNetV2-S
model_eff = SkinToneClassifier("efficientnet_v2_s", num_classes=3, pretrained=False)
model_eff.load_state_dict(torch.load("checkpoints/efficientnet_v2_s_final.pt", map_location=DEVICE))
model_eff = model_eff.to(DEVICE)

y_true, y_pred_eff, y_proba_eff = get_predictions(model_eff, test_loader, DEVICE)
metrics_eff = compute_all_metrics(y_true, y_pred_eff, y_proba_eff, CLASS_NAMES)

print("EfficientNetV2-S Results:")
print(f"  Accuracy: {metrics_eff['accuracy']:.4f}")
print(f"  Macro F1: {metrics_eff['macro_f1']:.4f}")
print(f"  ROC-AUC:  {metrics_eff['roc_auc']:.4f}")
for cls in CLASS_NAMES:
    m = metrics_eff["per_class"][cls]
    print(f"  {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

# Cell 7 — Confusion matrix: EfficientNetV2
plot_confusion_matrix(
    metrics_eff["confusion_matrix"],
    DISPLAY_NAMES,
    title="EfficientNetV2-S — Confusion Matrix",
    save_path="results/cm_efficientnet.png",
)

# Cell 8 — Evaluate ResNet50
model_res = SkinToneClassifier("resnet50", num_classes=3, pretrained=False)
model_res.load_state_dict(torch.load("checkpoints/resnet50_final.pt", map_location=DEVICE))
model_res = model_res.to(DEVICE)

_, y_pred_res, y_proba_res = get_predictions(model_res, test_loader, DEVICE)
metrics_res = compute_all_metrics(y_true, y_pred_res, y_proba_res, CLASS_NAMES)

print("\nResNet50 Results:")
print(f"  Accuracy: {metrics_res['accuracy']:.4f}")
print(f"  Macro F1: {metrics_res['macro_f1']:.4f}")
for cls in CLASS_NAMES:
    m = metrics_res["per_class"][cls]
    print(f"  {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

# Cell 9 — Confusion matrix: ResNet50
plot_confusion_matrix(
    metrics_res["confusion_matrix"],
    DISPLAY_NAMES,
    title="ResNet50 — Confusion Matrix",
    save_path="results/cm_resnet50.png",
)

# Cell 10 — Load AutoML results (from notebook 04)
# AutoML metrics must be manually entered or loaded from Vertex AI API
# Example placeholder — replace with actual AutoML results
automl_per_class = {
    "12": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
    "34": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
    "56": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
}
# TODO: Fill in from AutoML evaluation output

# Cell 11 — Fairness gap analysis
print("\n" + "="*60)
print("FAIRNESS ANALYSIS")
print("="*60)

fairness_eff = compute_fairness_gap(metrics_eff["per_class"])
fairness_res = compute_fairness_gap(metrics_res["per_class"])

print(f"\nEfficientNetV2-S Fairness Gap: {fairness_eff['gap']:.2%}")
print(f"  Best:  {fairness_eff['best_class']} ({fairness_eff['best_value']:.2%})")
print(f"  Worst: {fairness_eff['worst_class']} ({fairness_eff['worst_value']:.2%})")
print(f"  Significant: {fairness_eff['significant']}")

print(f"\nResNet50 Fairness Gap: {fairness_res['gap']:.2%}")
print(f"  Best:  {fairness_res['best_class']} ({fairness_res['best_value']:.2%})")
print(f"  Worst: {fairness_res['worst_class']} ({fairness_res['worst_value']:.2%})")
print(f"  Significant: {fairness_res['significant']}")

# Cell 12 — Cross-model fairness comparison
model_metrics = {
    "EfficientNetV2-S": metrics_eff["per_class"],
    "ResNet50": metrics_res["per_class"],
    # "AutoML": automl_per_class,  # Uncomment when AutoML results are ready
}

fairness_results = compare_model_fairness(model_metrics)

plot_fairness_comparison(
    fairness_results,
    save_path="results/fairness_comparison.png",
)

# Cell 13 — Summary table
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Fairness Gap':<15} {'Significant?'}")
print("-"*65)
for name, m, fg in [
    ("EfficientNetV2-S", metrics_eff, fairness_eff),
    ("ResNet50", metrics_res, fairness_res),
]:
    print(f"{name:<20} {m['accuracy']:<10.4f} {m['macro_f1']:<10.4f} {fg['gap']:<15.2%} {fg['significant']}")
```

**Step 2: Commit**

```bash
git add notebooks/03_evaluation.ipynb
git commit -m "feat: evaluation and fairness analysis notebook"
```

---

## Task 17: Notebook 04 — AutoML Baseline

**Files:**
- Create: `notebooks/04_automl_baseline.ipynb`

**Step 1: Create the AutoML baseline notebook**

```python
# Cell 1 — Setup
"""
# 04 — Vertex AutoML Baseline
Train an AutoML Image Classification model on Vertex AI as a zero-effort baseline.
"""

# Cell 2 — Install (Colab)
# !pip install -q google-cloud-aiplatform google-cloud-storage

# Cell 3 — Imports
import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage
from src.utils.gcs import upload_directory_to_gcs, upload_file_to_gcs, generate_automl_csv

# Cell 4 — Configuration
PROJECT_ID = "YOUR_PROJECT_ID"  # Replace with your GCP project
REGION = "us-central1"
BUCKET_NAME = "skin-tone-project"  # Replace with your bucket name
GCS_IMAGE_PREFIX = f"gs://{BUCKET_NAME}/images"

aiplatform.init(project=PROJECT_ID, location=REGION)

# Cell 5 — Upload images to GCS
# Only run this once!
# upload_directory_to_gcs("data/images", BUCKET_NAME, "images")
# print("Images uploaded to GCS")

# Cell 6 — Prepare AutoML manifest
train_df = pd.read_csv("data/cleaned/train.csv")
val_df = pd.read_csv("data/cleaned/val.csv")
test_df = pd.read_csv("data/cleaned/test.csv")

# Add split column
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

manifest_path = generate_automl_csv(
    full_df,
    image_gcs_prefix=GCS_IMAGE_PREFIX,
    output_path="data/automl_manifest.csv",
)
print(f"Manifest created at {manifest_path}")

# Cell 7 — Upload manifest to GCS
manifest_gcs_uri = upload_file_to_gcs(
    manifest_path,
    BUCKET_NAME,
    "automl/manifest.csv",
)
print(f"Manifest uploaded to {manifest_gcs_uri}")

# Cell 8 — Create Vertex AI Dataset
dataset = aiplatform.ImageDataset.create(
    display_name="fitzpatrick17k-skin-tone",
    gcs_source=manifest_gcs_uri,
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
)
print(f"Dataset created: {dataset.resource_name}")

# Cell 9 — Train AutoML model
job = aiplatform.AutoMLImageTrainingJob(
    display_name="skin-tone-automl-baseline",
    prediction_type="classification",
    multi_label=False,
    model_type="CLOUD",
    base_model=None,
)

model = job.run(
    dataset=dataset,
    model_display_name="skin-tone-automl-v1",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=8000,  # ~8 node hours
)
print(f"Model trained: {model.resource_name}")

# Cell 10 — Evaluate AutoML model
model_eval = model.list_model_evaluations()[0]
print("\nAutoML Evaluation Metrics:")
print(f"  Model: {model.display_name}")

metrics = model_eval.metrics
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Cell 11 — Extract per-class metrics for fairness comparison
# Parse AutoML evaluation output into the format needed for fairness analysis
# The exact format depends on AutoML output structure
print("\nPer-class metrics (copy these to notebook 03):")
if "confusionMatrix" in metrics:
    cm = metrics["confusionMatrix"]
    print(cm)

# Cell 12 — Register model ID
print(f"\nVertex Model ID: {model.resource_name}")
print("Use this ID in notebook 03 for comparison.")
```

**Step 2: Commit**

```bash
git add notebooks/04_automl_baseline.ipynb
git commit -m "feat: AutoML baseline notebook with Vertex AI training"
```

---

## Task 18: Vertex AI Training Script (`scripts/train.py`)

**Files:**
- Create: `scripts/train.py`

**Step 1: Implement CLI training entrypoint**

```python
# scripts/train.py
"""CLI entrypoint for training — usable locally or as Vertex AI custom job."""
import argparse
import logging
import sys
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import SkinToneClassifier
from src.training.config import TrainingConfig
from src.training.trainer import Trainer, compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train skin tone classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default="data/cleaned", help="Directory with train/val CSVs")
    parser.add_argument("--image-dir", type=str, default="data/images", help="Directory with images")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to save model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Load config
    config = TrainingConfig.from_yaml(args.config)

    # Apply overrides
    if args.backbone:
        config.backbone = args.backbone
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    config.checkpoint_dir = args.output_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")

    train_dataset = FitzpatrickDataset(train_df, args.image_dir, transform=get_train_transforms(config.image_size))
    val_dataset = FitzpatrickDataset(val_df, args.image_dir, transform=get_eval_transforms(config.image_size))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # Class weights
    class_weights = None
    if config.use_class_weights:
        weights = compute_class_weights(train_df["skin_tone_label"].tolist(), num_classes=config.num_classes)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights: {weights}")

    # Model
    model = SkinToneClassifier(config.backbone, config.num_classes, pretrained=config.pretrained)
    if config.freeze_backbone:
        model.freeze_backbone()

    # W&B
    wandb_run = None
    if not args.no_wandb:
        from src.utils.logging import init_wandb
        wandb_run = init_wandb(config.wandb_project, vars(config), run_name=f"{config.backbone}_vertex")

    # Train
    trainer = Trainer(model, config, train_loader, val_loader, class_weights, device, wandb_run)
    history = trainer.train()

    # Save final model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    final_path = f"{args.output_dir}/{config.backbone}_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/train.py
git commit -m "feat: CLI training script for local and Vertex AI custom jobs"
```

---

## Task 19: Vertex AI Upload Script (`scripts/upload_to_vertex.py`)

**Files:**
- Create: `scripts/upload_to_vertex.py`

**Step 1: Implement upload script**

```python
# scripts/upload_to_vertex.py
"""Upload a trained model to Vertex AI Model Registry."""
import argparse
import logging

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload model to Vertex AI")
    parser.add_argument("--model-path", type=str, required=True, help="GCS path to model artifact directory")
    parser.add_argument("--display-name", type=str, required=True, help="Model display name")
    parser.add_argument("--project", type=str, required=True, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--serving-container", type=str, default="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest", help="Serving container image URI")
    args = parser.parse_args()

    aiplatform.init(project=args.project, location=args.region)

    model = aiplatform.Model.upload(
        display_name=args.display_name,
        artifact_uri=args.model_path,
        serving_container_image_uri=args.serving_container,
    )

    logger.info(f"Model uploaded: {model.resource_name}")
    logger.info(f"Model ID: {model.name}")
    print(f"\nVertex Model ID: {model.resource_name}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/upload_to_vertex.py
git commit -m "feat: Vertex AI Model Registry upload script"
```

---

## Task 20: Dockerfile for Vertex AI Custom Training

**Files:**
- Create: `Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
# Dockerfile for Vertex AI Custom Training Job
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Default entrypoint for Vertex AI custom training
ENTRYPOINT ["python", "scripts/train.py"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: Dockerfile for Vertex AI custom training container"
```

---

## Task 21: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Smoke test: verify the full pipeline works end-to-end on synthetic data."""
import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.prepare import encode_grouped_labels, stratified_split
from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import SkinToneClassifier
from src.training.config import TrainingConfig
from src.training.trainer import Trainer, compute_class_weights
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.fairness import compute_fairness_gap


@pytest.fixture
def synthetic_data(tmp_path):
    """Generate a small synthetic dataset for smoke testing."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(60):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(image_dir / f"img_{i}.jpg")
        fitz = (i % 6) + 1
        rows.append({"hasher": f"img_{i}", "fitzpatrick": fitz})

    df = pd.DataFrame(rows)
    df = encode_grouped_labels(df)
    return df, str(image_dir)


def test_full_pipeline_smoke(synthetic_data):
    """Verify training → evaluation → fairness works end-to-end."""
    df, image_dir = synthetic_data

    # Split
    train_df, val_df, test_df = stratified_split(df, "skin_tone_label", (0.7, 0.15, 0.15))

    # Datasets
    train_ds = FitzpatrickDataset(train_df, image_dir, transform=get_train_transforms(64))
    val_ds = FitzpatrickDataset(val_df, image_dir, transform=get_eval_transforms(64))
    test_ds = FitzpatrickDataset(test_df, image_dir, transform=get_eval_transforms(64))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)

    # Model
    config = TrainingConfig(
        backbone="resnet50",
        pretrained=False,  # No pretrained weights for speed
        image_size=64,
        epochs=2,
        batch_size=8,
        freeze_backbone=False,
        early_stopping_patience=5,
    )

    model = SkinToneClassifier("resnet50", num_classes=3, pretrained=False)

    # Class weights
    weights = compute_class_weights(train_df["skin_tone_label"].tolist(), 3)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Train
    trainer = Trainer(model, config, train_loader, val_loader, class_weights, device="cpu")
    history = trainer.train()

    assert len(history["train"]) == 2
    assert len(history["val"]) == 2

    # Evaluate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    metrics = compute_all_metrics(all_labels, all_preds, np.array(all_probs))
    assert "accuracy" in metrics
    assert "per_class" in metrics

    # Fairness
    fairness = compute_fairness_gap(metrics["per_class"])
    assert "gap" in fairness
    assert "significant" in fairness
```

**Step 2: Run integration test**

```bash
python -m pytest tests/test_integration.py -v --timeout=120
```

Expected: PASS (may take ~30 seconds due to 2 training epochs)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: end-to-end integration smoke test"
```

---

## Task 22: Final — Run All Tests & Verify

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS

**Step 2: Verify project structure**

```bash
find . -type f -name "*.py" | sort
```

Expected output:
```
./scripts/train.py
./scripts/upload_to_vertex.py
./src/__init__.py
./src/data/__init__.py
./src/data/dataset.py
./src/data/prepare.py
./src/data/transforms.py
./src/evaluation/__init__.py
./src/evaluation/confusion.py
./src/evaluation/fairness.py
./src/evaluation/metrics.py
./src/models/__init__.py
./src/models/backbone.py
./src/models/classifier.py
./src/training/__init__.py
./src/training/config.py
./src/training/trainer.py
./src/utils/__init__.py
./src/utils/gcs.py
./src/utils/logging.py
./tests/__init__.py
./tests/test_backbone.py
./tests/test_classifier.py
./tests/test_dataset.py
./tests/test_fairness.py
./tests/test_integration.py
./tests/test_metrics.py
./tests/test_prepare.py
./tests/test_trainer.py
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete skin tone classifier pipeline — ready for training"
```

---

## Execution Order Summary

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | Project scaffolding | None |
| 2 | Data preparation (`prepare.py`) | Task 1 |
| 3 | Transforms (`transforms.py`) | Task 1 |
| 4 | Dataset (`dataset.py`) | Tasks 2, 3 |
| 5 | Backbones (`backbone.py`) | Task 1 |
| 6 | Classifier (`classifier.py`) | Task 5 |
| 7 | Training config (`config.py`) | Task 1 |
| 8 | Trainer (`trainer.py`) | Tasks 6, 7 |
| 9 | Metrics (`metrics.py`) | Task 1 |
| 10 | Fairness (`fairness.py`) | Task 9 |
| 11 | Confusion matrix (`confusion.py`) | Task 9 |
| 12 | GCS utils (`gcs.py`) | Task 1 |
| 13 | W&B logging (`logging.py`) | Task 1 |
| 14 | Notebook 01 — Data exploration | Tasks 2, 3 |
| 15 | Notebook 02 — Training | Tasks 4, 6, 7, 8, 13 |
| 16 | Notebook 03 — Evaluation | Tasks 9, 10, 11 |
| 17 | Notebook 04 — AutoML baseline | Task 12 |
| 18 | Train script | Tasks 4, 6, 7, 8 |
| 19 | Upload script | Task 12 |
| 20 | Dockerfile | Task 18 |
| 21 | Integration test | All src tasks |
| 22 | Final verification | All tasks |

## Parallelization Groups

These tasks can be worked on simultaneously:
- **Group A (data):** Tasks 2, 3 → then 4
- **Group B (models):** Tasks 5 → 6
- **Group C (training):** Task 7 → 8
- **Group D (evaluation):** Tasks 9 → 10, 11
- **Group E (utils):** Tasks 12, 13
