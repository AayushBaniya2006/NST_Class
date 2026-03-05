# NST_Class — Full Project Log

Everything that was built, changed, fixed, and decided across the entire development of this skin tone classification pipeline.

---

## What This Project Is

A PyTorch pipeline that classifies dermatology images into 6 Fitzpatrick skin tone types (I-VI) and measures fairness gaps across skin tones. Uses transfer learning with EfficientNetV2-S and ResNet50, trained on Google Colab, compared against a Vertex AI AutoML baseline.

**Repo:** https://github.com/AayushBaniya2006/NST_Class
**Dataset:** Fitzpatrick17k (~16,577 clinical dermatology images)

---

## Timeline of Work

### Phase 1: Initial Build (3-Class Design)

The project was originally built with **3 grouped classes**:
- Group 1: Fitzpatrick I-II (light skin)
- Group 2: Fitzpatrick III-IV (medium skin)
- Group 3: Fitzpatrick V-VI (dark skin)

Everything was built end-to-end:
- `src/data/prepare.py` — full data cleaning pipeline (download, validate, filter, deduplicate, split)
- `src/data/dataset.py` — PyTorch Dataset class
- `src/data/transforms.py` — train/eval augmentation pipelines
- `src/models/backbone.py` — EfficientNetV2-S and ResNet50 backbone wrappers
- `src/models/classifier.py` — SkinToneClassifier with dropout + linear head
- `src/training/trainer.py` — two-phase training loop (frozen → fine-tune)
- `src/training/config.py` — TrainingConfig dataclass, loadable from YAML
- `src/evaluation/metrics.py` — accuracy, F1, precision, recall, ROC-AUC
- `src/evaluation/fairness.py` — fairness gap analysis + cross-model comparison
- `src/evaluation/confusion.py` — confusion matrix and fairness bar chart plots
- `src/utils/gcs.py` — GCS upload/download + AutoML manifest generation
- `src/utils/logging.py` — Weights & Biases integration
- `scripts/train.py` — CLI training entrypoint
- `scripts/upload_to_vertex.py` — upload model to Vertex AI Model Registry
- `configs/default.yaml` — full experiment configuration
- `Dockerfile` — Vertex AI custom training container
- 4 notebooks (data exploration, training, evaluation, AutoML baseline)
- Full test suite

The project was completed through Phase 6 (evaluation) of an 8-phase plan.

---

### Phase 2: 3-Class → 6-Class Refactor

**Decision:** Switch from 3 grouped classes to all 6 individual Fitzpatrick types for full granularity and per-type fairness analysis.

**What changed (20+ files):**

#### Core Pipeline
- `src/data/prepare.py` — Removed `GROUP_MAP`, `GROUP_TO_LABEL`, `encode_grouped_labels()`. Added `FITZPATRICK_TO_LABEL` (1→0, 2→1, ..., 6→5), `NUM_CLASSES=6`, `encode_labels()`
- `src/training/config.py` — `num_classes=6`, `class_names=["1","2","3","4","5","6"]`
- `src/models/classifier.py` — `num_classes=6` default
- `src/evaluation/metrics.py` — 6 default class names
- `src/evaluation/confusion.py` — display names Fitz I-VI, dynamic tick labels
- `src/utils/gcs.py` — `label_column` default changed from `"skin_tone_group"` to `"fitzpatrick"`
- `configs/default.yaml` — `num_classes: 6`

#### Tests (all 7 test files updated)
- `test_prepare.py` — new TestConstants, TestEncodeLabels replaces TestEncodeGroupedLabels
- `test_classifier.py` — `num_classes=6`, output shapes `(4, 6)`
- `test_dataset.py` — labels `[0,1,2,3,4,5]`
- `test_trainer.py` — 6-class weight tests
- `test_metrics.py` — 6x6 confusion matrix
- `test_fairness.py` — 6 entries per per_class dict
- `test_integration.py` — 120 synthetic images, `num_classes=6`

#### Notebooks
- All 4 notebooks updated for 6-class labels

#### README
- Fully updated for 6-class design

All 69 tests passing after refactor (later grew to 71).

---

### Phase 3: Making Notebooks Self-Contained for Colab

**Problem:** User couldn't run notebooks in Colab because they required manual setup (cloning repo, installing deps, downloading CSV, setting sys.path).

**Solution:** Made all 4 notebooks fully self-contained. Cell 1 of each notebook now:
1. Clones the repo from GitHub (`git clone https://github.com/AayushBaniya2006/NST_Class.git`)
2. Installs all dependencies (`pip install -q -r requirements.txt`)
3. Downloads the Fitzpatrick17k CSV (notebook 01 only)
4. Sets up `sys.path`

---

### Phase 4: CSV Column Mismatch Discovery

**Problem:** The Fitzpatrick17k CSV from GitHub uses different column names than the pipeline expected.

| What pipeline expected | What CSV actually has |
|----------------------|---------------------|
| `hasher` | `url_alphanum` |
| `fitzpatrick` | `fitzpatrick_scale` |

**Fix:** Added `_COLUMN_RENAMES` dict to `load_metadata()` that auto-renames columns on load:
```python
_COLUMN_RENAMES = {
    "fitzpatrick_scale": "fitzpatrick",
    "url_alphanum": "hasher",
}
```

**How discovered:** Used Context7 and WebFetch to examine the actual CSV structure on GitHub.

---

### Phase 5: Image Download Issues

**Problem 1:** `download_images()` crashed on rows with NaN URL values (some Fitzpatrick17k rows have missing URLs).

**Fix:** Added `pd.isna()` check to skip rows with missing URL or hasher.

**Problem 2:** Downloading 16,577 images one at a time with 15s timeout was taking 4+ hours on Colab and timing out.

**Fix:** Rewrote `download_images()` to use:
- `ThreadPoolExecutor` with 20 concurrent workers (~15-20x faster)
- Batches of 500 with per-batch progress bars
- Automatic skip of already-downloaded images (survives Colab restarts)
- 10s timeout per request

---

### Phase 6: Deep Code Analysis ("Hyperanalysis")

Launched a comprehensive code exploration that analyzed every file in the project. Found 30+ issues across critical, high, and medium severity.

#### Critical Fixes Made
1. **`roc_auc_score` with invalid `average=None`** — scikit-learn's `multi_class` parameter doesn't support `average=None`. Replaced with manual binary one-vs-rest per-class AUC computation.

2. **`generate_automl_csv` label column** — Default `label_column="fitzpatrick"` was sending raw 1-6 values. Changed to `"skin_tone_label"` (encoded 0-5) for consistency.

3. **`dataset.py._find_image` missing bare path check** — Downloaded images can have double extensions (e.g., `hasher.jpg.jpg` where hasher already ends in `.jpg`). Added bare hasher path check before trying extensions.

#### High-Priority Fixes Made
4. **Stale comment** — `test_prepare.py` line 98 said "encode_grouped_labels", changed to "encode_labels".

5. **No tests for CSV column rename** — Added `test_renames_upstream_columns` and `test_already_renamed_columns_unchanged` to `TestLoadMetadata`.

6. **Notebook 04 AutoML split override** — `job.run()` was passing `training_fraction_split=0.8/0.1/0.1` which overrides the ML_USE column in the manifest. Removed explicit fractions.

#### Medium Fixes Made
7. **Unused `timm` dependency** — Listed in `requirements.txt` but never imported. Removed.

8. **GCS image extension detection** — Added `_find_image_extension()` helper and `image_dir` parameter to `generate_automl_csv()` to detect actual file extensions instead of hardcoding `.jpg`.

9. **Confusion matrix division by zero** — Added guard for empty classes in normalized confusion matrix.

#### Other Improvements
- Updated Dockerfile base image to `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- Added `ENV PYTHONPATH=/app` to Dockerfile
- Added `.dockerignore` and `pyproject.toml`
- Config simplified: removed `unfreeze_layers`, `optimizer`, `scheduler` (hardcoded in trainer)
- Added `unfreeze_n_blocks` config parameter

---

### Phase 7: Google Drive Integration

**Problem:** Colab runtimes disconnect between notebooks, losing all data.

**Solution:** Added Google Drive backup/restore cells to all notebooks:
- **Notebook 01** — saves cleaned CSVs to `MyDrive/NST_cleaned_data/` and images to `MyDrive/NST_images/` at the end
- **Notebook 02** — restores data from Drive at start, saves checkpoints to `MyDrive/NST_checkpoints/` at end
- **Notebook 03** — restores data + checkpoints from Drive at start
- **Notebook 04** — restores data from Drive at start

---

### Phase 8: Hyper-Accurate README

Rewrote the entire README with every claim verified against source code:

**Corrections made:**
- Test count: 69 → **71** (test_prepare.py: 43 → **45**)
- prepare.py functions: 11 → **12**
- Clone URL: `YOUR_USERNAME/skin-tone-classifier` → `AayushBaniya2006/NST_Class`
- Project root: `skin_tone_classifier/` → `NST_Class/`
- Config table: removed stale keys (`unfreeze_layers`, `optimizer`, `scheduler`), added all actual keys
- Added missing files to project tree (`.dockerignore`, `pyproject.toml`)
- Added technical details: per-class ROC-AUC method, image file discovery, CSV column auto-rename
- Added dead URLs to risk analysis
- Added full dependencies section with exact package versions
- Noted `CosineAnnealingLR` T_max is per-phase, not total epochs
- Noted early stopping resets at phase transition

---

## Current State

### What's Done
- All source code written and tested (71 tests passing)
- Data pipeline complete (download, clean, validate, filter, deduplicate, encode, split)
- Notebook 01 run successfully — 12,177 images survived cleaning (73% of 16,577)
  - 9.67x class imbalance ratio
  - Train: 8,523 / Val: 1,827 / Test: 1,827
- Data backed up to Google Drive
- Notebook 02 currently running (training EfficientNetV2-S)

### What's Left
- Finish training EfficientNetV2-S (notebook 02)
- Train ResNet50 (modify backbone in notebook 02 and re-run)
- Run evaluation and fairness analysis (notebook 03)
- Optionally run AutoML baseline (notebook 04, requires GCP project)
- Upload model to Vertex AI Model Registry (Phase 7-8 of original plan)

---

## Architecture Overview

```
Fitzpatrick17k CSV
        |
        v
  load_metadata()          ← auto-renames fitzpatrick_scale→fitzpatrick, url_alphanum→hasher
        |
        v
  download_images()        ← 20 parallel workers, batched, resumable
        |
        v
  validate_fitzpatrick_labels()  ← drops NaN, out-of-range (not 1-6)
        |
        v
  validate_images()        ← drops missing files, corrupted images
        |
        v
  filter_human_images()    ← min 50x50px, RGB only, color variance > 10
        |
        v
  deduplicate_images()     ← perceptual hash (pHash) deduplication
        |
        v
  encode_labels()          ← Fitzpatrick 1→0, 2→1, ..., 6→5
        |
        v
  stratified_split()       ← 70/15/15 train/val/test, stratified by label
        |
        v
  FitzpatrickDataset       ← PyTorch Dataset with transforms
        |
        v
  Trainer                  ← Phase 1: frozen backbone (epochs 1-5)
        |                     Phase 2: unfreeze last 2 blocks (epochs 6-20)
        |                     Adam optimizer, CosineAnnealingLR
        |                     CrossEntropyLoss with inverse-frequency class weights
        v
  compute_all_metrics()    ← accuracy, macro F1, per-class P/R/F1, ROC-AUC
        |
        v
  compute_fairness_gap()   ← max(recall) - min(recall) across 6 types
        |
        v
  compare_model_fairness() ← side-by-side: EfficientNet vs ResNet vs AutoML
```

---

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Labels | 6 individual Fitzpatrick types | Full granularity, per-type fairness |
| Primary backbone | EfficientNetV2-S | Best accuracy-to-compute ratio |
| Comparison backbone | ResNet50 | Well-studied baseline |
| Transfer learning | Two-phase (frozen → fine-tune) | Prevents catastrophic forgetting |
| Class imbalance | Inverse-frequency weighting | Upweights underrepresented types V-VI |
| Deduplication | Perceptual hash (pHash) | Catches visually identical images across encodings |
| Fairness metric | Recall gap across types | Directly measures missed diagnoses per skin tone |
| Training env | Google Colab → Vertex AI | Free GPU for development, production on GCP |
| Experiment tracking | Weights & Biases | Free for students, best-in-class |
| Data persistence | Google Drive backup/restore | Survives Colab runtime disconnects |

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/data/prepare.py` | ~360 | 12 functions: full data cleaning pipeline |
| `src/data/dataset.py` | ~57 | FitzpatrickDataset with extension-aware image finding |
| `src/data/transforms.py` | ~40 | Train (augmented) and eval (clean) transform pipelines |
| `src/models/backbone.py` | ~60 | EfficientNetV2-S and ResNet50 wrappers |
| `src/models/classifier.py` | ~80 | SkinToneClassifier: backbone + Dropout(0.3) + Linear(feat_dim, 6) |
| `src/training/config.py` | ~65 | TrainingConfig dataclass with YAML loading |
| `src/training/trainer.py` | ~230 | Two-phase training, early stopping, checkpointing, W&B logging |
| `src/evaluation/metrics.py` | ~75 | All metrics including per-class binary OVR ROC-AUC |
| `src/evaluation/fairness.py` | ~77 | Fairness gap + cross-model comparison |
| `src/evaluation/confusion.py` | ~103 | Confusion matrix heatmap + fairness bar chart |
| `src/utils/gcs.py` | ~120 | GCS helpers + AutoML manifest with extension detection |
| `src/utils/logging.py` | ~30 | W&B init wrapper |
| `scripts/train.py` | ~111 | CLI entrypoint with argparse overrides |
| `scripts/upload_to_vertex.py` | ~40 | Upload model to Vertex AI Model Registry |
| `configs/default.yaml` | ~55 | All default hyperparameters |
| `Dockerfile` | ~23 | pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime |
| `tests/test_prepare.py` | ~432 | 45 tests for data pipeline |
| `tests/test_backbone.py` | ~50 | 5 tests for backbone loading |
| `tests/test_classifier.py` | ~60 | 4 tests for classifier |
| `tests/test_dataset.py` | ~70 | 4 tests for dataset |
| `tests/test_metrics.py` | ~50 | 3 tests for metrics |
| `tests/test_fairness.py` | ~60 | 4 tests for fairness |
| `tests/test_trainer.py` | ~70 | 5 tests for training utils |
| `tests/test_integration.py` | ~80 | 1 end-to-end smoke test |
| `requirements.txt` | ~31 | 16 dependencies (no timm) |
| `README.md` | ~600 | Hyper-accurate project documentation |

**Total: 71 tests across 8 test files**

---

## Commit History

| Commit | Description |
|--------|-------------|
| `4285d9a` | refactor: switch from 3-class grouped to 6-class individual Fitzpatrick types |
| `85375f1` | fix: add Colab setup cell with git clone, pip install, and dataset download |
| `fee526e` | fix: auto-rename CSV columns and use correct dataset download URL |
| `210ad94` | fix: make all notebooks fully self-contained for Colab |
| `be8c717` | fix: uncomment image download step in notebook 01 |
| `25e97d5` | fix: address all hyperanalysis issues across codebase |
| `af38724` | fix: skip rows with NaN url/hasher in download_images |
| `249fd0a` | fix: parallel image downloads + hyper-accurate README |
| `e3b946e` | feat: add Google Drive backup/restore to all notebooks |
| `9c31b4a` | fix: match Google Drive folder names to user's actual layout |

---

## Known Issues / Future Work

1. **Dead source URLs** — The Fitzpatrick17k dataset is from 2021. Some image URLs are dead. ~73% of images downloaded successfully. For full dataset access, request the bulk archive from the dataset authors.

2. **Class imbalance** — 9.67x ratio between largest and smallest class. Handled by inverse-frequency weighting but could benefit from additional strategies (focal loss, mixup, oversampling).

3. **AutoML comparison** — Requires GCP project with billing enabled. Currently a placeholder in notebook 03.

4. **Vertex AI deployment** — Phases 7-8 of original plan. Code exists (`scripts/upload_to_vertex.py`, `Dockerfile`) but hasn't been executed yet.

5. **Potential improvements** — Lesion segmentation masking, Lab color space, multi-task learning, Monk Skin Tone scale comparison, adversarial debiasing.
