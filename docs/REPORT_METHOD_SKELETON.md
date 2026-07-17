# Report — Method (skeleton)

> Skeleton for role 5 to fill in. Person 1 has completed the *Dataset* and
> *Data preparation* parts below (marked ✅); the rest are placeholders `[...]`.

## 3. Method

### 3.1 Overview

We reproduce the pipeline of **Yuan & Chen (2024)**: a frozen ImageNet-pretrained
CNN backbone extracts deep features from each image, PCA reduces the feature
dimensionality, and a classical classifier (SVM / LDA / Bagging) predicts the
freshness class. We compare two backbones — **DenseNet-201** and
**ResNeXt-101** — across cases **B / C / F**.

`[insert a pipeline diagram: image → backbone → feature vector → PCA → classifier → class]`

### 3.2 Dataset ✅

- **Source:** Kaggle `muhriddinmuxiddinov/fruits-and-vegetables-dataset`.
- **Classes:** 20 (5 vegetables + 5 fruits, each in *fresh* and *rotten*).
- **Size:** ~12,000 images (see the exact count in
  `results/tables/format_size_summary.csv`).
- **Integrity:** every image was verified to decode with Pillow (and OpenCV
  where available); unreadable files are listed in `docs/corrupt_files.txt` and
  excluded before splitting. `[fill in: N corrupt files found]`.
- **Format / size:** `[fill in from format_size_summary.csv — formats present,
  size range; note whether images vary in size and are therefore resized]`.
- Class distribution: see Figure `class_distribution.png`.

`[Figure: results/figures/class_distribution.png]`

### 3.3 Data preparation ✅

- Images are resized to **224 × 224**, converted to RGB, and normalised with
  ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- The data is split **train 70% / val 15% / test 15%**, **stratified** by class,
  with a fixed seed of **42**. The split is stored in `docs/file_list.csv` and
  used identically by every experiment. Per-split counts:
  `[fill in from results/tables/split_summary.csv]`.
- No augmentation is applied by default (the backbone is used as a fixed feature
  extractor); if used, augmentation is train-only. Full rules: `EXPERIMENT_RULES.md`.

`[Figure: results/figures/split_distribution.png]`

### 3.4 Feature extraction `[roles 2 & 3]`

- **DenseNet-201:** `[which layer's output is used, feature dimension, batch size,
  device]`.
- **ResNeXt-101:** `[same details]`.
- Features are saved per split in `file_list.csv` row order as
  `{model}_{split}_features.npy` and validated with `validate_features.py`.

### 3.5 Dimensionality reduction (PCA) `[role 4]`

- `StandardScaler` + `PCA` are fit on **train only** and applied to val/test.
- Number of components: `[k / explained-variance target and how it was chosen]`.

### 3.6 Classification `[role 4]`

- Classifiers: **SVM** `[kernel, C, gamma]`, **LDA** `[solver]`,
  **Bagging** `[base estimator, n_estimators]`.
- Hyper-parameters selected on train/val; test used once at the end.

### 3.7 Experimental cases `[role 5]`

- **Case B:** `[describe]`
- **Case C:** `[describe]`
- **Case F:** `[describe]`

### 3.8 Evaluation metrics

- Accuracy, macro precision / recall / F1, and confusion matrices, computed with
  identical functions across all cases (see `EXPERIMENT_RULES.md` §8).

---

*References:* Yuan, L., & Chen, ... (2024). `[full citation — see sampleArticle.pdf]`.
