# Experiment Rules — COMP9444 Project 093

**Fruit & Vegetable Freshness Assessment (Deep features → PCA → SVM/LDA/Bagging)**

This document is the **shared contract** for the whole team. Roles 2 (DenseNet-201),
3 (ResNeXt-101), 4 (PCA + classifiers) and 5 (report) must all follow it so that
every experiment (Case B / Case C / Case F) is comparable and reproducible.

If anything here needs to change, change it **here first** and tell everyone —
do not silently deviate in your own script.

---

## 1. Global reproducibility

- **Global random seed = `42`.** Every step that involves randomness MUST use it:
  data split, classifier initialisation, cross-validation folds, any shuffling,
  PCA solvers that use randomness (`svd_solver="randomized"`), etc.
- In scikit-learn pass `random_state=42`. In NumPy use `np.random.default_rng(42)`.
- In PyTorch (roles 2/3) set, at the top of the script:
  ```python
  import torch, numpy as np, random
  SEED = 42
  random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  ```
  Feature extraction is inference-only, so this is mostly for any incidental
  randomness — still set it.

---

## 2. Input images & pre-processing

- **Resize** every image to **224 × 224** (the input size of DenseNet-201 and
  ResNeXt-101).
- **Channel order:** RGB. Convert with `Image.open(path).convert("RGB")` so that
  grayscale / RGBA / palette images become 3-channel.
- **Normalisation:** ImageNet statistics (the backbones are ImageNet-pretrained):
  - `mean = [0.485, 0.456, 0.406]`
  - `std  = [0.229, 0.224, 0.225]`
- Reference torchvision transform for **val / test** (no augmentation):
  ```python
  from torchvision import transforms
  eval_tf = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
  ])
  ```
  These constants are also defined once in
  [`src/preprocessing/dataset_utils.py`](../src/preprocessing/dataset_utils.py)
  (`IMAGE_SIZE`, `IMAGENET_MEAN`, `IMAGENET_STD`).

---

## 3. Data split (owned by Person 1 — do not re-split)

- Ratios: **train 70% / val 15% / test 15%**.
- **Stratified** by class label — each of the 20 classes keeps the same
  proportion in every split.
- Fixed `random_state=42`.
- The result is written to **[`docs/file_list.csv`](file_list.csv)**, the
  **single source of truth**. Columns:

  | column     | meaning                                   |
  |------------|-------------------------------------------|
  | `filepath` | absolute path to the image                |
  | `label`    | class name (lower-case, no spaces)        |
  | `split`    | one of `train` / `val` / `test`           |

- **Rule:** roles 2/3/4 read this CSV and honour the `split` column. **Nobody
  re-runs their own split.** Iterate the CSV **in row order** when extracting
  features so the feature matrix rows line up with the CSV.

---

## 4. Data augmentation

- Augmentation is applied to the **train split only**. `val` and `test` get the
  plain eval transform in §2 (resize + normalise, nothing else).
- **Split BEFORE augmenting.** Never augment first and then split — an augmented
  copy of an image must never land in a different split from its original, or the
  test set leaks into training (data leakage).
- For this project the reference pipeline (Yuan & Chen 2024) uses the frozen
  backbone as a fixed feature extractor, so **the default is NO augmentation** to
  keep features deterministic. If a case study explicitly adds train-only
  augmentation, document it in the report and keep val/test clean.

---

## 5. Feature standardisation & PCA (fit on train only)

This is the most common leakage trap — follow it exactly:

- `StandardScaler` and `PCA` are **`fit` on the TRAIN features only**.
- The **same fitted** scaler/PCA is then used to `transform` val and test.
- **Never** `fit` on the whole dataset (train+val+test together), and never fit
  PCA separately on val or test.
  ```python
  scaler = StandardScaler().fit(X_train)
  X_train_s = scaler.transform(X_train)
  X_val_s   = scaler.transform(X_val)
  X_test_s  = scaler.transform(X_test)

  # Agreed setting: keep enough components to retain 95% of the variance.
  pca = PCA(n_components=0.95, random_state=42).fit(X_train_s)
  X_train_p = pca.transform(X_train_s)
  X_val_p   = pca.transform(X_val_s)
  X_test_p  = pca.transform(X_test_s)
  ```
- **PCA target = 95% retained variance** (`n_components=0.95`), fit on train only.
  The resulting component count is decided by the train split; report it before
  touching test.

---

## 6. Test-set usage

- The **test split is used exactly ONCE**, at the very end, after all models and
  hyper-parameters are fixed using train + val.
- **Do not** use the test set to tune hyper-parameters, pick PCA components, do
  model selection, or "peek and adjust". That would invalidate the reported
  numbers.
- All tuning / cross-validation happens on train (and val where applicable).

---

## 7. File naming conventions

Consistent names let role 4/5 load everything in a loop.

**Feature files (roles 2 & 3)** — saved as `.npy`, one array per split, rows in
`file_list.csv` order:

```
{model}_{split}_features.npy      # 2D float array [n_images, feature_dim]
{model}_{split}_labels.npy        # 1D array [n_images] of class labels
```
where `{model} ∈ {densenet201, resnext101}` and `{split} ∈ {train, val, test}`.
Example: `densenet201_train_features.npy`, `resnext101_test_labels.npy`.

**Result / table files (roles 4 & 5)**:
```
results/tables/{model}_{classifier}_{case}_metrics.csv
results/figures/{model}_{classifier}_{case}_confusion.png
```
where `{classifier} ∈ {svm, lda, bagging}` and `{case} ∈ {B, C, F}`.

**Sanity check:** run
[`src/preprocessing/validate_features.py`](../src/preprocessing/validate_features.py)
on each model's feature folder before handing off to role 4.

---

## 8. Metrics (agreed up front so numbers are comparable)

Report at minimum: **accuracy**, **macro-precision / recall / F1**, and a
**confusion matrix**. Use the same metric functions across all cases. The 20
classes are balanced-ish but report macro averages so a few weak classes stay
visible.

---

*Owner: Person 1. Last updated when the split policy or constants change.*
