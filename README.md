# COMP9444 Project 093 — Fruit & Vegetable Freshness Assessment

Reproducing **Yuan & Chen (2024)**:

> image → frozen ImageNet-pretrained CNN (feature extractor) → PCA → classifier (SVM / LDA / Bagging)

We compare two backbones — **DenseNet-201** and **ResNeXt-101** — across cases **B / C / F**.
Dataset: 20 classes (5 fruits + 5 vegetables, each _fresh_ / _rotten_), ~12,000 images.

**The dataset is NOT committed to this repo.** Everyone downloads it locally via
`src/preprocessing/download_data.py` (see "Setup" below) — no Kaggle account needed, and this
keeps the repo small.

## Unified settings (both models must match)

| Item             | Unified setting                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| Image size       | 224 × 224                                                                                        |
| Data split       | 70% Train / 15% Validation / 15% Test                                                            |
| Random seed      | 42                                                                                               |
| PCA              | retain 95% variance (`PCA(n_components=0.95)`)                                                   |
| Classifiers      | SVM, LDA, Bagging                                                                                |
| Metrics          | Accuracy, Precision, Recall, F1-score                                                            |
| Model mode       | Pre-trained feature extraction (frozen backbone)                                                 |
| Data requirement | **Both models (DenseNet-201 & ResNeXt-101) must use the identical data split and preprocessing** |

Full details and leakage rules: [`docs/EXPERIMENT_RULES.md`](docs/EXPERIMENT_RULES.md).

## Team

| #   | Area                                               | Owner       |
| --- | --------------------------------------------------- | ----------- |
| 1   | Data split + project management + experiment rules | dacheng-liu |
| 2   | DenseNet-201 feature extraction                    | —           |
| 3   | ResNeXt-101 feature extraction                     | —           |
| 4   | PCA + classifiers (SVM / LDA / Bagging)            | —           |
| 5   | Report                                             | —           |

**Read [`docs/EXPERIMENT_RULES.md`](docs/EXPERIMENT_RULES.md) first** — the shared contract for
seeds, image size, normalisation, the split, PCA rules and file naming. The train/val/test split
is fixed by `src/preprocessing/make_split.py`; **run it locally, do not hand-edit
`docs/file_list.csv`** — with the fixed random seed, everyone's split will match row-for-row even
though the absolute file paths differ per machine.

## Setup

```bash
git clone <repo-url>
cd project

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Windows Git Bash:    source .venv/Scripts/activate
# macOS / Linux:       source .venv/bin/activate

pip install -r requirements.txt
```

## Data preparation (run this before anything else)

Run these three scripts **on your own machine**, in order. Each one prints the exact next
command (including the resolved data path) — just copy/paste it forward.

```bash
python src/preprocessing/download_data.py
python src/preprocessing/check_images.py --data-dir <path printed above>
python src/preprocessing/make_split.py   --data-dir <path printed above>
```

- The dataset downloads into a local cache directory (not the repo) — the exact path differs per
  machine, that's expected.
- Because `RANDOM_STATE = 42` is fixed, everyone's `docs/file_list.csv` has the **same split
  logic** (same image → same train/val/test assignment), even though the absolute paths inside
  the file differ. Don't copy someone else's `file_list.csv` — regenerate your own.
- Outputs produced:
  - `docs/file_list.csv` — the split (filepath, label, split columns)
  - `docs/corrupt_files.txt` — corrupted-file report
  - `results/figures/class_distribution.png`
  - `results/figures/split_distribution.png`
  - `results/tables/image_stats.csv`, `results/tables/split_summary.csv`

## Feature file handoff (roles 2 & 3)

Once DenseNet-201 / ResNeXt-101 features are extracted, commit the `.npy` files straight into
`results/features/` and open a PR — `.gitignore` already allows this, no shared drive needed:

```
results/features/densenet201_features.npy
results/features/densenet201_labels.npy
results/features/resnext101_features.npy
results/features/resnext101_labels.npy
```

Role 4 reads directly from `results/features/` — no need to re-download the dataset or
re-run feature extraction.

⚠️ **Row order matters.** Both models must iterate `docs/file_list.csv` in the same row order, so
that row *i* of the DenseNet features and row *i* of the ResNeXt features correspond to the same
image. `src/preprocessing/validate_features.py` sanity-checks this — run it before opening a PR.

## Collaboration workflow (branch → PR)

`main` is protected — never push to it directly. Work on your own branch and open a PR.

```bash
git checkout main && git pull            # start from latest main
git checkout -b <name>/<what>            # e.g. person2/densenet-features

# ... do your work, then ...
git add -A
git commit -m "brief message"
git push -u origin <name>/<what>         # then open a Pull Request on GitHub
```

Keep your branch fresh with `git pull --rebase origin main` before pushing.

## Layout

```
data/                      dataset images (local only, gitignored — see "Data preparation")
src/
    preprocessing/         [1] download, integrity check, split, feature validation
    feature_extraction/    [2,3] DenseNet-201 / ResNeXt-101
    classification/        [4] PCA + SVM / LDA / Bagging
    evaluation/             [5] metrics & figures
docs/                       EXPERIMENT_RULES.md, file_list.csv (local, gitignored), report skeleton
results/
    features/               [2,3] committed .npy feature files
    figures/, tables/       shared outputs
```

## Person 1 scripts (data preparation)

```bash
# download the dataset locally (prints the resolved --data-dir to use below)
python src/preprocessing/download_data.py

# integrity check + class distribution
python src/preprocessing/check_images.py --data-dir <path printed above>

# stratified 70/15/15 split -> docs/file_list.csv
python src/preprocessing/make_split.py --data-dir <path printed above>

# after roles 2/3 produce features, sanity-check the .npy files
python src/preprocessing/validate_features.py --features-dir results/features --model-name densenet201
```

Roles 2–5: load `docs/file_list.csv`, honour its `split` column, iterate **in row order**,
and follow the naming/PCA rules in `EXPERIMENT_RULES.md`. Feature extraction needs `torch` +
`torchvision` (already in `requirements.txt`).
