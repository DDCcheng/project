<<<<<<< HEAD
# COMP9444 Project 093 — Fruit & Vegetable Freshness Assessment

Reproducing **Yuan & Chen (2024)**:

> image → frozen ImageNet-pretrained CNN (feature extractor) → PCA → classifier (SVM / LDA / Bagging)

We compare two backbones — **DenseNet-201** and **ResNeXt-101** — across cases **B / C / F**.
Dataset: 20 classes (5 fruits + 5 vegetables, each *fresh* / *rotten*), ~12,000 images —
**committed under [`data/`](data/)**, so you get it straight from `git pull` (no Kaggle account needed).

## Unified settings (both models must match)

| Item | Unified setting |
|------|-----------------|
| Image size | 224 × 224 |
| Data split | 70% Train / 15% Validation / 15% Test |
| Random seed | 42 |
| PCA | retain 95% variance (`PCA(n_components=0.95)`) |
| Classifiers | SVM, LDA, Bagging |
| Metrics | Accuracy, Precision, Recall, F1-score |
| Model mode | Pre-trained feature extraction (frozen backbone) |
| Data requirement | **Both models (DenseNet-201 & ResNeXt-101) must use the identical data split and preprocessing** |

Full details and leakage rules: [`docs/EXPERIMENT_RULES.md`](docs/EXPERIMENT_RULES.md).

## Team

| # | Area | Owner |
|---|------|-------|
| 1 | Data split + project management + experiment rules | dacheng-liu |
| 2 | DenseNet-201 feature extraction | — |
| 3 | ResNeXt-101 feature extraction | — |
| 4 | PCA + classifiers (SVM / LDA / Bagging) | — |
| 5 | Report | — |

**Read [`docs/EXPERIMENT_RULES.md`](docs/EXPERIMENT_RULES.md) first** — the shared contract for
seeds, image size, normalisation, the split, PCA rules and file naming. The train/val/test split
is fixed in [`docs/file_list.csv`](docs/file_list.csv); **do not regenerate it** — just read it.

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
data/                     dataset images (committed)
src/
    preprocessing/        [1] data checks, split, feature validation
    feature_extraction/   [2,3] DenseNet-201 / ResNeXt-101
    classification/       [4] PCA + SVM / LDA / Bagging
    evaluation/           [5] metrics & figures
docs/                     EXPERIMENT_RULES.md, file_list.csv, report skeleton
results/                  figures/ and tables/ (shared outputs)
```

## Person 1 scripts (data preparation)

```bash
# integrity check + class distribution
python src/preprocessing/check_images.py --data-dir data/<dataset-folder>

# stratified 70/15/15 split -> docs/file_list.csv
python src/preprocessing/make_split.py --data-dir data/<dataset-folder>

# after roles 2/3 produce features, sanity-check the .npy files
python src/preprocessing/validate_features.py --features-dir <folder> --model-name densenet201
```

Roles 2–5: load `docs/file_list.csv`, honour its `split` column, iterate **in row order**,
and follow the naming/PCA rules in `EXPERIMENT_RULES.md`. Feature extraction needs `torch` +
`torchvision` (already in `requirements.txt`).
=======
Repository for COMP9444 PROJECT.
Project distribution:
1. Data split and writing project document: dacheng-liu
2. DenseNet-201
3. ResNeXt-101
4. PCA
5. Report: Xinye Pan
>>>>>>> 684225af4ac2766a7518d08c1abeeb5c3a0989ec
