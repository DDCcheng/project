"""Create an auditable DenseNet-201 feature handoff summary.

This stage intentionally does not train a classifier or report Accuracy,
Precision, Recall, or F1.  Those metrics belong to the PCA/classifier stage.

Usage:
  python src/evaluation/summarize_densenet201.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_LIST = REPO_ROOT / "docs" / "file_list.csv"
FEATURE_ROOT = REPO_ROOT / "results" / "features"
TABLE_ROOT = REPO_ROOT / "results" / "tables"
REPORT_PATH = REPO_ROOT / "docs" / "DENSENET201_ANALYSIS.md"


def load_case(case: str) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = FEATURE_ROOT / case
    with open(root / "metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    features = {
        split: np.load(root / f"densenet201_{split}_features.npy", allow_pickle=False)
        for split in ["train", "val", "test"]
    }
    labels = {
        split: np.load(root / f"densenet201_{split}_labels.npy", allow_pickle=True).astype(str)
        for split in ["train", "val", "test"]
    }
    return metadata, features, labels


def main() -> int:
    file_list = pd.read_csv(FILE_LIST)
    no_aug_meta, no_aug_features, no_aug_labels = load_case("no_aug")
    aug_meta, aug_features, aug_labels = load_case("aug")
    split_counts = file_list["split"].value_counts().to_dict()
    format_summary_path = TABLE_ROOT / "format_size_summary.csv"
    raw_total = int(pd.read_csv(format_summary_path)["total_images"].iloc[0]) if format_summary_path.is_file() else len(file_list)
    hash_path = TABLE_ROOT / "image_hashes.csv"
    conflicting_paths = 0
    if hash_path.is_file():
        hashes = pd.read_csv(hash_path)
        conflicting = set(
            hashes.groupby("sha256")["label"].nunique().loc[lambda s: s > 1].index
        )
        conflicting_paths = int(hashes["sha256"].isin(conflicting).sum())

    rows: list[dict[str, object]] = []
    for case, metadata, features, labels in [
        ("no_aug", no_aug_meta, no_aug_features, no_aug_labels),
        ("aug", aug_meta, aug_features, aug_labels),
    ]:
        for split in ["train", "val", "test"]:
            rows.append(
                {
                    "case": case,
                    "split": split,
                    "rows": int(features[split].shape[0]),
                    "feature_dim": int(features[split].shape[1]),
                    "fresh": int((labels[split] == "fresh").sum()),
                    "rotten": int((labels[split] == "rotten").sum()),
                    "seconds": metadata["splits"][split]["seconds"],
                }
            )
    summary = pd.DataFrame(rows)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_ROOT / "densenet201_feature_summary.csv", index=False)

    checks = {
        "file_list_rows": int(len(file_list)),
        "file_list_split_counts": file_list["split"].value_counts().to_dict(),
        "no_aug_aug_labels_identical": all(
            np.array_equal(no_aug_labels[split], aug_labels[split])
            for split in ["train", "val", "test"]
        ),
        "val_features_identical_between_cases": bool(
            np.array_equal(no_aug_features["val"], aug_features["val"])
        ),
        "test_features_identical_between_cases": bool(
            np.array_equal(no_aug_features["test"], aug_features["test"])
        ),
        "train_max_abs_difference": float(
            np.max(np.abs(no_aug_features["train"] - aug_features["train"]))
        ),
        "train_mean_l2_difference": float(
            np.linalg.norm(no_aug_features["train"] - aug_features["train"], axis=1).mean()
        ),
    }

    report = f"""# DenseNet-201 Feature Extraction Analysis

## Scope

This handoff covers the frozen ImageNet-pretrained DenseNet-201 feature
extraction stage for the **fresh/rotten binary classification** task. No PCA,
classifier, or test-set classification metric was fitted in this stage, so
Accuracy, Precision, Recall, and F1 are **待运行** until the classification role
uses these files.

## Data and split

- Raw images checked: {raw_total}.
- Readable images: {raw_total}; corrupt images: 0.
- Original source classes: 20.
- After removing {conflicting_paths} exact duplicate paths with conflicting
  source labels, the shared split contains {len(file_list)} images.
- Split counts: train {split_counts.get('train', 0)}; validation {split_counts.get('val', 0)}; test {split_counts.get('test', 0)}.
- Split policy: source-class stratification, seed 42, with exact duplicate byte
  groups kept within one split. No duplicate hash group crosses splits.
- Binary labels are derived as `fresh* -> fresh` and `rotten* -> rotten`.

## DenseNet-201 features

- Weights: `DenseNet201_Weights.DEFAULT` (ImageNet).
- Parameters frozen; `model.eval()` and `torch.no_grad()` used.
- Input: RGB, 224×224, ImageNet mean/std normalization.
- Global-average-pooled feature dimension: 1,920.
- Device: CPU.
- Files are in `results/features/no_aug/` and `results/features/aug/`, with
  rows matching `docs/file_list.csv` order within each split.

## Augmentation comparison

- `no_aug`: resize and normalization only.
- `aug`: one deterministic train-only view per training image using mild
  RandomResizedCrop, HorizontalFlip, Rotation, and ColorJitter.
- Validation and test features are reused from the clean evaluation pipeline.
- Validation and test features are identical between cases: `{checks['val_features_identical_between_cases']}` / `{checks['test_features_identical_between_cases']}`.
- Training features differ as expected: maximum absolute difference
  `{checks['train_max_abs_difference']:.6f}`, mean per-image L2 difference
  `{checks['train_mean_l2_difference']:.6f}`.

## Interpretation and limitations

The feature files are suitable for the next role to fit StandardScaler/PCA on
training features only and then train SVM, LDA, and Bagging. The augmentation
comparison isolates the effect of changing training representations while
keeping validation and test preprocessing fixed. This stage does not establish
which case or classifier is best; that requires validation-based selection and
one final test evaluation. The experiment was CPU-only, so the reported
extraction time is hardware-specific.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"Summary table: {TABLE_ROOT / 'densenet201_feature_summary.csv'}")
    print(f"Analysis report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
