"""Stage 4 -- sanity-check feature files handed off by roles 2 and 3.

Roles 2 (DenseNet-201) and 3 (ResNeXt-101) each save deep features as .npy
arrays, one file per split (train/val/test), extracted by iterating over
docs/file_list.csv *in row order*. Before role 4 (PCA + classifier) trusts those
arrays, this script verifies that:

  1. Each .npy row count matches the number of images in the corresponding split
     of file_list.csv (so no images were silently dropped/added).
  2. The label file (if provided) lines up with file_list.csv row-for-row, which
     is the cheap way to confirm the extraction kept the CSV ordering.

This does NOT open images or recompute features; it only checks shapes/labels so
2/3 can catch an off-by-one before 4 wastes time on a misaligned matrix.

Usage:
  python src/preprocessing/validate_features.py \
      --features-dir path/to/densenet_features \
      --model-name densenet201

Expected files in --features-dir (naming per docs/EXPERIMENT_RULES.md):
  {model}_{split}_features.npy   e.g. densenet201_train_features.npy
  {model}_{split}_labels.npy     (optional but recommended)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILE_LIST = os.path.join(REPO_ROOT, "docs", "file_list.csv")
SPLITS = ["train", "val", "test"]


def freshness_label(source_label: str) -> str:
    normalized = source_label.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if normalized.startswith("fresh"):
        return "fresh"
    if normalized.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Unknown freshness class: {source_label!r}")


def check_split(
    split: str, expected: pd.DataFrame, features_dir: str, model: str,
    expected_dim: int | None,
) -> list[str]:
    """Return a list of problem strings for one split (empty == all good)."""
    problems: list[str] = []
    feat_path = os.path.join(features_dir, f"{model}_{split}_features.npy")
    lbl_path = os.path.join(features_dir, f"{model}_{split}_labels.npy")

    if not os.path.isfile(feat_path):
        return [f"[{split}] missing feature file: {feat_path}"]

    feats = np.load(feat_path, allow_pickle=False)
    n_expected = len(expected)

    if feats.ndim != 2:
        problems.append(f"[{split}] expected a 2D array, got shape {feats.shape}")
    if feats.shape[0] != n_expected:
        problems.append(
            f"[{split}] row count mismatch: features={feats.shape[0]} "
            f"vs file_list={n_expected}"
        )
    if expected_dim is not None and feats.ndim == 2 and feats.shape[1] != expected_dim:
        problems.append(
            f"[{split}] feature dimension mismatch: features={feats.shape[1]} "
            f"vs expected={expected_dim}"
        )

    if os.path.isfile(lbl_path):
        labels = np.load(lbl_path, allow_pickle=True)
        if len(labels) != n_expected:
            problems.append(
                f"[{split}] label count mismatch: labels={len(labels)} "
                f"vs file_list={n_expected}"
            )
        else:
            expected_labels = expected["label"].map(freshness_label).to_numpy().astype(str)
            actual_labels = np.asarray(labels).astype(str)
            if np.array_equal(actual_labels, expected_labels):
                expected_labels = None
            else:
                n_bad = int((actual_labels != expected_labels).sum())
                problems.append(
                    f"[{split}] binary label ORDER mismatch on {n_bad} row(s): features were "
                    "likely not extracted in file_list.csv row order"
                )
        if os.path.isfile(lbl_path) and len(labels) == n_expected and not problems:
            actual_labels = np.asarray(labels).astype(str)
            if not set(actual_labels).issubset({"fresh", "rotten"}):
                problems.append(f"[{split}] labels must be binary fresh/rotten")
    else:
        problems.append(f"[{split}] no label file ({lbl_path}); order not verified")

    if not problems:
        print(f"  [{split}] OK  ({feats.shape[0]} rows, dim={feats.shape[1] if feats.ndim == 2 else '?'})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, help="Folder with the .npy files.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Feature file prefix, e.g. densenet201 or resnext101.",
    )
    parser.add_argument(
        "--expected-dim", type=int, default=None,
        help="Optional expected feature dimension, e.g. 1920 for DenseNet-201.",
    )
    args = parser.parse_args()

    if not os.path.isfile(FILE_LIST):
        print(f"file_list.csv not found at {FILE_LIST}. Run make_split.py first.", file=sys.stderr)
        return 1

    file_list = pd.read_csv(FILE_LIST)
    print(f"Validating '{args.model_name}' features against {FILE_LIST}\n")

    all_problems: list[str] = []
    for split in SPLITS:
        expected = file_list[file_list["split"] == split].reset_index(drop=True)
        all_problems.extend(
            check_split(split, expected, args.features_dir, args.model_name, args.expected_dim)
        )

    print("\n=== Validation result ===")
    if all_problems:
        for p in all_problems:
            print(f"  FAIL {p}")
        print(f"\n{len(all_problems)} problem(s) found -- do not proceed to PCA yet.")
        return 1
    print("  All splits match file_list.csv in count and order. Safe to hand off to role 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
