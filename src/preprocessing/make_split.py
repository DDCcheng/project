"""Stage 1.3 + 1.4 -- stratified train/val/test split and the master file list.

This produces docs/file_list.csv, the SINGLE SOURCE OF TRUTH that roles 2, 3 and
4 read. Nobody else re-splits the data: they load this CSV and honour the
``split`` column so that every experiment (Case B / C / F) uses identical folds.

Split policy (see docs/EXPERIMENT_RULES.md):
  - train 70% / val 15% / test 15%
  - stratified by class label (each class keeps the same proportion in each fold)
  - fixed random_state = 42 for full reproducibility
  - corrupt files (from check_images.py) are excluded before splitting

Outputs:
  docs/file_list.csv                       columns: filepath,label,split
  results/figures/split_distribution.png   per-class train/val/test bar chart
  results/tables/split_summary.csv         counts per split and per class

Usage:
  python src/preprocessing/make_split.py --data-dir path/to/dataset
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_utils as du  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
TABLE_DIR = os.path.join(REPO_ROOT, "results", "tables")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_corrupt_set() -> set[str]:
    """Read docs/corrupt_files.txt (if present) so we can exclude bad files."""
    path = os.path.join(DOCS_DIR, "corrupt_files.txt")
    if not os.path.isfile(path):
        return set()
    bad = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bad.add(line.split("\t", 1)[0])
    return bad


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``split`` column (train/val/test) with per-class stratification.

    Two-step split: first carve out test (15%), then split the remaining 85%
    into train and val so that val is 15% of the whole. Both steps are
    stratified on the label and seeded with RANDOM_STATE.
    """
    idx = df.index.to_numpy()
    labels = df["label"].to_numpy()

    # Step 1: hold out the test set (15% of everything).
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_FRAC,
        stratify=labels,
        random_state=du.RANDOM_STATE,
    )

    # Step 2: split the remaining 85% into train and val.
    # val must be 15% of the whole, i.e. 0.15 / 0.85 of this remainder.
    val_relative = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        stratify=df.loc[train_val_idx, "label"].to_numpy(),
        random_state=du.RANDOM_STATE,
    )

    split = pd.Series(index=df.index, dtype="object")
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    df = df.copy()
    df["split"] = split
    return df


def plot_split_distribution(df: pd.DataFrame, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = (
        df.pivot_table(index="label", columns="split", values="filepath", aggfunc="count")
        .reindex(columns=["train", "val", "test"])
        .fillna(0)
        .sort_index()
    )

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(13, 6),
        color={"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"},
    )
    ax.set_title("Train / Val / Test image count per class (stratified, seed=42)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.legend(title="split")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Path to extracted dataset.")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    records = du.scan_images(args.data_dir)
    if not records:
        print(f"No images found under {args.data_dir!r}.", file=sys.stderr)
        return 1

    corrupt = load_corrupt_set()
    df = pd.DataFrame(
        [{"filepath": r.filepath, "label": r.label} for r in records]
    )
    before = len(df)
    if corrupt:
        df = df[~df["filepath"].isin(corrupt)].reset_index(drop=True)
        print(f"Excluded {before - len(df)} corrupt file(s) listed in docs/corrupt_files.txt")

    # Guard: stratification needs at least 2 samples per class per step.
    tiny = df["label"].value_counts()
    tiny = tiny[tiny < 3]
    if not tiny.empty:
        print(
            f"[ERROR] These classes have <3 images, cannot stratify: {dict(tiny)}",
            file=sys.stderr,
        )
        return 1

    df = stratified_split(df)

    # Order columns exactly as agreed and keep rows deterministic.
    df = df[du.FILE_LIST_COLUMNS].sort_values(["split", "label", "filepath"])
    csv_path = os.path.join(DOCS_DIR, "file_list.csv")
    df.to_csv(csv_path, index=False)

    # Split summary table.
    summary = (
        df.groupby(["label", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
    summary["total"] = summary.sum(axis=1)
    summary.loc["ALL"] = summary.sum(axis=0)
    summary.to_csv(os.path.join(TABLE_DIR, "split_summary.csv"))

    plot_split_distribution(df, os.path.join(FIG_DIR, "split_distribution.png"))

    # Console report.
    counts = df["split"].value_counts()
    total = len(df)
    print("\n=== Split summary ===")
    for name in ["train", "val", "test"]:
        n = int(counts.get(name, 0))
        print(f"  {name:5}: {n:6d}  ({n / total:.1%})")
    print(f"  total: {total:6d}")
    print(f"  classes: {df['label'].nunique()}")
    print(f"\nfile_list.csv -> {csv_path}")
    print(f"split figure  -> {os.path.join(FIG_DIR, 'split_distribution.png')}")
    print(f"split summary -> {os.path.join(TABLE_DIR, 'split_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
