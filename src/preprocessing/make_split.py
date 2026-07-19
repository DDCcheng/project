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
import numpy as np
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


def load_duplicate_audit() -> tuple[dict[str, str], set[str]]:
    """Load content hashes and identify conflicting-label duplicate groups."""
    path = os.path.join(TABLE_DIR, "image_hashes.csv")
    if not os.path.isfile(path):
        return {}, set()
    hashes = pd.read_csv(path)
    required = {"sha256", "filepath", "label"}
    if not required.issubset(hashes.columns):
        return {}, set()
    path_to_hash = dict(zip(hashes["filepath"], hashes["sha256"]))
    conflicting = set(
        hashes.groupby("sha256")["label"].nunique().loc[lambda s: s > 1].index
    )
    return path_to_hash, conflicting


def grouped_stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split by source class while keeping exact duplicate groups together.

    Most groups contain one file, so this preserves the requested 70/15/15
    ratios closely.  Duplicate groups are assigned atomically, which prevents
    identical image bytes from crossing a split boundary.
    """
    rng = np.random.default_rng(du.RANDOM_STATE)
    assignments: dict[str, str] = {}
    fractions = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}

    for label, class_df in df.groupby("label", sort=True):
        groups = (
            class_df.groupby("group_key", as_index=False)
            .size()
            .rename(columns={"size": "group_size"})
        )
        groups["tie_break"] = rng.random(len(groups))
        groups = groups.sort_values(
            ["group_size", "tie_break", "group_key"],
            ascending=[False, True, True],
        )
        target = {name: fractions[name] * len(class_df) for name in fractions}
        assigned = {name: 0 for name in fractions}
        for row in groups.itertuples(index=False):
            remaining = {
                name: target[name] - assigned[name] for name in fractions
            }
            nonnegative = [name for name in fractions if remaining[name] >= 0]
            candidates = nonnegative or list(fractions)
            chosen = max(
                candidates,
                key=lambda name: (remaining[name] / fractions[name], fractions[name]),
            )
            assignments[row.group_key] = chosen
            assigned[chosen] += int(row.group_size)

    result = df.copy()
    result["split"] = result["group_key"].map(assignments)
    # Correct the small rounding drift from the greedy pass.  Move whole
    # duplicate groups only; singleton groups make the requested integer
    # counts attainable for this dataset while preserving each source class.
    grouped = (
        result.groupby(["label", "group_key", "split"], as_index=False)
        .size()
        .rename(columns={"size": "group_size"})
    )
    labels_sorted = sorted(result["label"].unique())
    global_targets = {
        "train": int(np.floor(len(result) * TRAIN_FRAC + 0.5)),
        "val": int(np.floor(len(result) * VAL_FRAC + 0.5)),
    }

    def allocate_targets(fraction: float, total_target: int) -> dict[str, int]:
        raw = {label: fraction * int((result["label"] == label).sum()) for label in labels_sorted}
        allocated = {label: int(np.floor(value)) for label, value in raw.items()}
        remainder = total_target - sum(allocated.values())
        order = sorted(
            labels_sorted,
            key=lambda label: (raw[label] - allocated[label], label),
            reverse=True,
        )
        for label in order[:remainder]:
            allocated[label] += 1
        return allocated

    train_targets = allocate_targets(TRAIN_FRAC, global_targets["train"])
    val_targets = allocate_targets(VAL_FRAC, global_targets["val"])
    target_by_label: dict[str, dict[str, int]] = {}
    for label in labels_sorted:
        total = int((result["label"] == label).sum())
        train_target = train_targets[label]
        val_target = min(val_targets[label], total - train_target)
        target_by_label[label] = {
            "train": train_target,
            "val": val_target,
            "test": total - train_target - val_target,
        }

    for label, target in target_by_label.items():
        counts = result.loc[result["label"] == label, "split"].value_counts().to_dict()
        for split in ["train", "val", "test"]:
            counts.setdefault(split, 0)
        for _ in range(100):
            deficits = {s: target[s] - counts[s] for s in target}
            if all(value == 0 for value in deficits.values()):
                break
            destination = max(deficits, key=deficits.get)
            if deficits[destination] <= 0:
                break
            source = min(
                (s for s in target if deficits[s] < 0),
                key=lambda s: deficits[s],
            )
            candidates = grouped[
                (grouped["label"] == label) & (grouped["split"] == source)
            ].copy()
            if candidates.empty:
                break
            fits = candidates[candidates["group_size"] <= deficits[destination]]
            pool = fits if not fits.empty else candidates
            chosen = pool.sort_values(["group_size", "group_key"]).iloc[0]
            group_key = chosen["group_key"]
            group_size = int(chosen["group_size"])
            result.loc[
                (result["label"] == label) & (result["group_key"] == group_key),
                "split",
            ] = destination
            grouped.loc[
                (grouped["label"] == label) & (grouped["group_key"] == group_key),
                "split",
            ] = destination
            counts[source] -= group_size
            counts[destination] += group_size
        else:
            raise RuntimeError(f"Could not reach target grouped split counts for {label}")

    return result.drop(columns=["group_key"])


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``split`` column (train/val/test) with per-class stratification.

    Two-step split: first carve out test (15%), then split the remaining 85%
    into train and val so that val is 15% of the whole. Both steps are
    stratified on the label and seeded with RANDOM_STATE.
    """
    if "group_key" in df.columns:
        return grouped_stratified_split(df)

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
    path_to_hash, conflicting_hashes = load_duplicate_audit()
    df = pd.DataFrame(
        [{"filepath": r.filepath, "label": r.label} for r in records]
    )
    before = len(df)
    if corrupt:
        df = df[~df["filepath"].isin(corrupt)].reset_index(drop=True)
        print(f"Excluded {before - len(df)} corrupt file(s) listed in docs/corrupt_files.txt")

    if conflicting_hashes:
        conflict_paths = {
            path for path, digest in path_to_hash.items() if digest in conflicting_hashes
        }
        df = df[~df["filepath"].isin(conflict_paths)].reset_index(drop=True)
        print(
            f"Excluded {len(conflict_paths)} exact-duplicate file(s) with conflicting labels"
        )

    if path_to_hash:
        df["group_key"] = df["filepath"].map(path_to_hash).fillna(df["filepath"])

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
