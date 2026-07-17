"""Stage 1.1 + 1.2 -- dataset integrity check and class distribution.

For every image in the dataset this script:
  1. Tries to open/decode it with Pillow (and, if installed, OpenCV as a second
     opinion). Any file that fails is written to a "corrupt files" report.
  2. Records its format and (width, height) so we can see the size distribution
     and decide whether a global resize is enough.
  3. Counts images per class and saves a class-distribution bar chart.

Outputs (all under results/ and docs/):
  results/figures/class_distribution.png
  results/tables/image_stats.csv         (per-image: path, label, format, w, h, ok)
  results/tables/format_size_summary.csv (aggregate format + size summary)
  docs/corrupt_files.txt                 (one bad file path per line; may be empty)

Usage:
  python src/preprocessing/check_images.py --data-dir path/to/dataset
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd
from PIL import Image

# Make "import dataset_utils" work whether run as a module or a plain script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_utils as du  # noqa: E402

# Pillow refuses truncated files by default with a warning; treat as an error.
warnings.simplefilter("error", Image.DecompressionBombWarning)

try:
    import cv2  # optional second-opinion decoder

    _HAS_CV2 = True
except Exception:  # pragma: no cover - opencv is optional
    _HAS_CV2 = False

# Repo root = two levels up from this file (src/preprocessing/ -> repo/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
TABLE_DIR = os.path.join(REPO_ROOT, "results", "tables")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


def inspect_image(path: str) -> dict:
    """Open one image and return its format/size, or mark it as corrupt.

    ``Image.verify()`` checks the file is not truncated; we then re-open to read
    the real size (verify() leaves the file object unusable). If OpenCV is
    available we also require it to decode the file, catching a class of files
    Pillow accepts but that break other libraries downstream.
    """
    info = {"format": None, "width": None, "height": None, "ok": False, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            info["format"] = im.format
            info["width"], info["height"] = im.size

        if _HAS_CV2:
            arr = cv2.imread(path)
            if arr is None:
                raise ValueError("cv2 failed to decode")

        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def plot_class_distribution(counts: dict[str, int], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: never try to open a window
    import matplotlib.pyplot as plt

    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_title(f"Image count per class (20 classes, total={sum(values)})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=90)
    ax.bar_label(bars, padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the extracted Kaggle dataset (folders named by class).",
    )
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    records = du.scan_images(args.data_dir)
    if not records:
        print(f"No images found under {args.data_dir!r}.", file=sys.stderr)
        return 1
    print(f"Scanning {len(records)} images from {args.data_dir} ...")

    rows = []
    corrupt = []
    for i, rec in enumerate(records, 1):
        info = inspect_image(rec.filepath)
        rows.append(
            {
                "filepath": rec.filepath,
                "label": rec.label,
                "format": info["format"],
                "width": info["width"],
                "height": info["height"],
                "ok": info["ok"],
                "error": info["error"],
            }
        )
        if not info["ok"]:
            corrupt.append(f"{rec.filepath}\t{info['error']}")
        if i % 1000 == 0:
            print(f"  ...{i}/{len(records)} checked")

    df = pd.DataFrame(rows)

    # Per-image stats table (full detail for auditing).
    stats_path = os.path.join(TABLE_DIR, "image_stats.csv")
    df.to_csv(stats_path, index=False)

    # Aggregate format + size summary (quick "do we need resizing?" answer).
    good = df[df["ok"]]
    summary = {
        "total_images": len(df),
        "readable_images": int(good.shape[0]),
        "corrupt_images": int((~df["ok"]).sum()),
        "num_classes": df["label"].nunique(),
        "formats": ", ".join(f"{k}:{v}" for k, v in good["format"].value_counts().items()),
        "width_min": int(good["width"].min()) if len(good) else None,
        "width_max": int(good["width"].max()) if len(good) else None,
        "height_min": int(good["height"].min()) if len(good) else None,
        "height_max": int(good["height"].max()) if len(good) else None,
        "unique_sizes": int(good[["width", "height"]].drop_duplicates().shape[0]),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(TABLE_DIR, "format_size_summary.csv"), index=False
    )

    # Corrupt file report (single source of truth for files to drop later).
    corrupt_path = os.path.join(DOCS_DIR, "corrupt_files.txt")
    with open(corrupt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corrupt))

    # Class distribution figure.
    counts = du.summarise_labels([r for r in records])
    fig_path = os.path.join(FIG_DIR, "class_distribution.png")
    plot_class_distribution(counts, fig_path)

    # Console summary.
    print("\n=== Integrity check summary ===")
    for k, v in summary.items():
        print(f"  {k:18}: {v}")
    print(f"\nPer-image stats  -> {stats_path}")
    print(f"Corrupt file list -> {corrupt_path} ({len(corrupt)} bad files)")
    print(f"Class distribution figure -> {fig_path}")
    if df["label"].nunique() != 20:
        print(
            f"\n[WARN] Expected 20 classes but found {df['label'].nunique()}. "
            "Check that --data-dir points at the class folders."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
