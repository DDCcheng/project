"""Optional helper -- download the Kaggle dataset with kagglehub.

You do NOT need this if you already have the dataset extracted somewhere; in
that case just pass that folder as --data-dir to the other scripts.

kagglehub caches the download under ~/.cache/kagglehub and returns the local
path. It needs Kaggle credentials: either a ~/.kaggle/kaggle.json file or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables.

Usage:
  python src/preprocessing/download_data.py
  # -> prints the local path; feed it to check_images.py / make_split.py
"""

from __future__ import annotations

import sys

DATASET = "muhriddinmuxiddinov/fruits-and-vegetables-dataset"


def main() -> int:
    try:
        import kagglehub
    except ImportError:
        print(
            "kagglehub is not installed. Run: pip install kagglehub\n"
            "Or download the dataset manually from:\n"
            f"  https://www.kaggle.com/datasets/{DATASET}\n"
            "and pass the extracted folder via --data-dir.",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {DATASET} (cached after first run) ...")
    path = kagglehub.dataset_download(DATASET)
    print("\nDataset available at:")
    print(f"  {path}")
    print("\nNext step, e.g.:")
    print(f'  python src/preprocessing/check_images.py --data-dir "{path}"')
    print(f'  python src/preprocessing/make_split.py  --data-dir "{path}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
