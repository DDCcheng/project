"""Audit exact duplicate image files before the shared split is trusted.

The audit uses SHA-256 content hashes.  It never changes the dataset or split;
it writes a report so the team can see whether multiple paths contain the same
image bytes.

Usage:
  python src/preprocessing/check_duplicates.py --data-dir path/to/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_utils as du  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLE_DIR = os.path.join(REPO_ROOT, "results", "tables")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Path to extracted dataset.")
    args = parser.parse_args()

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    records = du.scan_images(args.data_dir)
    hashes: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, str]] = []
    for i, record in enumerate(records, 1):
        digest = sha256_file(record.filepath)
        hashes[digest].append(record.filepath)
        rows.append({"sha256": digest, "filepath": record.filepath, "label": record.label})
        if i % 1000 == 0:
            print(f"  ...{i}/{len(records)} hashed")

    duplicates = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    duplicate_rows = [
        {"sha256": digest, "filepath": path, "label": next(r["label"] for r in rows if r["filepath"] == path)}
        for digest, paths in duplicates.items()
        for path in paths
    ]
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "image_hashes.csv"), index=False)
    pd.DataFrame(duplicate_rows).to_csv(
        os.path.join(TABLE_DIR, "duplicate_hashes.csv"), index=False
    )

    report_path = os.path.join(DOCS_DIR, "duplicate_files.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        for digest, paths in sorted(duplicates.items()):
            handle.write(f"sha256={digest}\n")
            handle.write("\n".join(paths))
            handle.write("\n\n")

    duplicate_file_count = sum(len(paths) for paths in duplicates.values())
    print("\n=== Duplicate audit summary ===")
    print(f"  images hashed       : {len(records)}")
    print(f"  unique hashes       : {len(hashes)}")
    print(f"  duplicate groups    : {len(duplicates)}")
    print(f"  duplicate file paths: {duplicate_file_count}")
    print(f"  report              : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
