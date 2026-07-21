"""Create fresh/rotten labels for existing ResNeXt-101 feature exports.

This migration script does not alter features, original 20-class labels, or
splits.  It writes new ``*_binary_labels.npy`` files alongside them, allowing
role 4 to use the agreed fresh/rotten task without re-running feature
extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "results" / "features"


def to_binary(label: str) -> str:
    value = str(label).lower().replace(" ", "").replace("_", "").replace("-", "")
    if value.startswith("fresh"):
        return "fresh"
    if value.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Cannot derive a fresh/rotten label from {label!r}")


def main() -> int:
    for stem in ("resnext101", "resnext101_aug"):
        source = FEATURES_DIR / f"{stem}_labels.npy"
        target = FEATURES_DIR / f"{stem}_binary_labels.npy"
        if not source.is_file():
            print(f"Skip missing {source}")
            continue
        labels = np.load(source, allow_pickle=True)
        binary = np.asarray([to_binary(label) for label in labels], dtype=str)
        np.save(target, binary)
        print(f"Wrote {target} ({len(binary)} labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
