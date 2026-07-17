"""Shared helpers for the preprocessing stage (Person 1).

Everything the team relies on for reading the raw dataset lives here so that
the "single source of truth" (docs/file_list.csv) is produced from one place.

Reference pipeline: Yuan & Chen (2024) -- pretrained CNN feature extraction ->
PCA -> SVM/LDA/Bagging. This module only covers reading/scanning the images and
the constants that describe how they must be pre-processed downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Global constants (kept in sync with docs/EXPERIMENT_RULES.md)
# --------------------------------------------------------------------------

# The single global random seed. Every random step in the whole project
# (data split, classifier init, cross-validation, ...) must use this value.
RANDOM_STATE = 42

# Input size fed to the pretrained backbones (DenseNet-201 / ResNeXt-101).
IMAGE_SIZE = (224, 224)

# ImageNet normalisation statistics (models are pretrained on ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# File extensions we treat as images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Column order for docs/file_list.csv -- do NOT change without telling 2/3/4.
FILE_LIST_COLUMNS = ["filepath", "label", "split"]


@dataclass
class ImageRecord:
    """One image on disk together with the class label inferred from its folder."""

    filepath: str  # absolute, normalised path
    label: str  # class name (immediate parent folder name)


def _normalise_label(name: str) -> str:
    """Normalise a class-folder name to a stable, lower-case label.

    Different subsets of the Kaggle dataset sometimes spell folders as
    ``freshApples`` / ``fresh_apples`` / ``Fresh Apples``.  We collapse those to
    a single canonical form so the 20 classes do not get double-counted.
    """
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def scan_images(data_dir: str) -> list[ImageRecord]:
    """Recursively scan ``data_dir`` and return one record per image file.

    The class label is taken from the *immediate parent folder* of each image,
    which matches the layout of the Kaggle
    ``muhriddinmuxiddinov/fruits-and-vegetables-dataset`` (one folder per class,
    optionally nested under a train/test folder we intentionally ignore because
    Person 1 re-splits everything from scratch).

    Files are returned sorted by path so the scan is deterministic across runs
    and machines -- this is what makes the resulting split reproducible.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir!r}. "
            "Point --data-dir at the extracted Kaggle dataset "
            "(see src/preprocessing/download_data.py)."
        )

    records: list[ImageRecord] = []
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            parent = os.path.basename(os.path.normpath(root))
            label = _normalise_label(parent)
            filepath = os.path.abspath(os.path.join(root, fname))
            records.append(ImageRecord(filepath=filepath, label=label))

    records.sort(key=lambda r: r.filepath)
    return records


def summarise_labels(records: list[ImageRecord]) -> dict[str, int]:
    """Return an ordered {label: count} dict (sorted by label name)."""
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.label] = counts.get(rec.label, 0) + 1
    return dict(sorted(counts.items()))
