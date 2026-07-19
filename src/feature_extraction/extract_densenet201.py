"""Extract frozen ImageNet DenseNet-201 features for fresh/rotten classification.

The input rows come from docs/file_list.csv.  The CSV's existing 20-class
stratified split is the single source of truth; labels saved by this script are
the binary freshness labels derived from those source class names.

Two deterministic cases are supported:
  no_aug: plain resize + ImageNet normalization for every split.
  aug:    one fixed, mild train-only augmentation per training image; val/test
          always use the plain evaluation transform.

Usage:
  python src/feature_extraction/extract_densenet201.py \
      --data-dir path/to/dataset --case no_aug
  python src/feature_extraction/extract_densenet201.py \
      --data-dir path/to/dataset --case aug
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_LIST = REPO_ROOT / "docs" / "file_list.csv"
SEED = 42
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEATURE_DIM = 1920


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freshness_label(source_label: str) -> str:
    normalized = source_label.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if normalized.startswith("fresh"):
        return "fresh"
    if normalized.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Unknown freshness class: {source_label!r}")


def build_transform(case: str, split: str) -> transforms.Compose:
    common = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if case == "aug" and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03
                ),
                *common,
            ]
        )
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), *common])


class ImageFeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, case: str, split: str):
        self.frame = frame.reset_index(drop=True)
        self.case = case
        self.split = split
        self.transform = build_transform(case, split)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        # Seed the transform per CSV row, so the augmented feature file is
        # deterministic and independent of DataLoader worker scheduling.
        if self.case == "aug" and self.split == "train":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED + index)
                image = Image.open(row.filepath).convert("RGB")
                tensor = self.transform(image)
        else:
            image = Image.open(row.filepath).convert("RGB")
            tensor = self.transform(image)
        return tensor, freshness_label(row.label), row.filepath


def make_model(device: torch.device) -> torch.nn.Module:
    weights = models.DenseNet201_Weights.DEFAULT
    backbone = models.densenet201(weights=weights)
    backbone.classifier = torch.nn.Identity()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval().to(device)
    return backbone


def extract_split(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    case: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    dataset = ImageFeatureDataset(frame, case, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_chunks: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(loader, desc=f"{case}/{split}"):
            outputs = model(images.to(device, non_blocking=True))
            outputs = outputs.reshape(outputs.shape[0], -1)
            if outputs.shape[1] != FEATURE_DIM:
                raise RuntimeError(f"Expected {FEATURE_DIM} features, got {tuple(outputs.shape)}")
            feature_chunks.append(outputs.cpu().numpy().astype(np.float32, copy=False))
            labels.extend(list(batch_labels))
            paths.extend(list(batch_paths))
    elapsed = time.perf_counter() - started
    features = np.concatenate(feature_chunks, axis=0)
    return features, np.asarray(labels, dtype=str), paths, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset path for metadata validation.")
    parser.add_argument("--case", choices=["no_aug", "aug"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit-per-split", type=int, default=None, help="Debug smoke-test limit.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)
    if not FILE_LIST.is_file():
        raise FileNotFoundError(f"Missing {FILE_LIST}; run make_split.py first.")

    set_seed()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    output_dir = REPO_ROOT / "results" / "features" / args.case
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = pd.read_csv(FILE_LIST)
    required = {"filepath", "label", "split"}
    if not required.issubset(file_list.columns):
        raise ValueError(f"file_list.csv must contain {sorted(required)}")
    if file_list["filepath"].duplicated().any():
        raise ValueError("file_list.csv contains duplicate file paths")
    if not file_list["filepath"].map(os.path.isfile).all():
        missing = file_list.loc[~file_list["filepath"].map(os.path.isfile), "filepath"].iloc[0]
        raise FileNotFoundError(f"Missing image referenced by file_list.csv: {missing}")

    model = make_model(device)
    summary: dict[str, object] = {
        "model": "DenseNet-201",
        "weights": "DenseNet201_Weights.DEFAULT",
        "case": args.case,
        "task_mode": "fresh/rotten binary classification",
        "seed": SEED,
        "device": str(device),
        "feature_dim": FEATURE_DIM,
        "image_size": [224, 224],
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "augmentation": (
            "train-only deterministic RandomResizedCrop(scale=(0.85,1.0)), "
            "HorizontalFlip(0.5), Rotation(10), ColorJitter(0.15,0.15,0.10,0.03)"
            if args.case == "aug"
            else "none"
        ),
        "splits": {},
    }
    splits_to_extract = ["train", "val", "test"]
    reused_eval_splits: set[str] = set()
    no_aug_dir = REPO_ROOT / "results" / "features" / "no_aug"
    if args.case == "aug" and args.limit_per_split is None:
        # Validation and test never use augmentation, so reuse the already
        # extracted no_aug arrays exactly instead of recomputing them.
        for split in ["val", "test"]:
            source_features = no_aug_dir / f"densenet201_{split}_features.npy"
            source_labels = no_aug_dir / f"densenet201_{split}_labels.npy"
            if source_features.is_file() and source_labels.is_file():
                np.save(output_dir / source_features.name, np.load(source_features, allow_pickle=False))
                np.save(output_dir / source_labels.name, np.load(source_labels, allow_pickle=True))
                reused_eval_splits.add(split)
        splits_to_extract = ["train"] + [s for s in ["val", "test"] if s not in reused_eval_splits]

    for split in splits_to_extract:
        frame = file_list[file_list["split"] == split].reset_index(drop=True)
        if args.limit_per_split is not None:
            frame = frame.iloc[: args.limit_per_split].copy()
        features, labels, paths, elapsed = extract_split(
            model, frame, args.case, split, device, args.batch_size, args.num_workers
        )
        expected_labels = frame["label"].map(freshness_label).to_numpy(dtype=str)
        if not np.array_equal(labels, expected_labels):
            raise RuntimeError(f"Label order mismatch during {split} extraction")
        if paths != frame["filepath"].tolist():
            raise RuntimeError(f"Path order mismatch during {split} extraction")
        np.save(output_dir / f"densenet201_{split}_features.npy", features)
        np.save(output_dir / f"densenet201_{split}_labels.npy", labels)
        summary["splits"][split] = {
            "rows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "seconds": round(elapsed, 3),
            "fresh": int((labels == "fresh").sum()),
            "rotten": int((labels == "rotten").sum()),
        }

    for split in sorted(reused_eval_splits):
        features = np.load(output_dir / f"densenet201_{split}_features.npy", allow_pickle=False)
        labels = np.load(output_dir / f"densenet201_{split}_labels.npy", allow_pickle=True)
        summary["splits"][split] = {
            "rows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "seconds": 0.0,
            "fresh": int((labels == "fresh").sum()),
            "rotten": int((labels == "rotten").sum()),
            "reused_from": "no_aug",
        }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
