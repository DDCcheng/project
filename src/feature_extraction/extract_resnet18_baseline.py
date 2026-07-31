"""Extract frozen ResNet-18 baseline features for train and validation only.

The baseline uses ImageNet-pretrained ResNet-18 with its final classifier
removed. It follows the official file_list.csv order, applies no augmentation,
and deliberately does not load the test split.

Usage:
    python src/feature_extraction/extract_resnet18_baseline.py --data-dir <dataset>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
FILE_LIST = ROOT / "docs" / "file_list.csv"
OUTPUT_DIR = ROOT / "results" / "features" / "resnet18_baseline"
SEED = 42
FEATURE_DIM = 512
SPLITS = ("train", "val")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def binary_label(label: str) -> str:
    value = str(label).lower().replace(" ", "").replace("_", "").replace("-", "")
    if value.startswith("fresh"):
        return "fresh"
    if value.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Unsupported freshness label: {label}")


class OfficialDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row.filepath).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(row.label), binary_label(row.label), str(row.filepath)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing the images.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)
    if not FILE_LIST.is_file():
        raise FileNotFoundError(FILE_LIST)
    set_seed()
    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    frame = pd.read_csv(FILE_LIST)
    if not {"filepath", "label", "split"}.issubset(frame.columns):
        raise ValueError("file_list.csv must contain filepath, label, split")
    for split in SPLITS:
        split_frame = frame[frame["split"] == split]
        if not split_frame["filepath"].map(os.path.isfile).all():
            raise FileNotFoundError(f"Missing image path in {split} file list")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "ResNet-18 baseline",
        "weights": "ResNet18_Weights.DEFAULT",
        "case": "no_aug",
        "task_mode": "fresh/rotten binary classification",
        "seed": SEED,
        "device": str(device),
        "feature_dim": FEATURE_DIM,
        "input": "RGB, 224x224, ImageNet mean/std, no augmentation",
        "source_of_split": "docs/file_list.csv",
        "splits": {},
        "test_loaded": False,
    }

    for split in SPLITS:
        split_frame = frame[frame["split"] == split].reset_index(drop=True)
        loader = DataLoader(
            OfficialDataset(split_frame),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        features: list[np.ndarray] = []
        source_labels: list[str] = []
        binary_labels: list[str] = []
        paths: list[str] = []
        started = time.perf_counter()
        with torch.no_grad():
            for images, source_batch, binary_batch, path_batch in tqdm(loader, desc=f"resnet18/{split}"):
                values = model(images.to(device, non_blocking=True)).reshape(images.shape[0], -1)
                if values.shape[1] != FEATURE_DIM:
                    raise RuntimeError(f"Expected {FEATURE_DIM}, got {tuple(values.shape)}")
                features.append(values.cpu().numpy().astype(np.float32, copy=False))
                source_labels.extend(source_batch)
                binary_labels.extend(binary_batch)
                paths.extend(path_batch)

        expected_paths = split_frame["filepath"].tolist()
        expected_labels = split_frame["label"].astype(str).tolist()
        if paths != expected_paths or source_labels != expected_labels:
            raise RuntimeError(f"Order mismatch in official {split} file_list")

        matrix = np.concatenate(features, axis=0)
        np.save(OUTPUT_DIR / f"resnet18_{split}_features.npy", matrix)
        np.save(OUTPUT_DIR / f"resnet18_{split}_labels.npy", np.asarray(source_labels, dtype=str))
        np.save(OUTPUT_DIR / f"resnet18_{split}_binary_labels.npy", np.asarray(binary_labels, dtype=str))
        metadata["splits"][split] = {
            "rows": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1]),
            "seconds": round(time.perf_counter() - started, 3),
            "fresh": int(binary_labels.count("fresh")),
            "rotten": int(binary_labels.count("rotten")),
        }

    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
