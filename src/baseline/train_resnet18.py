"""
ResNet-18 baseline training and validation.

Task:
    Fresh / rotten binary classification.

Cases:
    no_aug:
        No data augmentation.
    aug:
        Mild data augmentation is applied only to training images.

Unified experiment rules:
    - Use docs/file_list.csv.
    - Keep the existing 70/15/15 stratified split.
    - Random seed = 42.
    - RGB images.
    - Image size = 224 x 224.
    - ImageNet mean and standard deviation.
    - Validation set selects the best epoch.
    - Test set is not accessed by this script.

Example:
    python src/baseline/train_resnet18.py --case no_aug
    python src/baseline/train_resnet18.py --case aug
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_LIST_PATH = PROJECT_ROOT / "docs" / "file_list.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "results" / "models"
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "tables"
FIGURE_DIRECTORY = PROJECT_ROOT / "results" / "figures"


# ============================================================
# Unified settings
# ============================================================

RANDOM_SEED = 42
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["fresh", "rotten"]
CLASS_TO_INDEX = {
    "fresh": 0,
    "rotten": 1,
}

# These parameters must stay identical for no_aug and aug.
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def set_random_seed(seed: int) -> None:
    """Set all available random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic seed."""

    worker_seed = RANDOM_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================
# Label conversion
# ============================================================

def convert_to_binary_label(original_label: str) -> str:
    """
    Convert an original fruit/vegetable label into fresh or rotten.

    Examples:
        freshapple -> fresh
        rottenbanana -> rotten
    """

    normalised_label = (
        str(original_label)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    if normalised_label.startswith("fresh"):
        return "fresh"

    if normalised_label.startswith("rotten"):
        return "rotten"

    raise ValueError(
        f"Cannot convert label to fresh/rotten: {original_label!r}"
    )


# ============================================================
# Transforms
# ============================================================

def create_transform(case: str, split: str) -> transforms.Compose:
    """
    Create preprocessing for one split.

    Augmentation is applied only when:
        case == 'aug' and split == 'train'
    """

    if case == "aug" and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    IMAGE_SIZE,
                    scale=(0.85, 1.0),
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5,
                ),
                transforms.RandomRotation(
                    degrees=10,
                ),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ============================================================
# Dataset
# ============================================================

class FreshnessDataset(Dataset):
    """Dataset backed by the existing docs/file_list.csv."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        case: str,
        split: str,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.case = case
        self.split = split
        self.transform = create_transform(
            case=case,
            split=split,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, str]:
        row = self.dataframe.iloc[index]

        image_path = Path(str(row["filepath"]))

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image was not found: {image_path}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                # Make augmentation deterministic per image and epoch run.
                if self.case == "aug" and self.split == "train":
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(
                            RANDOM_SEED + index
                        )
                        image_tensor = self.transform(image)
                else:
                    image_tensor = self.transform(image)

        except UnidentifiedImageError as error:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            ) from error

        binary_label = convert_to_binary_label(
            row["label"]
        )

        label_index = CLASS_TO_INDEX[binary_label]

        return (
            image_tensor,
            label_index,
            str(image_path),
        )


# ============================================================
# Data loading
# ============================================================

def load_file_list() -> pd.DataFrame:
    """Load and validate the shared data split."""

    if not FILE_LIST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing shared split file: {FILE_LIST_PATH}"
        )

    dataframe = pd.read_csv(FILE_LIST_PATH)

    required_columns = {
        "filepath",
        "label",
        "split",
    }

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "file_list.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["split"] = (
        dataframe["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unexpected_splits = (
        set(dataframe["split"])
        - {"train", "val", "test"}
    )

    if unexpected_splits:
        raise ValueError(
            "Unexpected split labels: "
            f"{sorted(unexpected_splits)}"
        )

    dataframe["binary_label"] = (
        dataframe["label"]
        .map(convert_to_binary_label)
    )

    duplicate_paths = dataframe[
        "filepath"
    ].duplicated().sum()

    if duplicate_paths:
        raise ValueError(
            f"Found {duplicate_paths} duplicated image paths."
        )

    missing_paths = [
        path
        for path in dataframe["filepath"]
        if not Path(str(path)).is_file()
    ]

    if missing_paths:
        examples = missing_paths[:3]

        raise FileNotFoundError(
            f"{len(missing_paths)} image paths do not exist. "
            f"Examples: {examples}. "
            "Regenerate docs/file_list.csv on this computer "
            "using the agreed split procedure; do not create "
            "a new random split."
        )

    split_counts = (
        dataframe["split"]
        .value_counts()
        .to_dict()
    )

    print("Shared split counts:")
    print(split_counts)

    print("\nBinary class counts by split:")
    print(
        pd.crosstab(
            dataframe["split"],
            dataframe["binary_label"],
        )
    )

    return dataframe


def create_data_loaders(
    dataframe: pd.DataFrame,
    case: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    """Create train and validation loaders only."""

    loaders: dict[str, DataLoader] = {}

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    for split in ["train", "val"]:
        split_frame = dataframe[
            dataframe["split"] == split
        ].copy()

        dataset = FreshnessDataset(
            dataframe=split_frame,
            case=case,
            split=split,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=seed_worker,
            generator=generator,
        )

    return loaders


# ============================================================
# Device and model
# ============================================================

def select_device(requested_device: str) -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        return torch.device("cuda")

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable."
            )
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_resnet18() -> nn.Module:
    """Create an ImageNet-pretrained, end-to-end ResNet-18."""

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        len(CLASS_NAMES),
    )

    # End-to-end fine-tuning: all parameters remain trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
) -> dict[str, float]:
    """Calculate consistent binary and macro metrics."""

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


# ============================================================
# One epoch
# ============================================================

def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    description: str,
) -> tuple[float, dict[str, float]]:
    """
    Run one training or validation epoch.

    optimizer is None during validation.
    """

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    true_labels: list[int] = []
    predicted_labels: list[int] = []

    progress_bar = tqdm(
        data_loader,
        desc=description,
        leave=False,
    )

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:
        for images, labels, _paths in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            true_labels.extend(
                labels.detach().cpu().tolist()
            )

            predicted_labels.extend(
                predictions.detach().cpu().tolist()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    average_loss = total_loss / total_samples

    metrics = calculate_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    return average_loss, metrics


# ============================================================
# Plotting
# ============================================================

def plot_training_history(
    history: pd.DataFrame,
    case: str,
) -> None:
    """Save loss and Macro-F1 training curves."""

    output_path = (
        FIGURE_DIRECTORY
        / f"resnet18_{case}_training_curves.png"
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training Loss",
    )

    axis.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )

    axis.set_title(
        f"ResNet-18 {case}: Training and Validation Loss"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved figure: {output_path}")

    f1_output_path = (
        FIGURE_DIRECTORY
        / f"resnet18_{case}_macro_f1_curve.png"
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        history["epoch"],
        history["train_macro_f1"],
        marker="o",
        label="Training Macro-F1",
    )

    axis.plot(
        history["epoch"],
        history["validation_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )

    axis.set_title(
        f"ResNet-18 {case}: Training and Validation Macro-F1"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        f1_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved figure: {f1_output_path}")


# ============================================================
# Training
# ============================================================

def train_model(
    case: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    requested_device: str,
) -> None:
    """Train ResNet-18 and choose the best epoch by validation Macro-F1."""

    set_random_seed(RANDOM_SEED)

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = select_device(
        requested_device
    )

    print(f"Using device: {device}")
    print(f"Case: {case}")
    print("Task: fresh/rotten binary classification")
    print("The test set will not be accessed.")

    dataframe = load_file_list()

    loaders = create_data_loaders(
        dataframe=dataframe,
        case=case,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    model = create_resnet18().to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_macro_f1 = -1.0
    best_epoch = -1

    checkpoint_path = (
        MODEL_DIRECTORY
        / f"resnet18_{case}_best.pt"
    )

    history_rows: list[dict[str, float | int]] = []

    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        train_loss, train_metrics = run_epoch(
            model=model,
            data_loader=loaders["train"],
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            description=f"Train {epoch}",
        )

        validation_loss, validation_metrics = run_epoch(
            model=model,
            data_loader=loaders["val"],
            loss_function=loss_function,
            device=device,
            optimizer=None,
            description=f"Validation {epoch}",
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_precision": (
                train_metrics["macro_precision"]
            ),
            "train_macro_recall": (
                train_metrics["macro_recall"]
            ),
            "train_macro_f1": (
                train_metrics["macro_f1"]
            ),
            "train_weighted_f1": (
                train_metrics["weighted_f1"]
            ),
            "validation_loss": validation_loss,
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "validation_macro_precision": (
                validation_metrics["macro_precision"]
            ),
            "validation_macro_recall": (
                validation_metrics["macro_recall"]
            ),
            "validation_macro_f1": (
                validation_metrics["macro_f1"]
            ),
            "validation_weighted_f1": (
                validation_metrics["weighted_f1"]
            ),
        }

        history_rows.append(history_row)

        print(
            "Train:"
            f" loss={train_loss:.4f},"
            f" accuracy={train_metrics['accuracy']:.4f},"
            f" macro_f1={train_metrics['macro_f1']:.4f}"
        )

        print(
            "Validation:"
            f" loss={validation_loss:.4f},"
            f" accuracy={validation_metrics['accuracy']:.4f},"
            f" macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        # Validation Macro-F1 is the model-selection metric.
        if (
            validation_metrics["macro_f1"]
            > best_validation_macro_f1
        ):
            best_validation_macro_f1 = (
                validation_metrics["macro_f1"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_name": "ResNet-18",
                    "case": case,
                    "task_mode": (
                        "fresh/rotten binary classification"
                    ),
                    "random_seed": RANDOM_SEED,
                    "class_names": CLASS_NAMES,
                    "class_to_index": CLASS_TO_INDEX,
                    "epoch": epoch,
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "training_parameters": {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": (
                            learning_rate
                        ),
                        "weight_decay": (
                            weight_decay
                        ),
                    },
                },
                checkpoint_path,
            )

            print(
                "Saved new best checkpoint:"
                f" epoch={best_epoch},"
                f" validation_macro_f1="
                f"{best_validation_macro_f1:.6f}"
            )

    total_seconds = (
        time.perf_counter()
        - training_started
    )

    history = pd.DataFrame(
        history_rows
    )

    history_path = (
        TABLE_DIRECTORY
        / f"resnet18_{case}_history.csv"
    )

    history.to_csv(
        history_path,
        index=False,
    )

    print(f"\nSaved history: {history_path}")

    best_row = history.loc[
        history["epoch"] == best_epoch
    ].iloc[0]

    validation_result = {
        "model": "ResNet-18",
        "case": case,
        "task_mode": (
            "fresh/rotten binary classification"
        ),
        "split": "val",
        "selection_metric": (
            "validation_macro_f1"
        ),
        "best_epoch": int(best_epoch),
        "accuracy": float(
            best_row["validation_accuracy"]
        ),
        "macro_precision": float(
            best_row[
                "validation_macro_precision"
            ]
        ),
        "macro_recall": float(
            best_row[
                "validation_macro_recall"
            ]
        ),
        "macro_f1": float(
            best_row["validation_macro_f1"]
        ),
        "weighted_f1": float(
            best_row[
                "validation_weighted_f1"
            ]
        ),
        "training_seconds": float(
            total_seconds
        ),
        "random_seed": RANDOM_SEED,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "test_used": False,
    }

    validation_metrics_path = (
        TABLE_DIRECTORY
        / f"resnet18_{case}_validation_metrics.csv"
    )

    pd.DataFrame(
        [validation_result]
    ).to_csv(
        validation_metrics_path,
        index=False,
    )

    config_path = (
        TABLE_DIRECTORY
        / f"resnet18_{case}_config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation_result,
            file,
            indent=2,
        )

    plot_training_history(
        history=history,
        case=case,
    )

    print("\n=== ResNet-18 validation training complete ===")
    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation Macro-F1: "
        f"{best_validation_macro_f1:.6f}"
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(
        f"Saved validation metrics: "
        f"{validation_metrics_path}"
    )
    print("The test set was not used.")


# ============================================================
# Command line
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line parameters."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--case",
        choices=["no_aug", "aug"],
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    """Run ResNet-18 baseline training."""

    arguments = parse_arguments()

    train_model(
        case=arguments.case,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        requested_device=arguments.device,
    )


if __name__ == "__main__":
    main()