"""Evaluate the ResNet-18 baseline on the validation split only.

Pipeline: frozen ResNet-18 features -> StandardScaler -> PCA(95%) -> RBF SVM.
The test split is deliberately not loaded.

Usage:
    python src/evaluation/analyze_resnet18_baseline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "features" / "resnet18_baseline"
FILE_LIST = ROOT / "docs" / "file_list.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
ERROR_DIR = FIGURE_DIR / "error_samples"
SEED = 42
CLASS_NAMES = ("fresh", "rotten")


def binary_label(label: str) -> str:
    value = str(label).lower().replace(" ", "").replace("_", "").replace("-", "")
    if value.startswith("fresh"):
        return "fresh"
    if value.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Unsupported freshness label: {label}")


def load_arrays(file_list: pd.DataFrame) -> dict[str, np.ndarray | pd.DataFrame]:
    arrays: dict[str, np.ndarray | pd.DataFrame] = {}
    for split in ("train", "val"):
        feature_path = FEATURE_DIR / f"resnet18_{split}_features.npy"
        label_path = FEATURE_DIR / f"resnet18_{split}_binary_labels.npy"
        if not feature_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing ResNet-18 {split} features or labels")
        features = np.load(feature_path, allow_pickle=False).astype(np.float32, copy=False)
        labels = np.load(label_path, allow_pickle=False).astype(str)
        frame = file_list[file_list["split"] == split].reset_index(drop=True)
        expected = frame["label"].map(binary_label).to_numpy(dtype=str)
        if features.ndim != 2 or len(features) != len(frame) or not np.array_equal(labels, expected):
            raise ValueError(f"ResNet-18 {split} features, labels, and file_list order do not match")
        arrays[f"{split}_features"] = features
        arrays[f"{split}_labels"] = labels
        arrays[f"{split}_frame"] = frame
    return arrays


def save_confusion(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(2),
        yticks=range(2),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="ResNet-18 baseline validation confusion matrix",
    )
    threshold = matrix.max() / 2.0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_error_sheet(errors: pd.DataFrame, path: Path) -> None:
    selected = errors.sort_values("confidence_margin", ascending=True).head(24)
    if selected.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No validation errors", ha="center", va="center", fontsize=18)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, record) in zip(axes, selected.iterrows()):
        ax.axis("off")
        image_path = Path(str(record["filepath"]))
        try:
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.contain(image, (300, 220))
            ax.imshow(image)
        except Exception as exc:  # pragma: no cover
            ax.text(0.5, 0.5, f"Image unavailable\n{type(exc).__name__}", ha="center", va="center")
        ax.set_title(
            f"true={record['true_binary']} → pred={record['pred_binary']}\n{image_path.name[:32]}",
            fontsize=8,
        )
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.suptitle("ResNet-18 baseline validation error samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.fillna("").itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, matrix: np.ndarray, errors: pd.DataFrame, path: Path) -> None:
    baseline = metrics.iloc[0]
    lines = [
        "# ResNet-18 Baseline Analysis",
        "",
        "## Baseline Definition",
        "",
        "The baseline is a frozen ImageNet-pretrained ResNet-18 feature extractor with no data augmentation,",
        "followed by StandardScaler, PCA retaining 95% of the variance, and an RBF SVM classifier.",
        "The official `docs/file_list.csv` split and seed 42 are used throughout.",
        "Only train and validation data are used in this stage; test features are not loaded.",
        "",
        "## Validation Result",
        "",
        dataframe_to_markdown(metrics),
        "",
        f"The baseline achieved Accuracy={baseline['accuracy']:.4f}, Macro-F1={baseline['macro_f1']:.4f}, "
        f"and Weighted-F1={baseline['weighted_f1']:.4f} on validation. PCA produced "
        f"{int(baseline['pca_components'])} components from the 512-dimensional ResNet-18 features.",
        "",
        "## Comparison with DenseNet-201",
        "",
    ]
    dense_metrics_path = TABLE_DIR / "densenet201_augmentation_metrics_val.csv"
    if dense_metrics_path.is_file():
        dense = pd.read_csv(dense_metrics_path)
        dense_svm = dense[(dense["case"] == "no_aug") & (dense["classifier"] == "svm")]
        if not dense_svm.empty:
            dense_row = dense_svm.iloc[0]
            lines.append(
                f"Compared with the existing DenseNet-201 no_aug + SVM result, the baseline Macro-F1 changes by "
                f"{baseline['macro_f1'] - dense_row['macro_f1']:+.4f} and Accuracy changes by "
                f"{baseline['accuracy'] - dense_row['accuracy']:+.4f}."
            )
    lines += [
        "",
        "## Confusion Matrix and Errors",
        "",
        f"The validation confusion matrix is `[[{matrix[0, 0]}, {matrix[0, 1]}], [{matrix[1, 0]}, {matrix[1, 1]}]]` "
        f"using row order `fresh, rotten`. The baseline made {len(errors)} validation errors out of {int(baseline['n_val'])} samples.",
        "The complete error list is stored in `results/tables/resnet18_baseline_val_errors.csv`, and the visual error sheet is stored in `results/figures/error_samples/resnet18_baseline_error_samples_val.png`.",
        "",
        "## Reproducibility and Limitations",
        "",
        "The feature extractor is frozen and uses the default ImageNet weights. StandardScaler and PCA are fit on train only.",
        "The reported result is a validation baseline, not a final test score. No hyperparameter search or test-set model selection was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(FILE_LIST)
    arrays = load_arrays(file_list)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(arrays["train_features"])
    val_scaled = scaler.transform(arrays["val_features"])
    pca = PCA(n_components=0.95, random_state=SEED)
    train_pca = pca.fit_transform(train_scaled)
    val_pca = pca.transform(val_scaled)

    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    started = time.perf_counter()
    model.fit(train_pca, arrays["train_labels"])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(val_pca).astype(str)
    scores = np.asarray(model.decision_function(val_pca), dtype=float).reshape(-1)
    predict_seconds = time.perf_counter() - started
    true = arrays["val_labels"]
    confidence = np.abs(scores)

    prediction_frame = arrays["val_frame"][["filepath", "label"]].copy()
    prediction_frame["true_binary"] = true
    prediction_frame["pred_binary"] = predictions
    prediction_frame["correct"] = true == predictions
    prediction_frame["decision_score"] = scores
    prediction_frame["confidence_margin"] = confidence
    errors = prediction_frame[~prediction_frame["correct"]].copy()
    matrix = confusion_matrix(true, predictions, labels=list(CLASS_NAMES))
    macro = precision_recall_fscore_support(
        true, predictions, labels=list(CLASS_NAMES), average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        true, predictions, labels=list(CLASS_NAMES), average="weighted", zero_division=0
    )[2]
    metrics = pd.DataFrame(
        [
            {
                "model": "resnet18",
                "case": "no_aug_baseline",
                "classifier": "svm",
                "pca_target": "0.95_variance",
                "pca_components": int(pca.n_components_),
                "accuracy": accuracy_score(true, predictions),
                "macro_precision": macro[0],
                "macro_recall": macro[1],
                "macro_f1": macro[2],
                "weighted_f1": weighted_f1,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": fit_seconds + predict_seconds,
                "n_train": len(arrays["train_labels"]),
                "n_val": len(arrays["val_labels"]),
                "seed": SEED,
            }
        ]
    )
    report = pd.DataFrame(
        classification_report(
            true,
            predictions,
            labels=list(CLASS_NAMES),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="label")

    prediction_frame.to_csv(TABLE_DIR / "resnet18_baseline_val_predictions.csv", index=False)
    errors.to_csv(TABLE_DIR / "resnet18_baseline_val_errors.csv", index=False)
    metrics.to_csv(TABLE_DIR / "resnet18_baseline_metrics_val.csv", index=False)
    report.to_csv(TABLE_DIR / "resnet18_baseline_val_classification_report.csv", index=False)
    save_confusion(matrix, FIGURE_DIR / "resnet18_baseline_confusion_val.png")
    save_error_sheet(errors, ERROR_DIR / "resnet18_baseline_error_samples_val.png")
    write_report(metrics, matrix, errors, ROOT / "docs" / "RESNET18_BASELINE_ANALYSIS.md")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
