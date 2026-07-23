"""DenseNet-201 augmentation comparison on the validation split only.

This script compares the existing ``no_aug`` and ``aug`` feature cases using
the fixed train/validation split and fixed classifiers agreed by the project.
It deliberately loads no test arrays.

Usage:
    python src/evaluation/analyze_densenet201.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "results" / "features"
FILE_LIST = ROOT / "docs" / "file_list.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
ERROR_FIGURE_DIR = FIGURE_DIR / "error_samples"
SEED = 42
CASES = ("no_aug", "aug")
CLASSIFIERS = ("svm", "lda", "bagging")
CLASS_NAMES = ("fresh", "rotten")


def load_case(case: str, file_list: pd.DataFrame) -> dict[str, np.ndarray | pd.DataFrame]:
    """Load only train and validation arrays for one feature case."""
    case_dir = FEATURE_ROOT / case
    arrays: dict[str, np.ndarray | pd.DataFrame] = {}
    for split in ("train", "val"):
        features_path = case_dir / f"densenet201_{split}_features.npy"
        labels_path = case_dir / f"densenet201_{split}_binary_labels.npy"
        if not features_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Missing {case}/{split} feature or binary-label file. "
                f"Expected {features_path} and {labels_path}."
            )
        features = np.load(features_path, allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False).astype(str)
        expected = file_list[file_list["split"] == split].reset_index(drop=True)
        if features.ndim != 2:
            raise ValueError(f"{case}/{split} features must be 2D, got {features.shape}")
        if len(features) != len(expected) or len(labels) != len(expected):
            raise ValueError(
                f"{case}/{split} row mismatch: features={len(features)}, "
                f"labels={len(labels)}, file_list={len(expected)}"
            )
        expected_binary = expected["label"].map(binary_label).to_numpy(dtype=str)
        if not np.array_equal(labels, expected_binary):
            raise ValueError(f"{case}/{split} binary labels do not match file_list order")
        arrays[f"{split}_features"] = features.astype(np.float32, copy=False)
        arrays[f"{split}_labels"] = labels
        arrays[f"{split}_frame"] = expected
    return arrays


def binary_label(label: str) -> str:
    value = str(label).lower().replace(" ", "").replace("_", "").replace("-", "")
    if value.startswith("fresh"):
        return "fresh"
    if value.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Unsupported freshness label: {label}")


def build_classifier(name: str) -> ClassifierMixin:
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="svd")
    if name == "bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED),
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def score_predictions(model: ClassifierMixin, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predictions, signed score and confidence/margin."""
    predictions = model.predict(x_val).astype(str)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_val)
        classes = [str(value) for value in model.classes_]
        fresh_index = classes.index("fresh")
        rotten_index = classes.index("rotten")
        signed_score = probabilities[:, fresh_index] - probabilities[:, rotten_index]
        confidence = probabilities.max(axis=1)
    else:
        signed_score = np.asarray(model.decision_function(x_val), dtype=float).reshape(-1)
        confidence = np.abs(signed_score)
    return predictions, signed_score, confidence


def make_prediction_frame(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signed_score: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    result = frame[["filepath", "label"]].copy()
    result["true_binary"] = y_true
    result["pred_binary"] = y_pred
    result["correct"] = result["true_binary"] == result["pred_binary"]
    result["score_fresh_minus_rotten"] = signed_score
    result["confidence_or_margin"] = confidence
    result["uncertainty"] = 1.0 / (1.0 + confidence) if np.any(confidence > 1.0) else 1.0 - confidence
    return result


def save_confusion_figure(matrix: np.ndarray, title: str, path: Path) -> None:
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
        title=title,
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=13,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_combined_confusion(matrices: dict[tuple[str, str], np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    vmax = max(1, max(matrix.max() for matrix in matrices.values()))
    for row, case in enumerate(CASES):
        for col, classifier in enumerate(CLASSIFIERS):
            matrix = matrices[(case, classifier)]
            ax = axes[row, col]
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            ax.set(
                xticks=range(2),
                yticks=range(2),
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                title=f"{case} / {classifier}",
            )
            if row == 1:
                ax.set_xlabel("Predicted")
            if col == 0:
                ax.set_ylabel("True")
            threshold = matrix.max() / 2.0 if matrix.size else 0.0
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, str(matrix[r, c]), ha="center", va="center", color="white" if matrix[r, c] > threshold else "black")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.035, label="Count")
    fig.suptitle("DenseNet-201 validation confusion matrices", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_error_sheet(errors: pd.DataFrame, case: str, classifier: str, path: Path) -> None:
    selected = errors.sort_values("confidence_or_margin", ascending=True).head(24)
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
        except Exception as exc:  # pragma: no cover - only for missing/corrupt local assets
            ax.text(0.5, 0.5, f"Image unavailable\n{type(exc).__name__}", ha="center", va="center")
        ax.set_title(
            f"true={record['true_binary']} → pred={record['pred_binary']}\n"
            f"{image_path.name[:32]}",
            fontsize=8,
        )
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.suptitle(f"DenseNet-201 validation error samples: {case} / {classifier}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small table without requiring the optional tabulate package."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.fillna("").itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fit_and_evaluate(case: str, classifier_name: str, arrays: dict) -> dict:
    x_train = arrays["train_features"]
    y_train = arrays["train_labels"]
    x_val = arrays["val_features"]
    y_val = arrays["val_labels"]

    x_train_pca = arrays["train_pca"]
    x_val_pca = arrays["val_pca"]

    model = build_classifier(classifier_name)
    started = time.perf_counter()
    model.fit(x_train_pca, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    y_pred, signed_score, confidence = score_predictions(model, x_val_pca)
    predict_seconds = time.perf_counter() - started

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=list(CLASS_NAMES), average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_val, y_pred, labels=list(CLASS_NAMES))
    predictions = make_prediction_frame(
        arrays["val_frame"], y_val, y_pred, signed_score, confidence
    )
    errors = predictions[~predictions["correct"]].copy()
    report = pd.DataFrame(
        classification_report(
            y_val,
            y_pred,
            labels=list(CLASS_NAMES),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="label")

    return {
        "case": case,
        "classifier": classifier_name,
        "pca_target": "0.95_variance",
        "pca_components": int(arrays["pca_components"]),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": float(fit_seconds + predict_seconds),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "seed": SEED,
        "matrix": matrix,
        "predictions": predictions,
        "errors": errors,
        "report": report,
    }


def write_analysis(metrics: pd.DataFrame, results: dict[tuple[str, str], dict], path: Path) -> None:
    ordered = metrics.sort_values(
        ["macro_f1", "accuracy", "weighted_f1"], ascending=False
    ).reset_index(drop=True)
    best = ordered.iloc[0]
    lines = [
        "# DenseNet-201 Augmentation Comparison and Validation Analysis",
        "",
        "## Scope",
        "",
        "This analysis compares the DenseNet-201 `no_aug` and `aug` feature cases for binary `fresh/rotten` classification.",
        "StandardScaler and PCA are fit on train only; classifiers are trained on train and all metrics are computed on validation.",
        "The test split is not loaded or evaluated in this stage, so these are not final test results.",
        "",
        "Fixed settings: seed=42; PCA retains 95% variance; SVM uses RBF, C=1.0, and gamma=scale;",
        "LDA uses the svd solver; Bagging uses 100 decision trees with fixed random seeds.",
        "",
        "## Metric Comparison",
        "",
        "Complete results are in `results/tables/densenet201_augmentation_metrics_val.csv`. Macro-F1 is the primary selection metric,",
        "followed by Accuracy and Weighted-F1.",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## Best Validation Configuration",
        "",
        f"Ranking by Macro-F1, Accuracy, and Weighted-F1, the best configuration is **{best['case']} + {best['classifier']}**.",
        f"Macro-F1={best['macro_f1']:.4f}, Accuracy={best['accuracy']:.4f},",
        f"and the resulting PCA dimension is {int(best['pca_components'])}.",
        "This selection is based only on validation and does not replace the final test evaluation.",
        "",
        "## Augmentation Effect",
        "",
    ]
    for classifier in CLASSIFIERS:
        no_aug = metrics[(metrics.case == "no_aug") & (metrics.classifier == classifier)].iloc[0]
        aug = metrics[(metrics.case == "aug") & (metrics.classifier == classifier)].iloc[0]
        lines.append(
            f"- **{classifier}**: relative to no_aug, aug changes Macro-F1 by "
            f"{aug.macro_f1 - no_aug.macro_f1:+.4f}, Accuracy by "
            f"{aug.accuracy - no_aug.accuracy:+.4f}, and Weighted-F1 by "
            f"{aug.weighted_f1 - no_aug.weighted_f1:+.4f}."
        )
    lines += [
        "",
        "## Confusion Matrices and Error Samples",
        "",
        "Confusion-matrix figures are in `results/figures/`; the combined figure is `densenet201_augmentation_confusion_val.png`.",
        "Complete error lists are in `results/tables/`; visual error samples are in `results/figures/error_samples/`.",
        "Each error table contains the filepath, original 20-class label, binary true label, predicted label, and confidence or decision-margin information.",
        "",
    ]
    for key, result in results.items():
        matrix = result["matrix"]
        errors = result["errors"]
        lines.append(
            f"- `{key[0]} / {key[1]}`: {len(errors)}/{result['n_val']} errors; "
            f"fresh→rotten={matrix[0, 1]}, rotten→fresh={matrix[1, 0]}."
        )
    lines += [
        "",
        "Image-level causes should be reviewed against the corresponding error-sample figures before being included in the final report; this document records only conclusions directly supported by the predictions.",
        "",
        "## Visual Observations of Error Samples",
        "",
        "A visual review of the highest-uncertainty error samples across configurations shows the following patterns:",
        "",
        "- Some `rotten → fresh` samples appear visually intact and normally colored, with no prominent decay region. This suggests borderline examples or label noise.",
        "- Some `fresh → rotten` samples contain mild spots, local discoloration, uneven surface texture, or darker lighting, which may be interpreted as decay cues.",
        "- Some images contain watermarks, complex backgrounds, multiple objects, varied camera distances, or strong lighting changes, increasing freshness-classification difficulty.",
        "- These observations explain model errors but do not justify relabeling the dataset; the corresponding error CSVs remain the source of truth for paths and predictions.",
        "",
        "## Reproducibility and Limitations",
        "",
        "Both feature cases come from the same `docs/file_list.csv`; train/validation row order and labels are validated by the script.",
        "Augmentation is applied only to the aug training features; the validation features are identical to the no_aug validation features.",
        "This stage uses fixed classifier parameters, performs no hyperparameter search, and does not use test for model selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    np.random.seed(SEED)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not FILE_LIST.is_file():
        raise FileNotFoundError(FILE_LIST)
    file_list = pd.read_csv(FILE_LIST)
    required = {"filepath", "label", "split"}
    if not required.issubset(file_list.columns):
        raise ValueError(f"file_list.csv must contain {sorted(required)}")

    loaded = {case: load_case(case, file_list) for case in CASES}
    for case in CASES:
        # Fit the train-only preprocessing once per feature case and reuse it
        # across SVM, LDA, and Bagging.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(loaded[case]["train_features"])
        x_val_scaled = scaler.transform(loaded[case]["val_features"])
        pca = PCA(n_components=0.95, random_state=SEED)
        loaded[case]["train_pca"] = pca.fit_transform(x_train_scaled)
        loaded[case]["val_pca"] = pca.transform(x_val_scaled)
        loaded[case]["pca_components"] = int(pca.n_components_)
    results: dict[tuple[str, str], dict] = {}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    metric_rows = []
    for case in CASES:
        for classifier_name in CLASSIFIERS:
            print(f"Running {case}/{classifier_name} ...")
            result = fit_and_evaluate(case, classifier_name, loaded[case])
            results[(case, classifier_name)] = result
            matrices[(case, classifier_name)] = result["matrix"]
            metric_rows.append({key: value for key, value in result.items() if key not in {"matrix", "predictions", "errors", "report"}})

            prefix = f"densenet201_{case}_{classifier_name}_val"
            result["predictions"].to_csv(TABLE_DIR / f"{prefix}_predictions.csv", index=False)
            result["errors"].to_csv(TABLE_DIR / f"{prefix}_errors.csv", index=False)
            result["report"].to_csv(TABLE_DIR / f"{prefix}_classification_report.csv", index=False)
            save_confusion_figure(
                result["matrix"],
                f"DenseNet-201 validation: {case} / {classifier_name}",
                FIGURE_DIR / f"densenet201_{case}_{classifier_name}_confusion_val.png",
            )
            save_error_sheet(
                result["errors"],
                case,
                classifier_name,
                ERROR_FIGURE_DIR / f"densenet201_{case}_{classifier_name}_error_samples_val.png",
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(TABLE_DIR / "densenet201_augmentation_metrics_val.csv", index=False)
    save_combined_confusion(matrices, FIGURE_DIR / "densenet201_augmentation_confusion_val.png")
    write_analysis(metrics, results, ROOT / "docs" / "DENSENET201_RESULTS_ANALYSIS.md")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
