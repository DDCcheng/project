"""Create the report Results figure for PCA(95%) classifier comparisons.

Panel (a) compares DenseNet-201, ResNeXt-101, and feature fusion under
training augmentation.  Panel (b) compares SVM, LDA, and Bagging for
no-augmentation DenseNet-201 features.  The frozen ResNet-18 no-augmentation
SVM baseline is shown only as a reference line in panel (b), because it does
not satisfy panel (a)'s augmentation condition.

All values are validation-only results.  The test set is not accessed.

Inputs:
    results/tables/stage4_validation_ranked.csv
    results/tables/densenet201_augmentation_metrics_val.csv
    results/tables/resnet18_baseline_metrics_val.csv

Output:
    results/figures/report_results_pca95_classifier_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

STAGE4_PATH = TABLE_DIR / "stage4_validation_ranked.csv"
DENSENET_PATH = TABLE_DIR / "densenet201_augmentation_metrics_val.csv"
RESNET18_PATH = TABLE_DIR / "resnet18_baseline_metrics_val.csv"

OUTPUT_PATH = (
    FIGURE_DIR / "report_results_pca95_classifier_comparison.png"
)

MODELS = ["DenseNet-201", "ResNeXt-101", "Feature fusion"]
MODEL_CASES = {
    "DenseNet-201": "B",
    "ResNeXt-101": "D",
    "Feature fusion": "F",
}
CLASSIFIERS = ["svm", "lda", "bagging"]
CLASSIFIER_LABELS = {
    "svm": "SVM",
    "lda": "LDA",
    "bagging": "Bagging",
}
COLORS = {
    "svm": "#2F6B9A",
    "lda": "#E1812C",
    "bagging": "#3A923A",
}


def select_one(
    frame: pd.DataFrame, mask: pd.Series, description: str
) -> pd.Series:
    """Return exactly one requested validation result row."""

    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one {description} row, found {len(rows)}."
        )
    return rows.iloc[0]


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load the augmented grid, no-augmentation DenseNet, and baseline."""

    stage4 = pd.read_csv(STAGE4_PATH)
    densenet = pd.read_csv(DENSENET_PATH)
    resnet18 = pd.read_csv(RESNET18_PATH)

    for frame, path in [
        (stage4, STAGE4_PATH),
        (densenet, DENSENET_PATH),
        (resnet18, RESNET18_PATH),
    ]:
        if "macro_f1" not in frame.columns:
            raise ValueError(f"{path} is missing the macro_f1 column.")

    augmented_rows = []
    for model in MODELS:
        case = MODEL_CASES[model]
        for classifier in CLASSIFIERS:
            row = select_one(
                stage4,
                (stage4["case"].astype(str).str.upper() == case)
                & (
                    stage4["feature_case"].astype(str).str.lower()
                    == "aug"
                )
                & (
                    stage4["classifier"].astype(str).str.lower()
                    == classifier
                )
                & (stage4["split"].astype(str).str.lower() == "val")
                & stage4["pca_feature_dim"].notna(),
                f"{model}/augmentation/PCA95%/{classifier} validation",
            )
            augmented_rows.append(
                {
                    "model": model,
                    "classifier": classifier,
                    "macro_f1": float(row["macro_f1"]),
                }
            )

    no_aug_rows = []
    for classifier in CLASSIFIERS:
        row = select_one(
            densenet,
            (densenet["case"].astype(str).str.lower() == "no_aug")
            & (
                densenet["classifier"].astype(str).str.lower()
                == classifier
            )
            & (densenet["pca_target"].astype(str) == "0.95_variance"),
            f"DenseNet/no-augmentation/PCA95%/{classifier} validation",
        )
        no_aug_rows.append(
            {
                "model": "DenseNet-201",
                "classifier": classifier,
                "macro_f1": float(row["macro_f1"]),
            }
        )

    baseline_row = select_one(
        resnet18,
        (resnet18["case"].astype(str).str.lower() == "no_aug_baseline")
        & (resnet18["classifier"].astype(str).str.lower() == "svm")
        & (resnet18["pca_target"].astype(str) == "0.95_variance"),
        "frozen ResNet-18 no-augmentation/PCA95%/SVM baseline",
    )

    return (
        pd.DataFrame(augmented_rows),
        pd.DataFrame(no_aug_rows),
        float(baseline_row["macro_f1"]),
    )


def annotate_bars(axis: plt.Axes, bars: object) -> None:
    """Add four-decimal values above bars."""

    axis.bar_label(
        bars,
        labels=[f"{bar.get_height():.4f}" for bar in bars],
        padding=3,
        fontsize=8,
        rotation=90,
    )


def create_figure(
    augmented: pd.DataFrame,
    no_aug: pd.DataFrame,
    baseline_macro_f1: float,
) -> None:
    """Render the requested two-panel grouped bar chart."""

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": 0.14},
    )

    bar_width = 0.24
    left_x = np.arange(len(MODELS))
    for index, classifier in enumerate(CLASSIFIERS):
        values = []
        for model in MODELS:
            rows = augmented[
                (augmented["model"] == model)
                & (augmented["classifier"] == classifier)
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"Expected one augmented result for {model}/{classifier}."
                )
            values.append(float(rows.iloc[0]["macro_f1"]))

        offset = (index - 1) * bar_width
        bars = axes[0].bar(
            left_x + offset,
            values,
            bar_width,
            color=COLORS[classifier],
            label=CLASSIFIER_LABELS[classifier],
        )
        annotate_bars(axes[0], bars)

    right_x = np.arange(1)
    for index, classifier in enumerate(CLASSIFIERS):
        rows = no_aug[no_aug["classifier"] == classifier]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one no-augmentation DenseNet result for {classifier}."
            )
        value = float(rows.iloc[0]["macro_f1"])
        offset = (index - 1) * bar_width
        bars = axes[1].bar(
            right_x + offset,
            [value],
            bar_width,
            color=COLORS[classifier],
            label=CLASSIFIER_LABELS[classifier],
        )
        annotate_bars(axes[1], bars)

    axes[1].axhline(
        baseline_macro_f1,
        color="#666666",
        linestyle="--",
        linewidth=1.5,
    )
    axes[1].text(
        0.98,
        baseline_macro_f1 + 0.0012,
        (
            "Frozen ResNet-18 baseline: "
            f"{baseline_macro_f1:.4f}\n"
            "(no aug, PCA95%, SVM)"
        ),
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    axes[0].set_title("(a) Training Augmentation + PCA(95%)")
    axes[1].set_title("(b) No Augmentation + PCA(95%)")

    axes[0].set_xticks(left_x)
    axes[0].set_xticklabels(MODELS)
    axes[1].set_xticks(right_x)
    axes[1].set_xticklabels(["DenseNet-201"])

    axes[0].set_ylabel("Validation Macro-F1")
    for axis in axes:
        axis.set_xlabel("Feature model")
        axis.set_ylim(0.88, 1.0)
        axis.set_yticks(np.arange(0.88, 1.001, 0.02))
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].legend(title="Classifier", loc="lower left")

    figure.suptitle(
        "Validation Macro-F1 by Classifier with PCA(95%)",
        fontsize=15,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    augmented, no_aug, baseline_macro_f1 = load_results()
    create_figure(augmented, no_aug, baseline_macro_f1)

    print(f"Saved figure: {OUTPUT_PATH}")
    print("\nAugmentation + PCA95% results:")
    print(augmented.to_string(index=False))
    print("\nNo-augmentation + PCA95% DenseNet results:")
    print(no_aug.to_string(index=False))
    print(f"\nFrozen ResNet-18 baseline: {baseline_macro_f1:.6f}")
    print("Test set used: False")


if __name__ == "__main__":
    main()
