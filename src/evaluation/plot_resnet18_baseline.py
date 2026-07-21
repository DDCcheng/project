"""
Evaluation and visualisation stage.

Purpose:
    Compare ResNet-18 validation performance between:
    - no augmentation
    - training augmentation

Inputs:
    results/tables/resnet18_no_aug_validation_metrics.csv
    results/tables/resnet18_aug_validation_metrics.csv

Outputs:
    results/tables/resnet18_validation_comparison.csv
    results/figures/resnet18_augmentation_validation_comparison.png
    results/figures/resnet18_training_time_comparison.png

Important:
    This script only reads validation results.
    The test set is not accessed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NO_AUG_PATH = Path(
    "results/tables/resnet18_no_aug_validation_metrics.csv"
)

AUG_PATH = Path(
    "results/tables/resnet18_aug_validation_metrics.csv"
)

OUTPUT_TABLE_PATH = Path(
    "results/tables/resnet18_validation_comparison.csv"
)

OUTPUT_METRICS_FIGURE_PATH = Path(
    "results/figures/resnet18_augmentation_validation_comparison.png"
)

OUTPUT_TIME_FIGURE_PATH = Path(
    "results/figures/resnet18_training_time_comparison.png"
)


def load_result(path: Path, expected_case: str) -> pd.DataFrame:
    """Load and validate one ResNet-18 validation result."""

    if not path.exists():
        raise FileNotFoundError(
            f"Result file was not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if len(dataframe) != 1:
        raise ValueError(
            f"Expected exactly one row in {path}, "
            f"but found {len(dataframe)}."
        )

    required_columns = {
        "model",
        "case",
        "split",
        "best_epoch",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_seconds",
        "epochs_requested",
        "test_used",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    row = dataframe.iloc[0]

    if str(row["case"]).strip().lower() != expected_case:
        raise ValueError(
            f"Expected case {expected_case!r} in {path}, "
            f"but found {row['case']!r}."
        )

    if str(row["split"]).strip().lower() != "val":
        raise ValueError(
            f"{path} does not contain a validation result."
        )

    test_used_value = str(row["test_used"]).strip().lower()

    if test_used_value not in {"false", "0"}:
        raise ValueError(
            f"Test set appears to have been used in {path}."
        )

    if int(row["epochs_requested"]) != 10:
        raise ValueError(
            f"{path} is not a formal 10-epoch result."
        )

    return dataframe


def create_comparison_table() -> pd.DataFrame:
    """Create and save the combined ResNet-18 validation table."""

    no_aug = load_result(
        path=NO_AUG_PATH,
        expected_case="no_aug",
    )

    aug = load_result(
        path=AUG_PATH,
        expected_case="aug",
    )

    comparison = pd.concat(
        [no_aug, aug],
        ignore_index=True,
    )

    display_names = {
        "no_aug": "No Augmentation",
        "aug": "Training Augmentation",
    }

    comparison["setting"] = (
        comparison["case"]
        .str.lower()
        .map(display_names)
    )

    comparison["macro_f1_change_vs_no_aug"] = (
        comparison["macro_f1"]
        - float(
            comparison.loc[
                comparison["case"] == "no_aug",
                "macro_f1",
            ].iloc[0]
        )
    )

    output_columns = [
        "model",
        "setting",
        "case",
        "best_epoch",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_seconds",
        "macro_f1_change_vs_no_aug",
        "test_used",
    ]

    comparison = comparison[output_columns]

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    print(f"Saved table: {OUTPUT_TABLE_PATH}")

    return comparison


def plot_metric_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Plot the main validation metrics for both settings."""

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    metric_names = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro-F1",
        "Weighted-F1",
    ]

    no_aug_row = comparison[
        comparison["case"] == "no_aug"
    ].iloc[0]

    aug_row = comparison[
        comparison["case"] == "aug"
    ].iloc[0]

    no_aug_values = [
        float(no_aug_row[column])
        for column in metric_columns
    ]

    aug_values = [
        float(aug_row[column])
        for column in metric_columns
    ]

    x_positions = np.arange(
        len(metric_columns)
    )

    bar_width = 0.36

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    no_aug_bars = axis.bar(
        x_positions - bar_width / 2,
        no_aug_values,
        bar_width,
        label="No Augmentation",
    )

    aug_bars = axis.bar(
        x_positions + bar_width / 2,
        aug_values,
        bar_width,
        label="Training Augmentation",
    )

    axis.bar_label(
        no_aug_bars,
        fmt="%.4f",
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        aug_bars,
        fmt="%.4f",
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "ResNet-18 Validation Performance: "
        "No Augmentation vs Training Augmentation"
    )

    axis.set_xlabel(
        "Evaluation Metric"
    )

    axis.set_ylabel(
        "Validation Score"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        metric_names,
        rotation=15,
        ha="right",
    )

    minimum_value = min(
        min(no_aug_values),
        min(aug_values),
    )

    axis.set_ylim(
        max(0.0, minimum_value - 0.02),
        1.0,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    OUTPUT_METRICS_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_METRICS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_METRICS_FIGURE_PATH}"
    )


def plot_training_time(
    comparison: pd.DataFrame,
) -> None:
    """Plot total training time for both baseline settings."""

    ordered = comparison.copy()

    ordered["case_order"] = ordered["case"].map(
        {
            "no_aug": 0,
            "aug": 1,
        }
    )

    ordered = ordered.sort_values(
        "case_order"
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    bars = axis.bar(
        ordered["setting"],
        ordered["training_seconds"],
    )

    axis.bar_label(
        bars,
        fmt="%.1f s",
        padding=4,
    )

    axis.set_title(
        "ResNet-18 Validation Training Time Comparison"
    )

    axis.set_xlabel(
        "Training Setting"
    )

    axis.set_ylabel(
        "Total Training Time (seconds)"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    OUTPUT_TIME_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_TIME_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_TIME_FIGURE_PATH}"
    )


def print_summary(
    comparison: pd.DataFrame,
) -> None:
    """Print the validation comparison summary."""

    display_columns = [
        "setting",
        "best_epoch",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_seconds",
        "macro_f1_change_vs_no_aug",
    ]

    print("\nResNet-18 validation comparison:\n")

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    best_index = comparison["macro_f1"].idxmax()
    best_row = comparison.loc[best_index]

    print(
        "\nHigher validation Macro-F1 setting:"
    )

    print(
        f"  {best_row['setting']}"
    )

    print(
        f"  Macro-F1: {best_row['macro_f1']:.6f}"
    )

    print(
        f"  Best epoch: {int(best_row['best_epoch'])}"
    )

    print(
        "\nThese are validation observations only."
    )

    print(
        "The test set was not used."
    )


def main() -> None:
    """Run the complete ResNet-18 baseline comparison."""

    comparison = create_comparison_table()

    plot_metric_comparison(
        comparison
    )

    plot_training_time(
        comparison
    )

    print_summary(
        comparison
    )


if __name__ == "__main__":
    main()