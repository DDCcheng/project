"""
Evaluation and visualisation stage.

Purpose:
    Compare SVM, LDA, and Bagging using validation Macro-F1
    across all six feature cases.

Cases:
    A: DenseNet-201
    B: DenseNet-201 + PCA retaining 95% variance
    C: ResNeXt-101
    D: ResNeXt-101 + PCA retaining 95% variance
    E: DenseNet-201 + ResNeXt-101 feature fusion
    F: Feature fusion + PCA retaining 95% variance

Input:
    results/tables/stage4_validation_ranked.csv

Outputs:
    results/tables/validation_classifier_comparison.csv
    results/figures/validation_classifier_macro_f1_comparison.png
    results/figures/validation_classifier_average_macro_f1.png

Important:
    This script only uses validation results.
    The test set is not accessed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/tables/stage4_validation_ranked.csv"
)

OUTPUT_TABLE_PATH = Path(
    "results/tables/validation_classifier_comparison.csv"
)

OUTPUT_GROUPED_FIGURE_PATH = Path(
    "results/figures/validation_classifier_macro_f1_comparison.png"
)

OUTPUT_AVERAGE_FIGURE_PATH = Path(
    "results/figures/validation_classifier_average_macro_f1.png"
)


CASE_ORDER = ["A", "B", "C", "D", "E", "F"]

CASE_NAMES = {
    "A": "DenseNet-201",
    "B": "DenseNet-201\n+ PCA(95%)",
    "C": "ResNeXt-101",
    "D": "ResNeXt-101\n+ PCA(95%)",
    "E": "Feature Fusion",
    "F": "Feature Fusion\n+ PCA(95%)",
}

CLASSIFIER_ORDER = [
    "svm",
    "lda",
    "bagging",
]

CLASSIFIER_NAMES = {
    "svm": "SVM",
    "lda": "LDA",
    "bagging": "Bagging",
}


def load_validation_results() -> pd.DataFrame:
    """Load and validate the existing validation result table."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file was not found: {INPUT_PATH}"
        )

    results = pd.read_csv(INPUT_PATH)

    required_columns = {
        "case",
        "classifier",
        "split",
        "macro_f1",
        "accuracy",
        "weighted_f1",
        "seconds",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    results["case"] = (
        results["case"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    results["classifier"] = (
        results["classifier"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    results["split"] = (
        results["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    validation_results = results[
        results["split"] == "val"
    ].copy()

    if validation_results.empty:
        raise ValueError(
            "No validation results were found."
        )

    expected_cases = set(CASE_ORDER)
    actual_cases = set(validation_results["case"])

    missing_cases = expected_cases - actual_cases

    if missing_cases:
        raise ValueError(
            f"Missing validation cases: {sorted(missing_cases)}"
        )

    return validation_results


def create_comparison_table(
    validation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create and save a classifier comparison table."""

    selected_rows = validation_results[
        validation_results["case"].isin(CASE_ORDER)
        & validation_results["classifier"].isin(
            CLASSIFIER_ORDER
        )
    ].copy()

    duplicate_counts = (
        selected_rows
        .groupby(["case", "classifier"])
        .size()
    )

    invalid_counts = duplicate_counts[
        duplicate_counts != 1
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "Each case and classifier combination must have "
            "exactly one validation result.\n"
            f"{invalid_counts}"
        )

    selected_rows["case_name"] = (
        selected_rows["case"].map(CASE_NAMES)
    )

    selected_rows["classifier_name"] = (
        selected_rows["classifier"].map(
            CLASSIFIER_NAMES
        )
    )

    selected_rows["case_order"] = (
        selected_rows["case"].map(
            {
                case_name: index
                for index, case_name in enumerate(
                    CASE_ORDER
                )
            }
        )
    )

    selected_rows["classifier_order"] = (
        selected_rows["classifier"].map(
            {
                classifier_name: index
                for index, classifier_name in enumerate(
                    CLASSIFIER_ORDER
                )
            }
        )
    )

    selected_rows = selected_rows.sort_values(
        ["case_order", "classifier_order"]
    )

    output_columns = [
        "case",
        "case_name",
        "classifier",
        "classifier_name",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "seconds",
    ]

    comparison_table = selected_rows[
        output_columns
    ].copy()

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    print(f"Saved table: {OUTPUT_TABLE_PATH}")

    return comparison_table


def plot_grouped_classifier_comparison(
    comparison_table: pd.DataFrame,
) -> None:
    """Plot each classifier's Macro-F1 within every feature case."""

    x_positions = np.arange(len(CASE_ORDER))
    bar_width = 0.24

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    all_values = []

    for classifier_index, classifier in enumerate(
        CLASSIFIER_ORDER
    ):
        classifier_rows = comparison_table[
            comparison_table["classifier"] == classifier
        ].copy()

        classifier_rows["case_order"] = (
            classifier_rows["case"].map(
                {
                    case_name: index
                    for index, case_name in enumerate(
                        CASE_ORDER
                    )
                }
            )
        )

        classifier_rows = classifier_rows.sort_values(
            "case_order"
        )

        macro_f1_values = classifier_rows[
            "macro_f1"
        ].to_numpy()

        all_values.extend(macro_f1_values.tolist())

        offset = (
            classifier_index
            - (len(CLASSIFIER_ORDER) - 1) / 2
        ) * bar_width

        bars = axis.bar(
            x_positions + offset,
            macro_f1_values,
            bar_width,
            label=CLASSIFIER_NAMES[classifier],
        )

        axis.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
            rotation=90,
        )

    axis.set_title(
        "Validation Macro-F1 Comparison of SVM, LDA and Bagging"
    )

    axis.set_xlabel(
        "Feature Setting"
    )

    axis.set_ylabel(
        "Validation Macro-F1"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [CASE_NAMES[case] for case in CASE_ORDER]
    )

    minimum_value = min(all_values)

    axis.set_ylim(
        max(0.0, minimum_value - 0.04),
        1.0,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        title="Classifier"
    )

    figure.tight_layout()

    OUTPUT_GROUPED_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_GROUPED_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_GROUPED_FIGURE_PATH}"
    )


def plot_average_classifier_performance(
    comparison_table: pd.DataFrame,
) -> None:
    """Plot mean validation Macro-F1 for each classifier."""

    average_results = (
        comparison_table
        .groupby(
            ["classifier", "classifier_name"],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            minimum_macro_f1=("macro_f1", "min"),
            maximum_macro_f1=("macro_f1", "max"),
            mean_seconds=("seconds", "mean"),
        )
    )

    average_results["classifier_order"] = (
        average_results["classifier"].map(
            {
                classifier_name: index
                for index, classifier_name in enumerate(
                    CLASSIFIER_ORDER
                )
            }
        )
    )

    average_results = average_results.sort_values(
        "classifier_order"
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    bars = axis.bar(
        average_results["classifier_name"],
        average_results["mean_macro_f1"],
    )

    axis.bar_label(
        bars,
        fmt="%.4f",
        padding=4,
    )

    axis.set_title(
        "Average Validation Macro-F1 by Classifier"
    )

    axis.set_xlabel(
        "Classifier"
    )

    axis.set_ylabel(
        "Mean Validation Macro-F1 Across Six Feature Cases"
    )

    minimum_value = float(
        average_results["mean_macro_f1"].min()
    )

    axis.set_ylim(
        max(0.0, minimum_value - 0.04),
        1.0,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    OUTPUT_AVERAGE_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_AVERAGE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_AVERAGE_FIGURE_PATH}"
    )

    print("\nAverage classifier performance:\n")

    print(
        average_results[
            [
                "classifier_name",
                "mean_macro_f1",
                "minimum_macro_f1",
                "maximum_macro_f1",
                "mean_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def print_best_classifier_by_case(
    comparison_table: pd.DataFrame,
) -> None:
    """Print the classifier with the highest Macro-F1 in each case."""

    best_indices = (
        comparison_table
        .groupby("case")["macro_f1"]
        .idxmax()
    )

    best_rows = comparison_table.loc[
        best_indices,
        [
            "case",
            "case_name",
            "classifier_name",
            "macro_f1",
            "accuracy",
            "weighted_f1",
            "seconds",
        ],
    ].sort_values("case")

    print("\nBest classifier for each feature case:\n")

    print(
        best_rows.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nThe test set was not used.")


def main() -> None:
    """Run the validation classifier comparison."""

    validation_results = load_validation_results()

    comparison_table = create_comparison_table(
        validation_results
    )

    plot_grouped_classifier_comparison(
        comparison_table
    )

    plot_average_classifier_performance(
        comparison_table
    )

    print_best_classifier_by_case(
        comparison_table
    )


if __name__ == "__main__":
    main()