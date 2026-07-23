"""
Evaluation and visualisation stage.

Purpose:
    Compare validation Macro-F1 between:
    - No PCA
    - PCA retaining 95% variance

Feature groups:
    - DenseNet-201
    - ResNeXt-101
    - DenseNet-201 + ResNeXt-101 feature fusion

Classifiers:
    - SVM
    - LDA
    - Bagging

Input:
    results/tables/stage4_validation_ranked.csv

Outputs:
    results/tables/validation_no_pca_vs_pca95_comparison.csv
    results/figures/validation_no_pca_vs_pca95_macro_f1.png

Important:
    This script only reads validation results.
    It does not use the test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/tables/stage4_validation_ranked.csv"
)

OUTPUT_TABLE_PATH = Path(
    "results/tables/validation_no_pca_vs_pca95_comparison.csv"
)

OUTPUT_FIGURE_PATH = Path(
    "results/figures/validation_no_pca_vs_pca95_macro_f1.png"
)


FEATURE_GROUPS = {
    "DenseNet-201": {
        "No PCA": "A",
        "PCA(95%)": "B",
    },
    "ResNeXt-101": {
        "No PCA": "C",
        "PCA(95%)": "D",
    },
    "Feature Fusion": {
        "No PCA": "E",
        "PCA(95%)": "F",
    },
}


CLASSIFIER_ORDER = [
    "svm",
    "lda",
    "bagging",
]


CLASSIFIER_DISPLAY_NAMES = {
    "svm": "SVM",
    "lda": "LDA",
    "bagging": "Bagging",
}


def load_validation_results() -> pd.DataFrame:
    """Load and validate the existing Stage 4 validation results."""

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
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    validation_results = results[
        results["split"].str.lower() == "val"
    ].copy()

    if validation_results.empty:
        raise ValueError(
            "No validation rows were found in the input file."
        )

    validation_results["case"] = (
        validation_results["case"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    validation_results["classifier"] = (
        validation_results["classifier"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return validation_results


def create_comparison_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create a tidy table for No PCA versus PCA(95%) comparison."""

    comparison_rows = []

    for feature_name, case_mapping in FEATURE_GROUPS.items():
        for classifier_name in CLASSIFIER_ORDER:
            no_pca_case = case_mapping["No PCA"]
            pca_case = case_mapping["PCA(95%)"]

            no_pca_rows = results[
                (results["case"] == no_pca_case)
                & (results["classifier"] == classifier_name)
            ]

            pca_rows = results[
                (results["case"] == pca_case)
                & (results["classifier"] == classifier_name)
            ]

            if len(no_pca_rows) != 1:
                raise ValueError(
                    "Expected exactly one No PCA result for "
                    f"{feature_name} + {classifier_name}, "
                    f"but found {len(no_pca_rows)}."
                )

            if len(pca_rows) != 1:
                raise ValueError(
                    "Expected exactly one PCA(95%) result for "
                    f"{feature_name} + {classifier_name}, "
                    f"but found {len(pca_rows)}."
                )

            no_pca_macro_f1 = float(
                no_pca_rows.iloc[0]["macro_f1"]
            )

            pca_macro_f1 = float(
                pca_rows.iloc[0]["macro_f1"]
            )

            macro_f1_change = (
                pca_macro_f1 - no_pca_macro_f1
            )

            comparison_rows.append(
                {
                    "feature_group": feature_name,
                    "classifier": (
                        CLASSIFIER_DISPLAY_NAMES[classifier_name]
                    ),
                    "no_pca_macro_f1": no_pca_macro_f1,
                    "pca95_macro_f1": pca_macro_f1,
                    "macro_f1_change": macro_f1_change,
                    "pca_effect": (
                        "Improved"
                        if macro_f1_change > 0
                        else "Decreased"
                        if macro_f1_change < 0
                        else "No change"
                    ),
                }
            )

    comparison_table = pd.DataFrame(comparison_rows)

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


def plot_pca_comparison(
    comparison_table: pd.DataFrame,
) -> None:
    """Create grouped bars comparing No PCA and PCA(95%)."""

    experiment_labels = (
        comparison_table["feature_group"]
        + "\n"
        + comparison_table["classifier"]
    )

    no_pca_values = comparison_table[
        "no_pca_macro_f1"
    ].to_numpy()

    pca_values = comparison_table[
        "pca95_macro_f1"
    ].to_numpy()

    x_positions = np.arange(
        len(comparison_table)
    )

    bar_width = 0.38

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    no_pca_bars = axis.bar(
        x_positions - bar_width / 2,
        no_pca_values,
        bar_width,
        label="No PCA",
    )

    pca_bars = axis.bar(
        x_positions + bar_width / 2,
        pca_values,
        bar_width,
        label="PCA(95%)",
    )

    axis.set_title(
        "Validation Macro-F1: No PCA vs PCA Retaining 95% Variance"
    )

    axis.set_xlabel(
        "Feature Model and Classifier"
    )

    axis.set_ylabel(
        "Validation Macro-F1"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        experiment_labels,
        rotation=30,
        ha="right",
    )

    minimum_value = min(
        no_pca_values.min(),
        pca_values.min(),
    )

    axis.set_ylim(
        max(0.0, minimum_value - 0.03),
        1.0,
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.bar_label(
        no_pca_bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        pca_bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    figure.tight_layout()

    OUTPUT_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved figure: {OUTPUT_FIGURE_PATH}")


def print_summary(
    comparison_table: pd.DataFrame,
) -> None:
    """Print the PCA effect for each feature and classifier pair."""

    print("\nNo PCA vs PCA(95%) comparison:\n")

    display_columns = [
        "feature_group",
        "classifier",
        "no_pca_macro_f1",
        "pca95_macro_f1",
        "macro_f1_change",
        "pca_effect",
    ]

    print(
        comparison_table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    improved_count = int(
        (
            comparison_table["macro_f1_change"] > 0
        ).sum()
    )

    decreased_count = int(
        (
            comparison_table["macro_f1_change"] < 0
        ).sum()
    )

    print(
        f"\nPCA improved Macro-F1 in "
        f"{improved_count} experiment(s)."
    )

    print(
        f"PCA decreased Macro-F1 in "
        f"{decreased_count} experiment(s)."
    )

    print("\nThe test set was not used.")


def main() -> None:
    """Run the complete validation PCA comparison."""

    validation_results = load_validation_results()

    comparison_table = create_comparison_table(
        validation_results
    )

    plot_pca_comparison(
        comparison_table
    )

    print_summary(
        comparison_table
    )


if __name__ == "__main__":
    main()