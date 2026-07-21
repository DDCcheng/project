"""
Evaluation and visualisation stage.

Purpose:
    Visualise the trade-off between validation Macro-F1
    and classifier runtime for all Stage 4 experiments.

Input:
    results/tables/stage4_validation_ranked.csv

Outputs:
    results/tables/validation_efficiency_tradeoff.csv
    results/figures/validation_macro_f1_vs_runtime.png
    results/figures/validation_macro_f1_vs_runtime_log.png

Important:
    This script only reads validation results.
    The test set is not accessed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "results/tables/stage4_validation_ranked.csv"
)

OUTPUT_TABLE_PATH = Path(
    "results/tables/validation_efficiency_tradeoff.csv"
)

OUTPUT_LINEAR_FIGURE_PATH = Path(
    "results/figures/validation_macro_f1_vs_runtime.png"
)

OUTPUT_LOG_FIGURE_PATH = Path(
    "results/figures/validation_macro_f1_vs_runtime_log.png"
)


CASE_NAMES = {
    "A": "DenseNet",
    "B": "DenseNet + PCA(95%)",
    "C": "ResNeXt",
    "D": "ResNeXt + PCA(95%)",
    "E": "Fusion",
    "F": "Fusion + PCA(95%)",
}

CLASSIFIER_NAMES = {
    "svm": "SVM",
    "lda": "LDA",
    "bagging": "Bagging",
}


def load_validation_results() -> pd.DataFrame:
    """Load and validate Stage 4 validation results."""

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
        "raw_feature_dim",
        "pca_feature_dim",
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

    if (validation_results["seconds"] <= 0).any():
        raise ValueError(
            "All runtime values must be greater than zero."
        )

    validation_results["feature_name"] = (
        validation_results["case"].map(CASE_NAMES)
    )

    validation_results["classifier_name"] = (
        validation_results["classifier"].map(
            CLASSIFIER_NAMES
        )
    )

    if validation_results["feature_name"].isna().any():
        raise ValueError(
            "Unknown feature case was found."
        )

    if validation_results["classifier_name"].isna().any():
        raise ValueError(
            "Unknown classifier was found."
        )

    validation_results["experiment"] = (
        validation_results["feature_name"]
        + " + "
        + validation_results["classifier_name"]
    )

    return validation_results


def create_output_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Save the performance-efficiency comparison table."""

    output_columns = [
        "case",
        "feature_name",
        "classifier",
        "classifier_name",
        "experiment",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "seconds",
        "raw_feature_dim",
        "pca_feature_dim",
    ]

    output_table = (
        results[output_columns]
        .sort_values(
            ["macro_f1", "seconds"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    print(f"Saved table: {OUTPUT_TABLE_PATH}")

    return output_table


def add_point_labels(
    axis: plt.Axes,
    results: pd.DataFrame,
) -> None:
    """Add concise experiment labels beside scatter points."""

    for _, row in results.iterrows():
        short_label = (
            f"{row['case']}-"
            f"{row['classifier_name']}"
        )

        axis.annotate(
            short_label,
            (
                row["seconds"],
                row["macro_f1"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


def plot_linear_tradeoff(
    results: pd.DataFrame,
) -> None:
    """Create a standard linear-runtime trade-off plot."""

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    for classifier_name, classifier_rows in results.groupby(
        "classifier_name"
    ):
        axis.scatter(
            classifier_rows["seconds"],
            classifier_rows["macro_f1"],
            s=80,
            label=classifier_name,
        )

    add_point_labels(
        axis,
        results,
    )

    axis.set_title(
        "Validation Performance–Efficiency Trade-off"
    )

    axis.set_xlabel(
        "Runtime (seconds)"
    )

    axis.set_ylabel(
        "Validation Macro-F1"
    )

    minimum_f1 = float(
        results["macro_f1"].min()
    )

    axis.set_ylim(
        max(0.0, minimum_f1 - 0.03),
        1.0,
    )

    axis.grid(
        alpha=0.3,
    )

    axis.legend(
        title="Classifier"
    )

    figure.tight_layout()

    OUTPUT_LINEAR_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_LINEAR_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_LINEAR_FIGURE_PATH}"
    )


def plot_log_tradeoff(
    results: pd.DataFrame,
) -> None:
    """Create a log-runtime trade-off plot."""

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    for classifier_name, classifier_rows in results.groupby(
        "classifier_name"
    ):
        axis.scatter(
            classifier_rows["seconds"],
            classifier_rows["macro_f1"],
            s=80,
            label=classifier_name,
        )

    add_point_labels(
        axis,
        results,
    )

    axis.set_xscale("log")

    axis.set_title(
        "Validation Performance–Efficiency Trade-off "
        "(Log Runtime Scale)"
    )

    axis.set_xlabel(
        "Runtime in Seconds (Log Scale)"
    )

    axis.set_ylabel(
        "Validation Macro-F1"
    )

    minimum_f1 = float(
        results["macro_f1"].min()
    )

    axis.set_ylim(
        max(0.0, minimum_f1 - 0.03),
        1.0,
    )

    axis.grid(
        alpha=0.3,
        which="both",
    )

    axis.legend(
        title="Classifier"
    )

    figure.tight_layout()

    OUTPUT_LOG_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_LOG_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_LOG_FIGURE_PATH}"
    )


def identify_tradeoff_candidates(
    results: pd.DataFrame,
) -> None:
    """
    Print useful candidates without claiming a final best model.

    The selection is based only on validation results.
    """

    highest_macro_f1_row = results.loc[
        results["macro_f1"].idxmax()
    ]

    fastest_row = results.loc[
        results["seconds"].idxmin()
    ]

    high_performance_threshold = (
        results["macro_f1"].max() - 0.005
    )

    high_performance_results = results[
        results["macro_f1"]
        >= high_performance_threshold
    ]

    efficient_high_performance_row = (
        high_performance_results
        .sort_values("seconds")
        .iloc[0]
    )

    print("\nValidation trade-off summary:\n")

    print(
        "Highest validation Macro-F1:\n"
        f"  {highest_macro_f1_row['experiment']}\n"
        f"  Macro-F1: "
        f"{highest_macro_f1_row['macro_f1']:.6f}\n"
        f"  Runtime: "
        f"{highest_macro_f1_row['seconds']:.3f} seconds\n"
    )

    print(
        "Fastest validation experiment:\n"
        f"  {fastest_row['experiment']}\n"
        f"  Macro-F1: "
        f"{fastest_row['macro_f1']:.6f}\n"
        f"  Runtime: "
        f"{fastest_row['seconds']:.3f} seconds\n"
    )

    print(
        "Fastest experiment within 0.005 Macro-F1 "
        "of the highest validation score:\n"
        f"  {efficient_high_performance_row['experiment']}\n"
        f"  Macro-F1: "
        f"{efficient_high_performance_row['macro_f1']:.6f}\n"
        f"  Runtime: "
        f"{efficient_high_performance_row['seconds']:.3f} seconds\n"
    )

    print(
        "These are validation observations only."
    )

    print(
        "The test set was not used."
    )


def main() -> None:
    """Run the validation performance-efficiency analysis."""

    validation_results = load_validation_results()

    output_table = create_output_table(
        validation_results
    )

    plot_linear_tradeoff(
        output_table
    )

    plot_log_tradeoff(
        output_table
    )

    identify_tradeoff_candidates(
        output_table
    )


if __name__ == "__main__":
    main()