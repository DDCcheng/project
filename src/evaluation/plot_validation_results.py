"""
Evaluation stage:
Create validation Macro-F1 comparison figures from the existing Stage 4 results.

Input:
    results/tables/stage4_validation_ranked.csv

Outputs:
    results/figures/validation_macro_f1_comparison.png
    results/figures/validation_runtime_comparison.png

This script only reads validation results.
It does not access or evaluate the test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RANKED_RESULTS_PATH = Path(
    "results/tables/stage4_validation_ranked.csv"
)
FIGURE_DIRECTORY = Path("results/figures")

CASE_NAMES = {
    "A": "DenseNet",
    "B": "DenseNet + PCA(95%)",
    "C": "ResNeXt",
    "D": "ResNeXt + PCA(95%)",
    "E": "Fusion",
    "F": "Fusion + PCA(95%)",
}


def load_results() -> pd.DataFrame:
    """Load and validate the ranked validation results."""

    if not RANKED_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {RANKED_RESULTS_PATH}"
        )

    results = pd.read_csv(RANKED_RESULTS_PATH)

    required_columns = {
        "case",
        "classifier",
        "macro_f1",
        "seconds",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    results["case_name"] = results["case"].map(CASE_NAMES)

    if results["case_name"].isna().any():
        unknown_cases = results.loc[
            results["case_name"].isna(), "case"
        ].unique()

        raise ValueError(
            f"Unknown case values: {unknown_cases.tolist()}"
        )

    results["experiment"] = (
        results["case_name"]
        + " + "
        + results["classifier"].str.upper()
    )

    return results


def plot_macro_f1(results: pd.DataFrame) -> None:
    """Plot validation Macro-F1 for all 18 experiments."""

    sorted_results = results.sort_values(
        "macro_f1",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(11, 8))

    axis.barh(
        sorted_results["experiment"],
        sorted_results["macro_f1"],
    )

    axis.set_title(
        "Validation Macro-F1 Across Feature and Classifier Settings"
    )
    axis.set_xlabel("Validation Macro-F1")
    axis.set_ylabel("Experiment")

    minimum_value = max(
        0.0,
        sorted_results["macro_f1"].min() - 0.02,
    )
    axis.set_xlim(minimum_value, 1.0)

    axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "validation_macro_f1_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_runtime(results: pd.DataFrame) -> None:
    """Plot runtime for all 18 validation experiments."""

    sorted_results = results.sort_values(
        "seconds",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(11, 8))

    axis.barh(
        sorted_results["experiment"],
        sorted_results["seconds"],
    )

    axis.set_title(
        "Validation Experiment Runtime Comparison"
    )
    axis.set_xlabel("Runtime (seconds)")
    axis.set_ylabel("Experiment")
    axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "validation_runtime_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    """Create validation comparison figures."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results()

    plot_macro_f1(results)
    plot_runtime(results)

    print("Validation figures completed.")
    print("The test set was not used.")


if __name__ == "__main__":
    main()
