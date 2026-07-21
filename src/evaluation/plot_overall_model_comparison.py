"""
Evaluation and visualisation stage.

Purpose:
    Add ResNet-18 baseline results to the overall validation
    comparison of DenseNet-201, ResNeXt-101 and feature fusion.

Selection rule:
    For each Stage 4 feature case A-F, select the classifier with
    the highest validation Macro-F1.

Inputs:
    results/tables/stage4_validation_ranked.csv
    results/tables/resnet18_validation_comparison.csv

Outputs:
    results/tables/overall_validation_model_comparison.csv
    results/figures/overall_validation_model_comparison.png
    results/figures/overall_validation_macro_f1_ranking.png

Important:
    This script only uses validation results.
    The test set is not accessed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE4_PATH = Path(
    "results/tables/stage4_validation_ranked.csv"
)

RESNET18_PATH = Path(
    "results/tables/resnet18_validation_comparison.csv"
)

OUTPUT_TABLE_PATH = Path(
    "results/tables/overall_validation_model_comparison.csv"
)

OUTPUT_METRICS_FIGURE_PATH = Path(
    "results/figures/overall_validation_model_comparison.png"
)

OUTPUT_RANKING_FIGURE_PATH = Path(
    "results/figures/overall_validation_macro_f1_ranking.png"
)


CASE_NAMES = {
    "A": "DenseNet-201",
    "B": "DenseNet-201 + PCA(95%)",
    "C": "ResNeXt-101",
    "D": "ResNeXt-101 + PCA(95%)",
    "E": "Feature Fusion",
    "F": "Feature Fusion + PCA(95%)",
}


CLASSIFIER_NAMES = {
    "svm": "SVM",
    "lda": "LDA",
    "bagging": "Bagging",
}


def load_stage4_results() -> pd.DataFrame:
    """Load and validate the Stage 4 validation results."""

    if not STAGE4_PATH.exists():
        raise FileNotFoundError(
            f"Stage 4 result file was not found: {STAGE4_PATH}"
        )

    results = pd.read_csv(STAGE4_PATH)

    required_columns = {
        "case",
        "classifier",
        "split",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "seconds",
        "raw_feature_dim",
        "pca_feature_dim",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            "Stage 4 results are missing columns: "
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

    results = results[
        results["split"] == "val"
    ].copy()

    if results.empty:
        raise ValueError(
            "No Stage 4 validation results were found."
        )

    return results


def select_best_stage4_by_case(
    stage4_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the best classifier for each feature case
    using validation Macro-F1 only.
    """

    expected_cases = set(CASE_NAMES)
    actual_cases = set(stage4_results["case"])

    missing_cases = expected_cases - actual_cases

    if missing_cases:
        raise ValueError(
            f"Missing Stage 4 cases: {sorted(missing_cases)}"
        )

    best_indices = (
        stage4_results
        .groupby("case")["macro_f1"]
        .idxmax()
    )

    best_results = stage4_results.loc[
        best_indices
    ].copy()

    best_results["model_setting"] = (
        best_results["case"].map(CASE_NAMES)
    )

    best_results["selected_classifier"] = (
        best_results["classifier"].map(
            CLASSIFIER_NAMES
        )
    )

    best_results["display_name"] = (
        best_results["model_setting"]
        + "\n"
        + best_results["selected_classifier"]
    )

    best_results["model_family"] = (
        best_results["case"].map(
            {
                "A": "DenseNet-201",
                "B": "DenseNet-201",
                "C": "ResNeXt-101",
                "D": "ResNeXt-101",
                "E": "Feature Fusion",
                "F": "Feature Fusion",
            }
        )
    )

    best_results["augmentation"] = (
        "Training augmentation features"
    )

    best_results["selection_source"] = (
        "Best classifier selected by validation Macro-F1"
    )

    best_results["runtime_type"] = (
        "Traditional classifier fitting and prediction time"
    )

    output = pd.DataFrame(
        {
            "model_family": best_results["model_family"],
            "model_setting": best_results["model_setting"],
            "display_name": best_results["display_name"],
            "augmentation": best_results["augmentation"],
            "classifier": best_results[
                "selected_classifier"
            ],
            "accuracy": best_results["accuracy"],
            "macro_precision": best_results[
                "macro_precision"
            ],
            "macro_recall": best_results["macro_recall"],
            "macro_f1": best_results["macro_f1"],
            "weighted_f1": best_results["weighted_f1"],
            "runtime_seconds": best_results["seconds"],
            "runtime_type": best_results["runtime_type"],
            "raw_feature_dim": best_results[
                "raw_feature_dim"
            ],
            "pca_feature_dim": best_results[
                "pca_feature_dim"
            ],
            "best_epoch": np.nan,
            "selection_source": best_results[
                "selection_source"
            ],
            "split": "val",
            "test_used": False,
        }
    )

    return output


def load_resnet18_results() -> pd.DataFrame:
    """Load the two formal ResNet-18 validation results."""

    if not RESNET18_PATH.exists():
        raise FileNotFoundError(
            f"ResNet-18 comparison file was not found: "
            f"{RESNET18_PATH}"
        )

    results = pd.read_csv(RESNET18_PATH)

    required_columns = {
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
        "test_used",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            "ResNet-18 results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(results) != 2:
        raise ValueError(
            "Expected exactly two ResNet-18 settings."
        )

    expected_cases = {"no_aug", "aug"}
    actual_cases = set(
        results["case"].astype(str).str.lower()
    )

    if actual_cases != expected_cases:
        raise ValueError(
            "ResNet-18 results must contain no_aug and aug."
        )

    test_values = (
        results["test_used"]
        .astype(str)
        .str.lower()
    )

    if not test_values.isin({"false", "0"}).all():
        raise ValueError(
            "The ResNet-18 comparison appears to use test data."
        )

    display_names = {
        "no_aug": "ResNet-18\nNo Augmentation",
        "aug": "ResNet-18\nTraining Augmentation",
    }

    augmentation_names = {
        "no_aug": "No augmentation",
        "aug": "Training augmentation",
    }

    results["case"] = (
        results["case"]
        .astype(str)
        .str.lower()
    )

    output = pd.DataFrame(
        {
            "model_family": "ResNet-18",
            "model_setting": results["case"].map(
                {
                    "no_aug": "ResNet-18 No Augmentation",
                    "aug": "ResNet-18 Training Augmentation",
                }
            ),
            "display_name": results["case"].map(
                display_names
            ),
            "augmentation": results["case"].map(
                augmentation_names
            ),
            "classifier": "End-to-end CNN",
            "accuracy": results["accuracy"],
            "macro_precision": results[
                "macro_precision"
            ],
            "macro_recall": results["macro_recall"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],
            "runtime_seconds": results[
                "training_seconds"
            ],
            "runtime_type": (
                "Complete end-to-end CNN training time"
            ),
            "raw_feature_dim": np.nan,
            "pca_feature_dim": np.nan,
            "best_epoch": results["best_epoch"],
            "selection_source": (
                "Best epoch selected by validation Macro-F1"
            ),
            "split": "val",
            "test_used": False,
        }
    )

    return output


def create_overall_table() -> pd.DataFrame:
    """Combine Stage 4 and ResNet-18 validation results."""

    stage4_results = load_stage4_results()

    best_stage4 = select_best_stage4_by_case(
        stage4_results
    )

    resnet18_results = load_resnet18_results()

    overall = pd.concat(
        [
            best_stage4,
            resnet18_results,
        ],
        ignore_index=True,
    )

    overall = overall.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    overall.insert(
        0,
        "validation_rank",
        range(1, len(overall) + 1),
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    print(f"Saved table: {OUTPUT_TABLE_PATH}")

    return overall


def plot_overall_metrics(
    overall: pd.DataFrame,
) -> None:
    """Compare Accuracy, Macro-F1 and Weighted-F1."""

    ordered = overall.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    x_positions = np.arange(len(ordered))
    bar_width = 0.25

    figure, axis = plt.subplots(
        figsize=(15, 7)
    )

    accuracy_bars = axis.bar(
        x_positions - bar_width,
        ordered["accuracy"],
        bar_width,
        label="Accuracy",
    )

    macro_f1_bars = axis.bar(
        x_positions,
        ordered["macro_f1"],
        bar_width,
        label="Macro-F1",
    )

    weighted_f1_bars = axis.bar(
        x_positions + bar_width,
        ordered["weighted_f1"],
        bar_width,
        label="Weighted-F1",
    )

    axis.bar_label(
        accuracy_bars,
        fmt="%.3f",
        padding=3,
        fontsize=7,
        rotation=90,
    )

    axis.bar_label(
        macro_f1_bars,
        fmt="%.3f",
        padding=3,
        fontsize=7,
        rotation=90,
    )

    axis.bar_label(
        weighted_f1_bars,
        fmt="%.3f",
        padding=3,
        fontsize=7,
        rotation=90,
    )

    axis.set_title(
        "Overall Validation Performance Comparison"
    )

    axis.set_xlabel(
        "Model Configuration"
    )

    axis.set_ylabel(
        "Validation Score"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        ordered["display_name"],
        rotation=25,
        ha="right",
    )

    minimum_value = float(
        ordered[
            [
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].min().min()
    )

    axis.set_ylim(
        max(0.0, minimum_value - 0.025),
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


def plot_macro_f1_ranking(
    overall: pd.DataFrame,
) -> None:
    """Create a validation Macro-F1 ranking chart."""

    ordered = overall.sort_values(
        "macro_f1",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    bars = axis.barh(
        ordered["display_name"],
        ordered["macro_f1"],
    )

    axis.bar_label(
        bars,
        fmt="%.4f",
        padding=4,
        fontsize=9,
    )

    axis.set_title(
        "Overall Validation Macro-F1 Ranking"
    )

    axis.set_xlabel(
        "Validation Macro-F1"
    )

    axis.set_ylabel(
        "Model Configuration"
    )

    minimum_value = float(
        ordered["macro_f1"].min()
    )

    axis.set_xlim(
        max(0.0, minimum_value - 0.025),
        1.0,
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    OUTPUT_RANKING_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_RANKING_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved figure: {OUTPUT_RANKING_FIGURE_PATH}"
    )


def print_summary(
    overall: pd.DataFrame,
) -> None:
    """Print the overall validation ranking."""

    display_columns = [
        "validation_rank",
        "model_setting",
        "classifier",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "best_epoch",
    ]

    print("\nOverall validation comparison:\n")

    print(
        overall[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    highest_row = overall.iloc[0]

    print(
        "\nHighest Validation Macro-F1 configuration:"
    )

    print(
        f"  {highest_row['model_setting']}"
    )

    print(
        f"  Classifier: {highest_row['classifier']}"
    )

    print(
        f"  Accuracy: {highest_row['accuracy']:.6f}"
    )

    print(
        f"  Macro-F1: {highest_row['macro_f1']:.6f}"
    )

    print(
        "\nRuntime values are not directly comparable between "
        "traditional classifiers and ResNet-18."
    )

    print(
        "Stage 4 runtime records classifier fitting and prediction, "
        "whereas ResNet-18 runtime records complete CNN training."
    )

    print(
        "\nThese are validation observations only."
    )

    print(
        "The test set was not used."
    )


def main() -> None:
    """Run the overall validation model comparison."""

    overall = create_overall_table()

    plot_overall_metrics(
        overall
    )

    plot_macro_f1_ranking(
        overall
    )

    print_summary(
        overall
    )


if __name__ == "__main__":
    main()