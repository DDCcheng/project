"""Role 4: validation-first PCA and traditional-classifier experiments.

Cases follow the team-approved plan:
  A DenseNet-201                 B DenseNet-201 + PCA(95%)
  C ResNeXt-101                  D ResNeXt-101 + PCA(95%)
  E DenseNet-201 + ResNeXt-101   F Fusion + PCA(95%)

All scalers and PCA objects are fit on training features only.

Default behaviour:
    Evaluate validation data only.

Final-test behaviour:
    When --evaluate-test is supplied, skip validation experiments and evaluate
    exactly one previously selected case/classifier configuration on the
    held-out test split.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Shared settings and paths
# ============================================================

SEED = 42

LABELS = np.asarray(
    [
        "fresh",
        "rotten",
    ]
)

ROOT = Path(__file__).resolve().parents[2]

FEATURES = ROOT / "results" / "features"

FILE_LIST = ROOT / "docs" / "file_list.csv"

TABLES = ROOT / "results" / "tables"


# ============================================================
# Label conversion
# ============================================================

def freshness(label: str) -> str:
    """Convert an original class label to fresh or rotten."""

    value = (
        str(label)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    if value.startswith("fresh"):
        return "fresh"

    if value.startswith("rotten"):
        return "rotten"

    raise ValueError(
        f"Invalid source label: {label!r}"
    )


# ============================================================
# Feature container
# ============================================================

@dataclass
class SplitFeatures:
    """Store train, validation, and test features and labels."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


# ============================================================
# Shared expected labels
# ============================================================

def expected_labels() -> dict[str, np.ndarray]:
    """Read the fixed split and return binary labels in row order."""

    if not FILE_LIST.is_file():
        raise FileNotFoundError(
            f"Missing file list: {FILE_LIST}"
        )

    frame = pd.read_csv(FILE_LIST)

    required_columns = {
        "label",
        "split",
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "file_list.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    frame["split"] = (
        frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return {
        split: np.asarray(
            [
                freshness(label)
                for label in frame.loc[
                    frame["split"] == split,
                    "label",
                ]
            ]
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }


# ============================================================
# DenseNet loading
# ============================================================

def load_densenet(
    feature_case: str,
    expected: dict[str, np.ndarray],
) -> SplitFeatures:
    """Load DenseNet features for one augmentation case."""

    directory = FEATURES / feature_case

    if not directory.is_dir():
        raise FileNotFoundError(
            f"DenseNet feature directory not found: {directory}"
        )

    feature_arrays: dict[str, np.ndarray] = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        feature_path = (
            directory
            / f"densenet201_{split}_features.npy"
        )

        label_path = (
            directory
            / f"densenet201_{split}_binary_labels.npy"
        )

        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Missing DenseNet features: {feature_path}"
            )

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing DenseNet labels: {label_path}"
            )

        feature_arrays[split] = np.load(
            feature_path,
            allow_pickle=False,
        )

        labels = (
            np.load(
                label_path,
                allow_pickle=True,
            )
            .astype(str)
        )

        if not np.array_equal(
            labels,
            expected[split],
        ):
            raise ValueError(
                f"DenseNet {split} labels do not match "
                "docs/file_list.csv."
            )

        if feature_arrays[split].shape[0] != len(labels):
            raise ValueError(
                f"DenseNet {split} feature/label row mismatch."
            )

    return SplitFeatures(
        train=feature_arrays["train"],
        val=feature_arrays["val"],
        test=feature_arrays["test"],
        y_train=expected["train"],
        y_val=expected["val"],
        y_test=expected["test"],
    )


# ============================================================
# ResNeXt loading
# ============================================================

def load_resnext(
    feature_case: str,
    expected: dict[str, np.ndarray],
) -> SplitFeatures:
    """Load ResNeXt features for one augmentation case."""

    prefix = (
        "resnext101"
        if feature_case == "no_aug"
        else "resnext101_aug"
    )

    feature_path = (
        FEATURES
        / f"{prefix}_features.npy"
    )

    label_path = (
        FEATURES
        / f"{prefix}_labels.npy"
    )

    split_path = (
        FEATURES
        / f"{prefix}_splits.npy"
    )

    for path in (
        feature_path,
        label_path,
        split_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing ResNeXt file: {path}"
            )

    matrix = np.load(
        feature_path,
        allow_pickle=False,
    )

    source_labels = (
        np.load(
            label_path,
            allow_pickle=True,
        )
        .astype(str)
    )

    splits = (
        np.load(
            split_path,
            allow_pickle=True,
        )
        .astype(str)
    )

    binary_labels = np.asarray(
        [
            freshness(label)
            for label in source_labels
        ]
    )

    if not (
        len(matrix)
        == len(binary_labels)
        == len(splits)
    ):
        raise ValueError(
            "ResNeXt feature, label, and split arrays "
            "have different lengths."
        )

    result: dict[str, np.ndarray] = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        split_mask = splits == split

        result[split] = matrix[split_mask]

        if not np.array_equal(
            binary_labels[split_mask],
            expected[split],
        ):
            raise ValueError(
                f"ResNeXt {split} labels/order do not match "
                "docs/file_list.csv."
            )

    return SplitFeatures(
        train=result["train"],
        val=result["val"],
        test=result["test"],
        y_train=expected["train"],
        y_val=expected["val"],
        y_test=expected["test"],
    )


# ============================================================
# Feature fusion
# ============================================================

def fuse(
    dense: SplitFeatures,
    resnext: SplitFeatures,
) -> SplitFeatures:
    """Standardise each backbone separately and concatenate features."""

    for split in (
        "train",
        "val",
        "test",
    ):
        dense_labels = getattr(
            dense,
            f"y_{split}",
        )

        resnext_labels = getattr(
            resnext,
            f"y_{split}",
        )

        if not np.array_equal(
            dense_labels,
            resnext_labels,
        ):
            raise ValueError(
                "Cannot fuse features because DenseNet and "
                f"ResNeXt {split} labels are not aligned."
            )

    dense_scaler = StandardScaler().fit(
        dense.train
    )

    resnext_scaler = StandardScaler().fit(
        resnext.train
    )

    def joined(split: str) -> np.ndarray:
        dense_features = dense_scaler.transform(
            getattr(dense, split)
        )

        resnext_features = resnext_scaler.transform(
            getattr(resnext, split)
        )

        return np.concatenate(
            (
                dense_features,
                resnext_features,
            ),
            axis=1,
        )

    return SplitFeatures(
        train=joined("train"),
        val=joined("val"),
        test=joined("test"),
        y_train=dense.y_train,
        y_val=dense.y_val,
        y_test=dense.y_test,
    )


# ============================================================
# Classifier construction
# ============================================================

def classifier(
    name: str,
    bagging_n_jobs: int = -1,
):
    """Create one traditional classifier."""

    if name == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=SEED,
        )

    if name == "lda":
        return LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )

    if name == "bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(
                random_state=SEED,
            ),
            n_estimators=100,
            random_state=SEED,
            n_jobs=bagging_n_jobs,
        )

    raise ValueError(
        f"Unknown classifier: {name}"
    )


# ============================================================
# Model fitting and prediction
# ============================================================

def fit_predict(
    data: SplitFeatures,
    use_pca: bool,
    classifier_name: str,
    target: str,
    bagging_n_jobs: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    int | None,
]:
    """Fit on training features and predict one target split."""

    # Fusion has 3968 features and has already been block-standardised.
    # Single-backbone features are standardised here.
    steps = (
        []
        if data.train.shape[1] == 3968
        else [StandardScaler()]
    )

    if use_pca:
        steps.append(
            PCA(
                n_components=0.95,
                svd_solver="full",
            )
        )

    steps.append(
        classifier(
            classifier_name,
            bagging_n_jobs,
        )
    )

    model = make_pipeline(
        *steps
    )

    target_features = getattr(
        data,
        target,
    )

    target_labels = getattr(
        data,
        f"y_{target}",
    )

    print(
        f"Training {classifier_name.upper()} "
        f"with PCA={use_pca} and evaluating split={target}..."
    )

    print(
        "Training feature shape:",
        data.train.shape,
    )

    print(
        f"{target.capitalize()} feature shape:",
        target_features.shape,
    )

    started = time.perf_counter()

    model.fit(
        data.train,
        data.y_train,
    )

    print(
        "Training completed. Generating predictions..."
    )

    predicted = model.predict(
        target_features
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    pca_dimension = next(
        (
            step.n_components_
            for step in model.named_steps.values()
            if isinstance(step, PCA)
        ),
        None,
    )

    print(
        f"Completed in {elapsed:.2f} seconds."
    )

    return (
        predicted,
        target_labels,
        elapsed,
        pca_dimension,
    )


# ============================================================
# Metrics
# ============================================================

def metric_rows(
    case: str,
    feature_case: str,
    classifier_name: str,
    split: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    seconds: float,
    raw_dimension: int,
    pca_dimension: int | None,
) -> tuple[
    dict,
    pd.DataFrame,
    np.ndarray,
]:
    """Create summary metrics, per-class report, and confusion matrix."""

    macro = precision_recall_fscore_support(
        truth,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted = precision_recall_fscore_support(
        truth,
        predicted,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    summary = {
        "case": case,
        "feature_case": feature_case,
        "classifier": classifier_name,
        "split": split,
        "accuracy": accuracy_score(
            truth,
            predicted,
        ),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
        "raw_feature_dim": raw_dimension,
        "pca_feature_dim": pca_dimension,
        "seconds": seconds,
    }

    report = pd.DataFrame(
        classification_report(
            truth,
            predicted,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    report = (
        report
        .reset_index()
        .rename(
            columns={
                "index": "class",
            }
        )
    )

    report.insert(
        0,
        "classifier",
        classifier_name,
    )

    report.insert(
        0,
        "case",
        case,
    )

    matrix = confusion_matrix(
        truth,
        predicted,
        labels=LABELS,
    )

    return (
        summary,
        report,
        matrix,
    )


# ============================================================
# Configuration creation
# ============================================================

def create_selected_configuration(
    case: str,
    feature_case: str,
    expected: dict[str, np.ndarray],
) -> tuple[SplitFeatures, bool]:
    """Load only the feature data required by one selected case."""

    if case in {
        "A",
        "B",
    }:
        dense = load_densenet(
            feature_case,
            expected,
        )

        return (
            dense,
            case == "B",
        )

    if case in {
        "C",
        "D",
    }:
        resnext = load_resnext(
            feature_case,
            expected,
        )

        return (
            resnext,
            case == "D",
        )

    if case in {
        "E",
        "F",
    }:
        dense = load_densenet(
            feature_case,
            expected,
        )

        resnext = load_resnext(
            feature_case,
            expected,
        )

        fusion = fuse(
            dense,
            resnext,
        )

        return (
            fusion,
            case == "F",
        )

    raise ValueError(
        f"Unknown case: {case}"
    )


def create_all_configurations(
    feature_case: str,
    expected: dict[str, np.ndarray],
) -> dict[str, tuple[SplitFeatures, bool]]:
    """Load all feature types for validation comparison."""

    dense = load_densenet(
        feature_case,
        expected,
    )

    resnext = load_resnext(
        feature_case,
        expected,
    )

    fusion = fuse(
        dense,
        resnext,
    )

    return {
        "A": (
            dense,
            False,
        ),
        "B": (
            dense,
            True,
        ),
        "C": (
            resnext,
            False,
        ),
        "D": (
            resnext,
            True,
        ),
        "E": (
            fusion,
            False,
        ),
        "F": (
            fusion,
            True,
        ),
    }


# ============================================================
# Validation evaluation
# ============================================================

def run_validation(
    feature_case: str,
    run_name: str,
    classifier_names: list[str],
    bagging_n_jobs: int,
    expected: dict[str, np.ndarray],
) -> None:
    """Run all selected classifiers on all six validation cases."""

    print(
        "Validation mode selected."
    )

    print(
        "The test split will not be evaluated."
    )

    configurations = create_all_configurations(
        feature_case,
        expected,
    )

    metrics: list[dict] = []

    reports: list[pd.DataFrame] = []

    for case, (
        data,
        use_pca,
    ) in configurations.items():
        for classifier_name in classifier_names:
            print(
                "\n"
                f"Validation case {case}, "
                f"classifier={classifier_name}, "
                f"PCA={use_pca}"
            )

            (
                predicted,
                truth,
                seconds,
                pca_dimension,
            ) = fit_predict(
                data=data,
                use_pca=use_pca,
                classifier_name=classifier_name,
                target="val",
                bagging_n_jobs=bagging_n_jobs,
            )

            summary, report, _matrix = metric_rows(
                case=case,
                feature_case=feature_case,
                classifier_name=classifier_name,
                split="val",
                truth=truth,
                predicted=predicted,
                seconds=seconds,
                raw_dimension=data.train.shape[1],
                pca_dimension=pca_dimension,
            )

            metrics.append(
                summary
            )

            reports.append(
                report
            )

    metrics_path = (
        TABLES
        / (
            f"stage4_{run_name}_{feature_case}"
            "_validation_metrics.csv"
        )
    )

    reports_path = (
        TABLES
        / (
            f"stage4_{run_name}_{feature_case}"
            "_validation_per_class.csv"
        )
    )

    pd.DataFrame(
        metrics
    ).to_csv(
        metrics_path,
        index=False,
    )

    pd.concat(
        reports,
        ignore_index=True,
    ).to_csv(
        reports_path,
        index=False,
    )

    print(
        "\nValidation evaluation completed."
    )

    print(
        f"Metrics saved to: {metrics_path}"
    )

    print(
        f"Per-class results saved to: {reports_path}"
    )


# ============================================================
# Final test evaluation
# ============================================================

def run_final_test(
    feature_case: str,
    run_name: str,
    test_case: str,
    test_classifier: str,
    bagging_n_jobs: int,
    expected: dict[str, np.ndarray],
) -> None:
    """Evaluate exactly one validation-selected configuration on test."""

    metrics_path = (
        TABLES
        / f"stage4_{run_name}_FINAL_test_metrics.csv"
    )

    report_path = (
        TABLES
        / f"stage4_{run_name}_FINAL_test_per_class.csv"
    )

    matrix_path = (
        TABLES
        / (
            f"stage4_{run_name}"
            "_FINAL_test_confusion_matrix.csv"
        )
    )

    existing_outputs = [
        path
        for path in (
            metrics_path,
            report_path,
            matrix_path,
        )
        if path.exists()
    ]

    if existing_outputs:
        existing_text = "\n".join(
            str(path)
            for path in existing_outputs
        )

        raise FileExistsError(
            "Final test output already exists. "
            "The script stopped to avoid overwriting a "
            "previous final test result:\n"
            f"{existing_text}"
        )

    print(
        "FINAL TEST MODE"
    )

    print(
        "Validation experiments will be skipped."
    )

    print(
        "This mode must be used only for an already "
        "validation-selected configuration."
    )

    print(
        f"Selected case: {test_case}"
    )

    print(
        f"Selected classifier: {test_classifier}"
    )

    print(
        f"Feature case: {feature_case}"
    )

    data, use_pca = create_selected_configuration(
        case=test_case,
        feature_case=feature_case,
        expected=expected,
    )

    (
        predicted,
        truth,
        seconds,
        pca_dimension,
    ) = fit_predict(
        data=data,
        use_pca=use_pca,
        classifier_name=test_classifier,
        target="test",
        bagging_n_jobs=bagging_n_jobs,
    )

    summary, report, matrix = metric_rows(
        case=test_case,
        feature_case=feature_case,
        classifier_name=test_classifier,
        split="test",
        truth=truth,
        predicted=predicted,
        seconds=seconds,
        raw_dimension=data.train.shape[1],
        pca_dimension=pca_dimension,
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        metrics_path,
        index=False,
    )

    report.to_csv(
        report_path,
        index=False,
    )

    pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    ).to_csv(
        matrix_path,
    )

    print(
        "\nFinal test result:"
    )

    print(
        f"Accuracy: {summary['accuracy']:.6f}"
    )

    print(
        f"Macro precision: "
        f"{summary['macro_precision']:.6f}"
    )

    print(
        f"Macro recall: "
        f"{summary['macro_recall']:.6f}"
    )

    print(
        f"Macro F1: {summary['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1: "
        f"{summary['weighted_f1']:.6f}"
    )

    print(
        "\nFinal test files saved:"
    )

    print(
        f" - {metrics_path}"
    )

    print(
        f" - {report_path}"
    )

    print(
        f" - {matrix_path}"
    )

    print(
        "\nDo not use the test result for further "
        "model or parameter selection."
    )


# ============================================================
# Metadata
# ============================================================

def save_config(
    feature_case: str,
    run_name: str,
    classifier_names: list[str],
    bagging_n_jobs: int,
    test_used: bool,
    test_case: str | None,
    test_classifier: str | None,
) -> None:
    """Save experiment configuration metadata."""

    config_path = (
        TABLES
        / f"stage4_{run_name}_{feature_case}_config.json"
    )

    config = {
        "seed": SEED,
        "pca": (
            "95% variance; fitted on training features only"
        ),
        "classifiers": classifier_names,
        "bagging_n_jobs": bagging_n_jobs,
        "test_used": test_used,
        "test_case": test_case,
        "test_classifier": test_classifier,
        "cases": {
            "A": "DenseNet",
            "B": "DenseNet + PCA",
            "C": "ResNeXt",
            "D": "ResNeXt + PCA",
            "E": "Fusion",
            "F": "Fusion + PCA",
        },
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2,
        )

    print(
        f"Configuration saved to: {config_path}"
    )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--feature-case",
        choices=[
            "no_aug",
            "aug",
        ],
        default="no_aug",
        help=(
            "Feature exports to assess. Run augmentation "
            "as a separate comparable experiment."
        ),
    )

    parser.add_argument(
        "--run-name",
        default=time.strftime(
            "%Y%m%d_%H%M%S"
        ),
    )

    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=[
            "svm",
            "lda",
            "bagging",
        ],
        default=[
            "svm",
            "lda",
            "bagging",
        ],
        help=(
            "Classifier subset for validation mode."
        ),
    )

    parser.add_argument(
        "--bagging-n-jobs",
        type=int,
        default=-1,
        help=(
            "Bagging worker count. Use 1 for a "
            "stable single-core verification."
        ),
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Skip validation and evaluate exactly one "
            "already selected configuration on test."
        ),
    )

    parser.add_argument(
        "--test-case",
        choices=list(
            "ABCDEF"
        ),
    )

    parser.add_argument(
        "--test-classifier",
        choices=[
            "svm",
            "lda",
            "bagging",
        ],
    )

    arguments = parser.parse_args()

    test_options_complete = bool(
        arguments.test_case
        and arguments.test_classifier
    )

    if arguments.evaluate_test != test_options_complete:
        parser.error(
            "Final test evaluation requires all three options: "
            "--evaluate-test, --test-case, and "
            "--test-classifier. Otherwise omit all three."
        )

    return arguments


# ============================================================
# Main
# ============================================================

def main() -> int:
    """Run validation experiments or one final test evaluation."""

    arguments = parse_arguments()

    TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = expected_labels()

    if arguments.evaluate_test:
        run_final_test(
            feature_case=arguments.feature_case,
            run_name=arguments.run_name,
            test_case=arguments.test_case,
            test_classifier=arguments.test_classifier,
            bagging_n_jobs=arguments.bagging_n_jobs,
            expected=expected,
        )

    else:
        run_validation(
            feature_case=arguments.feature_case,
            run_name=arguments.run_name,
            classifier_names=arguments.classifiers,
            bagging_n_jobs=arguments.bagging_n_jobs,
            expected=expected,
        )

    save_config(
        feature_case=arguments.feature_case,
        run_name=arguments.run_name,
        classifier_names=arguments.classifiers,
        bagging_n_jobs=arguments.bagging_n_jobs,
        test_used=arguments.evaluate_test,
        test_case=arguments.test_case,
        test_classifier=arguments.test_classifier,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )