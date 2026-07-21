"""Role 4: validation-first PCA and traditional-classifier experiments.

Cases follow the team-approved plan:
  A DenseNet-201                 B DenseNet-201 + PCA(95%)
  C ResNeXt-101                  D ResNeXt-101 + PCA(95%)
  E DenseNet-201 + ResNeXt-101   F Fusion + PCA(95%)

All scalers and PCA objects are fit on training features only.  By default the
script evaluates validation data only.  Test evaluation is deliberately opt-in
and accepts exactly one pre-selected case/classifier combination.
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
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 42
LABELS = np.asarray(["fresh", "rotten"])
ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "results" / "features"
FILE_LIST = ROOT / "docs" / "file_list.csv"


def freshness(label: str) -> str:
    value = str(label).lower().replace(" ", "").replace("_", "").replace("-", "")
    if value.startswith("fresh"):
        return "fresh"
    if value.startswith("rotten"):
        return "rotten"
    raise ValueError(f"Invalid source label: {label!r}")


@dataclass
class SplitFeatures:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def expected_labels() -> dict[str, np.ndarray]:
    frame = pd.read_csv(FILE_LIST)
    if not {"label", "split"}.issubset(frame.columns):
        raise ValueError("file_list.csv must contain label and split columns")
    return {
        split: np.asarray([freshness(label) for label in frame.loc[frame.split == split, "label"]])
        for split in ("train", "val", "test")
    }


def load_densenet(feature_case: str, expected: dict[str, np.ndarray]) -> SplitFeatures:
    directory = FEATURES / feature_case
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        feature_path = directory / f"densenet201_{split}_features.npy"
        label_path = directory / f"densenet201_{split}_binary_labels.npy"
        arrays[split] = np.load(feature_path, allow_pickle=False)
        labels = np.load(label_path, allow_pickle=True).astype(str)
        if not np.array_equal(labels, expected[split]):
            raise ValueError(f"DenseNet {split} labels do not match file_list.csv")
        if arrays[split].shape[0] != len(labels):
            raise ValueError(f"DenseNet {split} feature/label row mismatch")
    return SplitFeatures(arrays["train"], arrays["val"], arrays["test"], expected["train"], expected["val"], expected["test"])


def load_resnext(feature_case: str, expected: dict[str, np.ndarray]) -> SplitFeatures:
    prefix = "resnext101" if feature_case == "no_aug" else "resnext101_aug"
    matrix = np.load(FEATURES / f"{prefix}_features.npy", allow_pickle=False)
    source_labels = np.load(FEATURES / f"{prefix}_labels.npy", allow_pickle=True).astype(str)
    splits = np.load(FEATURES / f"{prefix}_splits.npy", allow_pickle=True).astype(str)
    binary = np.asarray([freshness(label) for label in source_labels])
    if not (len(matrix) == len(binary) == len(splits)):
        raise ValueError("ResNeXt feature, label, and split arrays have different lengths")
    result: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        result[split] = matrix[mask]
        if not np.array_equal(binary[mask], expected[split]):
            raise ValueError(f"ResNeXt {split} labels/order do not match file_list.csv")
    return SplitFeatures(result["train"], result["val"], result["test"], expected["train"], expected["val"], expected["test"])


def fuse(dense: SplitFeatures, resnext: SplitFeatures) -> SplitFeatures:
    if not all(np.array_equal(getattr(dense, f"y_{split}"), getattr(resnext, f"y_{split}")) for split in ("train", "val", "test")):
        raise ValueError("Cannot fuse: DenseNet and ResNeXt labels are not aligned")
    # Each backbone is standardised independently, using training rows only.
    dense_scaler = StandardScaler().fit(dense.train)
    resnext_scaler = StandardScaler().fit(resnext.train)
    def joined(split: str) -> np.ndarray:
        return np.concatenate((dense_scaler.transform(getattr(dense, split)), resnext_scaler.transform(getattr(resnext, split))), axis=1)
    return SplitFeatures(joined("train"), joined("val"), joined("test"), dense.y_train, dense.y_val, dense.y_test)


def classifier(name: str, bagging_n_jobs: int = -1):
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "bagging":
        return BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED), n_estimators=100, random_state=SEED, n_jobs=bagging_n_jobs)
    raise ValueError(name)


def fit_predict(data: SplitFeatures, use_pca: bool, classifier_name: str, target: str, bagging_n_jobs: int):
    # Fusion is already block-standardised.  Single-model features need scaling.
    steps = [] if data.train.shape[1] == 3968 else [StandardScaler()]
    if use_pca:
        steps.append(PCA(n_components=0.95, svd_solver="full"))
    steps.append(classifier(classifier_name, bagging_n_jobs))
    model = make_pipeline(*steps)
    X_target = getattr(data, target)
    y_target = getattr(data, f"y_{target}")
    started = time.perf_counter()
    model.fit(data.train, data.y_train)
    predicted = model.predict(X_target)
    elapsed = time.perf_counter() - started
    pca_dim = next((step.n_components_ for step in model.named_steps.values() if isinstance(step, PCA)), None)
    return predicted, y_target, elapsed, pca_dim


def metric_rows(case: str, feature_case: str, classifier_name: str, split: str, truth: np.ndarray, predicted: np.ndarray, seconds: float, raw_dim: int, pca_dim: int | None) -> tuple[dict, pd.DataFrame, np.ndarray]:
    macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)
    summary = {"case": case, "feature_case": feature_case, "classifier": classifier_name, "split": split, "accuracy": accuracy_score(truth, predicted), "macro_precision": macro[0], "macro_recall": macro[1], "macro_f1": macro[2], "weighted_precision": weighted[0], "weighted_recall": weighted[1], "weighted_f1": weighted[2], "raw_feature_dim": raw_dim, "pca_feature_dim": pca_dim, "seconds": seconds}
    report = pd.DataFrame(classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)).transpose().reset_index(names="class")
    report.insert(0, "classifier", classifier_name)
    report.insert(0, "case", case)
    return summary, report, confusion_matrix(truth, predicted, labels=LABELS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-case", choices=["no_aug", "aug"], default="no_aug", help="Feature exports to assess. Run augmentation as a separate comparable experiment.")
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--classifiers", nargs="+", choices=["svm", "lda", "bagging"], default=["svm", "lda", "bagging"], help="Subset of classifiers to run; default runs all three.")
    parser.add_argument("--bagging-n-jobs", type=int, default=-1, help="Bagging worker count. Use 1 to perform a stable single-core verification.")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate exactly one already selected configuration on the held-out test split.")
    parser.add_argument("--test-case", choices=list("ABCDEF"))
    parser.add_argument("--test-classifier", choices=["svm", "lda", "bagging"])
    args = parser.parse_args()
    if args.evaluate_test != bool(args.test_case and args.test_classifier):
        parser.error("Test evaluation requires both --test-case and --test-classifier; otherwise omit all three test options.")

    expected = expected_labels()
    dense = load_densenet(args.feature_case, expected)
    resnext = load_resnext(args.feature_case, expected)
    configurations = {"A": (dense, False), "B": (dense, True), "C": (resnext, False), "D": (resnext, True), "E": (fuse(dense, resnext), False), "F": (fuse(dense, resnext), True)}
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics, reports = [], []
    for case, (data, use_pca) in configurations.items():
        for name in args.classifiers:
            predicted, truth, seconds, pca_dim = fit_predict(data, use_pca, name, "val", args.bagging_n_jobs)
            summary, report, _ = metric_rows(case, args.feature_case, name, "val", truth, predicted, seconds, data.train.shape[1], pca_dim)
            metrics.append(summary); reports.append(report)
    pd.DataFrame(metrics).to_csv(tables / f"stage4_{args.run_name}_{args.feature_case}_validation_metrics.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(tables / f"stage4_{args.run_name}_{args.feature_case}_validation_per_class.csv", index=False)
    with open(tables / f"stage4_{args.run_name}_{args.feature_case}_config.json", "w", encoding="utf-8") as handle:
        json.dump({"seed": SEED, "pca": "95% variance; fit on training features only", "classifiers": args.classifiers, "bagging_n_jobs": args.bagging_n_jobs, "test_used": args.evaluate_test, "cases": {"A": "DenseNet", "B": "DenseNet + PCA", "C": "ResNeXt", "D": "ResNeXt + PCA", "E": "fusion", "F": "fusion + PCA"}}, handle, indent=2)
    print(f"Validation results saved with run name {args.run_name}. Test set not used: {not args.evaluate_test}")
    if args.evaluate_test:
        data, use_pca = configurations[args.test_case]
        predicted, truth, seconds, pca_dim = fit_predict(data, use_pca, args.test_classifier, "test", args.bagging_n_jobs)
        summary, report, matrix = metric_rows(args.test_case, args.feature_case, args.test_classifier, "test", truth, predicted, seconds, data.train.shape[1], pca_dim)
        pd.DataFrame([summary]).to_csv(tables / f"stage4_{args.run_name}_FINAL_test_metrics.csv", index=False)
        report.to_csv(tables / f"stage4_{args.run_name}_FINAL_test_per_class.csv", index=False)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(tables / f"stage4_{args.run_name}_FINAL_test_confusion_matrix.csv")
        print("Final test result saved. Do not re-run this option for parameter selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
