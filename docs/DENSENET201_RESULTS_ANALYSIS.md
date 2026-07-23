# DenseNet-201 Augmentation Comparison and Validation Analysis

## Scope

This analysis compares the DenseNet-201 `no_aug` and `aug` feature cases for binary `fresh/rotten` classification.
StandardScaler and PCA are fit on train only; classifiers are trained on train and all metrics are computed on validation.
The test split is not loaded or evaluated in this stage, so these are not final test results.

Fixed settings: seed=42; PCA retains 95% variance; SVM uses RBF, C=1.0, and gamma=scale;
LDA uses the svd solver; Bagging uses 100 decision trees with fixed random seeds.

## Metric Comparison

Complete results are in `results/tables/densenet201_augmentation_metrics_val.csv`. Macro-F1 is the primary selection metric,
followed by Accuracy and Weighted-F1.

| case | classifier | pca_target | pca_components | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | fit_seconds | predict_seconds | total_seconds | n_train | n_val | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_aug | svm | 0.95_variance | 690 | 0.9878 | 0.9878 | 0.9878 | 0.9878 | 0.9878 | 4.7478 | 1.6841 | 6.4319 | 8399 | 1801 | 42 |
| no_aug | lda | 0.95_variance | 690 | 0.9700 | 0.9704 | 0.9698 | 0.9700 | 0.9700 | 0.5254 | 0.0025 | 0.5279 | 8399 | 1801 | 42 |
| no_aug | bagging | 0.95_variance | 690 | 0.9439 | 0.9439 | 0.9439 | 0.9439 | 0.9439 | 61.4784 | 0.4619 | 61.9402 | 8399 | 1801 | 42 |
| aug | svm | 0.95_variance | 738 | 0.9772 | 0.9773 | 0.9772 | 0.9772 | 0.9772 | 5.9411 | 2.2466 | 8.1877 | 8399 | 1801 | 42 |
| aug | lda | 0.95_variance | 738 | 0.9678 | 0.9683 | 0.9675 | 0.9678 | 0.9678 | 0.5975 | 0.0028 | 0.6003 | 8399 | 1801 | 42 |
| aug | bagging | 0.95_variance | 738 | 0.9162 | 0.9172 | 0.9157 | 0.9160 | 0.9161 | 48.8231 | 0.4644 | 49.2875 | 8399 | 1801 | 42 |

## Best Validation Configuration

Ranking by Macro-F1, Accuracy, and Weighted-F1, the best configuration is **no_aug + svm**.
Macro-F1=0.9878, Accuracy=0.9878,
and the resulting PCA dimension is 690.
This selection is based only on validation and does not replace the final test evaluation.

## Augmentation Effect

- **svm**: relative to no_aug, aug changes Macro-F1 by -0.0106, Accuracy by -0.0105, and Weighted-F1 by -0.0106.
- **lda**: relative to no_aug, aug changes Macro-F1 by -0.0022, Accuracy by -0.0022, and Weighted-F1 by -0.0022.
- **bagging**: relative to no_aug, aug changes Macro-F1 by -0.0279, Accuracy by -0.0278, and Weighted-F1 by -0.0278.

## Confusion Matrices and Error Samples

Confusion-matrix figures are in `results/figures/`; the combined figure is `densenet201_augmentation_confusion_val.png`.
Complete error lists are in `results/tables/`; visual error samples are in `results/figures/error_samples/`.
Each error table contains the filepath, original 20-class label, binary true label, predicted label, and confidence or decision-margin information.

- `no_aug / svm`: 22/1801 errors; fresh→rotten=10, rotten→fresh=12.
- `no_aug / lda`: 54/1801 errors; fresh→rotten=17, rotten→fresh=37.
- `no_aug / bagging`: 101/1801 errors; fresh→rotten=49, rotten→fresh=52.
- `aug / svm`: 41/1801 errors; fresh→rotten=18, rotten→fresh=23.
- `aug / lda`: 58/1801 errors; fresh→rotten=18, rotten→fresh=40.
- `aug / bagging`: 151/1801 errors; fresh→rotten=56, rotten→fresh=95.

Image-level causes should be reviewed against the corresponding error-sample figures before being included in the final report; this document records only conclusions directly supported by the predictions.

## Visual Observations of Error Samples

A visual review of the highest-uncertainty error samples across configurations shows the following patterns:

- Some `rotten → fresh` samples appear visually intact and normally colored, with no prominent decay region. This suggests borderline examples or label noise.
- Some `fresh → rotten` samples contain mild spots, local discoloration, uneven surface texture, or darker lighting, which may be interpreted as decay cues.
- Some images contain watermarks, complex backgrounds, multiple objects, varied camera distances, or strong lighting changes, increasing freshness-classification difficulty.
- These observations explain model errors but do not justify relabeling the dataset; the corresponding error CSVs remain the source of truth for paths and predictions.

## Reproducibility and Limitations

Both feature cases come from the same `docs/file_list.csv`; train/validation row order and labels are validated by the script.
Augmentation is applied only to the aug training features; the validation features are identical to the no_aug validation features.
This stage uses fixed classifier parameters, performs no hyperparameter search, and does not use test for model selection.
