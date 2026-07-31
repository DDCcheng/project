# ResNet-18 Baseline Analysis

## Baseline Definition

The baseline is a frozen ImageNet-pretrained ResNet-18 feature extractor with no data augmentation,
followed by StandardScaler, PCA retaining 95% of the variance, and an RBF SVM classifier.
The official `docs/file_list.csv` split and seed 42 are used throughout.
Only train and validation data are used in this stage; test features are not loaded.

## Validation Result

| model | case | classifier | pca_target | pca_components | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | fit_seconds | predict_seconds | total_seconds | n_train | n_val | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | no_aug_baseline | svm | 0.95_variance | 245 | 0.9761 | 0.9762 | 0.9761 | 0.9761 | 0.9761 | 1.3051 | 0.7899 | 2.0950 | 8399 | 1801 | 42 |

The baseline achieved Accuracy=0.9761, Macro-F1=0.9761, and Weighted-F1=0.9761 on validation. PCA produced 245 components from the 512-dimensional ResNet-18 features.

## Comparison with DenseNet-201

Compared with the existing DenseNet-201 no_aug + SVM result, the baseline Macro-F1 changes by -0.0117 and Accuracy changes by -0.0117.

## Confusion Matrix and Errors

The validation confusion matrix is `[[899, 19], [24, 859]]` using row order `fresh, rotten`. The baseline made 43 validation errors out of 1801 samples.
The complete error list is stored in `results/tables/resnet18_baseline_val_errors.csv`, and the visual error sheet is stored in `results/figures/error_samples/resnet18_baseline_error_samples_val.png`.

## Reproducibility and Limitations

The feature extractor is frozen and uses the default ImageNet weights. StandardScaler and PCA are fit on train only.
The reported result is a validation baseline, not a final test score. No hyperparameter search or test-set model selection was performed.
