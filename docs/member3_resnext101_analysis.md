# Member 3: ResNeXt-101 Result Analysis

## ResNeXt-101 Cases

According to the Stage 4 configuration:

- Case C: ResNeXt-101 features without PCA
- Case D: ResNeXt-101 features with PCA

The PCA setting keeps 95% of the variance and is fitted only on the training features.

## Overall Result

For ResNeXt-101, SVM achieved the best performance among the three classifiers.

| Case             | Classifier | Accuracy | Macro F1 |
| ---------------- | ---------- | -------: | -------: |
| C: ResNeXt       | SVM        |   0.9495 |   0.9494 |
| C: ResNeXt       | LDA        |   0.9395 |   0.9393 |
| C: ResNeXt       | Bagging    |   0.9273 |   0.9272 |
| D: ResNeXt + PCA | SVM        |   0.9528 |   0.9528 |
| D: ResNeXt + PCA | LDA        |   0.9311 |   0.9310 |
| D: ResNeXt + PCA | Bagging    |   0.8901 |   0.8900 |

## PCA Effect

Applying PCA slightly improved the SVM result. The accuracy increased from 94.95% in Case C to 95.28% in Case D.

This suggests that PCA helped remove redundant information from the ResNeXt-101 features while keeping useful discriminative information.

However, PCA reduced the performance of LDA and Bagging. This means that not all classifiers benefited from dimensionality reduction. In this experiment, the best ResNeXt-101 setting was:

**ResNeXt-101 + PCA + SVM**

## Fresh vs Rotten Performance

For Case D with SVM:

- Fresh F1-score: 0.9540
- Rotten F1-score: 0.9515

Both classes achieved similar performance, which shows that the model was relatively balanced between fresh and rotten classification.

## Visualization

Two feature visualizations were generated:

- `results/figures/resnext101_pca_visualization.png`
- `results/figures/resnext101_tsne_visualization.png`

These figures show how ResNeXt-101 features are distributed in a lower-dimensional space.
