# DenseNet-201 Feature Extraction Analysis

## Scope

This handoff covers the frozen ImageNet-pretrained DenseNet-201 feature
extraction stage for the **fresh/rotten binary classification** task. No PCA,
classifier, or test-set classification metric was fitted in this stage, so
Accuracy, Precision, Recall, and F1 are **待运行** until the classification role
uses these files.

## Data and split

- Raw images checked: 12000.
- Readable images: 12000; corrupt images: 0.
- Original source classes: 20.
- After removing 5 exact duplicate paths with conflicting
  source labels, the shared split contains 11995 images.
- Split counts: train 8397; validation 1799; test 1799.
- Split policy: source-class stratification, seed 42, with exact duplicate byte
  groups kept within one split. No duplicate hash group crosses splits.
- Binary labels are derived as `fresh* -> fresh` and `rotten* -> rotten`.

## DenseNet-201 features

- Weights: `DenseNet201_Weights.DEFAULT` (ImageNet).
- Parameters frozen; `model.eval()` and `torch.no_grad()` used.
- Input: RGB, 224×224, ImageNet mean/std normalization.
- Global-average-pooled feature dimension: 1,920.
- Device: CPU.
- Files are in `results/features/no_aug/` and `results/features/aug/`, with
  rows matching `docs/file_list.csv` order within each split.

## Augmentation comparison

- `no_aug`: resize and normalization only.
- `aug`: one deterministic train-only view per training image using mild
  RandomResizedCrop, HorizontalFlip, Rotation, and ColorJitter.
- Validation and test features are reused from the clean evaluation pipeline.
- Validation and test features are identical between cases: `True` / `True`.
- Training features differ as expected: maximum absolute difference
  `11.895671`, mean per-image L2 difference
  `15.314991`.

## Interpretation and limitations

The feature files are suitable for the next role to fit StandardScaler/PCA on
training features only and then train SVM, LDA, and Bagging. The augmentation
comparison isolates the effect of changing training representations while
keeping validation and test preprocessing fixed. This stage does not establish
which case or classifier is best; that requires validation-based selection and
one final test evaluation. The experiment was CPU-only, so the reported
extraction time is hardware-specific.
