# Fruit and Vegetable Freshness Assessment Using Frozen CNN Features and Classical Classifiers

## 1. Introduction

Freshness is an important quality attribute for fruit and vegetables, but manual inspection is subjective, slow, and difficult to scale. This project investigates whether visual features extracted by ImageNet-pretrained convolutional neural networks can classify produce as fresh or rotten. The study compares DenseNet-201 and ResNeXt-101 feature extractors and evaluates classical classifiers on the resulting high-dimensional representations.

The project has three objectives: establish a reproducible baseline, determine which CNN backbone produces the most useful features, and analyse whether classifier choice, PCA, and training augmentation improve validation performance. The final model is selected using validation Macro-F1 and evaluated once on the held-out test set.

## 2. Literature review

The experimental design follows the frozen-feature pipeline described by Yuan and Chen (2024), in which a pretrained CNN extracts image representations and a conventional classifier performs the final prediction. This design is suitable for a moderate-sized dataset because it avoids training a large network from scratch and makes the effect of feature representation easier to analyse.

DenseNet-201 uses dense connections so that later layers can reuse information from earlier layers. This encourages feature reuse and supports the representation of both local texture and higher-level appearance cues. ResNeXt-101 uses grouped convolutions and increased cardinality to learn diverse transformations with a strong computational trade-off. Comparing these architectures tests whether dense feature reuse or grouped feature transformations are more suitable for freshness cues.

SVM is a suitable classifier for high-dimensional CNN features because an RBF kernel can learn a nonlinear decision boundary while remaining effective when the number of features is large. LDA provides a fast linear discriminant baseline. Bagging tests whether an ensemble of decision trees can model nonlinear feature interactions, although it is expected to be less efficient on dense continuous feature vectors. PCA is used as a dimensionality-reduction ablation rather than assumed to improve every classifier.

## 3. Dataset and experimental setup

The dataset contains 12,000 readable images from 20 original classes: five fruits and five vegetables, each represented by fresh and rotten variants. For the main modelling task, the original labels are remapped to two classes, `fresh` and `rotten`. Images are converted to RGB, resized to 224 x 224, and normalized using ImageNet mean and standard deviation.

The dataset is split stratified by class into 8,399 training images, 1,801 validation images, and 1,800 test images using random seed 42. The same split is used across experiments. StandardScaler and PCA are fitted on training features only. Validation Macro-F1 is the primary model-selection metric, with Accuracy, Macro Precision, Macro Recall, and Weighted-F1 reported as supporting metrics. The test set is reserved for the final selected configuration.

The controlled backbone and classifier comparison uses the shared augmented feature case so that DenseNet-201, ResNeXt-101, Fusion, and the three classifiers use the same feature convention. A separate DenseNet-201 ablation compares no augmentation with training augmentation. The baseline is a frozen ImageNet-pretrained ResNet-18 followed by StandardScaler, PCA retaining 95% variance, and an RBF SVM.

## 4. Models and methods

The pipeline is:

`image -> frozen CNN backbone -> feature vector -> optional scaling/PCA -> classifier -> freshness label`

DenseNet-201 produces 1,920-dimensional features and ResNeXt-101 produces 2,048-dimensional features. Fusion concatenates standardized DenseNet and ResNeXt representations. PCA is fitted only on the training features and retains 95% of the variance. The classifiers use fixed settings: RBF SVM with `C=1.0` and `gamma=scale`, LDA with the SVD solver, and Bagging with 100 decision trees and seed 42.

The baseline provides a reference point for the proposed frozen-feature approach. The main model comparison then holds the classifier fixed to SVM while comparing DenseNet-201 and ResNeXt-101. Classifier analysis is performed on the DenseNet feature case. PCA and augmentation are evaluated through targeted ablations so that the final report remains focused rather than presenting every available experiment as an independent claim.

## 5. Results and analysis

### 5.1 Baseline

The frozen ResNet-18 baseline obtains validation Accuracy 0.9761 and Macro-F1 0.9761. This establishes that a compact pretrained backbone combined with a conventional classifier already provides a strong reference solution.

### 5.2 Backbone comparison

Using the shared augmented feature case and SVM, DenseNet-201 without PCA obtains validation Macro-F1 0.9783. ResNeXt-101 without PCA obtains 0.9494, while ResNeXt-101 with PCA obtains 0.9528. Therefore, DenseNet-201 produces more discriminative features for this task under the controlled comparison. The difference suggests that dense feature reuse may capture the colour and texture patterns associated with freshness more effectively than the ResNeXt representation in this dataset.

Feature Fusion with SVM obtains Macro-F1 0.9733 without PCA and 0.9739 with PCA. It does not outperform DenseNet-201, so Fusion is treated as a supplementary result rather than the central model claim.

### 5.3 Classifier comparison

For DenseNet-201 without PCA, SVM achieves Macro-F1 0.9783, LDA achieves 0.9772, and Bagging achieves 0.9405. SVM is therefore selected as the primary classifier. LDA is competitive but slightly weaker. Bagging is both less accurate and more computationally expensive, with a runtime of approximately 144 seconds compared with approximately 18 seconds for SVM in the corresponding validation experiment.

The results are consistent with the expected properties of the methods. CNN feature vectors are dense, continuous, and high-dimensional, which favours a kernel classifier. Tree ensembles can be effective for heterogeneous tabular data, but they are less effective here without extensive feature engineering or hyperparameter tuning.

### 5.4 PCA and augmentation ablations

PCA has a backbone-dependent effect. For DenseNet-201 with SVM, Macro-F1 changes from 0.9783 without PCA to 0.9772 with PCA, a decrease of 0.0011. For ResNeXt-101 with SVM, PCA changes Macro-F1 from 0.9494 to 0.9528, an improvement of 0.0033. PCA can therefore reduce the representation size and runtime, but it should not be described as universally performance-enhancing.

The DenseNet-201 augmentation ablation gives a clearer result. With PCA and SVM, the no-augmentation case reaches validation Macro-F1 0.9878, whereas the augmented case reaches 0.9772. In this dataset, the additional training transformations appear to introduce variation that is not useful for the frozen-feature classifier. The no-augmentation PCA-SVM configuration is consequently selected for final test evaluation.

### 5.5 Final test result and error analysis

The final selected configuration is DenseNet-201 with no augmentation, PCA retaining 95% variance, and an RBF SVM. It is evaluated once on the 1,800-image test split. The final results are Accuracy 0.9828, Macro Precision 0.9829, Macro Recall 0.9827, Macro-F1 0.9828, and Weighted-F1 0.9828.

The selected DenseNet configuration makes 22 errors out of 1,801 validation images. The error samples include visually borderline images, mild spots, local discoloration, uneven surface texture, lighting changes, watermarks, complex backgrounds, and multiple objects. Some rotten examples appear visually intact, while some fresh examples contain cues that can reasonably be interpreted as early decay. These observations indicate that the remaining errors are partly caused by visual ambiguity and possible label uncertainty rather than only by classifier failure.

## 6. Conclusions and limitations

The experiments show that frozen CNN features combined with a classical classifier provide an effective and reproducible freshness-classification pipeline. DenseNet-201 is the strongest backbone in the controlled comparison, and RBF SVM is the most reliable classifier for the extracted features. PCA is useful for reducing dimensionality and can improve ResNeXt performance, but its effect is model-dependent. Augmentation is not automatically beneficial; the no-augmentation DenseNet configuration performs best in the targeted ablation.

The main limitations are the binary remapping of the original 20 classes, the use of frozen ImageNet features, fixed classifier hyperparameters, and evaluation on one dataset. The results should therefore be interpreted as evidence for this experimental setting rather than as a universal ranking of CNN architectures. Future work should evaluate the original 20-class task, tune the CNN or classifier jointly, test on an external dataset, and investigate calibration or confidence-aware rejection for visually ambiguous samples.

## References

1. Yuan, L., and Chen, [full citation in `sampleArticle.pdf`].
2. Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *CVPR*.
3. Xie, S., Girshick, R., Dollar, P., Tu, Z., and He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks. *CVPR*.
4. Cortes, C., and Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273-297.
5. Jolliffe, I. T., and Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374.
