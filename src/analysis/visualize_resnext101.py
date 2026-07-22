import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

FEATURE_PATH = "results/features/resnext101_aug_features.npy"
LABEL_PATH = "results/features/resnext101_aug_labels.npy"
OUT_DIR = "results/figures"

os.makedirs(OUT_DIR, exist_ok=True)

features = np.load(FEATURE_PATH)
labels = np.load(LABEL_PATH, allow_pickle=True)

print("features shape:", features.shape)
print("labels shape:", labels.shape)

np.random.seed(42)
max_points = 2000

if len(features) > max_points:
    idx = np.random.choice(len(features), max_points, replace=False)
    features_vis = features[idx]
    labels_vis = labels[idx]
else:
    features_vis = features
    labels_vis = labels

labels_vis = np.array([str(x) for x in labels_vis])
unique_labels = sorted(np.unique(labels_vis))

pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(features_vis)

plt.figure(figsize=(8, 6))
for lab in unique_labels:
    mask = labels_vis == lab
    plt.scatter(
        pca_result[mask, 0],
        pca_result[mask, 1],
        s=8,
        alpha=0.6,
        label=lab
    )

plt.title("ResNeXt-101 Feature Visualization using PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "resnext101_pca_visualization.png"), dpi=300)
plt.close()

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init="pca",
    learning_rate="auto"
)

tsne_result = tsne.fit_transform(features_vis)

plt.figure(figsize=(8, 6))
for lab in unique_labels:
    mask = labels_vis == lab
    plt.scatter(
        tsne_result[mask, 0],
        tsne_result[mask, 1],
        s=8,
        alpha=0.6,
        label=lab
    )

plt.title("ResNeXt-101 Feature Visualization using t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "resnext101_tsne_visualization.png"), dpi=300)
plt.close()

print("Saved:")
print("results/figures/resnext101_pca_visualization.png")
print("results/figures/resnext101_tsne_visualization.png")