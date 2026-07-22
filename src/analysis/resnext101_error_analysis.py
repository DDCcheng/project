import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

FEATURE_PATH = "results/features/resnext101_aug_features.npy"
LABEL_PATH = "results/features/resnext101_aug_labels.npy"
SPLIT_PATH = "results/features/resnext101_aug_splits.npy"
FILE_LIST_PATH = "docs/file_list.csv"

DATA_DIR = "/Users/yangshiyu/.cache/kagglehub/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset/versions/2"

OUT_FIG = "results/figures/resnext101_confusion_matrix.png"
OUT_ERR = "results/tables/resnext101_error_samples.csv"

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


def convert_to_local_path(path):
    path = str(path).replace("\\", "/")

    marker = "Fruits_Vegetables_Dataset(12000)/"
    if marker in path:
        relative_path = path.split(marker, 1)[1]
        return os.path.join(DATA_DIR, relative_path)

    if os.path.isabs(path):
        return path

    return os.path.join(DATA_DIR, path)


features = np.load(FEATURE_PATH)
labels_original = np.load(LABEL_PATH, allow_pickle=True)
splits = np.load(SPLIT_PATH, allow_pickle=True)

print("features:", features.shape)
print("labels:", labels_original.shape)
print("splits:", splits.shape)

# Convert 20-class labels into binary fresh/rotten labels
labels_binary = np.array([
    "fresh" if str(label).startswith("fresh") else "rotten"
    for label in labels_original
])

train_mask = splits == "train"
val_mask = splits == "val"

X_train = features[train_mask]
y_train = labels_binary[train_mask]

X_val = features[val_mask]
y_val = labels_binary[val_mask]
original_val_labels = labels_original[val_mask]

# Case D: ResNeXt + PCA + SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print("PCA dimension:", X_train_pca.shape[1])

# Match Stage 4 SVM setting
clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_val_pca)

labels_order = ["fresh", "rotten"]
cm = confusion_matrix(y_val, y_pred, labels=labels_order)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("ResNeXt-101 + PCA + SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(labels_order)), labels_order)
plt.yticks(range(len(labels_order)), labels_order)

for i in range(len(labels_order)):
    for j in range(len(labels_order)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

errors = pd.DataFrame({
    "original_class": original_val_labels,
    "true_label": y_val,
    "predicted_label": y_pred
})

# Add filepath if available
if os.path.exists(FILE_LIST_PATH):
    file_list = pd.read_csv(FILE_LIST_PATH)
    if "filepath" in file_list.columns:
        val_filepaths = file_list.loc[val_mask, "filepath"].values
        errors["filepath"] = val_filepaths

errors = errors[errors["true_label"] != errors["predicted_label"]]
errors.to_csv(OUT_ERR, index=False)

accuracy = (y_pred == y_val).mean()

print("Saved:")
print(OUT_FIG)
print(OUT_ERR)
print("Validation accuracy:", accuracy)
print("Number of validation errors:", len(errors))
print("Confusion matrix:")
print(cm)

# 20-class error distribution based on original labels
error_distribution = (
    errors["original_class"]
    .value_counts()
    .reset_index()
)

error_distribution.columns = ["original_class", "num_errors"]

OUT_ERR_DIST = "results/tables/resnext101_20class_error_distribution.csv"
error_distribution.to_csv(OUT_ERR_DIST, index=False)

print("20-class error distribution saved:")
print(OUT_ERR_DIST)

# Additional 20-class confusion matrix using original class labels
OUT_20CM_FIG = "results/figures/resnext101_20class_confusion_matrix.png"
OUT_20CM_CSV = "results/tables/resnext101_20class_confusion_matrix.csv"

y_train_20 = labels_original[train_mask]
y_val_20 = labels_original[val_mask]

clf_20 = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
clf_20.fit(X_train_pca, y_train_20)

y_pred_20 = clf_20.predict(X_val_pca)

labels_20_order = sorted(np.unique(labels_original))
cm_20 = confusion_matrix(y_val_20, y_pred_20, labels=labels_20_order)

pd.DataFrame(cm_20, index=labels_20_order, columns=labels_20_order).to_csv(OUT_20CM_CSV)

plt.figure(figsize=(14, 12))
plt.imshow(cm_20)
plt.title("ResNeXt-101 + PCA + SVM 20-Class Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(range(len(labels_20_order)), labels_20_order, rotation=90, fontsize=7)
plt.yticks(range(len(labels_20_order)), labels_20_order, fontsize=7)

for i in range(len(labels_20_order)):
    for j in range(len(labels_20_order)):
        if cm_20[i, j] > 0:
            plt.text(j, i, cm_20[i, j], ha="center", va="center", fontsize=6)

plt.colorbar()
plt.tight_layout()
plt.savefig(OUT_20CM_FIG, dpi=300)
plt.close()

accuracy_20 = (y_pred_20 == y_val_20).mean()

print("20-class confusion matrix saved:")
print(OUT_20CM_FIG)
print(OUT_20CM_CSV)
print("20-class validation accuracy:", accuracy_20)

# Error sample image grid
from PIL import Image

OUT_ERROR_GRID = "results/figures/resnext101_error_samples_grid.png"

# Build an index from local dataset image filenames to real local paths
image_index = {}
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mpo")):
            image_index[file] = os.path.join(root, file)

print("Indexed local images:", len(image_index))


def find_local_image_path(path):
    path = str(path).replace("\\", "/")
    filename = os.path.basename(path)

    if filename in image_index:
        return image_index[filename]

    return None


if "filepath" in errors.columns and len(errors) > 0:
    sample_errors = errors.head(12)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    found_count = 0

    for ax, (_, row) in zip(axes, sample_errors.iterrows()):
        image_path = find_local_image_path(row["filepath"])

        if image_path is not None and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(
                f"True: {row['true_label']}\nPred: {row['predicted_label']}\nClass: {row['original_class']}",
                fontsize=8
            )
            ax.axis("off")
            found_count += 1
        else:
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax.axis("off")

    for ax in axes[len(sample_errors):]:
        ax.axis("off")

    plt.suptitle("ResNeXt-101 Error Sample Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_ERROR_GRID, dpi=300)
    plt.close()

    print("Error sample image grid saved:")
    print(OUT_ERROR_GRID)
    print("Found error images:", found_count, "/", len(sample_errors))
else:
    print("No filepath column found in error samples. Error image grid was not generated.")