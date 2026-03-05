#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ------------------------------------------------------------------
#  ADVANCED HYBRID FACE RECOGNITION USING PCA + ANN
#  Author: Manoj S (ECE, JIT Davangere)
#  Dataset: AT&T ORL Face Database (40 persons × 10 images each)
# ------------------------------------------------------------------

# 🧩 Prerequisites:
# pip install opencv-python scikit-learn matplotlib numpy

import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------------
# 1️⃣ Data Loading
# ------------------------------------------------------------------
def load_images(dataset_path, image_size=(100, 100)):
    """
    Reads every image in each subfolder of the dataset path.
    Each folder = one person's face class.
    Returns flattened grayscale image data + labels.
    """
    images, labels = [], []
    label_map, current_label = {}, 0

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        label_map[current_label] = folder
        # Auto-detect image formats (.pgm, .jpg, .png)
        for file in glob(os.path.join(folder_path, "*.*")):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Warning] could not read: {file}")
                continue

            img = cv2.resize(img, image_size)
            images.append(img.flatten())
            labels.append(current_label)
        current_label += 1

    X = np.array(images).T   # shape: (d, n)
    labels = np.array(labels)
    return X, labels, label_map


# ------------------------------------------------------------------
# 2️⃣ PCA Feature Extraction
# ------------------------------------------------------------------
def pca(X, k):
    """
    Computes top-k eigenfaces using PCA.
    Returns: mean_face, eigfaces, X_centered
    """
    mean_face = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_face

    # Compact PCA trick
    cov_small = X_centered.T @ X_centered
    eigvals, eigvecs_small = np.linalg.eigh(cov_small)
    eigvecs = X_centered @ eigvecs_small
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

    idx = np.argsort(eigvals)[::-1][:k]
    eigfaces = eigvecs[:, idx]
    return mean_face, eigfaces, X_centered


# ------------------------------------------------------------------
# 3️⃣ Generate Face Signatures
# ------------------------------------------------------------------
def generate_signatures(eigfaces, X_centered):
    return eigfaces.T @ X_centered   # shape: (k, n)


# ------------------------------------------------------------------
# 4️⃣ ANN Training (Tuned for High Accuracy)
# ------------------------------------------------------------------
def train_ann(signatures, labels):
    """
    Trains an optimized ANN classifier using the PCA features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        signatures.T, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Normalize feature space
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tuned ANN for better convergence
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc


# ------------------------------------------------------------------
# 5️⃣ Accuracy vs. k Plot
# ------------------------------------------------------------------
def plot_accuracy_vs_k(X, labels, k_list):
    accs = []
    for k in k_list:
        _, eigfaces, X_c = pca(X, k)
        sigs = generate_signatures(eigfaces, X_c)
        _, acc = train_ann(sigs, labels)
        accs.append(acc)
        print(f"k = {k}, accuracy = {acc:.2%}")

    plt.figure(figsize=(7, 4))
    plt.plot(k_list, accs, marker='o', linewidth=2)
    plt.xlabel('k (Number of Eigenfaces)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Number of Principal Components')
    plt.grid(True)
    plt.savefig("results/accuracy_vs_k.png")
    plt.show()


# ------------------------------------------------------------------
# 6️⃣ MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = r"E:\att_faces"

    # Step A: Load Dataset
    X, labels, label_map = load_images(dataset_path)
    print("✅ Dataset Loaded Successfully!")
    print("X shape       :", X.shape)
    print("Labels shape  :", labels.shape)
    print("Classes found :", len(label_map))

    # Step B: PCA + ANN
    k = 60  # optimal number of components
    mean_face, eigfaces, X_centered = pca(X, k)
    signatures = generate_signatures(eigfaces, X_centered)
    model, acc = train_ann(signatures, labels)

    print(f"\n🎯 Initial accuracy with k = {k}: {acc:.2%}")

    # Step C: Evaluate across k values
    k_vals = list(range(10, 110, 10))
    plot_accuracy_vs_k(X, labels, k_vals)


# In[ ]:




