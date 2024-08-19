# Machine Learning Exercise 4: K-Means and PCA

**Author:** Guy Reuveni

**Course:** Machine Learning

## Overview

This project, completed as part of a Machine Learning course, implements two important unsupervised machine learning algorithms: **K-Means Clustering** and **Principal Component Analysis (PCA)**. The notebook is divided into two parts:

1. **K-Means Clustering:** Implementation and experimentation on a synthetic 2D dataset.
2. **PCA:** Implementation and application on the MNIST dataset.

## Part 1: K-Means Clustering

### Description

In this section, the K-Means algorithm is implemented using only the `numpy` package. The algorithm iteratively assigns data points to the nearest cluster centroids and recalculates the centroids until convergence.

### Steps:

1. **Data Generation:** A synthetic 2D dataset is generated with multiple clusters.
2. **Implementation:** The K-Means algorithm is implemented in Python.
3. **Experimentation:** The algorithm is run with different numbers of clusters, and the final costs for each experiment are recorded.
4. **Visualization:** The clusters and centroids are plotted after each iteration to visualize the clustering process.

### Key Functions:

- `fit(X)`: Fits the K-Means model to the dataset `X`.
- `predict(X)`: Predicts the cluster labels for a given dataset `X`.
- `run_and_plot_kmeans(X, n_clusters)`: Runs K-Means and plots the clusters and centroids at each iteration.

### Results

The results include plots of the costs over iterations and visualizations of the clusters and centroids for different numbers of clusters.

## Part 2: Principal Component Analysis (PCA)

### Description

This section implements the PCA algorithm to reduce the dimensionality of the MNIST dataset, which consists of images of handwritten digits. The algorithm is implemented using `numpy` and involves calculating the covariance matrix, eigenvectors, and eigenvalues.

### Steps:

1. **Data Loading:** The MNIST dataset is loaded from `sklearn.datasets`.
2. **Implementation:** The PCA algorithm is implemented to reduce the dimensionality of the dataset.
3. **Visualization:** The dataset is projected into a 2D space using the top two principal components, and the result is visualized.
4. **Reconstruction:** Images from the dataset are reconstructed from reduced dimensions, and the quality of the reconstructions is analyzed.

### Key Functions:

- `fit(X)`: Computes the principal components for the dataset `X`.
- `transform(X, n_dimensions)`: Transforms the dataset `X` into a lower-dimensional space.
- `reconstruct(X, n_dimensions)`: Reconstructs the dataset `X` from its lower-dimensional representation.
- `pca_reconstruction(X, n_dimensions)`: Projects a sample to a lower-dimensional space and reconstructs it to the original space.

### Results

The results include visualizations of the MNIST data in the top two principal components and comparisons of original and reconstructed images using different numbers of dimensions.

## Conclusion

This exercise, completed as part of a Machine Learning course, demonstrates the effectiveness of K-Means for clustering and PCA for dimensionality reduction. It provides hands-on experience in implementing these algorithms from scratch using `numpy` and applying them to real-world datasets.
