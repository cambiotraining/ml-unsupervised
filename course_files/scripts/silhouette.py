'''
Code for silhouette score calculation.
'''

# Silhouette Plots for Unsupervised Learning
# Simple tutorial for biologists
# Install required packages: pip install scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# ============================================
# CREATE EXAMPLE DATA (like gene expression)
# ============================================
# Imagine this is gene expression data for 150 samples and 2 genes
X, true_labels = make_blobs(n_samples=150, n_features=2, 
                            centers=3, cluster_std=0.5, 
                            random_state=42)

# ============================================
# PART 1: K-MEANS WITH SILHOUETTE PLOT
# ============================================
print("=" * 50)
print("K-MEANS CLUSTERING")
print("=" * 50)

# Try different numbers of clusters (k=2, 3, 4)
for n_clusters in [2, 3, 4]:
    
    # Step 1: Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Step 2: Calculate silhouette scores
    # Overall score (average for all samples)
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    # Individual scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    print(f"\nFor k={n_clusters} clusters:")
    print(f"Average silhouette score: {silhouette_avg:.3f}")
    print("(Values close to 1 = good clustering, close to 0 = overlapping clusters)")
    
    # Step 3: Create silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Silhouette plot
    y_lower = 10
    for i in range(n_clusters):
        # Get silhouette scores for samples in cluster i
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f"Average = {silhouette_avg:.2f}")
    ax1.set_title(f"Silhouette Plot (k={n_clusters})")
    ax1.legend()
    
    # Plot 2: Actual clusters
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1],
                marker='X', s=200, c='red', edgecolors='black', 
                label='Centroids')
    ax2.set_xlabel("Feature 1 (e.g., Gene 1 expression)")
    ax2.set_ylabel("Feature 2 (e.g., Gene 2 expression)")
    ax2.set_title(f"K-means Clusters (k={n_clusters})")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'kmeans_silhouette_k{n_clusters}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================
# PART 2: HIERARCHICAL CLUSTERING WITH SILHOUETTE
# ============================================
print("\n" + "=" * 50)
print("HIERARCHICAL CLUSTERING")
print("=" * 50)

# Try different linkage methods
linkage_methods = ['ward', 'complete', 'average']

for linkage in linkage_methods:
    
    # Step 1: Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    cluster_labels = hierarchical.fit_predict(X)
    
    # Step 2: Calculate silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    print(f"\nLinkage: {linkage}")
    print(f"Average silhouette score: {silhouette_avg:.3f}")
    
    # Step 3: Create silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Silhouette plot
    y_lower = 10
    for i in range(3):
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / 3)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f"Average = {silhouette_avg:.2f}")
    ax1.set_title(f"Silhouette Plot ({linkage} linkage)")
    ax1.legend()
    
    # Plot 2: Actual clusters
    colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
    ax2.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7)
    ax2.set_xlabel("Feature 1 (e.g., Gene 1 expression)")
    ax2.set_ylabel("Feature 2 (e.g., Gene 2 expression)")
    ax2.set_title(f"Hierarchical Clusters ({linkage})")
    
    plt.tight_layout()
    plt.savefig(f'hierarchical_silhouette_{linkage}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================
# KEY INTERPRETATION GUIDE
# ============================================
print("\n" + "=" * 50)
print("HOW TO INTERPRET SILHOUETTE PLOTS")
print("=" * 50)
print("""
1. Silhouette Score Range: -1 to +1
   - Close to +1: Sample is well-matched to its cluster
   - Close to 0: Sample is on the border between clusters
   - Negative: Sample might be in the wrong cluster

2. In the Silhouette Plot:
   - Width of each colored section = number of samples in that cluster
   - Length of bars = how well samples fit in their cluster
   - Red dashed line = average score across all samples

3. Good Clustering Signs:
   - All clusters have scores above the average (red line)
   - Clusters have similar widths (balanced sizes)
   - Few or no negative values

4. Poor Clustering Signs:
   - Many samples below the average
   - Very different cluster widths (imbalanced)
   - Negative values present

5. Choosing k (number of clusters):
   - Higher average silhouette score is better
   - But also consider biological meaning!
""")
# End of silhouette.py