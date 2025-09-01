import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Figure 6: t-SNE vs PCA Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Create sample data with clear clusters in high dimensions
np.random.seed(42)

# Generate 3 distinct clusters in high-dimensional space
n_samples_per_cluster = 20
n_features = 50

# Cluster 1: High in first half of features
cluster1 = np.random.normal(0, 0.5, (n_samples_per_cluster, n_features))
cluster1[:, :25] += 3  # Boost first half

# Cluster 2: High in second half of features  
cluster2 = np.random.normal(0, 0.5, (n_samples_per_cluster, n_features))
cluster2[:, 25:] += 3  # Boost second half

# Cluster 3: Moderate in all features
cluster3 = np.random.normal(1.5, 0.3, (n_samples_per_cluster, n_features))

# Combine data
X = np.vstack([cluster1, cluster2, cluster3])
colors = ['#e74c3c'] * n_samples_per_cluster + \
         ['#3498db'] * n_samples_per_cluster + \
         ['#2ecc71'] * n_samples_per_cluster
labels = ['Type A'] * n_samples_per_cluster + \
         ['Type B'] * n_samples_per_cluster + \
         ['Type C'] * n_samples_per_cluster

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Plot PCA results
for i, (color, label) in enumerate(zip(['#e74c3c', '#3498db', '#2ecc71'], 
                                     ['Type A', 'Type B', 'Type C'])):
    start_idx = i * n_samples_per_cluster
    end_idx = (i + 1) * n_samples_per_cluster
    ax1.scatter(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], 
               c=color, label=label, alpha=0.7, s=50)

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax1.set_title('PCA: Linear Dimensionality Reduction\n' +
             'Preserves global structure & variance', fontsize=12, pad=15)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot t-SNE results  
for i, (color, label) in enumerate(zip(['#e74c3c', '#3498db', '#2ecc71'], 
                                     ['Type A', 'Type B', 'Type C'])):
    start_idx = i * n_samples_per_cluster
    end_idx = (i + 1) * n_samples_per_cluster
    ax2.scatter(X_tsne[start_idx:end_idx, 0], X_tsne[start_idx:end_idx, 1], 
               c=color, label=label, alpha=0.7, s=50)

ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.set_title('t-SNE: Non-linear Dimensionality Reduction\n' +
             'Preserves local neighborhoods & reveals clusters', fontsize=12, pad=15)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add comparison table below
fig.text(0.5, 0.02, 
"""
📊 PCA vs t-SNE Comparison:

                    PCA                                          t-SNE
🎯 Goal:           Preserve global variance                    Preserve local neighborhoods  
📈 Method:         Linear projection                          Non-linear embedding
⏱️ Speed:          Fast                                        Slower
🔍 Best for:       • Understanding data variance              • Finding hidden clusters
                   • Preprocessing for other methods          • Exploratory data analysis
                   • When global structure matters            • Visualizing complex patterns
❌ Limitations:    • May miss non-linear patterns             • Doesn't preserve global distances
                   • Clusters may overlap                     • Can create false clusters
                                                             • Sensitive to parameters
""", 
ha='center', va='bottom', fontsize=10, 
bbox=dict(boxstyle="round,pad=0.8", facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.show()

# Save the figure
plt.savefig('tsne_step6_comparison_with_pca.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_step6_comparison_with_pca.pdf', bbox_inches='tight')