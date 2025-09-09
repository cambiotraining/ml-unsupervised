"""
Quick Reference: sns.clustermap Examples for Biologists
======================================================

Essential code examples for hierarchical clustering with different parameters.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create sample biological data (gene expression)
np.random.seed(42)
genes = [f'Gene_{chr(65+i)}' for i in range(10)]
samples = [f'Sample_{i}' for i in range(1, 16)]
data = np.random.normal(0, 1, (10, 15))

# Add some structure
data[0:3, 0:5] += 2    # Group 1: High in samples 1-5
data[3:6, 5:10] += 1.5 # Group 2: High in samples 6-10
data[6:9, 10:15] -= 1.5 # Group 3: Low in samples 11-15

df = pd.DataFrame(data, index=genes, columns=samples)

# =============================================================================
# EXAMPLE 1: Basic clustermap with scaling
# =============================================================================
print("Example 1: Basic clustermap with scaling")
sns.clustermap(df, 
               standard_scale=0,  # Scale rows (genes)
               cmap='RdBu_r',
               center=0,
               figsize=(10, 8))
plt.title('Basic Clustermap with Scaling')
plt.show()

# =============================================================================
# EXAMPLE 2: Different linkage methods
# =============================================================================
print("\nExample 2: Different linkage methods")

linkage_methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for i, method in enumerate(linkage_methods):
    plt.subplot(2, 2, i+1)
    sns.clustermap(df,
                   method=method,
                   standard_scale=0,
                   cmap='RdBu_r',
                   center=0,
                   figsize=(6, 5))
    plt.title(f'Linkage: {method.title()}')

plt.tight_layout()
plt.show()

# =============================================================================
# EXAMPLE 3: Different distance metrics
# =============================================================================
print("\nExample 3: Different distance metrics")

metrics = ['euclidean', 'correlation', 'cosine', 'cityblock']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.clustermap(df,
                   metric=metric,
                   method='average',
                   standard_scale=0,
                   cmap='RdBu_r',
                   center=0,
                   figsize=(6, 5))
    plt.title(f'Distance: {metric.title()}')

plt.tight_layout()
plt.show()

# =============================================================================
# EXAMPLE 4: With vs Without scaling comparison
# =============================================================================
print("\nExample 4: With vs Without scaling")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Without scaling
plt.subplot(1, 2, 1)
sns.clustermap(df, cmap='RdBu_r', center=0, figsize=(6, 5))
plt.title('Without Scaling')

# With scaling
plt.subplot(1, 2, 2)
sns.clustermap(df, standard_scale=0, cmap='RdBu_r', center=0, figsize=(6, 5))
plt.title('With Scaling (standard_scale=0)')

plt.tight_layout()
plt.show()

# =============================================================================
# EXAMPLE 5: Comprehensive example with all parameters
# =============================================================================
print("\nExample 5: Comprehensive example")

g = sns.clustermap(df,
                   # Clustering parameters
                   method='average',           # Linkage method
                   metric='correlation',       # Distance metric
                   standard_scale=0,          # Scale rows (genes)
                   
                   # Visual parameters
                   cmap='RdBu_r',             # Color map
                   center=0,                  # Center color map
                   figsize=(12, 8),           # Figure size
                   
                   # Clustering options
                   row_cluster=True,          # Cluster rows
                   col_cluster=True,          # Cluster columns
                   
                   # Color bar
                   cbar_kws={'label': 'Standardized Expression',
                            'shrink': 0.8})

plt.title('Comprehensive Clustermap Example')
plt.show()

# =============================================================================
# PARAMETER REFERENCE GUIDE
# =============================================================================
print("""
PARAMETER REFERENCE GUIDE:
=========================

LINKAGE METHODS:
- 'single': Minimum distance (sensitive to outliers)
- 'complete': Maximum distance (robust to outliers)  
- 'average': Average distance (balanced approach)
- 'ward': Minimizes within-cluster variance (requires euclidean distance)

DISTANCE METRICS:
- 'euclidean': Absolute difference (good for magnitude)
- 'correlation': Correlation-based (good for patterns)
- 'cosine': Angle between vectors (ignores magnitude)
- 'cityblock': Sum of absolute differences (robust)

SCALING OPTIONS:
- standard_scale=0: Scale rows (genes) - most common for gene expression
- standard_scale=1: Scale columns (samples)
- standard_scale=None: No scaling (raw data)

VISUAL PARAMETERS:
- cmap: Color map ('RdBu_r', 'viridis', 'coolwarm', etc.)
- center: Center color map at this value (usually 0)
- figsize: Figure size (width, height)
- row_cluster/col_cluster: Whether to cluster rows/columns

BIOLOGICAL INTERPRETATION:
- Red regions: High expression
- Blue regions: Low expression
- Dendrograms show clustering relationships
- Look for biological patterns and functional groups
""")
