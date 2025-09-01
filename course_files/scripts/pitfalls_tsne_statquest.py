import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Figure 7: Common Pitfalls and Best Practices
fig = plt.figure(figsize=(16, 12))

# Create a 2x3 grid for different examples
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

# Example 1: Good t-SNE (top left)
ax1 = fig.add_subplot(gs[0, 0])
np.random.seed(42)

# Well-separated clusters
cluster1 = np.random.normal([2, 2], 0.3, (30, 2))
cluster2 = np.random.normal([6, 2], 0.3, (30, 2))  
cluster3 = np.random.normal([4, 5], 0.3, (30, 2))

ax1.scatter(cluster1[:, 0], cluster1[:, 1], c='#e74c3c', alpha=0.7, s=40)
ax1.scatter(cluster2[:, 0], cluster2[:, 1], c='#3498db', alpha=0.7, s=40)
ax1.scatter(cluster3[:, 0], cluster3[:, 1], c='#2ecc71', alpha=0.7, s=40)

ax1.set_title('✅ GOOD t-SNE\nClear, well-separated clusters', 
             fontsize=12, color='green', weight='bold')
ax1.set_xticks([])
ax1.set_yticks([])

# Example 2: Bad perplexity - too low (top right)
ax2 = fig.add_subplot(gs[0, 1])

# Simulate too-low perplexity (over-clustering)
n_points = 90
x = np.random.rand(n_points) * 8
y = np.random.rand(n_points) * 6
colors = ['#e74c3c', '#3498db', '#2ecc71'] * 30

# Create many small clusters
for i in range(0, len(x), 3):
    if i+2 < len(x):
        # Make small tight groups
        x[i:i+3] = np.random.normal(x[i], 0.1, 3)
        y[i:i+3] = np.random.normal(y[i], 0.1, 3)

ax2.scatter(x, y, c=colors, alpha=0.7, s=40)
ax2.set_title('❌ BAD: Perplexity Too Low\nToo many tiny clusters', 
             fontsize=12, color='red', weight='bold')
ax2.set_xticks([])
ax2.set_yticks([])

# Example 3: Bad perplexity - too high (middle left)
ax3 = fig.add_subplot(gs[1, 0])

# Simulate too-high perplexity (under-clustering)
all_points = np.vstack([cluster1, cluster2, cluster3])
# Add noise to simulate poor separation
noise = np.random.normal(0, 0.8, all_points.shape)
noisy_points = all_points + noise

colors_mixed = ['#e74c3c'] * 30 + ['#3498db'] * 30 + ['#2ecc71'] * 30
ax3.scatter(noisy_points[:, 0], noisy_points[:, 1], c=colors_mixed, alpha=0.7, s=40)
ax3.set_title('❌ BAD: Perplexity Too High\nClusters merge together', 
             fontsize=12, color='red', weight='bold')
ax3.set_xticks([])
ax3.set_yticks([])

# Example 4: Batch effects (middle right)
ax4 = fig.add_subplot(gs[1, 1])

# Simulate batch effects - same cell types in different "batches"
batch1_offset = [0, 0]
batch2_offset = [4, 3]

# Batch 1
b1_c1 = np.random.normal([1, 1], 0.2, (15, 2)) + batch1_offset
b1_c2 = np.random.normal([2, 2], 0.2, (15, 2)) + batch1_offset

# Batch 2 - same cell types but shifted
b2_c1 = np.random.normal([1, 1], 0.2, (15, 2)) + batch2_offset  
b2_c2 = np.random.normal([2, 2], 0.2, (15, 2)) + batch2_offset

ax4.scatter(b1_c1[:, 0], b1_c1[:, 1], c='#e74c3c', marker='o', alpha=0.7, s=40, label='Type A, Batch 1')
ax4.scatter(b1_c2[:, 0], b1_c2[:, 1], c='#3498db', marker='o', alpha=0.7, s=40, label='Type B, Batch 1')
ax4.scatter(b2_c1[:, 0], b2_c1[:, 1], c='#e74c3c', marker='^', alpha=0.7, s=40, label='Type A, Batch 2')
ax4.scatter(b2_c2[:, 0], b2_c2[:, 1], c='#3498db', marker='^', alpha=0.7, s=40, label='Type B, Batch 2')

ax4.set_title('❌ BAD: Batch Effects\nTechnical variation > biological', 
             fontsize=12, color='red', weight='bold')
ax4.legend(fontsize=8, loc='upper right')
ax4.set_xticks([])
ax4.set_yticks([])

# Bottom section: Best practices guide
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

best_practices = """
🛠️ t-SNE BEST PRACTICES FOR BIOLOGISTS:

🔧 BEFORE RUNNING t-SNE:
• Remove low-quality cells (high mitochondrial %, low gene count)
• Normalize your data (log-transform, scale)
• Select most variable genes (~2000-5000)
• Consider batch correction if needed

⚙️ PARAMETER TUNING:
• Perplexity: Start with 30-50, try 5-100 range
  - Small datasets: lower perplexity (5-30)
  - Large datasets: higher perplexity (50-100)
• Iterations: At least 1000, often need 5000+
• Learning rate: Usually 200-1000

🔍 INTERPRETING RESULTS:
✅ DO interpret:                           ❌ DON'T interpret:
• Cluster presence/absence                 • Exact distances between clusters
• Which cells group together              • Cluster sizes (can be misleading)
• Overall data structure                  • Absolute positions
• Cell type identification                • Trajectories between clusters

🧪 VALIDATION STEPS:
• Run multiple times with different random seeds
• Try different perplexity values
• Validate clusters with known markers
• Use complementary methods (UMAP, PCA)
• Check for batch effects

⚠️ COMMON MISTAKES:
• Using t-SNE distances for quantitative analysis
• Over-interpreting cluster sizes
• Not validating with biological knowledge  
• Ignoring parameter sensitivity
• Running only once with default settings
"""

ax5.text(0.05, 0.95, best_practices, transform=ax5.transAxes, 
         fontsize=11, va='top', ha='left',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))

plt.suptitle('t-SNE Pitfalls and Best Practices Guide', fontsize=16, weight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('tsne_step7_pitfalls_best_practices.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_step7_pitfalls_best_practices.pdf', bbox_inches='tight')