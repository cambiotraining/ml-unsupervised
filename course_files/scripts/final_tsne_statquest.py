import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# Figure 5: Final Result and How to Interpret It
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Final clustered positions
cells = {
    'Cell 1': {'pos': (2, 6), 'color': '#e74c3c', 'type': 'Type A'},
    'Cell 2': {'pos': (2.8, 6.5), 'color': '#e74c3c', 'type': 'Type A'},
    'Cell 3': {'pos': (6, 5.5), 'color': '#3498db', 'type': 'Type B'},
    'Cell 4': {'pos': (6.7, 6), 'color': '#3498db', 'type': 'Type B'},
    'Cell 5': {'pos': (4, 2), 'color': '#2ecc71', 'type': 'Type C'},
    'Cell 6': {'pos': (7, 1.5), 'color': '#f39c12', 'type': 'Type D'}
}

# Left plot: Clean final result
for cell_name, cell_info in cells.items():
    x, y = cell_info['pos']
    color = cell_info['color']
    
    # Draw cell
    circle = plt.Circle((x, y), 0.25, color=color, alpha=0.8, zorder=2)
    ax1.add_patch(circle)
    
    # Add cell label
    ax1.text(x, y-0.6, cell_name, ha='center', va='top', 
            fontweight='bold', fontsize=10)

# Draw cluster boundaries
red_center = (2.4, 6.25)
blue_center = (6.35, 5.75)

# Type A cluster
ellipse1 = Ellipse(red_center, 1.2, 0.8, fill=False, 
                   edgecolor='#e74c3c', linestyle='--', linewidth=3, alpha=0.8)
ax1.add_patch(ellipse1)
ax1.text(red_center[0], red_center[1] - 0.8, 'Cell Type A\nCluster', 
         ha='center', va='center', fontsize=11, weight='bold', 
         color='#e74c3c', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='white', alpha=0.8))

# Type B cluster  
ellipse2 = Ellipse(blue_center, 1.2, 0.8, fill=False, 
                   edgecolor='#3498db', linestyle='--', linewidth=3, alpha=0.8)
ax1.add_patch(ellipse2)
ax1.text(blue_center[0], blue_center[1] - 0.8, 'Cell Type B\nCluster', 
         ha='center', va='center', fontsize=11, weight='bold', 
         color='#3498db', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='white', alpha=0.8))

# Label individual/outlier cells
ax1.annotate('Individual/\nTransitional Cell', xy=(4, 2), xytext=(4, 0.5),
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
            ha='center', fontsize=10, color='#2ecc71', weight='bold')

ax1.annotate('Different\nCell Type', xy=(7, 1.5), xytext=(7.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2),
            ha='center', fontsize=10, color='#f39c12', weight='bold')

# Add success indicators
ax1.text(0.5, 7.5, '✅ SUCCESS!', ha='left', fontsize=14, 
         weight='bold', color='green')
ax1.text(0.5, 7, '• Similar cells are now close together', fontsize=11)
ax1.text(0.5, 6.5, '• Natural clusters are visible', fontsize=11)
ax1.text(0.5, 6, '• Cell types can be identified', fontsize=11)

ax1.set_xlim(0, 9)
ax1.set_ylim(0, 8)
ax1.set_aspect('equal')
ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax1.set_title('Final t-SNE Result: Clustered Cell Types', fontsize=14, pad=15)
ax1.grid(True, alpha=0.3)

# Right plot: Interpretation guide with examples
ax2.text(0.5, 0.95, 'How to Interpret Your t-SNE Plot', 
         ha='center', va='top', fontsize=16, weight='bold', 
         transform=ax2.transAxes)

# Create interpretation examples
interpretation_text = """
🎯 What Each Pattern Means:

🔴 TIGHT CLUSTERS
→ Cells with very similar gene expression
→ Likely the same cell type or state
→ High confidence grouping

🔵 LOOSE CLUSTERS  
→ Related cells with some variation
→ Could be cell subtypes
→ Or cells in transition

🟢 ISOLATED POINTS
→ Unique cells (outliers)
→ Rare cell types
→ Technical artifacts?

🟠 BRIDGE CELLS
→ Transitional states
→ Connecting different types
→ Developmental intermediates

⚠️ IMPORTANT LIMITATIONS:

❌ Don't interpret:
• Exact distances between clusters
• Cluster sizes (can be misleading)
• Absolute positions

✅ Do interpret:
• Which cells group together
• Presence of distinct clusters
• Overall data structure
• Potential cell types

🧬 Biological Examples:

• Stem cells → Differentiated cells
• Healthy cells → Disease states  
• Drug treated → Control cells
• Different tissue types
"""

ax2.text(0.05, 0.85, interpretation_text, ha='left', va='top', 
         fontsize=11, transform=ax2.transAxes, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

# Add small example plots
mini_examples_x = [0.1, 0.35, 0.6, 0.85]
mini_examples_y = [0.25, 0.25, 0.25, 0.25]
mini_titles = ['Good\nSeparation', 'Gradual\nTransition', 'No Clear\nStructure', 'Batch\nEffect']

for i, (x, y, title) in enumerate(zip(mini_examples_x, mini_examples_y, mini_titles)):
    # Create mini subplot
    mini_ax = fig.add_axes([x-0.06, y-0.08, 0.12, 0.12])
    
    if i == 0:  # Good separation
        mini_ax.scatter([1, 1.2, 0.8], [1, 0.9, 1.1], c='red', s=30, alpha=0.7)
        mini_ax.scatter([2, 2.1, 1.9], [2, 2.2, 1.8], c='blue', s=30, alpha=0.7)
    elif i == 1:  # Gradual transition
        x_grad = np.linspace(0.5, 2.5, 10)
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        mini_ax.scatter(x_grad, [1.5]*10, c=colors, s=30, alpha=0.7)
    elif i == 2:  # No structure
        mini_ax.scatter(np.random.rand(15)*2, np.random.rand(15)*2, 
                       c='gray', s=20, alpha=0.7)
    else:  # Batch effect
        mini_ax.scatter([0.5, 0.6, 0.7], [0.5, 0.6, 0.4], c='red', s=30, alpha=0.7)
        mini_ax.scatter([1.5, 1.6, 1.7], [1.5, 1.6, 1.4], c='red', s=30, alpha=0.7)
        mini_ax.scatter([0.5, 0.6, 0.7], [1.5, 1.6, 1.4], c='blue', s=30, alpha=0.7)
        mini_ax.scatter([1.5, 1.6, 1.7], [0.5, 0.6, 0.4], c='blue', s=30, alpha=0.7)
    
    mini_ax.set_xlim(0, 2.5)
    mini_ax.set_ylim(0, 2.5)
    mini_ax.set_xticks([])
    mini_ax.set_yticks([])
    mini_ax.set_title(title, fontsize=9, pad=5)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Create legend for main plot
legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Cell Type A'),
    mpatches.Patch(color='#3498db', label='Cell Type B'),
    mpatches.Patch(color='#2ecc71', label='Cell Type C'),
    mpatches.Patch(color='#f39c12', label='Cell Type D')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('tsne_step5_final_result_interpretation.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_step5_final_result_interpretation.pdf', bbox_inches='tight')