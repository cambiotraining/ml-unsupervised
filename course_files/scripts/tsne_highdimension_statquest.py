import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Figure 1: The High-Dimensional Data Problem
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Sample data for 6 cells across multiple dimensions
dimensions = ['Gene A', 'Gene B', 'Protein X', 'Protein Y', 'Gene C', 'Gene D']
cell_data = {
    'Cell 1': [8, 2, 9, 1, 5, 7],
    'Cell 2': [7, 3, 8, 2, 4, 8],
    'Cell 3': [2, 8, 3, 9, 8, 2],
    'Cell 4': [3, 9, 2, 8, 7, 1],
    'Cell 5': [5, 5, 5, 5, 5, 5],
    'Cell 6': [9, 1, 1, 9, 2, 9]
}

colors = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#2ecc71', '#f39c12']

# Left plot: Bar chart showing multi-dimensional data
x_pos = np.arange(len(dimensions))
width = 0.12

for i, (cell_name, values) in enumerate(cell_data.items()):
    ax1.bar(x_pos + i * width, values, width, label=cell_name, 
            color=colors[i], alpha=0.8)

ax1.set_xlabel('Measured Features', fontsize=12)
ax1.set_ylabel('Expression Level', fontsize=12)
ax1.set_title('Your Original High-Dimensional Data\n(6 cells × 6 features = hard to visualize!)', 
              fontsize=14, pad=20)
ax1.set_xticks(x_pos + width * 2.5)
ax1.set_xticklabels(dimensions, rotation=45)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Right plot: 3D representation attempt
ax2 = fig.add_subplot(122, projection='3d')

# Use first 3 dimensions for 3D plot
for i, (cell_name, values) in enumerate(cell_data.items()):
    ax2.scatter(values[0], values[1], values[2], 
               c=colors[i], s=100, alpha=0.8, label=cell_name)

ax2.set_xlabel('Gene A')
ax2.set_ylabel('Gene B') 
ax2.set_zlabel('Protein X')
ax2.set_title('Even in 3D, we\'re missing\n3 other dimensions!', fontsize=12, pad=20)

# Add text annotation
fig.text(0.5, 0.02, 
         '🧬 Biology Problem: You measured 50+ features per cell, but can only plot 2-3 dimensions!\n' +
         'How do you visualize all the relationships in your data?', 
         ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Save the figure
plt.savefig('tsne_step1_highdim_problem.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_step1_highdim_problem.pdf', bbox_inches='tight')