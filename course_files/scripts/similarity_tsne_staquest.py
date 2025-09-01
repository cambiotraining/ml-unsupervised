import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Figure 2: Similarity Calculation
fig, ax = plt.subplots(figsize=(12, 8))

# Cell positions for visualization (not the real high-dim positions)
cells = {
    'Cell 1': {'pos': (2, 6), 'color': '#e74c3c', 'type': 'Type A'},
    'Cell 2': {'pos': (3, 7), 'color': '#e74c3c', 'type': 'Type A'},
    'Cell 3': {'pos': (8, 6), 'color': '#3498db', 'type': 'Type B'},
    'Cell 4': {'pos': (9, 7), 'color': '#3498db', 'type': 'Type B'},
    'Cell 5': {'pos': (5, 3), 'color': '#2ecc71', 'type': 'Type C'},
    'Cell 6': {'pos': (7, 2), 'color': '#f39c12', 'type': 'Type D'}
}

# Draw similarity connections
similarities = [
    ('Cell 1', 'Cell 2', 0.9, 'Very Similar\n(Same cell type)'),
    ('Cell 3', 'Cell 4', 0.85, 'Very Similar\n(Same cell type)'),
    ('Cell 1', 'Cell 5', 0.3, 'Somewhat Similar'),
    ('Cell 2', 'Cell 6', 0.15, 'Not Similar'),
    ('Cell 1', 'Cell 3', 0.1, 'Very Different\n(Different types)')
]

# Draw connections first (so they appear behind points)
for cell1, cell2, strength, label in similarities:
    pos1 = cells[cell1]['pos']
    pos2 = cells[cell2]['pos']
    
    # Line thickness and opacity based on similarity
    linewidth = strength * 8
    alpha = strength * 0.8 + 0.2
    
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
            color='#3498db', linewidth=linewidth, alpha=alpha, zorder=1)
    
    # Add similarity labels
    mid_x = (pos1[0] + pos2[0]) / 2
    mid_y = (pos1[1] + pos2[1]) / 2
    
    # Create text box
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', 
                     edgecolor='#3498db', alpha=0.9)
    ax.text(mid_x, mid_y, f'{strength:.1f}\n{label}', 
            ha='center', va='center', fontsize=9, 
            bbox=bbox_props, zorder=3)

# Draw cells
for cell_name, cell_info in cells.items():
    x, y = cell_info['pos']
    color = cell_info['color']
    
    # Draw cell
    circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, zorder=2)
    ax.add_patch(circle)
    
    # Add cell label
    ax.text(x, y-0.8, cell_name, ha='center', va='top', 
            fontweight='bold', fontsize=10)

# Create legend for cell types
legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Cell Type A'),
    mpatches.Patch(color='#3498db', label='Cell Type B'),  
    mpatches.Patch(color='#2ecc71', label='Cell Type C'),
    mpatches.Patch(color='#f39c12', label='Cell Type D')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Formatting
ax.set_xlim(0, 11)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.set_title('Step 1: t-SNE Calculates Similarities Between Cells\n' +
             'Based on ALL measured features in high-dimensional space', 
             fontsize=14, pad=20)

# Remove axes for cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Add explanation box
explanation = (
    "🔍 Key Concept: Similarity Based on Distance\n\n"
    "• Cells with similar gene expression patterns get HIGH similarity scores\n"
    "• Cells with different patterns get LOW similarity scores\n" 
    "• t-SNE will try to preserve these neighborhood relationships in 2D"
)

props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
ax.text(0.5, 0.8, explanation, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('tsne_step2_similarity_calculation.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_step2_similarity_calculation.pdf', bbox_inches='tight')