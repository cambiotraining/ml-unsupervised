import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Figure 3: Random Initial Placement
fig, ax = plt.subplots(figsize=(10, 8))

# Random initial positions (what t-SNE starts with)
np.random.seed(42)  # For reproducible "random" positions
cells = {
    'Cell 1': {'pos': (2, 6), 'color': '#e74c3c', 'type': 'Type A'},
    'Cell 2': {'pos': (8, 2), 'color': '#e74c3c', 'type': 'Type A'}, 
    'Cell 3': {'pos': (1, 3), 'color': '#3498db', 'type': 'Type B'},
    'Cell 4': {'pos': (7, 7), 'color': '#3498db', 'type': 'Type B'},
    'Cell 5': {'pos': (5, 4), 'color': '#2ecc71', 'type': 'Type C'},
    'Cell 6': {'pos': (3, 1), 'color': '#f39c12', 'type': 'Type D'}
}

# Draw cells
for cell_name, cell_info in cells.items():
    x, y = cell_info['pos']
    color = cell_info['color']
    
    # Draw cell with larger circles for visibility
    circle = plt.Circle((x, y), 0.25, color=color, alpha=0.8, zorder=2)
    ax.add_patch(circle)
    
    # Add cell label
    ax.text(x, y-0.6, cell_name, ha='center', va='top', 
            fontweight='bold', fontsize=11)

# Highlight the problem with arrows and annotations
# Similar cells that are far apart
ax.annotate('', xy=(8, 2), xytext=(2, 6),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(5, 4.5, '❌ Same cell type\nbut FAR apart!', 
        ha='center', va='center', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffcccc', alpha=0.8))

ax.annotate('', xy=(7, 7), xytext=(1, 3),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(4, 5.5, '❌ Same cell type\nbut FAR apart!', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffcccc', alpha=0.8))

# Different cells that are close
ax.annotate('', xy=(3, 1), xytext=(5, 4),
            arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
ax.text(4, 2.5, '⚠️ Different types\nbut close together!', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffe6cc', alpha=0.8))

# Create legend
legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Cell Type A'),
    mpatches.Patch(color='#3498db', label='Cell Type B'),
    mpatches.Patch(color='#2ecc71', label='Cell Type C'), 
    mpatches.Patch(color='#f39c12', label='Cell Type D')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Formatting
ax.set_xlim(0, 9)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('Step 2: Random Initial Placement in 2D Space\n' +
             'This starting arrangement doesn\'t reflect the true similarities!', 
             fontsize=14, pad=20)

# Add grid for better visualization
ax.grid(True, alpha=0.3)

# Add explanation box
explanation = (
    "🎲 Starting Point Problem:\n\n"
    "• Cells are randomly scattered in 2D space\n"
    "• Similar cells (same colors) are far apart\n" 
    "• Different cells might be close together\n"
    "• This doesn't match our high-dimensional similarities!\n\n"
    "➡️ t-SNE will now gradually move points to fix this..."
)

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('tsne_step3_random_placement.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_step3_random_placement.pdf', bbox_inches='tight')