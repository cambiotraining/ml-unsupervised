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
ax1.set_title('Your Final t-SNE Result!\n(Similar cells are now close together)', 
              fontsize=14, pad=20)

plt.show()