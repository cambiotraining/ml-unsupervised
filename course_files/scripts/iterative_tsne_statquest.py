import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects

# Figure 4: Iterative Optimization (can be animated or shown as static frames)

def create_optimization_animation(save_gif=False):
    """Create an animation showing t-SNE optimization process"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Starting positions (random)
    start_positions = np.array([
        [2, 6],  # Cell 1 (red)
        [8, 2],  # Cell 2 (red) 
        [1, 3],  # Cell 3 (blue)
        [7, 7],  # Cell 4 (blue)
        [5, 4],  # Cell 5 (green)
        [3, 1]   # Cell 6 (orange)
    ])
    
    # Final positions (clustered)
    end_positions = np.array([
        [2, 6],    # Cell 1 (red cluster)
        [2.8, 6.5], # Cell 2 (red cluster)
        [6, 5.5],  # Cell 3 (blue cluster)
        [6.7, 6],  # Cell 4 (blue cluster) 
        [4, 2],    # Cell 5 (separate)
        [7, 1.5]   # Cell 6 (separate)
    ])
    
    colors = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#2ecc71', '#f39c12']
    cell_names = ['Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6']
    
    def animate(frame):
        ax.clear()
        
        # Calculate interpolated positions
        progress = frame / 100.0
        # Use smooth easing function
        smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
        
        current_positions = start_positions + (end_positions - start_positions) * smooth_progress
        
        # Draw cells
        for i, (pos, color, name) in enumerate(zip(current_positions, colors, cell_names)):
            circle = plt.Circle(pos, 0.25, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1]-0.6, name, ha='center', va='top', 
                   fontweight='bold', fontsize=10)
        
        # Draw arrows showing movement
        if frame > 10:  # Start showing arrows after a few frames
            for i, (start, current) in enumerate(zip(start_positions, current_positions)):
                if np.linalg.norm(current - start) > 0.1:  # Only if significant movement
                    ax.arrow(start[0], start[1], 
                            current[0] - start[0], current[1] - start[1],
                            head_width=0.15, head_length=0.15, 
                            fc='gray', ec='gray', alpha=0.5, linestyle='--')
        
        # Show clustering at the end
        if progress > 0.7:
            # Draw cluster boundaries
            if progress > 0.8:
                # Red cluster
                red_center = np.mean([current_positions[0], current_positions[1]], axis=0)
                cluster1 = plt.Circle(red_center, 0.6, fill=False, 
                                    edgecolor='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
                ax.add_patch(cluster1)
                ax.text(red_center[0], red_center[1] - 1, 'Type A Cluster', 
                       ha='center', fontsize=10, weight='bold', color='#e74c3c')
                
                # Blue cluster  
                blue_center = np.mean([current_positions[2], current_positions[3]], axis=0)
                cluster2 = plt.Circle(blue_center, 0.6, fill=False, 
                                    edgecolor='#3498db', linestyle='--', linewidth=2, alpha=0.7)
                ax.add_patch(cluster2)
                ax.text(blue_center[0], blue_center[1] - 1, 'Type B Cluster', 
                       ha='center', fontsize=10, weight='bold', color='#3498db')
        
        # Progress bar
        bar_width = 6
        bar_height = 0.3
        bar_x = 1.5
        bar_y = 0.5
        
        # Background
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, 
                                  facecolor='lightgray', edgecolor='black'))
        # Progress
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                  facecolor='#3498db', edgecolor='black'))
        ax.text(bar_x + bar_width/2, bar_y - 0.3, f'Optimization Progress: {progress*100:.0f}%', 
               ha='center', fontsize=10)
        
        # Formatting
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Step 3: t-SNE Gradually Moves Points to Preserve Neighborhoods\n' +
                    '🎯 Similar cells are pulled together, different cells pushed apart', 
                    fontsize=12, pad=15)
        ax.grid(True, alpha=0.3)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Cell Type A'),
            mpatches.Patch(color='#3498db', label='Cell Type B'),
            mpatches.Patch(color='#2ecc71', label='Cell Type C'),
            mpatches.Patch(color='#f39c12', label='Cell Type D')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=101, interval=100, repeat=True)
    
    if save_gif:
        anim.save('tsne_optimization_animation.gif', writer='pillow', fps=10)
    
    plt.show()
    return anim

# Also create static frames showing key stages
def create_static_optimization_frames():
    """Create static figures showing key stages of optimization"""
    
    stages = [
        (0, "Initial Random State"),
        (0.33, "Early Optimization"), 
        (0.66, "Mid Optimization"),
        (1.0, "Final Clustered State")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Positions
    start_positions = np.array([
        [2, 6], [8, 2], [1, 3], [7, 7], [5, 4], [3, 1]
    ])
    
    end_positions = np.array([
        [2, 6], [2.8, 6.5], [6, 5.5], [6.7, 6], [4, 2], [7, 1.5]
    ])
    
    colors = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#2ecc71', '#f39c12']
    cell_names = ['Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6']
    
    for idx, (progress, title) in enumerate(stages):
        ax = axes[idx]
        
        # Calculate positions
        current_positions = start_positions + (end_positions - start_positions) * progress
        
        # Draw cells
        for i, (pos, color, name) in enumerate(zip(current_positions, colors, cell_names)):
            circle = plt.Circle(pos, 0.25, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1]-0.6, name, ha='center', va='top', 
                   fontweight='bold', fontsize=9)
        
        # Show final clusters
        if progress == 1.0:
            red_center = np.mean([current_positions[0], current_positions[1]], axis=0)
            blue_center = np.mean([current_positions[2], current_positions[3]], axis=0)
            
            cluster1 = plt.Circle(red_center, 0.6, fill=False, 
                                edgecolor='#e74c3c', linestyle='--', linewidth=2)
            cluster2 = plt.Circle(blue_center, 0.6, fill=False, 
                                edgecolor='#3498db', linestyle='--', linewidth=2)
            ax.add_patch(cluster1)
            ax.add_patch(cluster2)
        
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('tsne_step4_optimization_stages.png', dpi=300, bbox_inches='tight')
    plt.savefig('tsne_step4_optimization_stages.pdf', bbox_inches='tight')

# Run the functions
if __name__ == "__main__":
    # Create static frames (easier to include in presentations)
    create_static_optimization_frames()
    
    # Uncomment to create animation (requires pillow: pip install pillow)
    # anim = create_optimization_animation(save_gif=True)