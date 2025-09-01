import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_high_dimensional_data():
    """Create synthetic data with 3 clusters in high dimensions"""
    # Create 3 well-separated clusters in 50 dimensions
    n_samples = 150
    n_features = 50
    
    # Generate clusters
    centers = [[0, 0, 0], [5, 5, 5], [-3, 3, -3]]
    cluster_std = [1.0, 1.0, 1.0]
    
    # Create the first 3 dimensions with clear separation
    X_3d, labels = make_blobs(n_samples=n_samples, centers=centers, 
                              cluster_std=cluster_std, random_state=42)
    
    # Add random noise for the remaining dimensions
    X_noise = np.random.randn(n_samples, n_features - 3) * 0.5
    
    # Combine to create high-dimensional data
    X_high_dim = np.hstack([X_3d, X_noise])
    
    return X_high_dim, labels

def plot_high_dimensional_challenge(X, labels):
    """Show why high-dimensional data is hard to visualize"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('The Challenge: High-Dimensional Data is Hard to Visualize', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: First 2 dimensions
    scatter1 = axes[0,0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    axes[0,0].set_title('First 2 Dimensions (X vs Y)', fontweight='bold')
    axes[0,0].set_xlabel('Dimension 1')
    axes[0,0].set_ylabel('Dimension 2')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Dimensions 1 and 3
    scatter2 = axes[0,1].scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', alpha=0.7, s=50)
    axes[0,1].set_title('Dimensions 1 vs 3 (X vs Z)', fontweight='bold')
    axes[0,1].set_xlabel('Dimension 1')
    axes[0,1].set_ylabel('Dimension 3')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Random pair of dimensions
    dim1, dim2 = np.random.choice(range(3, 50), 2, replace=False)
    scatter3 = axes[1,0].scatter(X[:, dim1], X[:, dim2], c=labels, cmap='viridis', alpha=0.7, s=50)
    axes[1,0].set_title(f'Random Dimensions {dim1} vs {dim2}', fontweight='bold')
    axes[1,0].set_xlabel(f'Dimension {dim1}')
    axes[1,0].set_ylabel(f'Dimension {dim2}')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Distance distribution
    distances = euclidean_distances(X[:50, :])  # Use subset for performance
    distances_flat = distances[np.triu_indices_from(distances, k=1)]
    
    axes[1,1].hist(distances_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,1].set_title('Distribution of Distances Between Points', fontweight='bold')
    axes[1,1].set_xlabel('Euclidean Distance')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add text explanation
    fig.text(0.02, 0.02, 
             'Problem: In high dimensions, all points appear equally distant!\n'
             'This is the "curse of dimensionality" - we need t-SNE to help us see patterns.',
             fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('tsne_high_dim_challenge.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_similarity_matrix(X, labels):
    """Show how t-SNE calculates similarities between points"""
    # Calculate similarities using Gaussian kernel (similar to t-SNE)
    n_samples = 100  # Use subset for visualization
    X_subset = X[:n_samples]
    labels_subset = labels[:n_samples]
    
    # Calculate pairwise distances
    distances = euclidean_distances(X_subset)
    
    # Convert to similarities using Gaussian kernel
    sigma = np.median(distances)  # Bandwidth parameter
    similarities = np.exp(-distances**2 / (2 * sigma**2))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Step 1: t-SNE Calculates Similarities Between All Points', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Similarity matrix heatmap
    im1 = ax1.imshow(similarities, cmap='viridis', aspect='auto')
    ax1.set_title('Similarity Matrix Heatmap', fontweight='bold')
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Point Index')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Similarity Score')
    
    # Plot 2: Similarity distribution
    similarities_flat = similarities[np.triu_indices_from(similarities, k=1)]
    ax2.hist(similarities_flat, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('Distribution of Similarity Scores', fontweight='bold')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add explanation
    fig.text(0.02, 0.02, 
             't-SNE calculates how similar each point is to every other point.\n'
             'High values (bright colors) = very similar points\n'
             'Low values (dark colors) = very different points',
             fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('tsne_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tsne_iterations(X, labels):
    """Show t-SNE optimization process step by step"""
    # Use subset for faster computation
    n_samples = 100
    X_subset = X[:n_samples]
    labels_subset = labels[:n_samples]
    
    # Run t-SNE with different numbers of iterations
    iterations = [250, 400, 500, 600, 800, 1000]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('t-SNE Optimization Process: From Random to Organized', 
                 fontsize=16, fontweight='bold')
    
    for i, n_iter in enumerate(iterations):
        row = i // 3
        col = i % 3
        
        if n_iter == 0:
            # Random initialization
            np.random.seed(42)
            X_2d = np.random.randn(n_samples, 2) * 2
            title = 'Random Start'
        else:
            # Run t-SNE
            tsne = TSNE(n_components=2, max_iter=n_iter, random_state=42, 
                        perplexity=30, learning_rate='auto')
            X_2d = tsne.fit_transform(X_subset)
            title = f'After {n_iter} iterations'
        
        # Plot
        scatter = axes[row, col].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_subset, 
                                       cmap='viridis', alpha=0.7, s=60)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].set_xlabel('t-SNE 1')
        axes[row, col].set_ylabel('t-SNE 2')
        axes[row, col].grid(True, alpha=0.3)
        
        # Remove axis ticks for cleaner look
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    # Add explanation
    fig.text(0.02, 0.02, 
             'Watch how t-SNE gradually organizes the data:\n'
             '1. Random chaos → 2. Points start moving → 3. Clusters form → 4. Clear separation!',
             fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('tsne_iterations.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_perplexity_effect(X, labels):
    """Show how different perplexity values affect t-SNE results"""
    perplexities = [5, 15, 30, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Perplexity Effect: How Many Neighbors to Consider?', 
                 fontsize=16, fontweight='bold')
    
    for i, perplexity in enumerate(perplexities):
        row = i // 2
        col = i % 2
        
        # Run t-SNE with different perplexity
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                    max_iter=1000, learning_rate='auto')
        X_2d = tsne.fit_transform(X[:100])  # Use subset for speed
        
        # Plot
        scatter = axes[row, col].scatter(X_2d[:, 0], X_2d[:, 1], c=labels[:100], 
                                       cmap='viridis', alpha=0.7, s=60)
        axes[row, col].set_title(f'Perplexity = {perplexity}', fontweight='bold')
        axes[row, col].set_xlabel('t-SNE 1')
        axes[row, col].set_ylabel('t-SNE 2')
        axes[row, col].grid(True, alpha=0.3)
        
        # Remove axis ticks
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    # Add explanation
    fig.text(0.02, 0.02, 
             'Perplexity controls how many neighbors each point considers:\n'
             '• Low (5-15): Many small clusters, preserves local structure\n'
             '• High (30-50): Fewer, larger clusters, more global view',
             fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('tsne_perplexity_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_tsne_animation(X, labels):
    """Create an animated visualization of t-SNE optimization"""
    # Use subset for animation
    n_samples = 80
    X_subset = X[:n_samples]
    labels_subset = labels[:n_samples]
    
    # Run t-SNE with many iterations to capture intermediate steps
    tsne = TSNE(n_components=2, max_iter=1000, random_state=42, 
                perplexity=30, learning_rate='auto', method='exact')
    
    # We'll simulate the process by running t-SNE multiple times with different iterations
    iterations = np.linspace(250, 1000, 20, dtype=int)
    positions = []
    
    for n_iter in iterations:
        if n_iter == 0:
            # Random start
            np.random.seed(42)
            pos = np.random.randn(n_samples, 2) * 2
        else:
            # Run t-SNE
            tsne_temp = TSNE(n_components=2, max_iter=n_iter, random_state=42, 
                            perplexity=30, learning_rate='auto', method='exact')
            pos = tsne_temp.fit_transform(X_subset)
        positions.append(pos)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('t-SNE Animation: Watch the Magic Happen!', fontsize=16, fontweight='bold')
    
    scatter = ax.scatter([], [], c=labels_subset, cmap='viridis', alpha=0.7, s=80)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    def animate(frame):
        pos = positions[frame]
        scatter.set_offsets(pos)
        ax.set_title(f'Iteration {iterations[frame]}', fontweight='bold')
        return scatter,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(iterations), 
                                 interval=500, blit=True, repeat=True)
    
    # Add explanation
    fig.text(0.02, 0.02, 
             'Watch how t-SNE gradually organizes your data from random chaos to clear patterns!\n'
             'This is the core magic of t-SNE: preserving local relationships while revealing structure.',
             fontsize=11, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink", alpha=0.7))
    
    plt.tight_layout()
    
    # Save animation
    anim.save('tsne_animation.gif', writer='pillow', fps=2, dpi=100)
    plt.show()
    
    return anim

def main():
    """Run all t-SNE visualizations"""
    print("Creating t-SNE visualizations inspired by StatQuest...")
    
    # Generate data
    X, labels = create_high_dimensional_data()
    
    # Create all visualizations
    plot_high_dimensional_challenge(X, labels)
    plot_similarity_matrix(X, labels)
    plot_tsne_iterations(X, labels)
    plot_perplexity_effect(X, labels)
    
    # Create animation (this takes longer)
    print("Creating animation (this may take a few minutes)...")
    anim = create_tsne_animation(X, labels)
    
    print("All visualizations created successfully!")
    print("Files saved:")
    print("- tsne_high_dim_challenge.png")
    print("- tsne_similarity_matrix.png") 
    print("- tsne_iterations.png")
    print("- tsne_perplexity_effect.png")
    print("- tsne_animation.gif")

if __name__ == "__main__":
    main()
