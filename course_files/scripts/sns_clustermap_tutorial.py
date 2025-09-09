"""
Comprehensive sns.clustermap Tutorial for Biologists
====================================================

This tutorial demonstrates hierarchical clustering using seaborn's clustermap
with various parameters that are commonly used in biological data analysis.

Key Concepts for Biologists:
- Clustering groups similar samples/genes together
- Different linkage methods affect how clusters are formed
- Distance metrics determine what "similar" means
- Scaling can dramatically affect clustering results
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_sample_data():
    """
    Create sample gene expression data for demonstration.
    Simulates microarray/RNA-seq data with some biological structure.
    """
    np.random.seed(42)
    
    # Create sample names
    samples = [f'Sample_{i}' for i in range(1, 21)]
    genes = [f'Gene_{chr(65+i)}' for i in range(15)]  # Gene_A to Gene_O
    
    # Create data with some structure
    data = np.random.normal(0, 1, (15, 20))
    
    # Add some biological structure:
    # Group 1: High expression in samples 1-5
    data[0:3, 0:5] += 3
    # Group 2: High expression in samples 6-10  
    data[3:6, 5:10] += 2.5
    # Group 3: High expression in samples 11-15
    data[6:9, 10:15] += 2
    # Group 4: Low expression in samples 16-20
    data[9:12, 15:20] -= 2
    
    df = pd.DataFrame(data, index=genes, columns=samples)
    return df

def plot_clustermap_comparison():
    """
    Compare clustering with and without scaling.
    This is crucial for biological data where genes have different expression ranges.
    """
    print("=" * 60)
    print("COMPARISON: With vs Without Scaling")
    print("=" * 60)
    
    df = create_sample_data()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Without scaling (raw data)
    plt.subplot(1, 2, 1)
    sns.clustermap(df, 
                   cmap='RdBu_r', 
                   center=0,
                   figsize=(10, 8),
                   cbar_kws={'label': 'Expression Level'})
    plt.title('Without Scaling (Raw Data)', fontsize=14, fontweight='bold')
    
    # With scaling (standardized data)
    plt.subplot(1, 2, 2)
    sns.clustermap(df, 
                   cmap='RdBu_r', 
                   center=0,
                   standard_scale=0,  # Scale rows (genes)
                   figsize=(10, 8),
                   cbar_kws={'label': 'Standardized Expression'})
    plt.title('With Scaling (Standardized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey Points:")
    print("- Without scaling: Genes with high variance dominate clustering")
    print("- With scaling: All genes contribute equally to clustering")
    print("- For gene expression: usually scale rows (genes)")
    print("- For samples: usually scale columns (samples)")

def demonstrate_linkage_methods():
    """
    Show different linkage methods and their effects on clustering.
    """
    print("\n" + "=" * 60)
    print("LINKAGE METHODS COMPARISON")
    print("=" * 60)
    
    df = create_sample_data()
    
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, method in enumerate(linkage_methods):
        plt.subplot(2, 2, i+1)
        
        # Create clustermap
        g = sns.clustermap(df, 
                          method=method,
                          cmap='RdBu_r',
                          center=0,
                          standard_scale=0,
                          figsize=(8, 6),
                          cbar_kws={'label': 'Standardized Expression'})
        
        plt.title(f'Linkage: {method.title()}', fontsize=14, fontweight='bold')
        
        # Add description
        descriptions = {
            'single': 'Min distance between clusters\n(creates loose clusters)',
            'complete': 'Max distance between clusters\n(creates tight clusters)', 
            'average': 'Average distance between clusters\n(balanced approach)',
            'ward': 'Minimizes within-cluster variance\n(creates compact clusters)'
        }
        
        plt.figtext(0.02, 0.85-i*0.4, descriptions[method], 
                   fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    print("\nLinkage Method Characteristics:")
    print("- Single: Sensitive to outliers, creates elongated clusters")
    print("- Complete: Robust to outliers, creates compact clusters")
    print("- Average: Balanced approach, good for most biological data")
    print("- Ward: Minimizes variance, creates spherical clusters (requires euclidean distance)")

def demonstrate_distance_metrics():
    """
    Show different distance metrics and their biological interpretations.
    """
    print("\n" + "=" * 60)
    print("DISTANCE METRICS COMPARISON")
    print("=" * 60)
    
    df = create_sample_data()
    
    # Different distance metrics
    metrics = ['euclidean', 'correlation', 'cosine', 'cityblock']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Create clustermap
        g = sns.clustermap(df, 
                          metric=metric,
                          method='average',  # Use average linkage for all
                          cmap='RdBu_r',
                          center=0,
                          standard_scale=0,
                          figsize=(8, 6),
                          cbar_kws={'label': 'Standardized Expression'})
        
        plt.title(f'Distance: {metric.title()}', fontsize=14, fontweight='bold')
        
        # Add biological interpretation
        interpretations = {
            'euclidean': 'Absolute difference in expression\n(good for magnitude)',
            'correlation': 'Correlation-based distance\n(good for patterns)',
            'cosine': 'Angle between expression vectors\n(good for direction)',
            'cityblock': 'Sum of absolute differences\n(robust to outliers)'
        }
        
        plt.figtext(0.02, 0.85-i*0.4, interpretations[metric], 
                   fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDistance Metric Biological Interpretations:")
    print("- Euclidean: Groups genes with similar absolute expression levels")
    print("- Correlation: Groups genes with similar expression patterns across samples")
    print("- Cosine: Groups genes with similar expression direction (ignores magnitude)")
    print("- Cityblock (Manhattan): Similar to Euclidean but more robust to outliers")

def comprehensive_example():
    """
    Show a comprehensive example with biological interpretation.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BIOLOGICAL EXAMPLE")
    print("=" * 60)
    
    df = create_sample_data()
    
    # Create a comprehensive clustermap
    g = sns.clustermap(df,
                      # Clustering parameters
                      method='average',           # Linkage method
                      metric='correlation',       # Distance metric
                      standard_scale=0,          # Scale rows (genes)
                      
                      # Visual parameters
                      cmap='RdBu_r',             # Color map
                      center=0,                  # Center color map at 0
                      figsize=(12, 10),          # Figure size
                      
                      # Dendrogram parameters
                      row_cluster=True,          # Cluster rows (genes)
                      col_cluster=True,          # Cluster columns (samples)
                      
                      # Color bar
                      cbar_kws={'label': 'Standardized Expression Level',
                               'shrink': 0.8},
                      
                      # Annotation
                      xticklabels=True,
                      yticklabels=True)
    
    # Customize the plot
    g.ax_heatmap.set_xlabel('Samples', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Genes', fontsize=12, fontweight='bold')
    g.fig.suptitle('Gene Expression Clustering Analysis', 
                   fontsize=16, fontweight='bold', y=0.95)
    
    plt.show()
    
    print("\nInterpretation Guide:")
    print("1. Red regions: High expression")
    print("2. Blue regions: Low expression") 
    print("3. Dendrograms show clustering relationships")
    print("4. Similar genes/samples cluster together")
    print("5. Look for biological patterns in the heatmap")

def parameter_effects_demo():
    """
    Demonstrate how different parameters affect clustering results.
    """
    print("\n" + "=" * 60)
    print("PARAMETER EFFECTS DEMONSTRATION")
    print("=" * 60)
    
    df = create_sample_data()
    
    # Test different combinations
    combinations = [
        {'method': 'ward', 'metric': 'euclidean', 'standard_scale': 0, 'title': 'Ward + Euclidean + Scaled'},
        {'method': 'complete', 'metric': 'correlation', 'standard_scale': 0, 'title': 'Complete + Correlation + Scaled'},
        {'method': 'average', 'metric': 'cosine', 'standard_scale': 0, 'title': 'Average + Cosine + Scaled'},
        {'method': 'single', 'metric': 'euclidean', 'standard_scale': None, 'title': 'Single + Euclidean + No Scaling'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, params in enumerate(combinations):
        plt.subplot(2, 2, i+1)
        
        # Create clustermap with specific parameters
        g = sns.clustermap(df,
                          method=params['method'],
                          metric=params['metric'],
                          standard_scale=params['standard_scale'],
                          cmap='RdBu_r',
                          center=0,
                          figsize=(8, 6),
                          cbar_kws={'label': 'Expression Level'})
        
        plt.title(params['title'], fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nParameter Effects Summary:")
    print("- Ward linkage creates compact, spherical clusters")
    print("- Complete linkage is robust to outliers")
    print("- Correlation metric focuses on expression patterns")
    print("- Cosine metric ignores expression magnitude")
    print("- Scaling is crucial for comparing genes with different expression ranges")

def biological_interpretation_guide():
    """
    Provide a guide for interpreting clustermap results in biological context.
    """
    print("\n" + "=" * 60)
    print("BIOLOGICAL INTERPRETATION GUIDE")
    print("=" * 60)
    
    print("""
    HOW TO INTERPRET CLUSTERMAP RESULTS:
    
    1. GENE CLUSTERING (Rows):
       - Genes in the same cluster have similar expression patterns
       - Look for functional relationships between clustered genes
       - Check if clustered genes share biological pathways
    
    2. SAMPLE CLUSTERING (Columns):
       - Samples in the same cluster have similar gene expression profiles
       - May represent different conditions, treatments, or cell types
       - Useful for identifying sample groups or outliers
    
    3. HEATMAP PATTERNS:
       - Red blocks: High expression of gene group in sample group
       - Blue blocks: Low expression of gene group in sample group
       - Checkered patterns: Complex, condition-specific expression
    
    4. DENDROGRAM HEIGHT:
       - Short branches: Very similar genes/samples
       - Long branches: More dissimilar genes/samples
       - Cut dendrograms at different heights to get different numbers of clusters
    
    5. PARAMETER CHOICE GUIDELINES:
       - For gene expression: Use correlation or cosine distance
       - For absolute expression levels: Use euclidean distance
       - For robust clustering: Use complete or average linkage
       - For compact clusters: Use ward linkage (with euclidean distance)
       - Always consider scaling: standard_scale=0 for genes, standard_scale=1 for samples
    """)

def main():
    """
    Run the complete clustermap tutorial.
    """
    print("SEABORN CLUSTERMAP TUTORIAL FOR BIOLOGISTS")
    print("=" * 60)
    print("This tutorial demonstrates hierarchical clustering using sns.clustermap")
    print("with various parameters commonly used in biological data analysis.")
    
    # Run all demonstrations
    plot_clustermap_comparison()
    demonstrate_linkage_methods()
    demonstrate_distance_metrics()
    comprehensive_example()
    parameter_effects_demo()
    biological_interpretation_guide()
    
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    print("You now have comprehensive examples of sns.clustermap for biological data.")
    print("Experiment with different parameters to find the best clustering for your data!")

if __name__ == "__main__":
    main()
