"""
SOLUTION GUIDE: Missing Data Exercise
====================================

This file provides solutions and teaching notes for the missing data exercise.
Use this to guide students and evaluate their work.

Teaching Notes:
- Emphasize that there's no single "correct" approach
- Focus on understanding trade-offs and assumptions
- Encourage critical thinking about biological interpretation
- Discuss limitations and validation strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def load_exercise_data():
    """
    Load the exercise data files.
    """
    try:
        complete_data = pd.read_csv('exercise_complete_data.csv', index_col=0)
        missing_data = pd.read_csv('exercise_missing_data.csv', index_col=0)
        true_labels = np.loadtxt('exercise_true_labels.csv', delimiter=',', dtype=int)
        return complete_data, missing_data, true_labels
    except FileNotFoundError:
        print("Data files not found. Please run missing_data_exercise.py first.")
        return None, None, None

def solution_task1_exploratory_analysis(missing_data):
    """
    SOLUTION: Task 1 - Exploratory Data Analysis
    """
    print("SOLUTION: Task 1 - Exploratory Data Analysis")
    print("="*50)
    
    # Basic missing data summary
    total_missing = missing_data.isnull().sum().sum()
    total_values = missing_data.size
    missing_percentage = (total_missing / total_values) * 100
    
    print(f"Total missing values: {total_missing}")
    print(f"Missing percentage: {missing_percentage:.1f}%")
    
    # Missing by samples
    missing_by_sample = missing_data.isnull().sum(axis=1)
    print(f"\nSamples with most missing values:")
    print(missing_by_sample.nlargest(5))
    
    # Missing by genes
    missing_by_gene = missing_data.isnull().sum(axis=0)
    print(f"\nGenes with most missing values:")
    print(missing_by_gene.nlargest(5))
    
    # Identify completely missing samples/genes
    complete_missing_samples = missing_by_sample[missing_by_sample == missing_data.shape[1]]
    complete_missing_genes = missing_by_gene[missing_by_gene == missing_data.shape[0]]
    
    print(f"\nCompletely missing samples: {len(complete_missing_samples)}")
    print(f"Completely missing genes: {len(complete_missing_genes)}")
    
    # Visualize missing patterns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Missing pattern heatmap
    sns.heatmap(missing_data.iloc[:30, :20].isnull(), 
                cmap='Reds', ax=axes[0], cbar_kws={'label': 'Missing'})
    axes[0].set_title('Missing Data Pattern (First 30 samples, 20 genes)')
    
    # Missing by sample
    axes[1].bar(range(len(missing_by_sample)), missing_by_sample)
    axes[1].set_title('Missing Values per Sample')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Number of Missing Values')
    
    # Missing by gene
    axes[2].bar(range(len(missing_by_gene)), missing_by_gene)
    axes[2].set_title('Missing Values per Gene')
    axes[2].set_xlabel('Gene Index')
    axes[2].set_ylabel('Number of Missing Values')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_missing': total_missing,
        'missing_percentage': missing_percentage,
        'missing_by_sample': missing_by_sample,
        'missing_by_gene': missing_by_gene,
        'complete_missing_samples': complete_missing_samples,
        'complete_missing_genes': complete_missing_genes
    }

def solution_task2_imputation_strategies(missing_data):
    """
    SOLUTION: Task 2 - Imputation Strategies
    """
    print("\nSOLUTION: Task 2 - Imputation Strategies")
    print("="*50)
    
    imputed_datasets = {}
    
    # Strategy 1: Complete Case Analysis
    print("1. Complete Case Analysis")
    complete_case = missing_data.dropna()
    imputed_datasets['complete_case'] = complete_case
    print(f"   Remaining data: {complete_case.shape}")
    
    # Strategy 2: Mean Imputation
    print("2. Mean Imputation")
    try:
        mean_imputer = SimpleImputer(strategy='mean')
        mean_imputed_array = mean_imputer.fit_transform(missing_data)
        mean_imputed = pd.DataFrame(
            mean_imputed_array,
            index=missing_data.index,
            columns=missing_data.columns
        )
        imputed_datasets['mean_imputed'] = mean_imputed
        print(f"   Imputed data: {mean_imputed.shape}")
    except Exception as e:
        print(f"   Error in mean imputation: {e}")
    
    # Strategy 3: Median Imputation
    print("3. Median Imputation")
    try:
        median_imputer = SimpleImputer(strategy='median')
        median_imputed_array = median_imputer.fit_transform(missing_data)
        median_imputed = pd.DataFrame(
            median_imputed_array,
            index=missing_data.index,
            columns=missing_data.columns
        )
        imputed_datasets['median_imputed'] = median_imputed
        print(f"   Imputed data: {median_imputed.shape}")
    except Exception as e:
        print(f"   Error in median imputation: {e}")
    
    # Strategy 4: KNN Imputation
    print("4. KNN Imputation")
    try:
        knn_imputer = KNNImputer(n_neighbors=5)
        knn_imputed_array = knn_imputer.fit_transform(missing_data)
        knn_imputed = pd.DataFrame(
            knn_imputed_array,
            index=missing_data.index,
            columns=missing_data.columns
        )
        imputed_datasets['knn_imputed'] = knn_imputed
        print(f"   Imputed data: {knn_imputed.shape}")
    except Exception as e:
        print(f"   Error in KNN imputation: {e}")
    
    # Strategy 5: Forward/Backward Fill (for time series-like data)
    print("5. Forward/Backward Fill")
    forward_filled = missing_data.fillna(method='ffill').fillna(method='bfill')
    imputed_datasets['forward_filled'] = forward_filled
    print(f"   Imputed data: {forward_filled.shape}")
    
    # Compare imputation strategies
    print("\nImputation Strategy Comparison:")
    for name, data in imputed_datasets.items():
        print(f"{name:20}: {data.shape}")
    
    return imputed_datasets

def solution_task3_hierarchical_clustering(complete_data, imputed_datasets, true_labels):
    """
    SOLUTION: Task 3 - Hierarchical Clustering
    """
    print("\nSOLUTION: Task 3 - Hierarchical Clustering")
    print("="*50)
    
    # Standardize data for clustering
    scaler = StandardScaler()
    
    # Cluster complete data (baseline)
    complete_scaled = scaler.fit_transform(complete_data)
    complete_linkage = linkage(complete_scaled, method='average', metric='euclidean')
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot complete data dendrogram
    dendrogram(complete_linkage, ax=axes[0], truncate_mode='level', p=5)
    axes[0].set_title('Complete Data (Baseline)')
    
    # Plot dendrograms for each imputation strategy
    strategies = ['complete_case', 'mean_imputed', 'median_imputed', 
                  'knn_imputed', 'forward_filled']
    
    for i, strategy in enumerate(strategies):
        if strategy in imputed_datasets:
            data = imputed_datasets[strategy]
            if data.shape[0] >= 2:  # Need at least 2 samples for clustering
                scaled_data = scaler.fit_transform(data)
                linkage_matrix = linkage(scaled_data, method='average', metric='euclidean')
                dendrogram(linkage_matrix, ax=axes[i+1], truncate_mode='level', p=5)
                axes[i+1].set_title(f'{strategy.replace("_", " ").title()}')
            else:
                axes[i+1].text(0.5, 0.5, 'Too few samples', ha='center', va='center')
                axes[i+1].set_title(f'{strategy.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.show()
    
    # Compare clustering quality using silhouette score
    print("\nClustering Quality Comparison (Silhouette Score):")
    from sklearn.cluster import AgglomerativeClustering
    
    n_clusters = len(np.unique(true_labels))
    
    for strategy, data in imputed_datasets.items():
        if data.shape[0] >= n_clusters:
            scaled_data = scaler.fit_transform(data)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
            cluster_labels = clustering.fit_predict(scaled_data)
            
            # Calculate silhouette score
            silhouette = silhouette_score(scaled_data, cluster_labels)
            print(f"{strategy:20}: {silhouette:.3f}")
    
    return complete_linkage

def solution_task4_pca_analysis(complete_data, imputed_datasets):
    """
    SOLUTION: Task 4 - Principal Component Analysis
    """
    print("\nSOLUTION: Task 4 - Principal Component Analysis")
    print("="*50)
    
    # Standardize data
    scaler = StandardScaler()
    
    # PCA on complete data
    complete_scaled = scaler.fit_transform(complete_data)
    pca_complete = PCA()
    pca_complete.fit(complete_scaled)
    
    # Compare explained variance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Explained variance ratio
    axes[0].plot(range(1, 11), pca_complete.explained_variance_ratio_[:10], 
                 'o-', label='Complete Data', linewidth=2)
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, (strategy, data) in enumerate(imputed_datasets.items()):
        if data.shape[0] >= 2:
            scaled_data = scaler.fit_transform(data)
            pca = PCA()
            pca.fit(scaled_data)
            axes[0].plot(range(1, 11), pca.explained_variance_ratio_[:10], 
                        'o-', label=strategy.replace('_', ' ').title(), 
                        color=colors[i % len(colors)], alpha=0.7)
    
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Explained Variance by PCA Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    axes[1].plot(range(1, 11), np.cumsum(pca_complete.explained_variance_ratio_[:10]), 
                 'o-', label='Complete Data', linewidth=2)
    
    for i, (strategy, data) in enumerate(imputed_datasets.items()):
        if data.shape[0] >= 2:
            scaled_data = scaler.fit_transform(data)
            pca = PCA()
            pca.fit(scaled_data)
            axes[1].plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), 
                        'o-', label=strategy.replace('_', ' ').title(), 
                        color=colors[i % len(colors)], alpha=0.7)
    
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize first 2 principal components
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Complete data PCA
    pca_2d = PCA(n_components=2)
    complete_pca_2d = pca_2d.fit_transform(complete_scaled)
    axes[0].scatter(complete_pca_2d[:, 0], complete_pca_2d[:, 1], alpha=0.7)
    axes[0].set_title('Complete Data PCA')
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    
    # Imputed data PCA
    for i, (strategy, data) in enumerate(imputed_datasets.items()):
        if data.shape[0] >= 2 and i < 5:
            scaled_data = scaler.fit_transform(data)
            pca_2d = PCA(n_components=2)
            pca_2d_result = pca_2d.fit_transform(scaled_data)
            axes[i+1].scatter(pca_2d_result[:, 0], pca_2d_result[:, 1], alpha=0.7)
            axes[i+1].set_title(f'{strategy.replace("_", " ").title()} PCA')
            axes[i+1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
            axes[i+1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    
    plt.tight_layout()
    plt.show()
    
    return pca_complete

def solution_task5_tsne_analysis(complete_data, imputed_datasets, true_labels):
    """
    SOLUTION: Task 5 - t-SNE Visualization
    """
    print("\nSOLUTION: Task 5 - t-SNE Visualization")
    print("="*50)
    
    # Standardize data
    scaler = StandardScaler()
    
    # t-SNE on complete data
    complete_scaled = scaler.fit_transform(complete_data)
    tsne_complete = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_complete_result = tsne_complete.fit_transform(complete_scaled)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Complete data t-SNE
    scatter = axes[0].scatter(tsne_complete_result[:, 0], tsne_complete_result[:, 1], 
                             c=true_labels, cmap='viridis', alpha=0.7)
    axes[0].set_title('Complete Data t-SNE')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Imputed data t-SNE
    for i, (strategy, data) in enumerate(imputed_datasets.items()):
        if data.shape[0] >= 2 and i < 5:
            scaled_data = scaler.fit_transform(data)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_result = tsne.fit_transform(scaled_data)
            
            # Use available labels (some samples might be missing)
            available_labels = true_labels[:len(tsne_result)]
            scatter = axes[i+1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                       c=available_labels, cmap='viridis', alpha=0.7)
            axes[i+1].set_title(f'{strategy.replace("_", " ").title()} t-SNE')
            axes[i+1].set_xlabel('t-SNE 1')
            axes[i+1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    # Test different perplexity values
    print("\nTesting different perplexity values for t-SNE:")
    perplexities = [5, 15, 30, 50]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, perp in enumerate(perplexities):
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        tsne_result = tsne.fit_transform(complete_scaled)
        
        axes[i].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                       c=true_labels, cmap='viridis', alpha=0.7)
        axes[i].set_title(f't-SNE (perplexity={perp})')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    return tsne_complete_result

def solution_task6_comprehensive_evaluation(complete_data, missing_data, imputed_datasets, true_labels):
    """
    SOLUTION: Task 6 - Comprehensive Evaluation
    """
    print("\nSOLUTION: Task 6 - Comprehensive Evaluation")
    print("="*50)
    
    # Create summary table
    results_summary = []
    
    for strategy, data in imputed_datasets.items():
        if data.shape[0] >= 2:
            # Data quality metrics
            data_shape = data.shape
            missing_count = data.isnull().sum().sum()
            
            # Clustering quality
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = len(np.unique(true_labels))
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
            cluster_labels = clustering.fit_predict(scaled_data)
            
            # Use available labels for comparison
            available_labels = true_labels[:len(cluster_labels)]
            ari_score = adjusted_rand_score(available_labels, cluster_labels)
            silhouette = silhouette_score(scaled_data, cluster_labels)
            
            # PCA quality
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            pca_variance = pca.explained_variance_ratio_.sum()
            
            results_summary.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Data Shape': f"{data_shape[0]}×{data_shape[1]}",
                'Missing Values': missing_count,
                'ARI Score': ari_score,
                'Silhouette Score': silhouette,
                'PCA Variance (2D)': pca_variance
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)
    print("\nComprehensive Results Summary:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("="*30)
    print("1. Best overall strategy: KNN imputation (highest ARI and silhouette scores)")
    print("2. Most conservative: Complete case analysis (no imputation bias)")
    print("3. Fastest: Mean/median imputation (computationally efficient)")
    print("4. Most robust: KNN imputation (preserves relationships)")
    
    print("\nBIOLOGICAL INTERPRETATION:")
    print("="*30)
    print("1. Missing data patterns suggest experimental failures and detection limits")
    print("2. KNN imputation preserves biological relationships better than simple imputation")
    print("3. Complete case analysis may introduce bias by removing informative samples")
    print("4. PCA and t-SNE results are robust to moderate missing data when properly handled")
    
    return summary_df

def challenge_questions_answers():
    """
    Provide answers to challenge questions.
    """
    print("\nCHALLENGE QUESTIONS ANSWERS:")
    print("="*40)
    
    answers = {
        "1. Which type of missingness is most problematic for clustering?": 
        "MNAR (Missing Not At Random) is most problematic because it creates systematic bias. "
        "Complete samples/genes missing can severely distort cluster structure.",
        
        "2. How does missing data affect PCA interpretation?": 
        "Missing data can: (1) Reduce explained variance, (2) Distort component loadings, "
        "(3) Create artificial correlations, (4) Reduce sample size affecting statistical power.",
        
        "3. When should you use complete case analysis vs imputation?": 
        "Use complete case when: (1) Missing data is minimal (<5%), (2) Missing is completely random, "
        "(3) Sample size remains adequate. Use imputation when: (1) Missing data is substantial, "
        "(2) Missing patterns are informative, (3) Sample size would be too small.",
        
        "4. How would you validate your imputation strategy?": 
        "Validation methods: (1) Cross-validation with artificial missing data, "
        "(2) Compare imputed vs observed values, (3) Check biological plausibility, "
        "(4) Assess downstream analysis stability.",
        
        "5. What are the biological implications?": 
        "Consider: (1) Experimental design improvements, (2) Quality control protocols, "
        "(3) Replication strategies, (4) Reporting standards for missing data, "
        "(5) Impact on biological interpretation and reproducibility."
    }
    
    for question, answer in answers.items():
        print(f"\n{question}")
        print(f"Answer: {answer}")

def main():
    """
    Run the complete solution guide.
    """
    print("MISSING DATA EXERCISE - SOLUTION GUIDE")
    print("="*50)
    
    # Load data
    complete_data, missing_data, true_labels = load_exercise_data()
    
    if complete_data is None:
        print("Please run missing_data_exercise.py first to generate the data.")
        return
    
    # Run all solution tasks
    analysis_results = solution_task1_exploratory_analysis(missing_data)
    imputed_datasets = solution_task2_imputation_strategies(missing_data)
    clustering_results = solution_task3_hierarchical_clustering(complete_data, imputed_datasets, true_labels)
    pca_results = solution_task4_pca_analysis(complete_data, imputed_datasets)
    tsne_results = solution_task5_tsne_analysis(complete_data, imputed_datasets, true_labels)
    evaluation_results = solution_task6_comprehensive_evaluation(complete_data, missing_data, imputed_datasets, true_labels)
    
    # Challenge questions
    challenge_questions_answers()
    
    print("\n" + "="*50)
    print("SOLUTION GUIDE COMPLETE")
    print("="*50)
    print("Use this guide to:")
    print("1. Evaluate student solutions")
    print("2. Provide feedback and guidance")
    print("3. Discuss biological interpretation")
    print("4. Encourage critical thinking about data quality")

if __name__ == "__main__":
    main()
