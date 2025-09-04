#!/usr/bin/env python3
"""
Network path anomaly visualization with dimensionality reduction
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from umap import UMAP

# Suppress warnings
warnings.filterwarnings('ignore')

# Optional UMAP import
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def create_anomaly_visualization(detector, baseline_embeddings, analysis_embeddings, 
                                anomaly_scores, analysis_results=None,
                                sample_size=3000, method='umap', random_state=42):
    """
    Network path anomaly visualization with dimensionality reduction
    
    Parameters:
    -----------
    detector : GraphAnomalyDetector
        Trained anomaly detector
    baseline_embeddings : np.array
        Baseline path embeddings (normal data)
    analysis_embeddings : np.array  
        Analysis path embeddings (test data)
    anomaly_scores : np.array
        Anomaly scores for analysis data
    analysis_results : DataFrame, optional
        Results from detect_anomalies() containing 'graph_is_anomaly' column
    sample_size : int
        Number of points to sample for visualization (default: 3000)
    method : str
        Dimensionality reduction method: 'umap', 'tsne', or 'both' (default: 'umap')
    random_state : int
        Random seed for reproducibility
        
    What to Expect with UMAP vs t-SNE:
    -----------------------------------
    UMAP (Uniform Manifold Approximation and Projection):
    - Faster than t-SNE, especially for large datasets
    - Better at preserving global structure
    - Often shows clearer cluster separation
    - More consistent results across runs
    - Can handle larger datasets efficiently
    
    t-SNE (t-distributed Stochastic Neighbor Embedding):
    - Better at revealing local structure
    - Can create artificial clusters sometimes
    - Slower and more memory-intensive
    - Results vary more between runs
    - Better for smaller datasets (<5000 points)
    """
    
    print("🎨 Creating Anomaly Visualization...")
    print("=" * 60)
    
    # Get actual anomaly flags from results if available
    if analysis_results is not None and 'graph_is_anomaly' in analysis_results.columns:
        anomaly_flags = analysis_results['graph_is_anomaly'].values
    else:
        # Use simple threshold
        threshold = np.percentile(anomaly_scores, 75)  # Top 25% as anomalies
        anomaly_flags = anomaly_scores > threshold
    
    # Intelligent sampling strategy
    print(f"📊 Sampling strategy for {sample_size} points...")
    
    # Sample baseline
    n_baseline = len(baseline_embeddings)
    n_analysis = len(analysis_embeddings)
    
    # Calculate sample sizes
    if n_baseline + n_analysis <= sample_size:
        # Use all data if it fits
        baseline_sample_idx = np.arange(n_baseline)
        analysis_sample_idx = np.arange(n_analysis)
        print(f"   Using all {n_baseline + n_analysis} points (no sampling needed)")
    else:
        # Proportional sampling with anomaly oversampling
        baseline_sample_size = min(n_baseline, sample_size // 3)
        analysis_sample_size = sample_size - baseline_sample_size
        
        # Sample baseline randomly
        baseline_sample_idx = np.random.RandomState(random_state).choice(
            n_baseline, baseline_sample_size, replace=False
        )
        
        # For analysis: oversample anomalies
        anomaly_indices = np.where(anomaly_flags)[0]
        normal_indices = np.where(~anomaly_flags)[0]
        
        # Take all anomalies if they fit, otherwise sample
        if len(anomaly_indices) <= analysis_sample_size // 2:
            anomaly_sample = anomaly_indices
            normal_sample_size = analysis_sample_size - len(anomaly_sample)
            normal_sample = np.random.RandomState(random_state).choice(
                normal_indices, min(normal_sample_size, len(normal_indices)), replace=False
            )
        else:
            # Sample half anomalies, half normal
            anomaly_sample = np.random.RandomState(random_state).choice(
                anomaly_indices, analysis_sample_size // 2, replace=False
            )
            normal_sample = np.random.RandomState(random_state).choice(
                normal_indices, analysis_sample_size // 2, replace=False
            )
        
        analysis_sample_idx = np.concatenate([anomaly_sample, normal_sample])
        
        print(f"   Sampled {len(baseline_sample_idx)} baseline points")
        print(f"   Sampled {len(anomaly_sample)} anomalies + {len(normal_sample)} normal from analysis")
    
    # Create sampled datasets
    baseline_sample = baseline_embeddings[baseline_sample_idx]
    analysis_sample = analysis_embeddings[analysis_sample_idx]
    analysis_scores_sample = anomaly_scores[analysis_sample_idx]
    analysis_flags_sample = anomaly_flags[analysis_sample_idx]
    
    # Combine for dimensionality reduction
    all_embeddings = np.vstack([baseline_sample, analysis_sample])
    
    # Create labels
    baseline_labels = ['Baseline'] * len(baseline_sample)
    analysis_labels = ['Anomaly' if flag else 'Normal' for flag in analysis_flags_sample]
    all_labels = baseline_labels + analysis_labels
    
    print(f"📈 Final sample composition:")
    print(f"   • Baseline: {len(baseline_sample)}")
    print(f"   • Normal (analysis): {sum(1 for l in analysis_labels if l == 'Normal')}")
    print(f"   • Anomaly (analysis): {sum(1 for l in analysis_labels if l == 'Anomaly')}")
    
    # Dimensionality reduction
    print(f"\n🔄 Computing dimensionality reductions (method: {method})...")
    
    # Always compute PCA for reference
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(all_embeddings)
    print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    
    # Compute requested method(s)
    umap_coords = None
    tsne_coords = None
    
    # Track what methods we'll actually run
    run_umap = False
    run_tsne = False
    
    # Handle UMAP with automatic fallback
    if method in ['umap', 'both']:
        if UMAP_AVAILABLE:
            print("   Running UMAP...")
            umap_model = UMAP(
                n_components=2,
                n_neighbors=15,  # Local neighborhood size
                min_dist=0.1,    # Minimum distance between points
                metric='euclidean',
                random_state=random_state
            )
            umap_coords = umap_model.fit_transform(all_embeddings)
            print("   ✓ UMAP complete")
            run_umap = True
        else:
            print("   ⚠️ UMAP not available, falling back to t-SNE")
            run_tsne = True
    
    # Handle t-SNE
    if method in ['tsne', 'both'] or run_tsne:
        print("   Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(50, len(all_embeddings) // 4),
            random_state=random_state,
            learning_rate='auto',
            init='pca',
            # n_iter parameter removed as it's not accepted
        )
        tsne_coords = tsne.fit_transform(all_embeddings)
        print("   ✓ t-SNE complete")
        run_tsne = True
    
    # Create visualizations
    n_methods = sum([umap_coords is not None, tsne_coords is not None])
    
    # Handle case where no methods are available (should not happen with fallback)
    if n_methods == 0:
        print("❌ No dimensionality reduction methods available")
        return None
    
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Anomaly Detection Visualization', fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {'Baseline': '#2ecc71', 'Normal': '#3498db', 'Anomaly': '#e74c3c'}
    
    col_idx = 0
    
    # UMAP visualization
    if umap_coords is not None:
        ax1 = axes[0, col_idx]
        ax2 = axes[1, col_idx]
        
        # Main UMAP plot
        for category in ['Baseline', 'Normal', 'Anomaly']:
            mask = np.array(all_labels) == category
            if mask.any():
                ax1.scatter(
                    umap_coords[mask, 0], umap_coords[mask, 1],
                    c=colors[category], s=20, alpha=0.7,
                    label=f'{category} ({mask.sum()})'
                )
        
        ax1.set_title('UMAP Projection', fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # UMAP vs PCA comparison
        ax2.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                   c=umap_coords[:, 0], cmap='viridis', s=20, alpha=0.6)
        ax2.set_title('PCA colored by UMAP-1', fontsize=14)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(ax2.collections[0], ax=ax2, label='UMAP-1')
        
        col_idx += 1
    
    # t-SNE visualization
    if tsne_coords is not None:
        ax1 = axes[0, col_idx]
        ax2 = axes[1, col_idx]
        
        # Main t-SNE plot
        for category in ['Baseline', 'Normal', 'Anomaly']:
            mask = np.array(all_labels) == category
            if mask.any():
                ax1.scatter(
                    tsne_coords[mask, 0], tsne_coords[mask, 1],
                    c=colors[category], s=20, alpha=0.7,
                    label=f'{category} ({mask.sum()})'
                )
        
        ax1.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE vs PCA comparison
        ax2.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                   c=tsne_coords[:, 0], cmap='plasma', s=20, alpha=0.6)
        ax2.set_title('PCA colored by t-SNE-1', fontsize=14)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(ax2.collections[0], ax=ax2, label='t-SNE-1')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📊 VISUALIZATION SUMMARY:")
    print("-" * 40)
    print(f"Total points visualized: {len(all_embeddings)}")
    print(f"Anomaly rate in sample: {sum(1 for l in all_labels if l == 'Anomaly')/len(all_labels)*100:.1f}%")
    
    if umap_coords is not None and tsne_coords is not None:
        print("\n📝 Method Comparison:")
        print("   UMAP advantages shown: Global structure preservation")
        print("   t-SNE advantages shown: Local cluster detail")
    
    # Extract clusters for analysis
    cluster_results = None
    if umap_coords is not None:
        cluster_results = extract_anomaly_clusters(
            umap_coords, all_labels, analysis_sample_idx
        )
    
    return {
        'pca_coords': pca_coords,
        'umap_coords': umap_coords,
        'tsne_coords': tsne_coords,
        'labels': all_labels,
        'sample_indices': {
            'baseline': baseline_sample_idx,
            'analysis': analysis_sample_idx
        },
        'clusters': cluster_results
    }

def extract_anomaly_clusters(umap_coords, all_labels, analysis_sample_idx):
    """
    Extract and analyze anomaly clusters from UMAP coordinates
    
    Returns information about clusters containing anomalous routing paths
    """
    print(f"\n🔍 CLUSTER ANALYSIS:")
    print("=" * 40)
    
    # Focus only on analysis points (not baseline)
    n_baseline = sum(1 for l in all_labels if l == 'Baseline')
    analysis_coords = umap_coords[n_baseline:]
    analysis_labels = all_labels[n_baseline:]
    
    # Run DBSCAN clustering on analysis points
    clustering = DBSCAN(eps=2.0, min_samples=10).fit(analysis_coords)
    cluster_labels = clustering.labels_
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"   📊 Found {n_clusters} clusters, {n_noise} noise points")
    
    # Analyze each cluster
    cluster_info = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_size = cluster_mask.sum()
        
        # Count anomalies in this cluster
        cluster_analysis_labels = np.array(analysis_labels)[cluster_mask]
        anomaly_count = sum(1 for l in cluster_analysis_labels if l == 'Anomaly')
        anomaly_rate = anomaly_count / cluster_size if cluster_size > 0 else 0
        
        cluster_info[cluster_id] = {
            'size': cluster_size,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'coords': analysis_coords[cluster_mask],
            'sample_indices': analysis_sample_idx[cluster_mask]
        }
        
        print(f"   🔸 Cluster {cluster_id}: {cluster_size} paths, {anomaly_count} anomalies ({anomaly_rate:.1%})")
    
    # Sort clusters by anomaly rate
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['anomaly_rate'], reverse=True)
    
    print(f"\n🚨 TOP ANOMALOUS CLUSTERS:")
    for i, (cluster_id, info) in enumerate(sorted_clusters[:3]):
        print(f"   #{i+1} Cluster {cluster_id}: {info['anomaly_rate']:.1%} anomalous ({info['anomaly_count']}/{info['size']})")
    
    return {
        'cluster_labels': cluster_labels,
        'cluster_info': cluster_info,
        'sorted_clusters': sorted_clusters,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

def compare_all_methods(detector, baseline_embeddings, analysis_embeddings, 
                        anomaly_scores, analysis_results=None, sample_size=2000):
    """
    Compare UMAP, t-SNE, and PCA side by side
    """
    print("🔬 Comparing All Dimensionality Reduction Methods...")
    print("=" * 60)
    
    return create_anomaly_visualization(
        detector, baseline_embeddings, analysis_embeddings,
        anomaly_scores, analysis_results,
        sample_size=sample_size,
        method='both'
    )