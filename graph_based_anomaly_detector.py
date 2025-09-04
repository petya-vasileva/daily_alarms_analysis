#!/usr/bin/env python3
"""
Graph-Based ASN Anomaly Detection

Implements graph-based approach for detecting routing anomalies:
1. Build ASN co-occurrence graph from observed paths
2. Learn node embeddings using node2vec
3. Generate path embeddings from hop embeddings
4. Train isolation forest on baseline embeddings
5. Detect anomalies using isolation forest scores
"""

# Standard library imports
import warnings
from collections import defaultdict, Counter

# Third-party imports
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Optional imports
try:
    from utils.asn_classifier import classify_asn_type
except ImportError:
    def classify_asn_type(asn):
        return 'UNKNOWN'

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    Node2Vec = None
    NODE2VEC_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')


class ASNGraph:
    
    def __init__(self, min_edge_weight=2, directed=True):
        """
        Initialize ASN graph builder
        
        Parameters:
        -----------
        min_edge_weight : int
            Minimum co-occurrence count to create edge
        directed : bool
            Whether to build directed graph (order matters)
        """
        self.min_edge_weight = min_edge_weight
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.asn_statistics = {}
        
    def build_from_paths(self, trace_df):
        """
        Build ASN co-occurrence graph from traceroute paths
        
        Parameters:
        -----------
        trace_df : DataFrame
            Traceroute data with 'asns' column
        """
        print(f"🕸️ Building ASN co-occurrence graph from {len(trace_df)} paths...")
        
        # Count ASN statistics and co-occurrences
        asn_counts = Counter()
        edge_weights = defaultdict(int)
        paths_processed = 0
        
        for _, trace in trace_df.iterrows():
            asn_path = trace.get('asns', [])
            if not asn_path or len(asn_path) < 2:
                continue
                
            # Filter out zero ASNs and ensure integers
            clean_path = [int(asn) for asn in asn_path if asn and asn != 0]
            if len(clean_path) < 2:
                continue
                
            paths_processed += 1
            
            # Count individual ASNs
            for asn in clean_path:
                asn_counts[asn] += 1
            
            # Count ASN co-occurrences (adjacent hops)
            for i in range(len(clean_path) - 1):
                asn1, asn2 = clean_path[i], clean_path[i + 1]
                if asn1 != asn2:  # Skip self-loops
                    if self.directed:
                        edge_weights[(asn1, asn2)] += 1
                    else:
                        edge = tuple(sorted([asn1, asn2]))
                        edge_weights[edge] += 1
        
        # Build NetworkX graph
        print(f"   📊 Processing {len(asn_counts)} unique ASNs and {len(edge_weights)} potential edges...")
        
        # Add nodes with frequency and provider type attributes
        for asn, count in asn_counts.items():
            provider_type = classify_asn_type(asn)
            is_ren = 1 if 'REN' in provider_type else 0
            is_commodity = 1 if provider_type == 'COMMODITY' else 0
            
            self.graph.add_node(asn, 
                              frequency=count,
                              provider_type=provider_type,
                              is_ren=is_ren,
                              is_commodity=is_commodity)
        
        # Add edges with weight >= threshold
        edges_added = 0
        for edge, weight in edge_weights.items():
            if weight >= self.min_edge_weight:
                self.graph.add_edge(edge[0], edge[1], weight=weight)
                edges_added += 1
        
        # Store statistics
        if self.directed and edges_added > 0:
            components = list(nx.weakly_connected_components(self.graph))
            largest_component_size = len(max(components, key=len)) if components else 0
        elif not self.directed and edges_added > 0:
            components = list(nx.connected_components(self.graph))
            largest_component_size = len(max(components, key=len)) if components else 0
        else:
            largest_component_size = 0
            
        self.asn_statistics = {
            'paths_processed': paths_processed,
            'unique_asns': len(asn_counts),
            'edges_added': edges_added,
            'graph_density': nx.density(self.graph),
            'largest_component_size': largest_component_size
        }
        
        print(f"   ✅ Graph construction complete:")
        print(f"      • ASN nodes: {self.graph.number_of_nodes()}")
        print(f"      • Edges: {self.graph.number_of_edges()}")
        print(f"      • Density: {self.asn_statistics['graph_density']:.4f}")
        print(f"      • Largest component: {self.asn_statistics['largest_component_size']} nodes")
        
        return self.asn_statistics
    
    def plot_network(self, max_nodes=100, figsize=(12, 10), save_path=None):
        """
        Visualize the ASN co-occurrence network
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ matplotlib not available - cannot plot network")
            return None
            
        if self.graph.number_of_nodes() == 0:
            print("⚠️ Empty graph - nothing to plot")
            return None
            
        print(f"🎨 Plotting ASN network ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)...")
        
        # For large networks, show only the most connected nodes
        if self.graph.number_of_nodes() > max_nodes:
            print(f"   📊 Network too large, showing top {max_nodes} highest-degree nodes")
            node_degrees = dict(self.graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            subgraph = self.graph.subgraph(top_node_ids)
        else:
            subgraph = self.graph
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Use spring layout for better visualization
        try:
            pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(subgraph, seed=42)
            
        # Node sizes based on degree
        node_degrees = dict(subgraph.degree())
        node_sizes = [max(50, min(1000, degree * 20)) for _, degree in node_degrees.items()]
        
        # Edge widths based on weight
        edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [max(0.5, min(3, weight / max_weight * 3)) for weight in edge_weights]
        
        # Node colors based on provider type
        node_colors = []
        for node in subgraph.nodes():
            provider_type = subgraph.nodes[node].get('provider_type', 'UNKNOWN')
            if 'REN' in provider_type:
                node_colors.append('lightblue')
            elif provider_type == 'COMMODITY':
                node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        # Draw the network
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_size=node_sizes,
                              node_color=node_colors,
                              alpha=0.7,
                              ax=ax)
                              
        nx.draw_networkx_edges(subgraph, pos,
                              width=edge_widths,
                              alpha=0.5,
                              edge_color='gray',
                              ax=ax)
                              
        # Add labels for high-degree nodes only
        high_degree_nodes = {node: f"AS{node}" for node, degree in node_degrees.items() if degree >= 5}
        nx.draw_networkx_labels(subgraph, pos, 
                               labels=high_degree_nodes,
                               font_size=8,
                               font_weight='bold',
                               ax=ax)
        
        # Add title and legend
        ax.set_title(f"ASN Co-occurrence Network\\n{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges", 
                    fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            mpatches.Circle((0, 0), 0.1, facecolor='lightblue', label='REN/WLCG Network'),
            mpatches.Circle((0, 0), 0.1, facecolor='orange', label='Commodity Network'),
            mpatches.Circle((0, 0), 0.1, facecolor='gray', label='Unknown Network'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_axis_off()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Network plot saved to: {save_path}")
            
        plt.show()
        return fig, ax


class PathEmbeddingGenerator:
    """Generate embeddings for ASN paths using node2vec + structural features"""
    
    def __init__(self, embedding_dim=64, walk_length=30, num_walks=200, 
                 workers=4, window=5, min_count=1, batch_words=4):
        """Initialize path embedding generator"""
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        
        self.node_embeddings = {}
        self.scaler = StandardScaler()
        
    def train_node2vec(self, asn_graph):
        """Train node2vec embeddings on ASN graph"""
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec library not available. Install with: pip install node2vec")
            
        print(f"🧠 Training node2vec embeddings...")
        
        if asn_graph.graph.number_of_nodes() == 0:
            raise ValueError("Empty graph - cannot train embeddings")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            asn_graph.graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            p=1,  # Return parameter (1 = unbiased)
            q=1   # In-out parameter (1 = unbiased)
        )
        
        # Train embeddings
        model = node2vec.fit(window=self.window, min_count=self.min_count, batch_words=self.batch_words)
        
        # Store embeddings
        self.node_embeddings = {}
        for asn in asn_graph.graph.nodes():
            try:
                self.node_embeddings[asn] = model.wv[str(asn)]
            except KeyError:
                # ASN not in vocabulary, use zero vector
                self.node_embeddings[asn] = np.zeros(self.embedding_dim)
        
        print(f"   ✅ Node2vec training complete:")
        print(f"      • Embedding dimension: {self.embedding_dim}")
        print(f"      • ASNs embedded: {len(self.node_embeddings)}")
        
        return len(self.node_embeddings)
    
    def generate_path_embedding(self, asn_path, include_structural=True):
        """Generate embedding for a single ASN path"""
        if not asn_path:
            return np.zeros(self.embedding_dim + (3 if include_structural else 0))
        
        # Clean path (remove zeros, convert to int)
        clean_path = [int(asn) for asn in asn_path if asn and asn != 0]
        if not clean_path:
            return np.zeros(self.embedding_dim + (3 if include_structural else 0))
        
        # Collect hop embeddings
        hop_embeddings = []
        for asn in clean_path:
            if asn in self.node_embeddings:
                hop_embeddings.append(self.node_embeddings[asn])
            else:
                # Unknown ASN - use zero vector
                hop_embeddings.append(np.zeros(self.embedding_dim))
        
        if not hop_embeddings:
            return np.zeros(self.embedding_dim + (3 if include_structural else 0))
        
        hop_embeddings = np.array(hop_embeddings)
        
        # Aggregate hop embeddings (mean is most stable)
        path_embedding = np.mean(hop_embeddings, axis=0)
        
        # Add structural features if requested
        if include_structural:
            path_length = len(clean_path)
            structural_features = np.array([
                path_length,                                     # Path length
                len(set(clean_path)),                           # Unique ASNs
                len(clean_path) / max(1, len(set(clean_path))), # ASN repetition ratio
            ])
            path_embedding = np.concatenate([path_embedding, structural_features])
        
        return path_embedding
    
    def generate_embeddings_batch(self, trace_df, include_structural=True):
        """Generate embeddings for batch of paths"""
        print(f"🔢 Generating path embeddings for {len(trace_df)} paths...")
        
        embeddings = []
        for _, trace in trace_df.iterrows():
            asn_path = trace.get('asns', [])
            embedding = self.generate_path_embedding(asn_path, include_structural)
            embeddings.append(embedding)
        
        embeddings_matrix = np.array(embeddings)
        
        print(f"   ✅ Generated embeddings: {embeddings_matrix.shape}")
        
        return embeddings_matrix


class GraphBasedAnomalyDetector:
    """Main class for graph-based ASN anomaly detection using Isolation Forest"""
    
    def __init__(self, embedding_dim=64, min_edge_weight=2, contamination=0.1):
        """
        Initialize graph-based anomaly detector
        
        Parameters:
        -----------
        embedding_dim : int
            Dimension of node embeddings
        min_edge_weight : int
            Minimum edge weight for graph construction
        contamination : float
            Expected proportion of anomalies (for Isolation Forest)
        """
        self.embedding_dim = embedding_dim
        self.min_edge_weight = min_edge_weight
        self.contamination = contamination
        
        # Components
        self.asn_graph = ASNGraph(min_edge_weight=min_edge_weight)
        self.path_embedder = PathEmbeddingGenerator(embedding_dim=embedding_dim)
        self.anomaly_model = None
        self.scaler = StandardScaler()
        
        # Training data for reference
        self.baseline_embeddings = None
        
    def train(self, trace_df):
        """Train the complete anomaly detection pipeline"""
        print(f"🚀 Training graph-based anomaly detector on {len(trace_df)} paths...")
        
        # Step 1: Build ASN co-occurrence graph
        graph_stats = self.asn_graph.build_from_paths(trace_df)
        
        if self.asn_graph.graph.number_of_nodes() == 0:
            raise ValueError("No valid ASN paths found in training data")
        
        # Step 2: Train node2vec embeddings
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec not available - install with: pip install node2vec")
            
        self.path_embedder.train_node2vec(self.asn_graph)
        
        # Step 3: Generate path embeddings
        self.baseline_embeddings = self.path_embedder.generate_embeddings_batch(trace_df)
        
        if len(self.baseline_embeddings) == 0:
            raise ValueError("No valid path embeddings generated")
        
        # Step 4: Fit scaler and isolation forest
        print(f"🤖 Training Isolation Forest anomaly detector...")
        
        # Fit scaler on baseline embeddings
        scaled_embeddings = self.scaler.fit_transform(self.baseline_embeddings)
        
        # Initialize and train Isolation Forest
        self.anomaly_model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_model.fit(scaled_embeddings)
        
        print(f"   ✅ Training complete!")
        print(f"      • Graph nodes: {self.asn_graph.graph.number_of_nodes()}")
        print(f"      • Graph edges: {self.asn_graph.graph.number_of_edges()}")
        print(f"      • Embedding dimension: {self.embedding_dim + 3}")  # Node embeddings + structural features
        print(f"      • Baseline paths: {len(self.baseline_embeddings)}")
        
        return {
            'graph_stats': graph_stats,
            'embedding_dim': self.embedding_dim + 3,
            'baseline_paths': len(self.baseline_embeddings),
            'anomaly_method': 'isolation_forest'
        }
    
    def detect_anomalies(self, trace_df):
        """Detect anomalies in new traceroute data"""
        print(f"🔍 Detecting anomalies in {len(trace_df)} paths...")
        
        if self.anomaly_model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Generate embeddings for new data
        test_embeddings = self.path_embedder.generate_embeddings_batch(trace_df)
        
        if len(test_embeddings) == 0:
            print("   ⚠️ No valid embeddings generated")
            return trace_df
        
        # Store analysis embeddings for reuse (avoid regenerating in visualization)
        self.analysis_embeddings = test_embeddings
        
        # Scale embeddings
        scaled_test = self.scaler.transform(test_embeddings)
        
        # Get anomaly scores and predictions
        anomaly_scores = self.anomaly_model.decision_function(scaled_test)
        anomaly_flags = self.anomaly_model.predict(scaled_test) == -1
        
        # Add results to DataFrame
        result_df = trace_df.copy()
        result_df['graph_anomaly_score'] = anomaly_scores
        result_df['graph_is_anomaly'] = anomaly_flags
        
        anomaly_count = anomaly_flags.sum()
        anomaly_rate = (anomaly_count / len(trace_df)) * 100
        
        print(f"   ✅ Anomaly detection complete:")
        print(f"      • Paths analyzed: {len(trace_df)}")
        print(f"      • Anomalies detected: {anomaly_count} ({anomaly_rate:.1f}%)")
        
        return result_df
    
    def get_anomaly_attribution(self, trace_row):
        """Get attribution for why a path was marked anomalous"""
        asn_path = trace_row.get('asns', [])
        if not asn_path:
            return {}
        
        clean_path = [int(asn) for asn in asn_path if asn and asn != 0]
        if not clean_path:
            return {}
        
        attribution = {
            'path_length': len(clean_path),
            'unique_asns': len(set(clean_path)),
            'hop_contributions': {}
        }
        
        # Check which ASNs are rare/unknown in the graph
        for i, asn in enumerate(clean_path):
            asn_info = {
                'position': i,
                'in_training_graph': asn in self.asn_graph.graph.nodes(),
                'has_embedding': asn in self.path_embedder.node_embeddings
            }
            
            if asn in self.asn_graph.graph.nodes():
                asn_info['graph_frequency'] = self.asn_graph.graph.nodes[asn].get('frequency', 0)
                asn_info['graph_degree'] = self.asn_graph.graph.degree(asn)
            else:
                asn_info['graph_frequency'] = 0
                asn_info['graph_degree'] = 0
                
            attribution['hop_contributions'][f'hop_{i}_asn_{asn}'] = asn_info
        
        return attribution
    
    def plot_asn_network(self, max_nodes=100, figsize=(12, 10), save_path=None):
        """Plot the ASN co-occurrence network"""
        if self.asn_graph is None:
            print("⚠️ ASN graph not built yet. Call train() first.")
            return None
            
        return self.asn_graph.plot_network(max_nodes=max_nodes, figsize=figsize, save_path=save_path)


def analyze_routing_with_graph_method(trace_df, baseline_days_before=2, 
                                    embedding_dim=32, min_edge_weight=2):
    """
    Complete graph-based routing anomaly analysis pipeline
    
    Parameters:
    -----------
    trace_df : DataFrame
        Traceroute data with 'asns' and 'dt' columns
    baseline_days_before : int
        Days before the main event period to use as baseline (default: 2)
    embedding_dim : int
        Node embedding dimension
    min_edge_weight : int
        Minimum co-occurrence for graph edges
        
    Returns:
    --------
    dict : Analysis results
    """
    print(f"\\n🕸️ GRAPH-BASED ROUTING ANOMALY ANALYSIS")
    print("=" * 60)
    
    if trace_df.empty:
        print("   ⚠️ No traceroute data provided")
        return {}
    
    # Sort by timestamp
    trace_df_sorted = trace_df.sort_values('dt').reset_index(drop=True)
    
    # Use days BEFORE the main event period as baseline
    trace_df_sorted['date'] = trace_df_sorted['dt'].dt.date
    daily_counts = trace_df_sorted.groupby('date').size()
    
    # Identify the peak event date
    peak_date = daily_counts.idxmax()
    peak_date_dt = pd.Timestamp(peak_date)
    
    # Use data from BEFORE the peak as baseline
    baseline_start = peak_date_dt - pd.Timedelta(days=baseline_days_before + 1)
    baseline_end = peak_date_dt - pd.Timedelta(hours=1)
    
    baseline_traces = trace_df_sorted[
        (trace_df_sorted['dt'] >= baseline_start) & 
        (trace_df_sorted['dt'] < baseline_end)
    ]
    analysis_traces = trace_df_sorted[trace_df_sorted['dt'] >= peak_date_dt]
    
    # Fallback if no baseline data
    if len(baseline_traces) < 50:
        print(f"   ⚠️ Insufficient baseline data before peak ({len(baseline_traces)} traces)")
        print(f"   📅 Falling back to first 30% of data as baseline")
        cutoff_idx = int(len(trace_df_sorted) * 0.3)
        baseline_traces = trace_df_sorted.iloc[:cutoff_idx]
        analysis_traces = trace_df_sorted.iloc[cutoff_idx:]
    
    print(f"📅 Time windows:")
    print(f"   • Baseline: {baseline_traces['dt'].min()} → {baseline_traces['dt'].max()}")
    print(f"   • Analysis: {analysis_traces['dt'].min()} → {analysis_traces['dt'].max()}")
    print(f"   • Baseline traces: {len(baseline_traces)}")
    print(f"   • Analysis traces: {len(analysis_traces)}")
    
    if len(baseline_traces) < 50:
        print("   ⚠️ Insufficient baseline data for graph construction")
        return {}
    
    # Initialize detector
    detector = GraphBasedAnomalyDetector(
        embedding_dim=embedding_dim,
        min_edge_weight=min_edge_weight
    )
    
    try:
        # Train on baseline data
        training_stats = detector.train(baseline_traces)
        
        # Detect anomalies in analysis window
        if not analysis_traces.empty:
            analysis_results = detector.detect_anomalies(analysis_traces)
            
            # Summarize results by site pair
            anomaly_summary = (analysis_results.groupby(['src_site', 'dest_site'])
                             .agg({
                                 'graph_is_anomaly': ['count', 'sum'],
                                 'graph_anomaly_score': ['mean', 'min', 'std']
                             }).round(3))
            
            # Flatten column names
            anomaly_summary.columns = ['total_traces', 'anomalies', 
                                     'mean_score', 'min_score', 'std_score']
            anomaly_summary['anomaly_rate'] = (anomaly_summary['anomalies'] / 
                                             anomaly_summary['total_traces'] * 100)
            anomaly_summary = anomaly_summary.round(3).sort_values('anomaly_rate', ascending=False)
            
        else:
            analysis_results = pd.DataFrame()
            anomaly_summary = pd.DataFrame()
            print("   ⚠️ No traces in analysis window")
        
        return {
            'detector': detector,
            'baseline_traces': baseline_traces,
            'analysis_traces': analysis_results,
            'anomaly_summary': anomaly_summary,
            'training_stats': training_stats,
            'total_anomalies': analysis_results['graph_is_anomaly'].sum() if not analysis_results.empty else 0,
            'anomaly_rate': (analysis_results['graph_is_anomaly'].sum() / len(analysis_results) * 100) 
                           if not analysis_results.empty else 0,
            'method': 'graph_based'
        }
        
    except Exception as e:
        print(f"   ❌ Graph-based analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}