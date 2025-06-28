import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from synqronix.dataproc.utils import process_mat
import networkx as nx
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

def load_session_data(session_path):
    """Load and process a session's data file"""
    mat_data = scipy.io.loadmat(session_path)
    return process_mat(mat_data)

def plot_3d_neurons_by_bf(data, planes=[2, 3, 4, 5], min_neurons=50, 
                          plot_type='matplotlib', interactive=False):
    """
    Plot neurons in 3D space colored by their best frequency (BF)
    
    Args:
        data: Processed mat data
        planes: Which imaging planes to include
        min_neurons: Minimum number of neurons required to process a plane
        plot_type: 'matplotlib' or 'plotly'
        interactive: Whether to show the plot or return the figure
    """
    # Collect neuron positions and BF values
    xs, ys, zs, bf_values = [], [], [], []
    
    for plane in planes:
        if plane not in data['allxc'] or data['allxc'][plane].shape[0] < min_neurons:
            continue
            
        # Get coordinates
        x_coords = data['allxc'][plane].flatten()
        y_coords = data['allyc'][plane].flatten()
        z_coords = data['allzc'][plane].flatten()
        
        # Get BF values (make sure they exist for this plane)
        if plane in data['BFInfo'] and 'BFval' in data['BFInfo'][plane]:
            bf = data['BFInfo'][plane]['BFval']
            
            # Only include neurons with valid BF values
            valid_idx = ~np.isnan(bf)
            
            xs.extend(x_coords[valid_idx])
            ys.extend(y_coords[valid_idx])
            zs.extend(z_coords[valid_idx])
            bf_values.extend(bf[valid_idx])
    
    if plot_type == 'matplotlib':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with color based on BF
        scatter = ax.scatter(xs, ys, zs, c=bf_values, cmap='viridis', 
                           s=30, alpha=0.7)
        
        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Best Frequency (BF) Index')
        
        # Set labels
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_zlabel('Z position (μm)')
        ax.set_title('3D Map of Neurons Colored by Best Frequency')
        
        if interactive:
            plt.show()
        return fig
        
    elif plot_type == 'plotly':
        # Create interactive plotly figure
        fig = go.Figure(data=[go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(
                size=5,
                color=bf_values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="BF Index")
            ),
            hovertext=[f"BF: {bf}" for bf in bf_values],
        )])
        
        fig.update_layout(
            title='3D Map of Neurons Colored by Best Frequency',
            scene=dict(
                xaxis_title='X Position (μm)',
                yaxis_title='Y Position (μm)',
                zaxis_title='Z Position (μm)',
            ),
            width=900,
            height=700,
        )
        
        if interactive:
            fig.show()
        return fig

def plot_correlation_matrix(data, plane=3, correlation_type='signal', threshold=None):
    """
    Plot correlation matrix between neurons
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to visualize
        correlation_type: 'signal' or 'noise'
        threshold: Optional threshold to mask weak correlations
    """
    if plane not in data['CorrInfo']:
        print(f"Plane {plane} not found in data")
        return None
        
    # Get correlation matrix
    if correlation_type == 'signal':
        corr_matrix = data['CorrInfo'][plane]['SigCorrs']
    elif correlation_type == 'noise':
        corr_matrix = data['CorrInfo'][plane]['NoiseCorrsVec']
    else:
        raise ValueError("correlation_type must be 'signal' or 'noise'")
    
    # Apply threshold if specified
    if threshold is not None:
        corr_matrix_masked = corr_matrix.copy()
        corr_matrix_masked[np.abs(corr_matrix) < threshold] = 0
    else:
        corr_matrix_masked = corr_matrix
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_masked, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1)
    plt.title(f"{correlation_type.capitalize()} Correlation Matrix, Plane {plane}")
    plt.tight_layout()
    return plt.gcf()

def plot_functional_graph(data, plane=3, threshold=0.5, layout='spring', 
                         color_by='bf', color_map='viridis', node_size=50):
    """
    Plot functional connectivity graph between neurons
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to visualize
        threshold: Correlation threshold for edge creation
        layout: Graph layout algorithm ('spring', 'kamada_kawai', etc.)
        color_by: 'bf' for Best Frequency or 'position' for depth
        color_map: Matplotlib colormap to use
    """
    if plane not in data['CorrInfo'] or plane not in data['BFInfo']:
        print(f"Required data for plane {plane} not found")
        return None
    
    # Get correlation matrix and create graph
    corr_matrix = data['CorrInfo'][plane]['SigCorrs']
    G = nx.Graph()
    
    # Add nodes
    num_neurons = corr_matrix.shape[0]
    G.add_nodes_from(range(num_neurons))
    
    # Add edges where correlation > threshold
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=corr_matrix[i, j])
    
    # Get node colors based on BF
    if color_by == 'bf' and 'BFval' in data['BFInfo'][plane]:
        node_colors = data['BFInfo'][plane]['BFval']
    elif color_by == 'position':
        node_colors = data['allzc'][plane].flatten()
    else:
        node_colors = '#1f78b4'  # Default blue
    
    # Get positions for visualization
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'original':
        # Use actual x,y coordinates
        pos = {i: (data['allxc'][plane][i][0], data['allyc'][plane][i][0]) 
               for i in range(num_neurons)}
    else:
        pos = nx.spring_layout(G)
    
    # Plot
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_size, alpha=0.8, 
                          cmap=plt.get_cmap(color_map))
    
    # Draw edges with width proportional to correlation strength
    edge_weights = [G[u][v]['weight']*2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    plt.title(f"Functional Connectivity Graph (Threshold={threshold})")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(color_map))
    sm.set_array(node_colors)
    # cbar = plt.colorbar(sm, cax=plt.gca())
    # cbar.set_label('Best Frequency Index' if color_by == 'bf' else 'Z Position')
    
    plt.axis('off')
    return plt.gcf()

def compare_structural_vs_functional(data, plane=3, k=5, corr_threshold=0.5):
    """
    Compare structural (k-NN) vs functional connectivity
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to visualize
        k: Number of nearest neighbors for structural graph
        corr_threshold: Threshold for functional graph
    """
    if plane not in data['allxc'] or plane not in data['CorrInfo']:
        print(f"Required data for plane {plane} not found")
        return None
    
    # Get neuron positions and correlations
    x_coords = data['allxc'][plane].flatten()
    y_coords = data['allyc'][plane].flatten()
    corr_matrix = data['CorrInfo'][plane]['SigCorrs']
    num_neurons = len(x_coords)
    
    # Create structural graph (k-NN)
    G_struct = nx.Graph()
    G_struct.add_nodes_from(range(num_neurons))
    
    # Compute pairwise distances and add k nearest neighbors
    for i in range(num_neurons):
        distances = []
        for j in range(num_neurons):
            if i != j:
                dist = np.sqrt((x_coords[i] - x_coords[j])**2 + 
                              (y_coords[i] - y_coords[j])**2)
                distances.append((j, dist))
        
        # Sort by distance and add edges to k nearest neighbors
        distances.sort(key=lambda x: x[1])
        for j, _ in distances[:k]:
            G_struct.add_edge(i, j)
    
    # Create functional graph
    G_func = nx.Graph()
    G_func.add_nodes_from(range(num_neurons))
    
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            if abs(corr_matrix[i, j]) > corr_threshold:
                G_func.add_edge(i, j)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Use same positions for both graphs
    pos = {i: (x_coords[i], y_coords[i]) for i in range(num_neurons)}
    
    # Plot structural graph
    nx.draw_networkx_nodes(G_struct, pos, node_size=30, ax=ax1)
    nx.draw_networkx_edges(G_struct, pos, alpha=0.5, ax=ax1)
    ax1.set_title(f"Structural Graph (k={k})")
    ax1.axis('off')
    
    # Plot functional graph
    nx.draw_networkx_nodes(G_func, pos, node_size=30, ax=ax2)
    nx.draw_networkx_edges(G_func, pos, alpha=0.5, ax=ax2)
    ax2.set_title(f"Functional Graph (Threshold={corr_threshold})")
    ax2.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def plot_distance_vs_correlation(data, plane=3, max_distance=200, n_bins=20):
    """
    Plot relationship between physical distance and functional correlation
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to visualize
        max_distance: Maximum distance to include (μm)
        n_bins: Number of distance bins for averaging
    """
    if plane not in data['allxc'] or plane not in data['CorrInfo']:
        print(f"Required data for plane {plane} not found")
        return None
    
    # Get coordinates and correlation matrix
    x_coords = data['allxc'][plane].flatten()
    y_coords = data['allyc'][plane].flatten()
    corr_matrix = data['CorrInfo'][plane]['SigCorrs']
    num_neurons = len(x_coords)
    
    # Calculate distances and collect correlations
    distances = []
    correlations = []
    
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            dist = np.sqrt((x_coords[i] - x_coords[j])**2 + 
                         (y_coords[i] - y_coords[j])**2)
            
            if dist <= max_distance:
                distances.append(dist)
                correlations.append(corr_matrix[i, j])
    
    # Convert to numpy arrays
    distances = np.array(distances)
    correlations = np.array(correlations)
    
    # Create distance bins and calculate average correlation
    bin_edges = np.linspace(0, max_distance, n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mean_corrs = []
    std_corrs = []
    
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        if np.sum(mask) > 0:
            mean_corrs.append(np.mean(correlations[mask]))
            std_corrs.append(np.std(correlations[mask]))
        else:
            mean_corrs.append(np.nan)
            std_corrs.append(np.nan)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of raw data (with alpha for density)
    plt.scatter(distances, correlations, alpha=0.1, color='gray', s=10)
    
    # Plot average correlation by distance
    plt.errorbar(bin_centers, mean_corrs, yerr=std_corrs, 
                capsize=3, color='red', linewidth=2)
    
    plt.xlabel('Physical Distance (μm)')
    plt.ylabel('Signal Correlation')
    plt.title('Distance vs Signal Correlation')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_bf_distribution(data, planes=[2, 3, 4, 5]):
    """
    Plot distribution of best frequencies across neurons
    """
    all_bfs = []
    
    for plane in planes:
        if plane in data['BFInfo'] and 'BFval' in data['BFInfo'][plane]:
            bf_values = data['BFInfo'][plane]['BFval']
            valid_bfs = bf_values[~np.isnan(bf_values)]
            all_bfs.extend(valid_bfs)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(all_bfs, kde=True)
    plt.xlabel('Best Frequency (BF) Index')
    plt.ylabel('Count')
    plt.title('Distribution of Best Frequencies')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_neuron_embeddings(data, plane=3, embedding_method='tsne', 
                          perplexity=30, n_iter=1000):
    """
    Create 2D embeddings of neurons based on functional similarity
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to visualize
        embedding_method: Currently supports 'tsne'
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
    """
    if plane not in data['CorrInfo'] or plane not in data['BFInfo']:
        print(f"Required data for plane {plane} not found")
        return None
    
    # Get correlation matrix
    corr_matrix = data['CorrInfo'][plane]['SigCorrs']
    
    # Convert correlation to distance: higher correlation = lower distance
    dist_matrix = 1 - corr_matrix
    
    # Ensure diagonal is 0 (self-distance)
    np.fill_diagonal(dist_matrix, 0)
    
    # Apply t-SNE for dimensionality reduction
    embedder = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                   metric='precomputed')
    embeddings = embedder.fit_transform(dist_matrix)
    
    # Get best frequency for coloring
    if 'BFval' in data['BFInfo'][plane]:
        bf_values = data['BFInfo'][plane]['BFval']
    else:
        bf_values = np.ones(corr_matrix.shape[0])
    
    # Plot embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=bf_values, cmap='viridis', s=50, alpha=0.7)
    
    plt.colorbar(scatter, label='Best Frequency Index')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title(f'Neuron Embeddings Based on Functional Similarity (t-SNE)')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def visualize_activity_over_time(data, plane=3, neuron_indices=None, n_neurons=5, 
                               stim_highlights=True):
    """
    Visualize neural activity over time for selected neurons
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to visualize
        neuron_indices: List of specific neurons to plot (if None, randomly select)
        n_neurons: Number of neurons to plot if neuron_indices is None
        stim_highlights: Whether to highlight stimulation periods
    """
    if plane not in data['zDFF'] or plane not in data['zStuff']:
        print(f"Required data for plane {plane} not found")
        return None
    
    # Get activity traces
    activity = data['zDFF'][plane]
    
    # Select neurons to plot
    if neuron_indices is None:
        n_total = activity.shape[0]
        if n_neurons > n_total:
            n_neurons = n_total
        neuron_indices = np.random.choice(n_total, n_neurons, replace=False)
    
    # Create time axis (assuming 30Hz frame rate from exptVars)
    n_frames = activity.shape[1]
    frame_rate = 30  # Default if not provided
    if 'frameRate' in data['exptVars']:
        frame_rate = data['exptVars']['frameRate']
    time = np.arange(n_frames) / frame_rate
    
    # Get stimulus timing info if available
    stim_frames = None
    if stim_highlights and 'zStimFrame' in data['zStuff'][plane]:
        stim_frames = data['zStuff'][plane]['zStimFrame'].astype(int)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    for i, idx in enumerate(neuron_indices):
        # Offset each trace for visibility
        offset = i * 3
        plt.plot(time, activity[idx] + offset, label=f'Neuron {idx}')
    
    # Highlight stimulus periods if available
    if stim_frames is not None:
        stim_duration = 1.0  # 1 second stimulus duration (customize if known)
        
        for stim_frame in stim_frames:
            stim_start = stim_frame / frame_rate
            plt.axvspan(stim_start, stim_start + stim_duration, 
                       color='lightgray', alpha=0.3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('ΔF/F (offset for visibility)')
    plt.title(f'Neural Activity Traces - Plane {plane}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def create_example_graphs_for_gnn(data, plane=3, method='functional', 
                                threshold=0.5, k=5, n_samples=6, 
                                min_nodes=10, max_nodes=20, seed=42):
    """
    Create example graphs for GNN tasks by sampling from the larger graph
    
    Args:
        data: Processed mat data
        plane: Which imaging plane to use
        method: 'functional' or 'structural' for edge creation
        threshold: Correlation threshold for functional graphs
        k: Number of nearest neighbors for structural graphs
        n_samples: Number of subgraphs to create
        min_nodes, max_nodes: Min/max size of sampled graphs
        seed: Random seed for reproducibility
    
    Returns:
        List of NetworkX graphs with node features
    """
    np.random.seed(seed)
    
    if plane not in data['allxc'] or plane not in data['CorrInfo'] or plane not in data['BFInfo']:
        print(f"Required data for plane {plane} not found")
        return []
    
    # Get coordinates and correlations
    x_coords = data['allxc'][plane].flatten()
    y_coords = data['allyc'][plane].flatten()
    z_coords = data['allzc'][plane].flatten() if plane in data['allzc'] else np.zeros_like(x_coords)
    corr_matrix = data['CorrInfo'][plane]['SigCorrs']
    
    # Get BF values if available
    if 'BFval' in data['BFInfo'][plane]:
        bf_values = data['BFInfo'][plane]['BFval']
    else:
        bf_values = np.ones_like(x_coords)
    
    # Get activity traces
    activity = data['zDFF'][plane]
    
    # Create list to store graphs
    graphs = []
    
    # Function to create a single graph
    def create_graph(node_indices):
        G = nx.Graph()
        
        # Add nodes with features
        for i, idx in enumerate(node_indices):
            G.add_node(i, 
                      pos_x=float(x_coords[idx]),
                      pos_y=float(y_coords[idx]), 
                      pos_z=float(z_coords[idx]),
                      bf=float(bf_values[idx]) if not np.isnan(bf_values[idx]) else 0.0,
                      activity=activity[idx].tolist() if idx < activity.shape[0] else []
                     )
        
        # Add edges based on method
        if method == 'functional':
            for i, idx_i in enumerate(node_indices):
                for j, idx_j in enumerate(node_indices):
                    if i != j and abs(corr_matrix[idx_i, idx_j]) > threshold:
                        G.add_edge(i, j, weight=float(corr_matrix[idx_i, idx_j]))
        
        elif method == 'structural':
            # Compute pairwise distances within the subgraph
            for i, idx_i in enumerate(node_indices):
                distances = []
                for j, idx_j in enumerate(node_indices):
                    if i != j:
                        dist = np.sqrt((x_coords[idx_i] - x_coords[idx_j])**2 + 
                                     (y_coords[idx_i] - y_coords[idx_j])**2)
                        distances.append((j, dist))
                
                # Connect to k nearest neighbors (or fewer if subgraph is small)
                k_local = min(k, len(distances))
                distances.sort(key=lambda x: x[1])
                for j, dist in distances[:k_local]:
                    G.add_edge(i, j, weight=float(1.0/dist if dist > 0 else 1.0))
        
        return G
    
    # Create n_samples subgraphs
    num_neurons = len(x_coords)
    
    for _ in range(n_samples):
        # Choose random subset size between min_nodes and max_nodes
        subset_size = np.random.randint(min_nodes, min(max_nodes+1, num_neurons))
        
        # Select random subset of neurons
        node_indices = np.random.choice(num_neurons, subset_size, replace=False)
        
        # Create and store graph
        G = create_graph(node_indices)
        graphs.append(G)
    
    return graphs

def plot_sample_graphs(graphs, colorby='bf', layout='spring', figsize=(15, 10)):
    """
    Plot sample graphs for visual inspection
    
    Args:
        graphs: List of NetworkX graphs
        colorby: Node attribute to use for coloring ('bf' or 'pos_z')
        layout: Graph layout algorithm
    """
    n_graphs = len(graphs)
    n_rows = (n_graphs + 2) // 3  # Adjust rows based on number of graphs
    
    fig, axes = plt.subplots(n_rows, min(n_graphs, 3), figsize=figsize)
    
    # Handle case of single graph (make axes iterable)
    if n_graphs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, G in enumerate(graphs):
        row, col = i // 3, i % 3
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Get position layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'original':
            pos = {n: (G.nodes[n]['pos_x'], G.nodes[n]['pos_y']) for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Get node colors
        if colorby == 'bf':
            node_colors = [G.nodes[n]['bf'] for n in G.nodes()]
        elif colorby == 'pos_z':
            node_colors = [G.nodes[n]['pos_z'] for n in G.nodes()]
        else:
            node_colors = 'skyblue'
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=100, alpha=1, cmap='viridis',
                              ax=ax)
        
        # Draw edges with width based on weight if available
        if 'weight' in G.edges[list(G.edges())[0]] if G.edges else False:
            edge_weights = [G[u][v]['weight']*2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, alpha=1, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, alpha=1, ax=ax)
        
        ax.set_title(f'Graph {i+1} ({len(G.nodes())} nodes, {len(G.edges())} edges)')
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(n_graphs, n_rows * 3):
        row, col = i // 3, i % 3
        if n_rows > 1:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Example usage of the visualization functions"""
    # Path to data directory
    data_dir = Path("data/Auditory_cortex_data")
    
    # Find first valid session
    for session_dir in data_dir.iterdir():
        if session_dir.is_dir():
            mat_file = session_dir / "allPlanesVariables27-Feb-2021.mat"
            if mat_file.exists():
                print(f"Loading data from {mat_file}")
                data = load_session_data(mat_file)
                break
    
    # Create some example visualizations
    plot_3d_neurons_by_bf(data, interactive=True)
    plot_correlation_matrix(data, plane=3, threshold=0.2)
    plot_functional_graph(data, plane=3, threshold=0.5)
    plot_distance_vs_correlation(data)
    plot_bf_distribution(data)
    
    # Create sample graphs for GNN tasks
    func_graphs = create_example_graphs_for_gnn(data, method='functional')
    struct_graphs = create_example_graphs_for_gnn(data, method='structural')
    
    plot_sample_graphs(func_graphs)
    plot_sample_graphs(struct_graphs)
    
    plt.show()

if __name__ == "__main__":
    main()