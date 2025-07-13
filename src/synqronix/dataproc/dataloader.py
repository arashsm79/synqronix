import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.io
from src.synqronix.dataproc.utils import process_mat, create_neuron_info_df
from torch_geometric.utils import to_undirected
from itertools import combinations

class NeuralGraphDataLoader:
    def __init__(self, data_dir, add_hyperedges=False, k=20, connectivity_threshold=0.5, batch_size=32):
        """
        Args:
            data_dir: Directory containing .mat files
            add_hyperedges: Whether to add hyperedges to the graph
            k: Number of top connected neurons to consider for each neuron
            connectivity_threshold: Threshold for functional connectivity
            batch_size: Batch size for DataLoader
        """
        self.data_dir = data_dir
        self.k = k
        self.connectivity_threshold = connectivity_threshold
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.add_hyperedges = add_hyperedges
        
    def load_session_data(self, mat_file_path):
        """Load and process a single session"""
        mat_data = scipy.io.loadmat(mat_file_path)
        processed_data = process_mat(mat_data)
        
        temp_path = mat_file_path.replace('.mat', '_neuron_info.pkl')
        neuron_df = create_neuron_info_df(processed_data, temp_path)
        
        return neuron_df
    
    def create_subgraph_for_neuron(self, neuron_df, center_neuron_idx):
        """Create a subgraph centered around a specific neuron"""
        center_neuron = neuron_df.iloc[center_neuron_idx]
        
        # Get correlations for center neuron
        correlations = np.array(center_neuron['global_corr'])
        
        # Find k most connected neurons above threshold
        valid_corrs = correlations.copy()
        valid_corrs[center_neuron_idx] = -1  # Exclude self-connection
        valid_indices = np.where(np.abs(valid_corrs) >= self.connectivity_threshold)[0]
        
        if len(valid_indices) == 0:
            # If no valid connections, create a minimal graph with just the center neuron
            subgraph_indices = [center_neuron_idx]
        else:
            # Get top k connections
            if len(valid_indices) > self.k:
                top_k_indices = valid_indices[np.argsort(np.abs(valid_corrs[valid_indices]))[-self.k:]]
            else:
                top_k_indices = valid_indices
            
            # Include center neuron and its top-k neighbors
            subgraph_indices = [center_neuron_idx] + list(top_k_indices)
        
        # Create node features for subgraph
        node_features = []
        node_labels = []
        
        for idx in subgraph_indices:
            row = neuron_df.iloc[idx]
            pc1, pc2 = row['PC'][0], row['PC'][1]
            x, y, z = row['x'], row['y'], row['z']
            bf_resp = row['BFresp']
            avg_activity = np.mean(row['activity'])
            
            features = [pc1, pc2, x, y, z, bf_resp, avg_activity]
            node_features.append(features)
            node_labels.append(row['BFval'])
        
        node_features = np.array(node_features)
        node_labels = np.array(node_labels)
        
        # Create mapping from original indices to subgraph indices
        # Maybe not necessary, but useful for later
        idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(subgraph_indices)}
        
        # Create edges within the subgraph
        edge_indices = []
        edge_weights = []
        
        for i, orig_i in enumerate(subgraph_indices):
            correlations_i = np.array(neuron_df.iloc[orig_i]['global_corr'])
            
            for j, orig_j in enumerate(subgraph_indices):
                if i != j:  # No self-loops
                    correlation = abs(correlations_i[orig_j])
                    if correlation >= self.connectivity_threshold:
                        edge_indices.append([i, j])
                        edge_weights.append(correlation)
        
        # If no edges found, create a self-loop for the center neuron to avoid empty graphs
        if len(edge_indices) == 0:
            edge_indices = [[0, 0]]  # Self-loop for center neuron (index 0)
            edge_weights = [0.1]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        # Make the graph undirected (except for self-loops)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        # Add hyperedges if required
        if self.add_hyperedges:
            # correlation (or distance) matrix *restricted to this subgraph*
            C_sub = np.vstack(neuron_df.iloc[subgraph_indices]['global_corr'])  # (n,n)

            hyperedges = []
            for i, j, k in combinations(range(len(subgraph_indices)), 3):
                if (abs(C_sub[i, j]) >= self.connectivity_threshold and
                    abs(C_sub[i, k]) >= self.connectivity_threshold and
                    abs(C_sub[j, k]) >= self.connectivity_threshold):
                    hyperedges.append([i, j, k])

            if not hyperedges:                               # guarantee non-empty
                hyperedges = [[0, 0, 0]] if x.size(0) < 3 else [[0, 1, 2]]

            # shape (3, num_triangles)
            hyperedge_index = torch.tensor(hyperedges,
                                        dtype=torch.long).t().contiguous()

            return Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        hyperedge_index=hyperedge_index)
        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def create_all_subgraphs_from_session(self, neuron_df):
        """Create all neuron-centered subgraphs from a session"""
        subgraphs = []
        num_neurons = len(neuron_df)
        
        for neuron_idx in range(num_neurons):
            subgraph = self.create_subgraph_for_neuron(neuron_df, neuron_idx)
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def load_all_sessions(self):
        """Load all .mat files and create subgraphs for each neuron"""
        mat_files = glob.glob(os.path.join(self.data_dir, '**', "*.mat"), recursive=True)
        all_subgraphs = []
        all_features = []
        all_labels = []
        
        print(f"Found {len(mat_files)} session files")
        
        for mat_file in mat_files:
            print(f"Processing {os.path.basename(mat_file)}")
            if "diffxy" in mat_file:
                print(f"Skipping {mat_file} because it has dubious values 🤨")
                continue
            try:
                neuron_df = self.load_session_data(mat_file)
                session_subgraphs = self.create_all_subgraphs_from_session(neuron_df)
                
                all_subgraphs.extend(session_subgraphs)
                
                # Collect features and labels for fitting scalers
                for subgraph in session_subgraphs:
                    all_features.append(subgraph.x.numpy())
                    all_labels.extend(subgraph.y.numpy())
                    
                print(f"Created {len(session_subgraphs)} subgraphs from session")
                
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                continue
        
        print(f"Total subgraphs created: {len(all_subgraphs)}")
        
        # Fit scalers on all data
        all_features_concat = np.vstack(all_features)
        self.scaler.fit(all_features_concat)
        self.label_encoder.fit(all_labels)
        
        # Apply scaling and label encoding to all subgraphs
        for subgraph in all_subgraphs:
            subgraph.x = torch.tensor(self.scaler.transform(subgraph.x.numpy()), dtype=torch.float)
            subgraph.y = torch.tensor(self.label_encoder.transform(subgraph.y.numpy()), dtype=torch.long)
        
        return all_subgraphs
    
    def split_data(self, subgraphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split subgraphs into train/val/test sets"""
        num_graphs = len(subgraphs)
        indices = np.random.permutation(num_graphs)
        
        train_end = int(train_ratio * num_graphs)
        val_end = int((train_ratio + val_ratio) * num_graphs)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_data = [subgraphs[i] for i in train_indices]
        val_data = [subgraphs[i] for i in val_indices]
        test_data = [subgraphs[i] for i in test_indices]
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def get_dataloaders(self, data_dir=None):
        """Get train, validation, and test dataloaders"""
        if data_dir:
            self.data_dir = data_dir
            
        subgraphs = self.load_all_sessions()
        train_data, val_data, test_data = self.split_data(subgraphs)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_num_features(self):
        """Get number of node features"""
        return 7  # PC1, PC2, x, y, z, BFresp, avg_activity
    
    def get_num_classes(self):
        """Get number of unique classes"""
        return len(self.label_encoder.classes_)

class ColumnarNeuralGraphDataLoader(NeuralGraphDataLoader):
    def __init__(self, data_dir, add_hyperedges=False, k=20, column_width=50, column_height=50, batch_size=32):
        """
        Args:
            data_dir: Directory containing .mat files
            k: Number of neurons to sample from the column for each neuron
            column_width: Width of the column in micrometers (15-100 um)
            column_height: Height of the column in micrometers (5-100 um)
            batch_size: Batch size for DataLoader
        """
        # Initialize parent class without connectivity_threshold
        super().__init__(data_dir, k, connectivity_threshold=0.0, batch_size=batch_size)
        self.column_width = column_width
        self.column_height = column_height
        self.add_hyperedges = add_hyperedges
        
    def find_columnar_neighbors(self, neuron_df, center_neuron_idx):
        """Find neurons within the same column as the center neuron"""
        center_neuron = neuron_df.iloc[center_neuron_idx]
        center_x, center_y, center_z = center_neuron['x'], center_neuron['y'], center_neuron['z']
        
        # Calculate spatial distances
        x_distances = np.abs(neuron_df['x'] - center_x)
        y_distances = np.abs(neuron_df['y'] - center_y)
        z_distances = np.abs(neuron_df['z'] - center_z)
        
        # Find neurons within columnar boundaries
        within_column_x = x_distances <= (self.column_width / 2)
        within_column_y = y_distances <= (self.column_width / 2)
        within_column_z = z_distances <= (self.column_height / 2)
        
        # Combine all conditions (exclude center neuron itself)
        columnar_mask = within_column_x & within_column_y & within_column_z
        columnar_indices = np.where(columnar_mask)[0]
        columnar_indices = columnar_indices[columnar_indices != center_neuron_idx]
        
        return columnar_indices
    
    def create_subgraph_for_neuron(self, neuron_df, center_neuron_idx):
        """Create a subgraph centered around a specific neuron using columnar sampling"""
        # Find neurons in the same column
        columnar_indices = self.find_columnar_neighbors(neuron_df, center_neuron_idx)
        
        if len(columnar_indices) == 0:
            # If no columnar neighbors, create a minimal graph with just the center neuron
            subgraph_indices = [center_neuron_idx]
        else:
            # Randomly sample k neurons from the column
            if len(columnar_indices) > self.k:
                sampled_indices = np.random.choice(columnar_indices, size=self.k, replace=False)
            else:
                sampled_indices = columnar_indices
            
            # Include center neuron and its sampled neighbors
            subgraph_indices = [center_neuron_idx] + list(sampled_indices)
        
        # Create node features for subgraph (same as parent class)
        node_features = []
        node_labels = []
        
        for idx in subgraph_indices:
            row = neuron_df.iloc[idx]
            pc1, pc2 = row['PC'][0], row['PC'][1]
            x, y, z = row['x'], row['y'], row['z']
            bf_resp = row['BFresp']
            avg_activity = np.mean(row['activity'])
            
            features = [pc1, pc2, x, y, z, bf_resp, avg_activity]
            node_features.append(features)
            node_labels.append(row['BFval'])
        
        node_features = np.array(node_features)
        node_labels = np.array(node_labels)
        
        # Create edges based on K-nearest neighbors in 3D space
        edge_indices = []
        edge_weights = []
        
        for i, orig_i in enumerate(subgraph_indices):
            neuron_i = neuron_df.iloc[orig_i]
            x_i, y_i, z_i = neuron_i['x'], neuron_i['y'], neuron_i['z']
            
            # Calculate distances to all other neurons
            distances = []
            for j, orig_j in enumerate(subgraph_indices):
                if i != j:
                    neuron_j = neuron_df.iloc[orig_j]
                    x_j, y_j, z_j = neuron_j['x'], neuron_j['y'], neuron_j['z']
                    distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2)
                    distances.append((j, distance))
            
            # Connect to K nearest neighbors (or all if fewer than K)
            k_neighbors = min(3, len(distances))  # Connect to 3 nearest neighbors
            distances.sort(key=lambda x: x[1])
            
            for j, distance in distances[:k_neighbors]:
                weight = 1.0 / (distance + 1e-6)
                edge_indices.append([i, j])
                edge_weights.append(weight)
        
        # If no edges found, create a self-loop for the center neuron
        if len(edge_indices) == 0:
            edge_indices = [[0, 0]]  # Self-loop for center neuron (index 0)
            edge_weights = [1.0]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        # Make the graph undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        # Add hyperedges
        if self.add_hyperedges:

            C_sub = np.vstack(neuron_df.iloc[subgraph_indices]['global_corr'])  # (n,n)

            hyperedges = []
            for i, j, k in combinations(range(len(subgraph_indices)), 3):
                if (abs(C_sub[i, j]) >= self.connectivity_threshold and
                    abs(C_sub[i, k]) >= self.connectivity_threshold and
                    abs(C_sub[j, k]) >= self.connectivity_threshold):
                    hyperedges.append([i, j, k])

            if not hyperedges:                               # guarantee non-empty
                hyperedges = [[0, 0, 0]] if x.size(0) < 3 else [[0, 1, 2]]

            # shape (3, num_triangles)
            hyperedge_index = torch.tensor(hyperedges,
                                        dtype=torch.long).t().contiguous()
            # ─────────────────────────────────────────────────────────────────── #

            return Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        hyperedge_index=hyperedge_index)   # ← only new field
        else: 
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        