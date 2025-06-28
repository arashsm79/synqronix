import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm

class NeuralGNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3, dropout_rate=0.5, model_type='GCN'):
        """
        Graph Neural Network for neural activity classification
        
        Args:
            num_features: Number of input node features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout_rate: Dropout rate
            model_type: Type of GNN layer ('GCN' or 'GAT')
        """
        super(NeuralGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if model_type == 'GCN':
            self.convs.append(GCNConv(num_features, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(num_features, hidden_dim, heads=8, concat=False))
        
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=False))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Last layer
        if model_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=False))
        
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_edge_features]
            batch: Batch vector [num_nodes] (for batched graphs)
        """
        # Apply GNN layers
        for i in range(self.num_layers):
            if self.model_type == 'GCN':
                x = self.convs[i](x, edge_index, edge_attr)
            else:  # GAT
                x = self.convs[i](x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.classifier(x)
        
        return x

class NeuralGNNWithAttention(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3, dropout_rate=0.5):
        """
        GNN with attention mechanism for better interpretability
        """
        super(NeuralGNNWithAttention, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # GAT layers with attention
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_features, hidden_dim, heads=8, concat=False, dropout=dropout_rate))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=False, dropout=dropout_rate))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Last layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout_rate))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Self-attention for final representation
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout_rate)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Apply GAT layers
        attention_weights = []
        for i in range(self.num_layers):
            x, att_weights = self.convs[i](x, edge_index, return_attention_weights=True)
            attention_weights.append(att_weights)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Self-attention
        x_att, _ = self.self_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x_att.squeeze(1)
        
        # Classification
        x = self.classifier(x)
        
        return x