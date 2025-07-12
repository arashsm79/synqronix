import os

from xml.parsers.expat import model
import torch
from torch.nn import (Module, ModuleList, Linear, LeakyReLU, 
                      Dropout, LogSoftmax, ReLU, Parameter, LayerNorm)
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing

from synqronix.quantum_models.qgcn_node_embedding import quantum_net
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import OptTensor



class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels=None, no_node_NN=False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        # to perform simple neighborhood aggregation (without NN to learn node embedding)
        self.no_NN = no_node_NN
        if not self.no_NN:
            assert out_channels is None or in_channels == out_channels
            out_channels = out_channels or in_channels
        else:
            out_channels = in_channels
            
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Apply GCN normalization (includes self-loops and symmetric normalization)
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight=edge_weight, num_nodes=x.size(0),
            improved=False, add_self_loops=True, flow='source_to_target',
            dtype=x.dtype
        )

        # Step 2: Linearly transform node feature matrix.
        if not self.no_NN:
            out = self.lin(x)
        else:
            out = x

        # Step 3: Start propagating messages.
        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

        # Step 4: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Apply edge weights from GCN normalization
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
class QGCNConv(GCNConv):
    def __init__(self, in_channels, nodeNN):
        # Tell the parent we don’t change feature size here
        super().__init__(in_channels,
                         out_channels=in_channels,
                         no_node_NN=False)
        self.nodeNN = nodeNN

    def update(self, aggr_out):
        # Example: use the quantum circuit instead of a Linear layer
        return self.nodeNN(aggr_out)


class QGCN(Module):

    def __init__(self, input_dims, q_depths, output_dims, hidden_dim=64, 
                 dropout_rate=0.2, activ_fn=LeakyReLU(0.2), readout=False,
                 quantum_device=None):

        super().__init__()
        layers = []
        self.n_qubits = input_dims

        for q_depth in q_depths:
            nodeNN = quantum_net(self.n_qubits, q_depth, quantum_device=quantum_device)
            QGCNConv_layer = QGCNConv(in_channels=self.n_qubits, nodeNN=nodeNN)
            layers.append(QGCNConv_layer)

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn
        self.norm = LayerNorm(self.n_qubits)

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        # self.classifier = Linear(input_dims, output_dims)
        self.classifier = torch.nn.Sequential(
            Linear(self.n_qubits, hidden_dim),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim, output_dims)
        )

        for m in self.classifier.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index, edge_attr)
            h = self.activ_fn(h)
            h = self.norm(h)

        # readout layer to get the embedding for each graph in batch
        # h = global_mean_pool(h, batch)
        h = self.classifier(h)

        # if self.readout is not None:
        #     h = self.readout(h)
        return h