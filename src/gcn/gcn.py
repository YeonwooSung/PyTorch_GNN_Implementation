import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import math


class GCN_Layer(nn.Module):
    def __init__(
        self, 
        adjacent_matrix, 
        input_size, 
        hidden_size, 
        dropout_prob=0.5, 
        node_dim=0, 
        bias=True
    ) -> None:
        super().__init__()
        self.adjacent_matrix = adjacent_matrix
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.node_dim = node_dim

        self.weight = Parameter(torch.FloatTensor(input_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        #
        # Step 1: Regularization
        #
        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.node_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N is the number of nodes in the graph, and FIN is the number of input features per node.
        # We apply the dropout to all of the input node features 
        # Note that for Cora features are already super sparse, so it is questionable how much this actually helps
        in_nodes_features = self.dropout1(in_nodes_features)

        # perform graph convolution
        support = torch.mm(in_nodes_features, self.weight)
        output = torch.spmm(self.adjacent_matrix, support)

        # add bias if needed
        if self.bias is not None:
            output += self.bias
        
        # pass output through activation function
        output = self.activation(output)
        output = self.dropout2(output)

        return (output, edge_index)



class GCN(nn.Module):
    def __init__(
        self,
        adjacent_matrix,
        num_of_layers,
        num_features_per_layer,
        dropout_prob=0.5, 
        node_dim=0, 
        bias=True
    ) -> None:
        super().__init__()
        assert num_of_layers == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        self.adjacent_matrix = adjacent_matrix
        self.num_of_layers = num_of_layers
        
        gcn_layers = []
        for i in range(num_of_layers):
            layer = GCN_Layer(
                adjacent_matrix,
                num_features_per_layer[i],
                num_features_per_layer[i+1],
                dropout_prob=dropout_prob,
                node_dim=node_dim,
                bias=bias
            )
            gcn_layers.append(layer)
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self, x):
        # The data is just a tuple of (node_features, edge_index)
        for i in range(self.num_of_layers):
            x = self.gcn_layers[i](x)
        output_features, edge_index = x
        output_features = F.log_softmax(output_features, dim=1)
        return (output_features, edge_index)
