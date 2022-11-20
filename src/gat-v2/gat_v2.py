import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionV2Layer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        share_weights: bool = False
    ) -> None:
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        if is_concat:
            assert out_features % n_heads == 0
            self.hidden = out_features // n_heads
        else:
            self.hidden = out_features
        
        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

        self.output_act = nn.ELU()
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, use_einsum=True) -> torch.Tensor:
        """
        * `h`, "h" is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """
        # Number of nodes
        n_nodes = h.shape[0]

        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # First, calculate g_li * g_rj for all pairs of i and j
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)

        # combine g_l and g_r
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # calculate attention score e_ij
        e = self.attn(self.activation(g_sum)).squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        # mask e_ij based on the adjacency matrix
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # apply softmax to calculate attention
        a = self.softmax(e)
        # apply dropout
        a = self.dropout(a)

        #
        # calculate the final output for each head
        #
        if use_einsum:
            h_prime = torch.einsum('ijh,jhf->ihf', a, g_r)
        else:
            h_prime = torch.matmul(a, g_r)

        # concatenate the output of each head
        if self.is_concat:
            h_prime = h_prime.view(n_nodes, -1)
        else:
            h_prime = torch.mean(h_prime, dim=1)

        # apply activation and dropout
        return self.output_dropout(self.output_act(h_prime))



class GATV2(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        n_hidden: int, 
        n_classes: int, 
        n_heads: int, 
        dropout: float,
        num_of_layers: int = 2,
        share_weights: bool = True
    ) -> None:
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `num_of_layers` is the number of graph attention layers
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()
        self.num_of_layers = num_of_layers
        
        self.layers = nn.ModuleList()

        # add input layer
        self.layers.append(GraphAttentionV2Layer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout, share_weights=share_weights))
        
        # add hidden layers
        for i in range(num_of_layers - 2):
            self.layers.append(GraphAttentionV2Layer(n_hidden, n_hidden, n_heads, share_weights=share_weights))

        # add output layer
        self.layers.append(GraphAttentionV2Layer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout, share_weights=share_weights))


    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        for i in range(self.num_of_layers):
            x = self.layers[i](x, adj_mat)
        return x
