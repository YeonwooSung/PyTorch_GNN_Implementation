import torch
from torch import nn
from torch.nn import functional as F
import dgl.function as fn


class GATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.graph = g
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        setattr(self, 'al', nn.Parameter(torch.randn(in_feats,1)))
        setattr(self, 'ar', nn.Parameter(torch.randn(in_feats,1)))

    def forward(self, feat):
        # equation (1)
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.ndata['el'] = feat.mm(getattr(self, 'al'))
        g.ndata['er'] = feat.mm(getattr(self, 'ar'))
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # message passing
        g.update_all(fn.src_mul_edge('h', 'w', 'm'), fn.sum('m', 'h'))
        e = F.leaky_relu(g.edata['e'])
        # compute softmax
        g.edata['w'] = F.softmax(e)
        rst = g.ndata['h']
        return rst


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, activation, num_heads=2, merge=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
        self.activation = activation

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            x = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            x = torch.mean(torch.stack(head_outs),dim=0)
        return self.activation(x)


class GAT(nn.Module):
    def __init__(
        self,
        g,
        input_dim,
        out_dim,
        hidden_dim=256,
        num_heads=2,
        num_hidden_layers=2,
    ):
        super(GAT, self).__init__()
        # Graph Attention Network could have one or more hidden layers. perhaps, only having input and output layers is enough.
        assert num_hidden_layers >= 0

        # define input and output layers
        self.layer_in = MultiHeadGATLayer(g, input_dim, hidden_dim, F.elu, num_heads=num_heads, merge='cat')
        self.layer_out = MultiHeadGATLayer(g, hidden_dim, out_dim, F.elu, num_heads=2, merge='cat')

        # define hidden layers
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(MultiHeadGATLayer(g, hidden_dim, hidden_dim, F.elu, num_heads=num_heads, merge='cat'))
    
    def forward(self, h):
        h = self.layer_in(h)
        for i in range(self.num_hidden_layers):
            h = self.hidden_layers[i](h)
        h = self.layer_out(h)
        return h
