import torch
from torch import nn


class GraphConvolutionLayer(nn.Module):
    def __init__(
        self,
        input_dim, 
        output_dim, 
        support, 
        act_func = None, 
        featureless = False, 
        dropout_rate = 0., 
        bias=False
    ):
        super(GraphConvolutionLayer, self).__init__()
        self.support = support
        self.featureless = featureless
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

        
    def forward(self, x):
        x = self.dropout(x)
        
        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            
            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__(self, 
        input_dim, 
        support,
        num_classes,
        dropout_rate=0., 
        hidden_dim=256,
        num_hidden_layers=2,
    ):
        super(GCN, self).__init__()
        
        # GraphConvolution
        self.layer_in = GraphConvolutionLayer(input_dim, hidden_dim, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer_out = GraphConvolutionLayer(hidden_dim, num_classes, support, dropout_rate=dropout_rate)

        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim, support, act_func=nn.ReLU(), dropout_rate=dropout_rate))
        
    
    def forward(self, x):
        out = self.layer_in(x)
        for i in range(self.num_hidden_layers):
            out = self.hidden_layers[i](out)
        out = self.layer_out(out)
        return out
