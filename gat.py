from torch import nn
from dgl.nn import GATv2Conv
import torch
import torch.nn.functional as F
import math


class GAT(nn.Module):

    def __init__(self,
                 num_layers=3,
                 in_dim=768,
                 out_dim=768,
                 num_head=8,
                 dropout=0.1,
                 use_edge_weights=True,
                 residual=True):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.use_edge_weights = use_edge_weights
        self.residual = residual

        hidden_dims = [in_dim]
        for i in range(num_layers):
            hidden_dim = in_dim // (2 ** i) if i > 0 else in_dim
            hidden_dims.append(hidden_dim)
            
            self.gat_layers.append(
                GATv2Conv(hidden_dims[i], 
                          hidden_dim // num_head,
                          num_head,
                          feat_drop=dropout,
                          attn_drop=dropout,
                          activation=nn.ELU()))
            
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.out_norm = nn.LayerNorm(out_dim)
        
        if hidden_dims[-1] != out_dim:
            self.transform = nn.Linear(hidden_dims[-1], out_dim)
        else:
            self.transform = nn.Identity()

        self._initialize_parameters()

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, input, graph):
        h = input
        
        if self.use_edge_weights and 'weight' in graph.edata:
            edge_weights = graph.edata['weight']
        else:
            edge_weights = None

        for i in range(self.num_layers):
            h_prev = h
            
            h_new = self.gat_layers[i](graph, h, edge_weight=edge_weights)
            
            if self.residual and h_new.shape == h_prev.shape:
                h_new = h_new + h_prev
            
            h_new = self.norms[i](h_new)
            h_new = self.dropout(h_new)
            
            h = h_new

        output = self.transform(h)
        
        output = self.out_norm(output)

        return output
