from torch import nn
from dgl.nn import GATv2Conv
import torch
import torch.nn.functional as F
import math

class GAT(nn.Module):
    def __init__(
            self,
            num_layers=2,
            in_dim=768,
            out_dim=768,
            num_head=8,
            dropout=0.1
        ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATv2Conv(in_dim, in_dim//num_head, num_head, activation=nn.ELU())
            )
            self.norms.append(nn.LayerNorm(in_dim))
        
        self.out_layer = GATv2Conv(in_dim, out_dim, 1, activation=nn.ELU())
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, input, graph):
        h = input

        for layer, norm in zip(self.gat_layers, self.norms):
            h_new = layer(graph, h).flatten(1)
            h_new = h_new + h
            h_new = norm(h_new)
            h_new = self.dropout(h_new)
            h = h_new

        logits = self.out_layer(graph, h).squeeze(1)
        logits = self.out_norm(logits)
        
        return logits
