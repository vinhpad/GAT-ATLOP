from torch import nn
from dgl.nn import GATConv
import dgl
import torch
class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, graphs):
        batched_graph = dgl.batch(graphs)  # gộp tất cả graph thành batch graph
        inputs = torch.cat([inputs[i][:g.num_nodes()] for i, g in enumerate(graphs)], dim=0)

        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](batched_graph, h).flatten(1)

        logits = self.gat_layers[-1](batched_graph, h).mean(1)  # [total_num_nodes, num_classes]
        
        # Tách logits ra theo từng graph
        output = []
        node_offset = 0
        for g in graphs:
            output.append(logits[node_offset: node_offset + g.num_nodes() - 1   ])
            node_offset += g.num_nodes()

        return output  # List logits cho từng graph
