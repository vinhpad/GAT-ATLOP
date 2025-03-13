from torch import nn
from dgl.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(
            self,
            num_layers=2,
            in_dim=768,
            out_dim=768,
            num_head=8
        ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATv2Conv(in_dim, in_dim//num_head , num_head, activation = nn.ELU())
            )
        
        self.out_layer = GATv2Conv(in_dim, out_dim, 1, activation = nn.ELU())

    def forward(self, input, graph):
        h = input

        for layer in self.gat_layers:
            h = layer(graph, h).flatten(1)
        logits = self.out_layer(graph, h).squeeze(1)
        return logits
