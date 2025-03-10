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

        self.gat_layers.append(
            GATv2Conv(
                in_dim, 
                in_dim // num_head, 
                num_head
            )
        )
        
        for _ in range(1, num_layers - 1):
            self.gat_layers.append(
                GATv2Conv(
                    in_dim,
                    in_dim // num_head,
                    num_head
                )
            )
        
        self.gat_layers.append(
            GATv2Conv(
                in_dim, 
                out_dim, 
                num_head
            )
        )

    def forward(self, input, graph):
        h = input

        for l in range(self.num_layers):
            h = self.gat_layers[l](graph, h).flatten(1)

        logits = self.gat_layers[-1](graph, h).mean(1)
        return logits
