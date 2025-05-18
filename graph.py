import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_size, hidden_size, graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)

        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, input):
        nodes_embed, node_adj = input
        h = torch.matmul(nodes_embed, self.W.unsqueeze(0))
        sum_nei = torch.zeros_like(h)
        sum_nei += torch.matmul(node_adj, h)
        degs = torch.sum(node_adj, dim=-1).float().unsqueeze(dim=-1)
        norm = 1.0 / degs
        dst = sum_nei * norm + self.bias
        out = self.drop(torch.relu(dst))
        return nodes_embed + out, node_adj

class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int, dropout=0.0):
        super().__init__()

        self.h = n_heads
        self.d_k = out_features // n_heads
        self.WQ = nn.Linear(in_features, out_features)
        self.WK = nn.Linear(in_features, out_features)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        N_bt, N_nodes, _ = h.shape
        adj_mat = adj_mat.unsqueeze(1)
        scores = torch.zeros(N_bt, self.h, N_nodes, N_nodes).cuda()
        q = self.WQ(h).view(N_bt, -1, self.h, self.d_k).transpose(1, 2)
        k = self.WK(h).view(N_bt, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(scores == 0, -1e9)
        scores = self.dropout(scores)
        attn = F.softmax(scores, dim=-1)
        return attn.transpose(0, 1)


class AttentionGCNLayer(nn.Module):
    def __init__(self, input_size, nhead=2, graph_drop=0.0, iters=2, attn_drop=0.0):
        super(AttentionGCNLayer, self).__init__()
        self.nhead = nhead
        self.fully_graph_attention = MultiHeadDotProductAttention(input_size, input_size, self.nhead, attn_drop)
        self.gcn_layers = nn.Sequential(
            *[GraphConvolutionLayer(input_size, input_size, graph_drop) for _ in range(iters)])
        self.blocks = nn.ModuleList([self.gcn_layers for _ in range(self.nhead)])

        self.aggregate_W = nn.Linear(input_size * nhead, input_size)

    def forward(self, nodes_embed):
        output = []
        graph_attention = self.fully_graph_attention(nodes_embed)
        for cnt in range(0, self.nhead):
            hi, _ = self.blocks[cnt]((nodes_embed, graph_attention[cnt]))
            output.append(hi)
        output = torch.cat(output, dim=-1)
        return self.aggregate_W(output)