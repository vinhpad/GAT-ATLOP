import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, conv_out_channels, num_heads, dropout=0.1):
        super(MLP, self).__init__()

        self.embed_dim = conv_out_channels  # Embed dim = conv_out_channels
        self.conv_out_channels = conv_out_channels

        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=conv_out_channels,
                                    kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        self.proj_MC_to_embed = nn.Identity()

        self.linear_Wd = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)

        self.linear_Wt = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm_attn = nn.LayerNorm(self.embed_dim)

        self.fnn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm_fnn = nn.LayerNorm(self.embed_dim)

    def forward(self, M_0_input, M_l_input=None):
        batch_size, n_dim1, n_dim2, d_channels = M_0_input.shape

        M_0_input_permuted = M_0_input.permute(0, 3, 1, 2) # (batch_size, d, n, n)

        M_C = self.relu(self.conv_layer(M_0_input_permuted))

        M_C_l_pooled = F.adaptive_avg_pool2d(M_C, (1, 1)).view(batch_size, -1)
        
        M_C_l_projected = self.proj_MC_to_embed(M_C_l_pooled)

        concatenated_M_C_l = torch.cat((M_C_l_projected, M_C_l_projected), dim=-1)
        
        F_l1_batch = self.linear_Wd(concatenated_M_C_l)

        Q = F_l1_batch.unsqueeze(1)
        K = F_l1_batch.unsqueeze(1)
        V = F_l1_batch.unsqueeze(1)
        
        attn_output, attn_weights = self.multihead_attn(Q, K, V)
        attn_output = attn_output.squeeze(1)

        attn_residual = Q.squeeze(1) + self.linear_Wt(attn_output)
        
        M_h_t_l1_norm = self.layer_norm_attn(attn_residual)

        if M_l_input is None:
            current_M_l = M_h_t_l1_norm
        else:
            current_M_l = M_l_input
        
        fnn_output = self.fnn(current_M_l)
        
        M_l_plus_1 = self.layer_norm_fnn(current_M_l + fnn_output)

        return M_l_plus_1