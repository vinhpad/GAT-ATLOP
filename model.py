import torch
import torch.nn as nn
from opt_einsum import contract
from gat import GAT
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F
import math

# import dgl
# import matplotlib.pyplot as plt
# import networkx as nx

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        # Query and Key transformation matrices for attention
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.gat = GAT(
            num_layers=2,
            in_dim=768,
            out_dim=768,
            num_head=8
        )

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_entity_representation(self, entity_mentions, local_context, hts):
        """
        Calculate entity representation using attention mechanism from paper
        entity_mentions: list of mentions for each entity [num_entity, num_mentions, hidden_size]
        local_context: context vector from hrt [total_num_pairs, hidden_size]
        hts: head-tail pairs [num_pairs, 2]
        """
        # Transform local context
        q_c = self.W_q(local_context)  # [num_pairs, hidden_size]
        
        # Get head mentions and tail mentions for all pairs
        h_indices = torch.tensor([ht[0] for ht in hts], device=local_context.device)
        t_indices = torch.tensor([ht[1] for ht in hts], device=local_context.device)
        
        # Process head entities
        h_mentions = entity_mentions[h_indices]  # [num_pairs, num_mentions, hidden_size]
        k_h = self.W_k(h_mentions)  # [num_pairs, num_mentions, hidden_size]
        
        # Calculate attention scores for head entities
        # a^i_(h,t) = W_q c_(h,t) W_k m_s / sqrt(d)
        h_scores = torch.bmm(k_h, q_c.unsqueeze(-1)).squeeze(-1)  # [num_pairs, num_mentions]
        h_scores = h_scores / math.sqrt(self.hidden_size)
        
        # Apply softmax to get attention weights
        h_attention_weights = F.softmax(h_scores, dim=-1)  # [num_pairs, num_mentions]
        
        # Calculate head entity representations
        # e^h_(h,t) = sum_i(a^i_(h,t) m_s)
        h_entity = torch.bmm(h_attention_weights.unsqueeze(1), h_mentions).squeeze(1)  # [num_pairs, hidden_size]
        
        # Process tail entities
        t_mentions = entity_mentions[t_indices]  # [num_pairs, num_mentions, hidden_size]
        k_t = self.W_k(t_mentions)  # [num_pairs, num_mentions, hidden_size]
        
        # Calculate attention scores for tail entities
        t_scores = torch.bmm(k_t, q_c.unsqueeze(-1)).squeeze(-1)  # [num_pairs, num_mentions]
        t_scores = t_scores / math.sqrt(self.hidden_size)
        
        # Apply softmax to get attention weights
        t_attention_weights = F.softmax(t_scores, dim=-1)  # [num_pairs, num_mentions]
        
        # Calculate tail entity representations
        t_entity = torch.bmm(t_attention_weights.unsqueeze(1), t_mentions).squeeze(1)  # [num_pairs, hidden_size]
        
        return h_entity, t_entity

    def graph(self, sequence_output, graphs, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        batch_size, h, _, c = attention.size()

        num_node = graphs.num_nodes()
        features = torch.zeros(num_node, self.config.hidden_size, device=sequence_output.device)

        # Get features for graph nodes
        node_offset = 0
        for i in range(batch_size):
            for entity_per_batch in entity_pos[i]:
                for start, _ in entity_per_batch:
                    if start + offset < c:
                        features[node_offset, :] = sequence_output[i, start + offset]
                    else:
                        features[node_offset, :] = torch.zeros(self.config.hidden_size).to(sequence_output)
                    node_offset = node_offset + 1
            
            features[node_offset, :] = sequence_output[i, 0]
            node_offset = node_offset + 1

        # Apply GAT
        features = self.gat(features, graphs.to(sequence_output.device))

        # Calculate local context
        rss = []  # [batch_size * num_pairs, hidden_size]
        for i in range(len(entity_pos)):
            entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_att = []
                    for start, end in e:
                        if start + offset < c:
                            e_att.append(attention[i, :, start + offset])
                    if len(e_att) > 0:
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_att = attention[i, :, start + offset]
                    else:
                        e_att = torch.zeros(h, c).to(attention)
                entity_atts.append(e_att)
            
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)  # [num_pairs, hidden_size]
            rss.append(rs)
        rss = torch.cat(rss, dim=0)  # [total_num_pairs, hidden_size]
        local_context = rss

        # Process entities with local context
        node_offset = 0
        h_entity, t_entity = [], []
        for i in range(len(entity_pos)):
            entity_embs = []
            mention_index = 0
            for e in entity_pos[i]:
                # Get mentions for current entity
                e_mentions = features[node_offset + mention_index: node_offset + mention_index + len(e), :]
                mention_index += len(e)
                entity_embs.append(e_mentions)
            node_offset = node_offset + (len(entity_pos[i]) + 1)
            
        
        h_entity, t_entity = self.get_entity_representation(entity_embs, local_context, hts)

        return h_entity, t_entity, rss
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                graphs=None
        ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        # GATv2 enhancement
        h, rs, t  = self.graph(sequence_output, graphs, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([h, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([t, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output
