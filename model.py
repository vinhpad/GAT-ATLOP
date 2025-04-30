import torch
import torch.nn as nn
from opt_einsum import contract
from gat import GAT
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F
import math
# import pickle


class DocREModel(nn.Module):

    def __init__(self,
                 config,
                 model,
                 emb_size=768,
                 block_size=64,
                 num_labels=-1):
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

        self.gat = GAT(num_layers=2,
                       in_dim=768,
                       num_hidden=500,
                       num_classes=768,
                       heads=([2] * 2) + [1],
                       activation=F.elu,
                       feat_drop=0,
                       attn_drop=0,
                       negative_slope=0.2,
                       residual=False)

        # Bilinear transformation for entity embeddings
        self.entity_bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, config.num_labels, bias=False)
        self.bilinear_graph_integration = nn.Linear(config.num_labels * 2, config.num_labels)
        # self.bilinear = nn.Linear(config.num_labels * 2, config.num_labels)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(
            self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_entity_representation(self, entity_mentions, local_context, hts):
        """
        Calculate entity representation using attention mechanism from paper
        entity_mentions: entity mentions tensor [num_entities, max_mentions, hidden_size]
        local_context: context vector from hrt [num_pairs, hidden_size]
        hts: list of head-tail pairs for each batch
        """
        # Transform local context
        q_c = self.W_q(local_context)  # [num_pairs, hidden_size]

        # Convert list of hts to flat indices
        all_hts = []
        for batch_hts in hts:
            all_hts.extend(batch_hts)
        indices = torch.tensor(all_hts, device=local_context.device)

        h_indices, t_indices = indices[:, 0], indices[:, 1]

        # Process head and tail mentions together
        ht_mentions = torch.cat(
            [
                entity_mentions[h_indices],  # [num_pairs, max_mentions, hidden_size]
                entity_mentions[t_indices]  # [num_pairs, max_mentions, hidden_size]
            ],
            dim=1)  # [num_pairs, 2*max_mentions, hidden_size]

        # Transform mentions
        k_ht = self.W_k(ht_mentions)  # [num_pairs, 2*max_mentions, hidden_size]

        # Calculate attention scores for both head and tail
        scores = torch.bmm(k_ht, q_c.unsqueeze(-1)).squeeze(-1)  # [num_pairs, 2*max_mentions]
        scores = scores / math.sqrt(self.hidden_size)

        # Split scores for head and tail
        h_scores, t_scores = scores.chunk(2, dim=1)

        # Apply softmax separately for head and tail
        h_attention_weights = F.softmax(h_scores, dim=-1)  # [num_pairs, max_mentions]
        t_attention_weights = F.softmax(t_scores, dim=-1)  # [num_pairs, max_mentions]

        # Calculate entity representations
        h_mentions, t_mentions = entity_mentions[h_indices], entity_mentions[t_indices]
        h_entity = torch.bmm(h_attention_weights.unsqueeze(1), h_mentions).squeeze(1)
        t_entity = torch.bmm(t_attention_weights.unsqueeze(1), t_mentions).squeeze(1)

        return h_entity, t_entity

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"
                                                       ] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0),
                                                dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(
                            self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(
                            self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def graph(self, sequence_output, graphs, attention, entity_pos, hts,
              local_context):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"
                                                       ] else 0
        batch_size, h, _, c = attention.size()

        num_node = graphs.num_nodes()
        features = torch.zeros(num_node,
                               self.config.hidden_size,
                               device=sequence_output.device)

        # Get features for graph nodes
        node_offset = 0
        for i in range(batch_size):
            for entity_per_batch in entity_pos[i]:
                for start, _ in entity_per_batch:
                    if start + offset < c:
                        features[node_offset, :] = sequence_output[i, start +
                                                                   offset]
                    else:
                        features[node_offset, :] = torch.zeros(
                            self.config.hidden_size).to(sequence_output)
                    node_offset = node_offset + 1

            features[node_offset, :] = sequence_output[i, 0]
            node_offset = node_offset + 1
        # label_embedding = features[node_offset:node_offset +
        #                            self.config.num_labels, :]

        # Apply GAT
        features = self.gat(features, graphs.to(sequence_output.device))

        # Process entities with local context
        all_entity_embs = []  # [num_entities, max_mentions, hidden_size]
        node_offset = 0
        max_mentions = max(len(e) for entities in entity_pos for e in entities)

        for i in range(len(entity_pos)):
            batch_entity_embs = []
            for e in entity_pos[i]:
                # Get mentions for current entity
                e_mentions = features[node_offset:node_offset + len(e), :]
                # Pad to max_mentions
                if len(e) < max_mentions:
                    padding = torch.zeros(max_mentions - len(e),
                                          self.config.hidden_size,
                                          device=features.device)
                    e_mentions = torch.cat([e_mentions, padding], dim=0)
                batch_entity_embs.append(e_mentions)
                node_offset += len(e)
            node_offset += 1  # Skip CLS token
            all_entity_embs.extend(batch_entity_embs)

        all_entity_embs = torch.stack(
            all_entity_embs,
            dim=0)  # [num_entities, max_mentions, hidden_size]

        # Get entity representations
        h_entity, t_entity = self.get_entity_representation(
            all_entity_embs, local_context, hts)

        return h_entity, t_entity

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                graphs=None):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        # GATv2 enhancement
        hs_enhacement, ts_enhancement = self.graph(
            sequence_output, graphs, attention, entity_pos, hts, rs)
        entity_scores = self.entity_bilinear(hs_enhacement, ts_enhancement)  # [batch_size, num_labels]

        hs = torch.tanh(
            self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(
            self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = self.bilinear((b1.unsqueeze(3) * b2.unsqueeze(2)).view(
            -1, self.emb_size * self.block_size))

        logits = self.bilinear_graph_integration(torch.cat([bl, entity_scores], dim=1))
        output = (self.loss_fnt.get_label(logits,
                                          num_labels=self.num_labels), )
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), ) + output
        return output
