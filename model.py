import torch
import torch.nn as nn
from opt_einsum import contract
from gat import GAT
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.gat = GAT(
            num_layers=2,
            in_dim=768,
            num_hidden=500,
            num_classes=768,
            heads=([2] * 2) + [1],
            activation=F.elu,
            feat_drop=0,
            attn_drop=0,
            negative_slope=0.2,
            residual=False
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

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
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
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
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

    def graph(self, sequence_output, graphs, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()

        num_node = sum([graph.num_nodes() for graph in graphs])
        graph_fea = torch.zeros(num_node, self.config.hidden_size, device=sequence_output.device)
        
        node_id = 0
        for i in range(len(entity_pos)):
            mention_index = 0
            for e in entity_pos[i]:
                for start, end in e:
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        graph_fea[node_id, :] = sequence_output[i, start + offset]
                    else:
                        graph_fea[node_id, :] = torch.zeros(self.config.hidden_size).to(sequence_output)
                    mention_index += 1
                    node_id = node_id + 1

        
        graph_fea = self.gat(graph_fea, graphs)

        node_offset = 0
        h_entity, t_entity = [], []
        
        for i in range(len(entity_pos)):
            entity_embs = []
            mention_index = 0
            for e in entity_pos[i]:
                e_emb = graph_fea[node_offset + mention_index: node_offset + mention_index + len(e), :]
                mention_index += len(e)

                e_emb = torch.logsumexp(e_emb, dim=0) if len(e) > 1 else e_emb.squeeze(0)
                entity_embs.append(e_emb)
            node_offset = node_offset + len(entity_pos)
            
            entity_embs = torch.stack(entity_embs, dim=0)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_entity.append(hs)
            t_entity.append(ts)

        h_entity = torch.cat(h_entity, dim=0)
        t_entity = torch.cat(t_entity, dim=0)
        return h_entity, t_entity
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                graphs=None
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        # GAT enhancement
        h, t = self.graph(sequence_output, graphs, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs, h], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs, t], dim=1)))
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
