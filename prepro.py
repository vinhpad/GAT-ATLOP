from tqdm import tqdm
import ujson as json
import dgl
import torch
import numpy as np

from spacy.tokens import Doc
import spacy
import dgl.data

nlp = spacy.load('en_core_web_sm')

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def get_anaphors(sents, mentions):
    potential_mentions = []

    for sent_id, sent in enumerate(sents):
        doc_spacy = Doc(nlp.vocab, words=sent)
        for name, tool in nlp.pipeline:
            if name != 'ner':
                tool(doc_spacy)

        for token in doc_spacy:
            potential_mention = ''
            if token.dep_ == 'det' and token.text.lower() == 'the':
                potential_name = doc_spacy.text[token.idx:token.head.idx + len(token.head.text)]
                pos_start, pos_end = token.i, token.i + len(potential_name.split(' '))
                potential_mention = {
                    'pos': [pos_start, pos_end],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }
            if token.pos_ == 'PRON':
                potential_name = token.text
                pos_start = sent.index(token.text)
                potential_mention = {
                    'pos': [pos_start, pos_start + 1],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }

            if potential_mention:
                if not any(mention in potential_mention['name'] for mention in mentions):
                    potential_mentions.append(potential_mention)
    return potential_mentions

def build_graph(entity_pos, sent_pos):
    u = []
    v = []

    mention_idx_offset = []

    mention_idx = 0
    for entity in entity_pos:
        mention_idx_offset.append([])
        for mention_id1, _ in enumerate(entity):
            mention_idx_offset[-1].append(mention_idx + mention_id1)

            for mention_id2, _ in enumerate(entity):
                if mention_id1 == mention_id2:
                    continue
                u.append(mention_idx + mention_id1)
                v.append(mention_idx + mention_id2)
        mention_idx += len(entity)

    # document to mention
    document_idx = mention_idx
    mention_idx = 0
    for entity in entity_pos:
        for mention_id, _ in enumerate(entity):
            u.append(document_idx)
            v.append(mention_idx + mention_id)

            v.append(document_idx)
            u.append(mention_idx + mention_id)
        mention_idx += len(entity)

    for entity_idx1, _ in enumerate(entity_pos):
        for entity_idx2, _ in enumerate(entity_pos):
            if entity_idx1 == entity_idx2:
                continue
            for mention_id1, _ in enumerate(entity_pos[entity_idx1]):
                for mention_id2, _ in enumerate(entity_pos[entity_idx2]):
                    for sent in sent_pos:
                        # if sent_id == entity_pos[entity_idx1][mention_id1][0] and sent_id == entity_pos[entity_idx2][mention_id2][0]:
                        if sent[0] <= entity_pos[entity_idx1][mention_id1][0] \
                            and sent[1] >= entity_pos[entity_idx1][mention_id1][1] \
                                and sent[0] <= entity_pos[entity_idx2][mention_id2][0] \
                                    and sent[1] >= entity_pos[entity_idx2][mention_id2][1]:
                            u.append(mention_idx_offset[entity_idx1][mention_id1])
                            v.append(mention_idx_offset[entity_idx2][mention_id2])
    
    # graph builder
    for edge_id in range(len(u)):
            assert u[edge_id] != v[edge_id], f"Exist self edge {u[edge_id]} to {v[edge_id]}"
    graph = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=document_idx + 1)
    graph = dgl.add_self_loop(graph)

    return graph

def read_docred(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        
        sent_pos = []
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            start_sent = len(sents)
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            end_sent = len(sents)
            sent_pos.append((start_sent, end_sent,))
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        mentions = set([m['name'] for e in entities for m in e])

        potential_mention = get_anaphors(sample['sents'], mentions)

        entities.append(potential_mention)


        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        graph = build_graph(entity_pos, sent_pos)
        i_line += 1
        feature = {
            'input_ids': input_ids,
            'entity_pos': entity_pos if entity_pos[-1] != [] else entity_pos[:-1],
            'labels': relations,
            'hts': hts,
            'title': sample['title'],
            'graph': graph
        }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features