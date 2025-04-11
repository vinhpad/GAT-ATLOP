import torch
import random
import numpy as np
import dgl


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [
        f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch
    ]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] *
                  (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    graphs = [f["graph"] for f in batch]

    doc_ids = [graph.num_nodes() - 1 for graph in graphs]

    graphs = dgl.batch(graphs)

    num_nodes = graphs.num_nodes()
    graphs = dgl.add_nodes(graphs, 97)

    for doc_id in doc_ids:
        for i in range(97):
            if not graphs.has_edge_between(doc_id, i + num_nodes):
                graphs.add_edge(doc_id, i + num_nodes)
            if not graphs.has_edge_between(i + num_nodes, doc_id):
                graphs.add_edge(i + num_nodes, doc_id)

    for i in range(97):
        for j in range(97):
            if i != j:
                if not graphs.has_edge_between(i + num_nodes, j + num_nodes):
                    graphs.add_edge(i + num_nodes, j + num_nodes)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts, graphs)
    return output
