# data/load_dataset.py

import networkx as nx
import numpy as np
import pickle
import os
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_graph(path="/Users/seinkim/bigdas/code/gnn_with_rwr_centrality/data", dataset="cora"):
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        with open(os.path.join(path, f"ind.{dataset}.{name}"), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(os.path.join(path, f"ind.{dataset}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_reorder - min(test_idx_reorder), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_reorder - min(test_idx_reorder), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).toarray()
    labels = np.vstack((ally, ty))

    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    G = nx.from_dict_of_lists(graph)
    id2idx = {nid: i for i, nid in enumerate(G.nodes())}

    for nid in G.nodes():
        G.nodes[nid]['label'] = np.argmax(labels[id2idx[nid]])

    return G, id2idx

# import networkx as nx
# import numpy as np
# import pickle
# import os
# import scipy.sparse as sp


# def parse_index_file(filename):
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index


# def load_graph(path="/Users/seinkim/bigdas/code/gnn_with_rwr_centrality/data", dataset="citeseer"):
#     names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
#     objects = []
#     for name in names:
#         with open(os.path.join(path, f"ind.{dataset}.{name}"), 'rb') as f:
#             objects.append(pickle.load(f, encoding='latin1'))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)

#     test_idx_reorder = parse_index_file(os.path.join(path, f"ind.{dataset}.test.index"))
#     test_idx_reorder = np.array(parse_index_file(os.path.join(path, f"ind.{dataset}.test.index")))

#     test_idx_range = np.sort(test_idx_reorder)

#     if dataset == "citeseer":
#         # 🛠️ Fix for missing test indices in Citeseer
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_reorder - min(test_idx_reorder), :] = tx
#         tx = tx_extended

#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_reorder - min(test_idx_reorder), :] = ty
#         ty = ty_extended

#     features = sp.vstack((allx, tx)).toarray()
#     labels = np.vstack((ally, ty))

#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     G = nx.from_dict_of_lists(graph)
#     id2idx = {nid: i for i, nid in enumerate(G.nodes())}

#     for nid in G.nodes():
#         G.nodes[nid]['label'] = np.argmax(labels[id2idx[nid]])

#     return G, id2idx

# # gnn_with_rwr/data/load_dataset.py
# import networkx as nx
# import numpy as np
# import pickle
# import os


# def parse_index_file(filename):
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index


# def load_cora_graph(path="/Users/seinkim/bigdas/code/gnn_with_rwr_centrality/data", dataset="cora"):
#     names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
#     objects = []
#     for name in names:
#         with open(os.path.join(path, f"ind.{dataset}.{name}"), 'rb') as f:
#             objects.append(pickle.load(f, encoding='latin1'))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)

#     test_idx_reorder = parse_index_file(os.path.join(path, f"ind.{dataset}.test.index"))
#     test_idx_range = np.sort(test_idx_reorder)

#     # Combine features and labels
#     features = np.vstack((allx.toarray(), tx.toarray()))
#     labels = np.vstack((ally, ty))

#     # Reorder
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     G = nx.from_dict_of_lists(graph)
#     id2idx = {nid: i for i, nid in enumerate(G.nodes())}

#     # 🟡 여기서 label 정보를 노드에 넣어줌
#     for nid in G.nodes():
#         G.nodes[nid]['label'] = np.argmax(labels[id2idx[nid]])

#     return G, id2idx




##################


# def load_cora_graph(path="/Users/seinkim/bigdas/code/gnn_with_rwr/data", dataset="cora"):
#     names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
#     objects = []
#     for name in names:
#         with open(os.path.join(path, f"ind.{dataset}.{name}"), 'rb') as f:
#             objects.append(pickle.load(f, encoding='latin1'))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)

#     test_idx_reorder = parse_index_file(os.path.join(path, f"ind.{dataset}.test.index"))
#     test_idx_range = np.sort(test_idx_reorder)

#     # Combine features and labels
#     features = np.vstack((allx.toarray(), tx.toarray()))
#     labels = np.vstack((ally, ty))

#     # Reorder test features
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     G = nx.from_dict_of_lists(graph)
#     id2idx = {nid: i for i, nid in enumerate(G.nodes())}

#     return G, id2idx
