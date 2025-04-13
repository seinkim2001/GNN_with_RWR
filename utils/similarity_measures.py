import numpy as np
import networkx as nx
import math
from scipy.sparse import csr_matrix


import numpy as np
import networkx as nx
import math
from scipy.sparse import csr_matrix

def rwr_scores(G, anchors):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    score = []
    for anchor in anchors:
        s = nx.pagerank(G, personalization={anchor: 1})
        s_list = [s[node] for node in nodes]
        score.append(s_list)
    rwr_score = np.array(score).T

    # 자기자신과는 1로 설정
    for i, node in enumerate(nodes):
        if node in anchors:
            idx = anchors.index(node)
            rwr_score[i][idx] = 1.0
    return rwr_score

def compute_AdaSim(graph, anchors, id2idx, decay_factor=0.7, iterations=1, alpha_val=0.7):
    nodes = list(graph.nodes())
    adj = nx.adjacency_matrix(graph, nodelist=nodes)
    degrees = adj.sum(axis=1).T
    weights = csr_matrix(1 / np.log(degrees + math.e))
    weight_matrix = csr_matrix(adj.multiply(weights))
    adamic_scores = weight_matrix @ adj.T
    adamic_scores.setdiag(0)
    adamic_scores = adamic_scores / np.max(adamic_scores)
    result_matrix = decay_factor * alpha_val * adamic_scores
    result_matrix.setdiag(1)

    for _ in range(2, iterations + 1):
        result_matrix.setdiag(0)
        result_matrix = decay_factor * (
            alpha_val * adamic_scores +
            (1 - alpha_val) * (weight_matrix @ result_matrix @ weight_matrix.T)
        )
        result_matrix.setdiag(1)

    AdaSim_array = np.array([
        [result_matrix[id2idx[u], id2idx[v]] for v in anchors] for u in graph
    ])

    # 자기자신과는 1로 설정
    for i, u in enumerate(graph):
        for j, v in enumerate(anchors):
            if u == v:
                AdaSim_array[i][j] = 1.0
    return AdaSim_array

def my_simrank(G, anchors):
    simrank_sim = nx.simrank_similarity(G)
    sim_array = np.array([
        [simrank_sim[u][v] if u != v else 1.0 for v in anchors] for u in G
    ])
    return sim_array

def jaccard(G, u, v):
    if u == v:
        return 1.0  # 자기자신이면 유사도 1
    union = set(G[u]) | set(G[v])
    if not union:
        return 0.0
    inter = len(list(nx.common_neighbors(G, u, v)))
    return inter / len(union)

def my_jaccard(G, anchors):
    sim_array = np.array([
        [jaccard(G, u, v) for v in anchors] for u in G
    ])
    return sim_array


import networkx as nx
import numpy as np

def adamic_adar_similarity(G, anchors, id2idx):
    n = len(G)
    matrix = np.zeros((n, len(anchors)))
    for j, a in enumerate(anchors):
        for u, v, score in nx.adamic_adar_index(G, [(a, node) for node in G if node != a]):
            matrix[id2idx[v]][j] = score
    return matrix

def resource_allocation_similarity(G, anchors, id2idx):
    n = len(G)
    matrix = np.zeros((n, len(anchors)))
    for j, a in enumerate(anchors):
        for u, v, score in nx.resource_allocation_index(G, [(a, node) for node in G if node != a]):
            matrix[id2idx[v]][j] = score
    return matrix

def preferential_attachment_similarity(G, anchors, id2idx):
    n = len(G)
    matrix = np.zeros((n, len(anchors)))
    for j, a in enumerate(anchors):
        for u, v, score in nx.preferential_attachment(G, [(a, node) for node in G if node != a]):
            matrix[id2idx[v]][j] = score
    return matrix

def common_neighbors_similarity(G, anchors, id2idx):
    n = len(G)
    matrix = np.zeros((n, len(anchors)))
    for j, a in enumerate(anchors):
        for v in G:
            if a == v:
                continue
            cn = len(list(nx.common_neighbors(G, a, v)))
            matrix[id2idx[v]][j] = cn
    return matrix

def salton_index_similarity(G, anchors, id2idx):
    n = len(G)
    matrix = np.zeros((n, len(anchors)))
    for j, a in enumerate(anchors):
        for v in G:
            if a == v:
                continue
            cn = len(list(nx.common_neighbors(G, a, v)))
            da = G.degree(a)
            dv = G.degree(v)
            if da == 0 or dv == 0:
                sim = 0
            else:
                sim = cn / np.sqrt(da * dv)
            matrix[id2idx[v]][j] = sim
    return matrix


# def rwr_scores(G, anchors):
#     n = G.number_of_nodes()
#     score = []
#     for i, anchor in enumerate(anchors):
#         s = nx.pagerank(G, personalization={anchor: 1})  # ✅ 여기 수정!
#         s_list = [0] * n

#         for j, node in enumerate(list(G.nodes())):
#             s_list[j] = s[node]

#         score.append(s_list)

#     rwr_score = np.array(score).T
#     return rwr_score


# def compute_AdaSim(graph, selected_anchor_node, id2idx, decay_factor=0.7, iterations=1, alpha_val=0.7, link_type='none'):
#     G = graph
#     nodes = list(G.nodes())
#     adj = nx.adjacency_matrix(G, nodelist=nodes, weight=None)
#     degrees = adj.sum(axis=1).T
#     weights = csr_matrix(1 / np.log(degrees + math.e))
#     weight_matrix = csr_matrix(adj.multiply(weights))
#     adamic_scores = weight_matrix * adj.T
#     adamic_scores.setdiag(0)
#     adamic_scores = adamic_scores / np.max(adamic_scores)
#     result_matrix = decay_factor * alpha_val * adamic_scores
#     result_matrix.setdiag(1)

#     for _ in range(2, iterations + 1):
#         result_matrix.setdiag(0)
#         result_matrix = decay_factor * (
#             alpha_val * adamic_scores +
#             (1 - alpha_val) * (weight_matrix * result_matrix * weight_matrix.T)
#         )
#         result_matrix.setdiag(1)

#     AdaSim_array = np.array([[result_matrix[id2idx[u], id2idx[v]] for v in selected_anchor_node] for u in G])
#     return AdaSim_array

# def my_simrank(G, selected_anchor_node):
#     simrank_sim = nx.simrank_similarity(G)
#     sim_array = np.array([[simrank_sim[u][v] for v in selected_anchor_node] for u in G])
#     return sim_array

# def jaccard(G, u, v):
#     union_size = len(set(G[u]) | set(G[v]))
#     if union_size == 0:
#         return 0
#     return len(list(nx.common_neighbors(G, u, v))) / union_size

# def my_jaccard(G, selected_anchor_node):
#     sim_array = np.array([[jaccard(G, u, v) for v in selected_anchor_node] for u in G])
#     return sim_array






