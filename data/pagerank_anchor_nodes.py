import os
import networkx as nx
import numpy as np
import pickle as pkl

DATA_DIR = "/Users/seinkim/bigdas/code/gnn_with_rwr_centrality/data"

# def load_graph(dataset="cora"):
def load_graph(dataset="pubmed"):

    graph_file = os.path.join(DATA_DIR, f"ind.{dataset}.graph")
    with open(graph_file, "rb") as f:
        graph = pkl.load(f, encoding="latin1")
    G = nx.from_dict_of_lists(graph)
    return G

def compute_topk_centrality(G, k_ratio=0.01, method="pagerank"):
    if method == "pagerank":
        centrality = nx.pagerank(G)
    elif method == "degree":
        centrality = nx.degree_centrality(G)
    elif method == "eigen":
        centrality = nx.eigenvector_centrality_numpy(G)
    elif method == "closeness":
        centrality = nx.closeness_centrality(G)
    else:
        raise ValueError(f"Unknown method: {method}")

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_k = int(len(G) * k_ratio)
    return [node for node, _ in sorted_nodes[:top_k]]

def save_topk_nodes(dataset="pubmed", method="pagerank"):
    G = load_graph(dataset)
    result = {}
    for k in range(1, 11):  # 1% ~ 10%
        ratio = k / 100
        key = f"{k}%"
        result[key] = compute_topk_centrality(G, ratio, method)
        print(f"[DEBUG] {dataset} | {method} | {key}: {len(result[key])} nodes")

    save_name = f"{method}_topk_{dataset}.pkl"
    save_path = os.path.join(DATA_DIR, save_name)
    with open(save_path, "wb") as f:
        pkl.dump(result, f)
    print(f"[✓] Saved Top-K nodes ({method}) for {dataset} at: {save_path}")

if __name__ == "__main__":
    # for method in ["pagerank", "degree", "eigen", "closeness"]:
    for method in ["pagerank"]:

        save_topk_nodes(dataset="pubmed", method=method)

        # save_topk_nodes(dataset="citeseer", method=method)
    #    save_topk_nodes(dataset="cora", method=method)


# # gnn_with_rwr/data/pagerank_anchor_nodes.py

# import os
# import networkx as nx
# import numpy as np
# import pickle as pkl

# DATA_DIR = "/Users/seinkim/bigdas/code/gnn_with_rwr_centrality/data"
# GRAPH_FILE = os.path.join(DATA_DIR, "ind.cora.graph")

# def load_cora_graph():
#     with open(GRAPH_FILE, "rb") as f:
#         graph = pkl.load(f, encoding="latin1")
#     G = nx.from_dict_of_lists(graph)
#     return G

# def compute_topk_centrality(G, k_ratio=0.01, method="pagerank"):
#     if method == "pagerank":
#         centrality = nx.pagerank(G)
#     elif method == "degree":
#         centrality = nx.degree_centrality(G)
#     elif method == "eigen":
#         centrality = nx.eigenvector_centrality_numpy(G)
#     elif method == "closeness":
#         centrality = nx.closeness_centrality(G)
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
#     top_k = int(len(G) * k_ratio)
#     return [node for node, _ in sorted_nodes[:top_k]]


# def save_topk_nodes(method="pagerank"):
#     G = load_cora_graph()
#     result = {}
#     for k in range(1, 11):  # 1% ~ 10%
#         ratio = k / 100
#         key = f"{k}%"
#         result[key] = compute_topk_centrality(G, ratio, method)
#         print(f"[DEBUG] {method} | {key}: {len(result[key])} nodes")

#     save_name = f"{method}_topk.pkl"
#     save_path = os.path.join(DATA_DIR, save_name)
#     with open(save_path, "wb") as f:
#         pkl.dump(result, f)
#     print(f"[✓] Saved Top-K nodes ({method}) at: {save_path}")

# if __name__ == "__main__":
#     for method in ["pagerank", "degree", "eigen", "closeness"]:
#         save_topk_nodes(method)





#####################






# if __name__ == "__main__":
#     save_topk_nodes()



# def save_topk_nodes():
#     G = load_cora_graph()
#     result = {}
#     for k in np.arange(0.01, 0.11, 0.01):
#         key = f"{int(k * 100)}%"
#         result[key] = compute_pagerank_topk(G, k)
#     save_path = os.path.join(DATA_DIR, "pagerank_topk.pkl")
#     with open(save_path, "wb") as f:
#         pkl.dump(result, f)
#     print(f"[✓] Saved Top-K nodes at: {save_path}")


# def save_topk_nodes():
#     G = load_cora_graph()
#     result = {}
#     for k in range(1, 11):  # 1 ~ 10
#         ratio = k / 100
#         key = f"{k}%"
#         result[key] = compute_pagerank_topk(G, ratio)
#         print(f"[DEBUG] {key}: {len(result[key])} nodes")
#     save_path = os.path.join(DATA_DIR, "pagerank_topk.pkl")
#     with open(save_path, "wb") as f:
#         pkl.dump(result, f)
#     print(f"[✓] Saved Top-K nodes at: {save_path}")

# def compute_pagerank_topk(G, k_ratio=0.01):
#     pr = nx.pagerank(G)
#     sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
#     top_k = int(len(G) * k_ratio)
#     top_nodes = [node for node, _ in sorted_nodes[:top_k]]
#     return top_nodes
