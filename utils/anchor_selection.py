# gnn_with_rwr/utils/anchor_selection.py
import networkx as nx
import numpy as np

def get_topk_pagerank_nodes(G: nx.Graph, top_k_ratio: float) -> list:
    """
    PageRank 기반 상위 top_k_ratio 비율의 노드 선택
    """
    pagerank_scores = nx.pagerank(G)
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    top_k = int(len(sorted_nodes) * top_k_ratio)
    top_k_nodes = [node for node, _ in sorted_nodes[:top_k]]
    return top_k_nodes
