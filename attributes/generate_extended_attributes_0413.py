import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import os
import pickle
import numpy as np
import networkx as nx
from data.load_dataset import load_graph
from utils.similarity_measures import (
    rwr_scores, compute_AdaSim, my_simrank, my_jaccard,
    adamic_adar_similarity, resource_allocation_similarity,
    preferential_attachment_similarity, common_neighbors_similarity,
    salton_index_similarity
)

def normalize_dict(d):
    values = np.array(list(d.values()))
    min_val, max_val = values.min(), values.max()
    return {k: (v - min_val) / (max_val - min_val + 1e-8) for k, v in d.items()}

def get_combined_topk_anchors(G, ratio, method1='pagerank', method2='betweenness'):
    # 두 centrality 계산
    c1 = normalize_dict(nx.pagerank(G))
    c2 = normalize_dict(nx.betweenness_centrality(G))

    combined = {k: c1.get(k, 0) + c2.get(k, 0) for k in G.nodes()}
    sorted_nodes = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    topk_count = max(1, int(len(G.nodes) * (ratio / 100)))
    anchors = [node for node, _ in sorted_nodes[:topk_count]]
    return anchors

# ✅ 설정
output_dir = "./attributes/generated_extended_CORA"
os.makedirs(output_dir, exist_ok=True)

G, id2idx = load_graph()

centrality_methods = ['pagerank', 'betweenness', 'pagerank_betweenness']
# similarity_map = {
#     'rwr': rwr_scores,
#     'adasim': lambda G, A: compute_AdaSim(G, A, id2idx),
#     'simrank': my_simrank,
#     'jaccard': my_jaccard,
#     'aa': adamic_adar_similarity,
#     'ra': resource_allocation_similarity,
#     'pa': preferential_attachment_similarity,
#     'cn': common_neighbors_similarity,
#     'salton': salton_index_similarity
# }

similarity_map = {
    'rwr': lambda G, A: rwr_scores(G, A),
    'adasim': lambda G, A: compute_AdaSim(G, A, id2idx),
    'simrank': lambda G, A: my_simrank(G, A),
    'jaccard': lambda G, A: my_jaccard(G, A),
    'aa': lambda G, A: adamic_adar_similarity(G, A, id2idx),
    'ra': lambda G, A: resource_allocation_similarity(G, A, id2idx),
    'pa': lambda G, A: preferential_attachment_similarity(G, A, id2idx),
    'cn': lambda G, A: common_neighbors_similarity(G, A, id2idx),
    'salton': lambda G, A: salton_index_similarity(G, A, id2idx)
}

for method in centrality_methods:
    print(f"\n=== Now Processing: {method.upper()} centrality ===")

    if method == 'pagerank_betweenness':
        # 직접 계산
        for topk in range(1, 11):
            anchors = get_combined_topk_anchors(G, ratio=topk)
            print(f"[Top-{topk}%] Anchors from combined centrality: {len(anchors)} nodes")

            results = {}
            for sim_key, sim_func in similarity_map.items():
                results[sim_key] = sim_func(G, anchors)

            for key, value in results.items():
                np.save(f"{output_dir}/attr_{key}_{method}_top{topk}.npy", value)

            combined = np.concatenate([results[k] for k in similarity_map], axis=1)
            np.save(f"{output_dir}/attr_all_{method}_top{topk}.npy", combined)
            print(f"✅ 저장 완료: {method}_top{topk}% (9개 measure 조합)")

    else:
        # 기존 방식 (pkl 기반)
        pkl_path = f"./data/{method}_topk.pkl"
        with open(pkl_path, "rb") as f:
            topk_dict = pickle.load(f)
            topk_dict = {int(k.replace('%', '')): v for k, v in topk_dict.items()}

        for topk in range(1, 11):
            if topk not in topk_dict:
                print(f"[Top-{topk}%] ❌ Skipped (anchor 없음)")
                continue

            anchors = topk_dict[topk]
            print(f"[Top-{topk}%] Anchors from {method}: {len(anchors)} nodes")

            results = {}
            for sim_key, sim_func in similarity_map.items():
                results[sim_key] = sim_func(G, anchors)

            for key, value in results.items():
                np.save(f"{output_dir}/attr_{key}_{method}_top{topk}.npy", value)

            combined = np.concatenate([results[k] for k in similarity_map], axis=1)
            np.save(f"{output_dir}/attr_all_{method}_top{topk}.npy", combined)
            print(f"✅ 저장 완료: {method}_top{topk}% (9개 measure 조합)")
