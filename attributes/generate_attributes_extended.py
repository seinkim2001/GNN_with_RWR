# import os
# import pickle
# import numpy as np
# import networkx as nx

# from utils.similarity_measures import (
#     rwr_scores, compute_AdaSim, my_simrank, my_jaccard,
#     adamic_adar_similarity, resource_allocation_similarity,
#     preferential_attachment_similarity, common_neighbors_similarity,
#     salton_index_similarity
# )
# from data.load_dataset import load_graph

# output_dir = "./attributes/generated_extended_similarity"
# os.makedirs(output_dir, exist_ok=True)

# # 1. Load Cora graph
# G, id2idx = load_graph()

# # ✅ 선택할 centrality 기준
# centrality_methods = ['degree', 'eigen', 'closeness']
# # centrality_methods = ['pagerank']

# # ✅ similarity 종류 (기존 + 확장 measure)
# measures = ['rwr', 'adasim', 'simrank', 'jaccard',
#             'aa', 'ra', 'pa', 'cn', 'salton']

# for method in centrality_methods:
#     print(f"\n=== Now Processing: {method.upper()} centrality ===")

#     pkl_path = f"./data/{method}_topk.pkl"
#     with open(pkl_path, "rb") as f:
#         topk_dict = pickle.load(f)
#         topk_dict = {int(k.replace('%', '')): v for k, v in topk_dict.items()}

#     for topk in range(1, 11):  # 1% ~ 10%
#         if topk not in topk_dict:
#             print(f"[Top-{topk}%] Skipped (anchor 없음)")
#             continue

#         anchors = topk_dict[topk]
#         print(f"[Top-{topk}%] Calculating {method} for {len(anchors)} anchors...")

#         results = {}
#         results['rwr'] = rwr_scores(G, anchors)
#         results['adasim'] = compute_AdaSim(G, anchors, id2idx)
#         results['simrank'] = my_simrank(G, anchors)
#         results['jaccard'] = my_jaccard(G, anchors)
#         results['aa'] = adamic_adar_similarity(G, anchors, id2idx)
#         results['ra'] = resource_allocation_similarity(G, anchors, id2idx)
#         results['pa'] = preferential_attachment_similarity(G, anchors, id2idx)
#         results['cn'] = common_neighbors_similarity(G, anchors, id2idx)
#         results['salton'] = salton_index_similarity(G, anchors, id2idx)

#         # ✅ 개별 저장
#         for key, value in results.items():
#             np.save(f"{output_dir}/attr_{key}_{method}_top{topk}.npy", value)

#         # ✅ 전체 measure 조합 저장
#         combined = np.concatenate([results[m] for m in measures], axis=1)
#         np.save(f"{output_dir}/attr_all_{method}_top{topk}.npy", combined)
#         print(f"✅ 저장 완료: {method}_top{topk}% (9개 measure 조합)")

import os
import pickle
import numpy as np
import networkx as nx

from utils.similarity_measures import (
    rwr_scores, compute_AdaSim, my_simrank, my_jaccard,
    adamic_adar_similarity, resource_allocation_similarity,
    preferential_attachment_similarity, common_neighbors_similarity,
    salton_index_similarity
)
from data.load_dataset import load_graph

# ✅ Citeseer 용 경로
output_dir = "./attributes/generated_extended_similarity_citeseer"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Citeseer graph
G, id2idx = load_graph(dataset="citeseer")

# ✅ centrality 기준
centrality_methods = ['pagerank']

# ✅ similarity measure 목록
measures = ['rwr', 'adasim', 'simrank', 'jaccard',
            'aa', 'ra', 'pa', 'cn', 'salton']

for method in centrality_methods:
    print(f"\n=== Now Processing: {method.upper()} centrality ===")

    pkl_path = f"./data/{method}_topk_citeseer.pkl"  # ✅ Citeseer 전용 경로
    with open(pkl_path, "rb") as f:
        topk_dict = pickle.load(f)
        topk_dict = {int(k.replace('%', '')): v for k, v in topk_dict.items()}

    for topk in range(1, 11):
        if topk not in topk_dict:
            print(f"[Top-{topk}%] Skipped (anchor 없음)")
            continue

        anchors = topk_dict[topk]
        print(f"[Top-{topk}%] Calculating {method} for {len(anchors)} anchors...")

        results = {
            'rwr': rwr_scores(G, anchors),
            'adasim': compute_AdaSim(G, anchors, id2idx),
            'simrank': my_simrank(G, anchors),
            'jaccard': my_jaccard(G, anchors),
            'aa': adamic_adar_similarity(G, anchors, id2idx),
            'ra': resource_allocation_similarity(G, anchors, id2idx),
            'pa': preferential_attachment_similarity(G, anchors, id2idx),
            'cn': common_neighbors_similarity(G, anchors, id2idx),
            'salton': salton_index_similarity(G, anchors, id2idx),
        }

        # 개별 저장
        for key, value in results.items():
            np.save(f"{output_dir}/attr_{key}_{method}_citeseer_top{topk}.npy", value)

        # 조합 저장
        combined = np.concatenate([results[m] for m in measures], axis=1)
        np.save(f"{output_dir}/attr_all_{method}_citeseer_top{topk}.npy", combined)
        print(f"✅ 저장 완료: {method}_top{topk}% (Citeseer + 9개 measure)")
