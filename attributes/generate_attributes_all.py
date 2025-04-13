import os
import pickle
import numpy as np
import networkx as nx

from utils.similarity_measures import rwr_scores, compute_AdaSim, my_simrank, my_jaccard
from data.load_dataset import load_cora_graph

output_dir = "./attributes/generated"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Cora graph
G, id2idx = load_cora_graph()

# ✅ 선택할 centrality 기준
centrality_methods = ['pagerank', 'degree', 'eigen', 'closeness']

# ✅ similarity 종류
measures = ['rwr', 'adasim', 'simrank', 'jaccard']

for method in centrality_methods:
    print(f"\n=== Now Processing: {method.upper()} centrality ===")

    pkl_path = f"./data/{method}_topk.pkl"
    with open(pkl_path, "rb") as f:
        topk_dict = pickle.load(f)
        topk_dict = {int(k.replace('%', '')): v for k, v in topk_dict.items()}

    for topk in range(1, 11):  # 1% ~ 10%
        if topk not in topk_dict:
            print(f"[Top-{topk}%] ❌ Skipped (anchor 없음)")
            continue

        anchors = topk_dict[topk]
        print(f"[Top-{topk}%] Calculating {method} for {len(anchors)} anchors...")

        results = {}
        results['rwr'] = rwr_scores(G, anchors)
        results['adasim'] = compute_AdaSim(G, anchors, id2idx)
        results['simrank'] = my_simrank(G, anchors)
        results['jaccard'] = my_jaccard(G, anchors)

        # 개별 저장
        for key, value in results.items():
            np.save(f"{output_dir}/attr_{key}_{method}_top{topk}.npy", value)

        # 조합 저장
        combined = np.concatenate([results[m] for m in measures], axis=1)
        np.save(f"{output_dir}/attr_rwr_adasim_simrank_jaccard_{method}_top{topk}.npy", combined)
        print(f"저장 완료: {method}_top{topk}% - 개별 + 조합 저장됨 ✅")
