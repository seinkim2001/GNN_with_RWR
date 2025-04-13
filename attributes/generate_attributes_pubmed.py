"""
PYTHONPATH=. python attributes/generate_attributes_pubmed.py
"""

import os
import pickle
import numpy as np
import networkx as nx
from utils.similarity_measures import rwr_scores, compute_AdaSim, my_simrank, my_jaccard
from data.load_dataset import load_graph
from torch_geometric.datasets import Planetoid

# # pubmed dataset을 Planetoid 형식으로 불러오고 저장
# dataset = Planetoid(root='./data', name='Pubmed')

# 저장 디렉토리
output_dir = "./attributes/generated_50_pubmed"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Pubmed graph
dataset = "pubmed"
G, id2idx = load_graph(dataset=dataset)

# ✅ similarity 종류
measures = ['rwr', 'adasim', 'simrank', 'jaccard']

# ✅ centrality 방식 (pagerank 기준)
method = "pagerank"
pkl_path = f"./data/{method}_topk_{dataset}.pkl"

with open(pkl_path, "rb") as f:
    topk_dict = pickle.load(f)
    topk_dict = {int(k.replace('%', '')): v for k, v in topk_dict.items()}

# ✅ 1% ~ 10% 비율별로 반복
for topk in range(1, 11):
    if topk not in topk_dict:
        print(f"[Top-{topk}%] ❌ Skipped (anchor 없음)")
        continue

    anchors = topk_dict[topk]
    print(f"\n🔍 [Top-{topk}%] Calculating {method.upper()} for {len(anchors)} anchors...")

    results = {}
    results['rwr'] = rwr_scores(G, anchors)
    results['adasim'] = compute_AdaSim(G, anchors, id2idx)
    results['simrank'] = my_simrank(G, anchors)
    results['jaccard'] = my_jaccard(G, anchors)

    # ✅ 단일 similarity 저장
    for key, value in results.items():
        save_path = f"{output_dir}/attr_{key}_top{topk}.npy"
        np.save(save_path, value)

    # ✅ 전체 조합 저장
    combined = np.concatenate([results[m] for m in measures], axis=1)
    comb_path = f"{output_dir}/attr_rwr_adasim_simrank_jaccard_top{topk}.npy"
    np.save(comb_path, combined)

    print(f"✅ 저장 완료: top{topk}% - 개별 + 조합 저장됨")
