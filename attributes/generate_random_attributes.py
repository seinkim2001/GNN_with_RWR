'''
PYTHONPATH=. python attributes/generate_random_attributes.py
'''
import os
import pickle
import numpy as np
import networkx as nx
import random
import time
from tqdm import tqdm

from utils.similarity_measures import rwr_scores, compute_AdaSim, my_simrank, my_jaccard
from data.load_dataset import load_cora_graph

output_dir = "./attributes/random_avg"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Cora graph
G, id2idx = load_cora_graph()
nodes = list(G.nodes())
num_nodes = len(nodes)

measures = ['rwr', 'adasim', 'simrank', 'jaccard']

for topk in range(1, 11):  # 1% ~ 10%
    k_num = int(num_nodes * topk / 100)
    print(f"\n🔍 Now Processing Random-{topk}% with 10 samples...")

    accum = {m: None for m in measures}

    for repeat in tqdm(range(10), desc=f"  ▶ Top-{topk}%"):  # tqdm 진행바
        random.seed(42 + repeat)
        anchors = random.sample(nodes, k_num)

        results = {}

        for name, func in zip(
            measures,
            [rwr_scores, lambda G, A: compute_AdaSim(G, A, id2idx), my_simrank, my_jaccard]
        ):
            start_time = time.time()
            results[name] = func(G, anchors)
            elapsed = time.time() - start_time
            print(f"    ⏳ {name.upper():8s} done in {elapsed:.2f} sec")

        # 누적
        for key in measures:
            if accum[key] is None:
                accum[key] = results[key]
            else:
                accum[key] += results[key]

    # 평균 및 저장
    for key in measures:
        avg_attr = accum[key] / 10
        np.save(f"{output_dir}/attr_{key}_randavg_top{topk}.npy", avg_attr)

    combined = np.concatenate([accum[m] / 10 for m in measures], axis=1)
    np.save(f"{output_dir}/attr_rwr_adasim_simrank_jaccard_randavg_top{topk}.npy", combined)

    print(f"✅ 저장 완료: 평균 속성 (top{topk}%) - 개별 + 조합 저장됨")
