# gnn_with_rwr/experiments/run_experiment.py
import time
import numpy as np
from code.gnn_with_rwr_centrality.attributes.generate_attributes_all import generate_attribute_matrix
from data.load_dataset import load_cora  # 이후 작성
import torch
import os

def run_and_log_experiment():
    start = time.time()

    G, id2idx = load_cora()
    attr = generate_attribute_matrix(G, id2idx, ratio=0.1)

    # 저장
    np.save("attributes/cora_attr.npy", attr)

    end = time.time()
    elapsed = end - start

    log_path = "logs/log.txt"
    with open(log_path, "a") as f:
        f.write(f"[Cora] Attribute 생성 완료, 시간: {elapsed:.2f}초, shape={attr.shape}\n")

    print("Attribute 생성 및 로그 기록 완료")

if __name__ == "__main__":
    run_and_log_experiment()
