# experiments/train_gcn.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from data.load_dataset import load_cora_graph

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train():
    # 데이터 불러오기
    G, id2idx = load_cora_graph()
    data = from_networkx(G)

    # attribute 불러오기 (Top-5%, rwr+adasim)
    attr_path = "./attributes/combined/attr_rwr_adasim_top5.npy"
    attr = np.load(attr_path)
    x = torch.tensor(attr, dtype=torch.float)
    data.x = x

    # 라벨 및 학습/검증/테스트 분할
    y = torch.tensor([G.nodes[n]["label"] for n in G.nodes()], dtype=torch.long)
    data.y = y

    idx = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx = torch.tensor(idx[:140])
    val_idx = torch.tensor(idx[140:640])
    test_idx = torch.tensor(idx[1708:])

    # 모델 정의
    model = GCN(input_dim=x.size(1), hidden_dim=64, num_classes=7)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            _, pred = out[val_idx].max(dim=1)
            acc = (pred == data.y[val_idx]).sum().item() / len(val_idx)
            print(f"[Epoch {epoch}] Val Acc: {acc:.4f}")
            model.train()

    # 테스트 성능 평가
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    acc = (pred[test_idx] == data.y[test_idx]).sum().item() / len(test_idx)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
