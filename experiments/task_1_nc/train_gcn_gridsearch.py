import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from data.load_dataset import load_cora_graph

# GCN 모델 정의
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 결과 저장할 리스트
results = []

# 데이터 불러오기
G, id2idx = load_cora_graph()
data = from_networkx(G)

# 라벨
y = torch.tensor([G.nodes[n]["label"] for n in G.nodes()], dtype=torch.long)
data.y = y

# 데이터 분할
idx = np.arange(len(y))
np.random.seed(42)
np.random.shuffle(idx)
train_idx = torch.tensor(idx[:140])
val_idx = torch.tensor(idx[140:640])
test_idx = torch.tensor(idx[1708:])

# 속성 조합 디렉토리
attr_dir = "./attributes/combined"
files = sorted([f for f in os.listdir(attr_dir) if f.endswith(".npy")])

print(f"총 {len(files)}개의 속성 조합 실험 시작...")

for fname in tqdm(files):
    attr_path = os.path.join(attr_dir, fname)
    attr = np.load(attr_path)
    x = torch.tensor(attr, dtype=torch.float)
    data.x = x

    model = GCN(input_dim=x.size(1), hidden_dim=64, num_classes=7)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

    # 평가
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    acc = (pred[test_idx] == data.y[test_idx]).sum().item() / len(test_idx)

    # 결과 저장
    results.append({"Attribute File": fname, "Test Accuracy": acc})

# CSV로 저장
df = pd.DataFrame(results)
df.to_csv("./results/gcn_results.csv", index=False)
print("모든 실험 완료! 결과는 results/gcn_results.csv에 저장됨.")