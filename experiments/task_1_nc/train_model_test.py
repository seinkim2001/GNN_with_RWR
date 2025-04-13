import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from data.load_dataset import load_cora_graph

# 다양한 GNN 모델 정의
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

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 모델 선택 옵션
gnn_models = {
    "gcn": GCN,
    "sage": GraphSAGE,
    "gin": GIN
}

# 사용 모델 선택
selected_model_name = "sage"  # 여기만 바꾸면 됨: "gcn", "sage", "gin"
ModelClass = gnn_models[selected_model_name]

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

print(f"총 {len(files)}개의 속성 조합 실험 시작... (모델: {selected_model_name})")

for fname in tqdm(files):
    attr_path = os.path.join(attr_dir, fname)
    attr = np.load(attr_path)
    x = torch.tensor(attr, dtype=torch.float)
    data.x = x

    model = ModelClass(input_dim=x.size(1), hidden_dim=64, num_classes=7)
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
    results.append({
        "Model": selected_model_name,
        "Attribute File": fname,
        "Test Accuracy": acc
    })

# CSV로 저장
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(f"./results/{selected_model_name}_results.csv", index=False)
print(f"모든 실험 완료! 결과는 results/{selected_model_name}_results.csv 에 저장됨.")
