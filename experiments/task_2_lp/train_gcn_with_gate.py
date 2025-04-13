import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from data.load_dataset import load_cora_graph

# ✅ GCN 인코더 + Selection Gate 정의
class GCNEncoderWithGate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNEncoderWithGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        gate_weight = self.gate(x)
        x = x * gate_weight  # ✅ selection gate 적용
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ✅ 디코더 정의 (dot product 방식)
def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# ✅ 실험 결과 저장 리스트
results = []

# ✅ 데이터셋 로딩
G, id2idx = load_cora_graph()
data = from_networkx(G)
data.train_mask = data.val_mask = data.test_mask = None

# ✅ 링크 예측용 데이터 분할
transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

# ✅ attribute 벡터 경로 지정
attr_dir = "./attributes/combined"  # 또는 ./attributes/combined
files = sorted([f for f in os.listdir(attr_dir) if f.endswith(".npy")])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"총 {len(files)}개의 속성 벡터에 대해 실험 시작...")

# ✅ 실험 반복
for fname in tqdm(files):
    attr = np.load(os.path.join(attr_dir, fname))
    x = torch.tensor(attr, dtype=torch.float)

    # 속성 적용
    train_data.x = x.to(device)
    val_data.x = x.to(device)
    test_data.x = x.to(device)

    # 모델 초기화
    # input_dim 별로 바뀌게끔 
    model = GCNEncoderWithGate(input_dim=x.size(1), hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 학습
    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        z = model(train_data.x, train_data.edge_index)
        score = decode(z, train_data.edge_label_index)
        labels = train_data.edge_label.to(device).float()
        loss = F.binary_cross_entropy_with_logits(score, labels)
        loss.backward()
        optimizer.step()

    # 평가
    model.eval()
    with torch.no_grad():
        z = model(test_data.x, test_data.edge_index)
        test_score = decode(z, test_data.edge_label_index).sigmoid().cpu().numpy()
        test_labels = test_data.edge_label.cpu().numpy()

        auc = roc_auc_score(test_labels, test_score)
        ap = average_precision_score(test_labels, test_score)

    # 결과 저장
    results.append({
        "Attribute File": fname,
        "ROC AUC": round(auc, 4),
        "Average Precision": round(ap, 4)
    })

# ✅ CSV로 결과 저장
os.makedirs("./results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("./results/link_prediction_results_with_gate_combined.csv", index=False)
print("✅ 모든 실험 완료! 결과는 ./results/link_prediction_results_with_gate_combined.csv 에 저장됨.")
