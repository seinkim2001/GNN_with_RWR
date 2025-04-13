'''
GAT
ROC AUC: 0.6999
Average Precision: 0.6509
'''
import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
from torch_geometric.nn import GCNConv, SAGEConv, GINConv

DATA_DIR = "./data"
GRAPH_FILE = os.path.join(DATA_DIR, "ind.cora.graph")
ALLX_FILE = os.path.join(DATA_DIR, "ind.cora.allx")

# GAT Encoder 정의
# class GATEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, heads=2):
#         super(GATEncoder, self).__init__()
#         self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
#         self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

#     def forward(self, x, edge_index):
#         x = self.gat1(x, edge_index)
#         x = F.elu(x)
#         x = self.gat2(x, edge_index)
#         return x

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


# Dot product decoder
def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# Cora 데이터 불러오기 (기본 feature 사용)
def load_cora_with_features(data_dir="./data", dataset="cora"):
    with open(os.path.join(data_dir, f"ind.{dataset}.allx"), 'rb') as f:
        allx = pkl.load(f, encoding='latin1')
    with open(os.path.join(data_dir, f"ind.{dataset}.graph"), 'rb') as f:
        graph = pkl.load(f, encoding='latin1')

    G = nx.from_dict_of_lists(graph)
    node_list = sorted(G.nodes())  # G에 실제로 존재하는 노드만
    id2idx = {nid: i for i, nid in enumerate(node_list)}
    G = nx.relabel_nodes(G, id2idx)  # 그래프 노드를 0~n-1로 재정렬

    # allx는 원래 indexing 기준으로 되어 있어서, id2idx를 활용해서 feature 맞춰줌
    feature_dim = allx.shape[1]
    features = np.zeros((len(G.nodes()), feature_dim))  # zero로 초기화
    for old_id, new_id in id2idx.items():
        if old_id < allx.shape[0]:
            features[new_id] = allx[old_id].toarray()[0]
        else:
            # 없는 노드는 zero vector 유지
            continue

    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float)
    return data

# 메인 실험 함수
def train_baseline_gat():
    print("🚀 GAT 기반 Link Prediction 실험 시작...")

    data = load_cora_with_features()

    # Edge split
    transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGE(input_dim=data.x.size(1), hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    x = data.x.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    # 학습
    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        z = model(x, train_data.edge_index)

        pos_score = decode(z, train_data.pos_edge_label_index)
        neg_score = decode(z, train_data.neg_edge_label_index)
        score = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones(pos_score.size(0)),
            torch.zeros(neg_score.size(0))
        ]).to(device)

        loss = F.binary_cross_entropy_with_logits(score, labels)
        loss.backward()
        optimizer.step()

    # 평가
    model.eval()
    with torch.no_grad():
        z = model(x, train_data.edge_index)
        pos_score = decode(z, test_data.pos_edge_label_index).sigmoid().cpu().numpy()
        neg_score = decode(z, test_data.neg_edge_label_index).sigmoid().cpu().numpy()
        scores = np.concatenate([pos_score, neg_score])
        labels = np.concatenate([
            np.ones(len(pos_score)),
            np.zeros(len(neg_score))
        ])
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

    print(f"\n📊 Test ROC AUC: {auc:.4f}, Average Precision: {ap:.4f}")

if __name__ == "__main__":
    train_baseline_gat()
