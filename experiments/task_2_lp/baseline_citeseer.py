'''
Link Prediction 결과 (Citeseer + 기본 feature)
ROC AUC: 0.7014
Average Precision: 0.6521
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pkl
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

# ✅ GCN 인코더 정의
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ✅ 디코더 (dot product 방식)
def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# ✅ 일반화된 graph + feature 로딩 함수
def load_graph_with_features(data_dir="./data", dataset="citeseer"):
    with open(os.path.join(data_dir, f"ind.{dataset}.allx"), 'rb') as f:
        allx = pkl.load(f, encoding='latin1')
    with open(os.path.join(data_dir, f"ind.{dataset}.graph"), 'rb') as f:
        graph = pkl.load(f, encoding='latin1')

    G = nx.from_dict_of_lists(graph)
    node_list = sorted(G.nodes())  # 실제 노드 기준
    id2idx = {nid: i for i, nid in enumerate(node_list)}
    G = nx.relabel_nodes(G, id2idx)  # 0 ~ N-1 인덱스로 재정렬

    feature_dim = allx.shape[1]
    features = np.zeros((len(G.nodes()), feature_dim))
    for old_id, new_id in id2idx.items():
        if old_id < allx.shape[0]:
            features[new_id] = allx[old_id].toarray()[0]
        else:
            continue  # 없는 노드는 0으로 유지

    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float)
    return data


# ✅ Link Prediction 실험 함수
def train_baseline(dataset="citeseer"):
    print(f"\n🔍 {dataset.upper()} 원본 feature 기반 Link Prediction 실험 시작...")

    data = load_graph_with_features(dataset=dataset)

    transform = RandomLinkSplit(is_undirected=True, split_labels=True,
                                 add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNEncoder(input_dim=data.x.size(1), hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_data = train_data.to(device)
    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        z = model(train_data.x, train_data.edge_index)
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
    test_data = test_data.to(device)
    with torch.no_grad():
        z = model(test_data.x, test_data.edge_index)
        pos_score = decode(z, test_data.pos_edge_label_index).sigmoid().cpu().numpy()
        neg_score = decode(z, test_data.neg_edge_label_index).sigmoid().cpu().numpy()

        scores = np.concatenate([pos_score, neg_score])
        labels = np.concatenate([
            np.ones(len(pos_score)),
            np.zeros(len(neg_score))
        ])
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

    print(f"\n✅ Link Prediction 결과 ({dataset.capitalize()} + 기본 feature)")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")


if __name__ == "__main__":
    train_baseline(dataset="citeseer")
