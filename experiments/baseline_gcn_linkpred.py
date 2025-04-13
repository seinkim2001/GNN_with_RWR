import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. Load data
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# 2. Edge split for Link Prediction
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)

# 3. Define GCN for embedding
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        # Dot product decoder
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# 4. Train GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, 32).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def get_link_labels(pos_edge_index, neg_edge_index):
    return torch.cat([torch.ones(pos_edge_index.size(1)),
                      torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    pos_out = model.decode(z, data.train_pos_edge_index)
    neg_out = model.decode(z, data.train_neg_edge_index)
    link_logits = torch.cat([pos_out, neg_out], dim=0)
    link_labels = get_link_labels(data.train_pos_edge_index, data.train_neg_edge_index)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

# 5. Evaluate
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    pos_out = model.decode(z, data.test_pos_edge_index).sigmoid()
    neg_out = model.decode(z, data.test_neg_edge_index).sigmoid()
    preds = torch.cat([pos_out, neg_out], dim=0).cpu()
    labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index).cpu()
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)

print(f"AUC: {auc:.4f}, AP: {ap:.4f}")
