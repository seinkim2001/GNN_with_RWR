import os
import urllib.request

# Create directory for data
data_dir = "/Users/seinkim/bigdas/code/gnn_with_rwr/data"
os.makedirs(data_dir, exist_ok=True)

# URLs for Cora dataset (Planetoid version used in PyTorch Geometric)
base_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
files = ["ind.cora.x", "ind.cora.tx", "ind.cora.allx", 
         "ind.cora.y", "ind.cora.ty", "ind.cora.ally", 
         "ind.cora.graph", "ind.cora.test.index"]

downloaded_files = []

# Download each file
for fname in files:
    url = f"{base_url}/{fname}"
    dest_path = os.path.join(data_dir, fname)
    urllib.request.urlretrieve(url, dest_path)
    downloaded_files.append(dest_path)

downloaded_files
