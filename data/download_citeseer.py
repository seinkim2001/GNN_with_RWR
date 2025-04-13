import os
import urllib.request

# CiteSeer 데이터를 저장할 경로 설정
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)

# CiteSeer 데이터셋의 각 파일명
files = [
    "ind.citeseer.allx",
    "ind.citeseer.ally",
    "ind.citeseer.graph",
    "ind.citeseer.test.index",
    "ind.citeseer.tx",
    "ind.citeseer.ty",
    "ind.citeseer.x",
    "ind.citeseer.y"
]

# GitHub raw URL prefix
base_url = "https://github.com/kimiyoung/planetoid/raw/master/data/"

# 파일 다운로드
for fname in files:
    url = base_url + fname
    save_path = os.path.join(save_dir, fname)
    print(f"⬇️ Downloading {fname}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to download {fname}: {e}")
