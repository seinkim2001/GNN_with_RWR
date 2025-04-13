import os
import itertools
import numpy as np
# from ace_tools import display_dataframe_to_user

# 경로 설정
attr_dir = "./attributes/generated"
output_dir = "./attributes/combined"
os.makedirs(output_dir, exist_ok=True)

# 사용 가능한 measure 종류
measures = ["rwr", "adasim", "simrank", "jaccard"]
topk_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  

# 생성할 파일 목록 기록
generated_files = []

# 2개, 3개, 4개 measure 조합에 대해 수행
for r in [2, 3, 4]:
    for combo in itertools.combinations(measures, r):
        combo_name = "_".join(combo)
        for topk in topk_values:
            matrices = []
            try:
                for m in combo:
                    path = os.path.join(attr_dir, f"attr_{m}_top{topk}.npy")
                    matrices.append(np.load(path))
                combined = np.concatenate(matrices, axis=1)
                out_path = os.path.join(output_dir, f"attr_{combo_name}_top{topk}.npy")
                np.save(out_path, combined)
                generated_files.append((combo_name, topk, out_path))
            except FileNotFoundError:
                continue

# 결과 출력
import pandas as pd
df = pd.DataFrame(generated_files, columns=["Measure Combination", "TopK %", "File Path"])
# display_dataframe_to_user(name="Combined Attribute Files", dataframe=df)
