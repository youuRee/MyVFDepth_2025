import pickle
from collections import Counter

# 파일 경로를 본인의 환경에 맞게 수정하세요
path = '/workspace/Multi-View_Depth/VFDepth/dataset/ddad_mask/mask_idx_dict.pkl' 

with open(path, 'rb') as f:
    mask_data = pickle.load(f)

# 1. 모든 Value 추출
all_values = list(mask_data.values())

# 2. 어떤 값들이 몇 개씩 있는지 확인
value_counts = Counter(all_values)

print("--- Value 값 분포 분석 ---")
for val, count in sorted(value_counts.items()):
    print(f"Value [{val}]: {count}개의 데이터가 이 값을 가짐")

# 3. 특정 범위 확인 (예: 0~10번 키의 값들)
print("\n--- 상위 10개 Key-Value 샘플 ---")
for i in sorted(list(mask_data.keys())):
    print(f"Key {i} -> Value {mask_data[i]}")