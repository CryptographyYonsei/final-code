# convert_raw_to_ascad.py
from ASCAD_generate import extract_traces  # ASCAD_generate.py 안에 정의된 함수
import os

# 원본 raw 파일 경로
raw_h5  = "./Data/ATMega8515_raw_traces.h5"
# 출력할 ASCAD 형식 파일 경로
out_h5  = "./Data/ASCAD_from_raw.h5"

# profiling / attack 인덱스 (v1 ASCAD와 동일하게 0~49999, 50000~59999)
profiling_idx = list(range(0, 50000))
attack_idx    = list(range(50000, 60000))
# 관심 포인트 전체 (예: 0~699)
target_pts    = list(range(0, 700))

# 디렉터리 생성
os.makedirs(os.path.dirname(out_h5), exist_ok=True)

# 변환 실행
extract_traces(
    traces_file         = raw_h5,
    labeled_traces_file = out_h5,
    profiling_index     = profiling_idx,
    attack_index        = attack_idx,
    target_points       = target_pts,
    profiling_desync    = 0,
    attack_desync       = 0,
    multilabel          = 0
)
print(">> raw → ASCAD 변환 완료:", out_h5)
