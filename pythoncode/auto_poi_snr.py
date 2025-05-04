# auto_poi_snr.py
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 1) 데이터 로드
with h5py.File("./Data/ASCAD.h5", "r") as hf:
    traces = hf["Profiling_traces/traces"][:]   # (N, 700)
    labels = hf["Profiling_traces/labels"][:]
if labels.dtype.names:
    labels = labels[labels.dtype.names[0]]
# Hamming Weight 레이블
hw = np.array([bin(int(x)).count("1") for x in labels])

# 2) 각 샘플 t마다 클래스별 평균 - 전체 평균, 클래스별 분산 - 전체 분산을 구해 SNR 계산
N, L = traces.shape
snr = np.zeros(L, dtype=np.float32)
overall_mean = traces.mean(axis=0)
overall_var  = traces.var(axis=0) + 1e-8

for w in range(9):           # HW 클래스 0~8
    idx = np.where(hw == w)[0]
    if len(idx)==0: continue
    class_mean = traces[idx].mean(axis=0)
    class_var  = traces[idx].var(axis=0)
    snr += ((class_mean - overall_mean)**2) / (class_var + 1e-8)

snr /= 9.0

# 3) SNR 플롯
plt.figure(figsize=(10,4))
plt.plot(snr, label="SNR")
plt.title("SNR per Sample Index")
plt.xlabel("Sample index (0‒699)")
plt.ylabel("SNR")
plt.grid(True)
plt.legend()
plt.show()

# 4) SNR 상위 k 지점 & 자동 POI 윈도우 생성
topk = 5
peak_idxs = np.argsort(snr)[-topk:][::-1]
print("Top SNR indices:", peak_idxs)

# 윈도우 길이 40, 80, 120 주위로 자동 생성
auto_pois = {}
for pi in peak_idxs:
    for w in [40, 80, 120]:
        s = max(0, pi - w//2)
        e = min(L,    pi + w//2)
        auto_pois[f"{pi-w//2:.0f}_{pi+w//2:.0f}"] = (s, e)

print("\nGenerated POI windows:")
for name,(s,e) in auto_pois.items():
    print(f"  {name}: ({s}, {e})")
