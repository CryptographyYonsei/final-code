import h5py
import numpy as np
import matplotlib.pyplot as plt

# 1) 데이터 로드
with h5py.File(r"./Data/ASCAD.h5", "r") as hf:
    traces = hf["Profiling_traces/traces"][:]  # shape (N, L)
    labels = hf["Profiling_traces/labels"][:]
# structured dtype이면 첫 필드 분리
if labels.dtype.names:
    labels = labels[labels.dtype.names[0]]
# Hamming Weight
hw = np.array([bin(int(x)).count("1") for x in labels], dtype=np.float32)

# 2) 각 시간축 t에 대해 상관계수 계산
#    pearsonr 대신 간단히 (cov/σxσy)
mean_hw = hw.mean()
std_hw  = hw.std()
corrs = []
for t in range(traces.shape[1]):
    x = traces[:, t].astype(np.float32)
    mx, sx = x.mean(), x.std()
    cov = np.mean((x - mx) * (hw - mean_hw))
    corrs.append(cov / (sx * std_hw + 1e-8))
corrs = np.array(corrs)

# 3) 플롯
plt.figure(figsize=(10,4))
plt.plot(corrs, linewidth=1)
plt.title("Correlation between trace[t] and Hamming Weight")
plt.xlabel("Sample index (t)")
plt.ylabel("Pearson correlation")
plt.grid(True)
plt.show()
