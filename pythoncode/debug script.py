# debug_ascad.py
# 디버깅용 전체 스크립트 — 데이터 로드, 포맷, 모델 순방향/역방향 확인

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# ASCADDatasetHWPOI 클래스 정의
# -----------------------------
class ASCADDatasetHWPOI(Dataset):
    """Hamming Weight 레이블, POI 자동 클램핑 포함"""
    def __init__(self, h5_path, mode, poi_start, poi_end):
        hf = h5py.File(h5_path, 'r')
        grp = hf['Profiling_traces'] if mode == 'profiling' else hf['Attack_traces']
        traces = grp['traces'][:]    # shape (N, L)
        labels = grp['labels'][:]
        hf.close()

        # structured dtype 처리
        if labels.dtype.names:
            labels = labels[labels.dtype.names[0]]
        # Hamming Weight 계산 (0~8)
        hw = np.array([bin(int(x)).count('1') for x in labels], dtype=np.int64)

        # POI 구간 클램핑
        L = traces.shape[1]
        start = min(max(0, poi_start), L - 1)
        end   = min(max(start + 1, poi_end), L)
        poi_len = end - start
        poi = traces if poi_len < 4 else traces[:, start:end]

        # tensor 변환 및 Min–Max 정규화
        t = torch.from_numpy(poi).float()
        tmin = t.min(dim=1, keepdim=True)[0]
        tmax = t.max(dim=1, keepdim=True)[0]
        self.traces = (t - tmin) / (tmax - tmin + 1e-8)
        self.labels = torch.from_numpy(hw).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # (1, seq_len), label
        return self.traces[idx].unsqueeze(0), self.labels[idx]


# -----------------------------
# 간단한 CNN 모델 정의
# -----------------------------
def get_cnn(trace_len, num_classes):
    """디버깅용 1D-CNN"""
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(16 * (trace_len // 2), num_classes)
    )


# -----------------------------
# 디버깅 로직
# -----------------------------
if __name__ == "__main__":
    # 1) 데이터셋 & 로더 준비
    ds = ASCADDatasetHWPOI(r"./Data/ASCAD.h5", mode="profiling", poi_start=100, poi_end=160)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    # 2) 첫 배치 확인
    x_batch, y_batch = next(iter(loader))
    print("▶ Batch shapes:", x_batch.shape, y_batch.shape)
    print("▶ Trace value range:", f"{x_batch.min().item():.4f} ~ {x_batch.max().item():.4f}")
    print("▶ Unique labels in batch:", torch.unique(y_batch).tolist())

    # 3) 모델 생성 및 손실함수, 옵티마이저 선언
    trace_len = x_batch.shape[-1]
    model = get_cnn(trace_len, num_classes=9)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    # 4) 순방향 + 손실, 정확도 계산
    model.train()
    out = model(x_batch)  # (batch_size, num_classes)
    loss = crit(out, y_batch)
    pred = out.argmax(dim=1)
    acc  = (pred == y_batch).float().mean()
    print(f"▶ Initial Loss: {loss.item():.4f}, Accuracy: {acc.item()*100:.2f}%")

    # 5) 역전파 후 gradient 확인
    opt.zero_grad()
    loss.backward()
    grad_means = [p.grad.abs().mean().item() if p.grad is not None else 0.0
                  for p in model.parameters()]
    print("▶ Gradient mean per parameter tensor:", grad_means)
