# ascad_train_compare.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# 사용자 설정 섹션 (여기만 수정하세요)
# -----------------------------
H5_PATH     = r"./Data/ASCAD.h5"   # ASCAD HDF5 파일 경로
BATCH_SIZE  = 64                   # 배치 크기
EPOCHS      = 20                   # 학습 에폭 수
LR          = 1e-3                 # 학습률
POI_START   = 3000                 # POI 구간 시작 (수정 버전만)
POI_END     = 3200                 # POI 구간 끝   (수정 버전만)
# -----------------------------

# -----------------------------------------------------------------------------
# 1) Dataset 클래스 정의
# -----------------------------------------------------------------------------
class ASCADDatasetFull(Dataset):
    """원본: 전체 trace, 256-way 분류"""
    def __init__(self, h5_path, mode='profiling'):
        hf = h5py.File(h5_path,'r')
        grp = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        traces = grp['traces'][:]    # (N, L)
        labels = grp['labels'][:]
        hf.close()

        if labels.dtype.names:
            labels = labels[labels.dtype.names[0]]
        self.traces = torch.from_numpy(traces).float()
        self.labels = torch.from_numpy(labels).long()

        # Min-Max 정규화
        tmin = self.traces.min(dim=1, keepdim=True)[0]
        tmax = self.traces.max(dim=1, keepdim=True)[0]
        self.traces = (self.traces - tmin) / (tmax - tmin + 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.traces[idx].unsqueeze(0), self.labels[idx]


class ASCADDatasetHWPOI(Dataset):
    """수정: Hamming Weight 9-way, POI 구간 슬라이싱 (자동 클램핑 + fallback)"""
    def __init__(self, h5_path, mode='profiling'):
        hf = h5py.File(h5_path,'r')
        grp = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        traces = grp['traces'][:]    # (N, L)
        labels = grp['labels'][:]
        hf.close()

        # raw labels 추출
        if labels.dtype.names:
            labels = labels[labels.dtype.names[0]]
        # Hamming Weight (0~8)
        hw = np.array([bin(int(x)).count('1') for x in labels], dtype=np.int64)

        # POI 구간을 trace 길이에 맞춰 클램핑
        L = traces.shape[1]
        start = min(max(0, POI_START), L-1)
        end   = min(max(start+1, POI_END), L)
        poi_len = end - start

        # 만약 POI 길이가 너무 작아서 풀링을 통과할 수 없으면 전체 trace fallback
        if poi_len < 4:
            print(f"[Warning] POI window too small ({poi_len}), using full trace of length {L}.")
            poi = traces
        else:
            poi = traces[:, start:end]

        self.traces = torch.from_numpy(poi).float()
        self.labels = torch.from_numpy(hw).long()

        # Min-Max 정규화
        tmin = self.traces.min(dim=1, keepdim=True)[0]
        tmax = self.traces.max(dim=1, keepdim=True)[0]
        self.traces = (self.traces - tmin) / (tmax - tmin + 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.traces[idx].unsqueeze(0), self.labels[idx]


# -----------------------------------------------------------------------------
# 2) 모델 정의 (공통)
# -----------------------------------------------------------------------------
class SideChannelCNN(nn.Module):
    def __init__(self, trace_len, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * (trace_len // 4), 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# 3) 학습/평가 함수 (공통)
# -----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss = tot_corr = tot_cnt = 0
    for x,y in tqdm(loader, desc='Train '):
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item()*x.size(0)
        tot_corr += (out.argmax(1)==y).sum().item()
        tot_cnt += x.size(0)
    print(f"Train  - Loss: {tot_loss/tot_cnt:.4f}, Acc: {100*tot_corr/tot_cnt:.2f}%")


def eval_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_corr = tot_cnt = 0
    with torch.no_grad():
        for x,y in tqdm(loader, desc='Eval  '):
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            tot_loss += loss.item()*x.size(0)
            tot_corr += (out.argmax(1)==y).sum().item()
            tot_cnt += x.size(0)
    print(f"Attack - Loss: {tot_loss/tot_cnt:.4f}, Acc: {100*tot_corr/tot_cnt:.2f}%")


# -----------------------------------------------------------------------------
# 4) 메인: 두 실험 순차 실행
# -----------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", device)

    # --- 실험 A: 전체 trace, 256-way ---
    print("\n===== Experiment A: Full Trace (256 classes) =====")
    dsA_train = ASCADDatasetFull(H5_PATH, mode='profiling')
    dsA_val   = ASCADDatasetFull(H5_PATH, mode='attack')
    L_A       = dsA_train.traces.shape[-1]
    numA      = int(dsA_train.labels.max().item()) + 1

    loaderA_tr = DataLoader(dsA_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    loaderA_va = DataLoader(dsA_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    modelA = SideChannelCNN(trace_len=L_A, num_classes=numA).to(device)
    optA   = optim.Adam(modelA.parameters(), lr=LR)
    crit   = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS+1):
        print(f"\n-- Epoch {ep}/{EPOCHS} --")
        train_epoch(modelA, loaderA_tr, crit, optA, device)
        eval_epoch(modelA, loaderA_va, crit, device)
    torch.save(modelA.state_dict(), 'ascad_cnn_full.pth')
    print("Saved Model: ascad_cnn_full.pth")

    # --- 실험 B: HW+POI, 9-way ---
    print("\n===== Experiment B: HW+POI (9 classes) =====")
    dsB_train = ASCADDatasetHWPOI(H5_PATH, mode='profiling')
    dsB_val   = ASCADDatasetHWPOI(H5_PATH, mode='attack')
    L_B       = dsB_train.traces.shape[-1]
    numB      = 9  # Hamming Weight 0~8

    loaderB_tr = DataLoader(dsB_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    loaderB_va = DataLoader(dsB_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    modelB = SideChannelCNN(trace_len=L_B, num_classes=numB).to(device)
    optB   = optim.Adam(modelB.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        print(f"\n-- Epoch {ep}/{EPOCHS} --")
        train_epoch(modelB, loaderB_tr, crit, optB, device)
        eval_epoch(modelB, loaderB_va, crit, device)
    torch.save(modelB.state_dict(), 'ascad_cnn_hw_poi.pth')
    print("Saved Model: ascad_cnn_hw_poi.pth")


if __name__=='__main__':
    main()
