# run_all_datasets.py

import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 사용자 설정 섹션
# -----------------------------
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 실험할 HDF5 파일 목록
DATASETS = {
    "no_desync":      "./Data/ASCAD.h5",
    "desync_50":      "./Data/ASCAD_desync50.h5",
    "desync_100":     "./Data/ASCAD_desync100.h5",
    # 원본 raw 로부터 변환한 ASCAD 포맷
    "raw_to_ascad":   "./Data/ASCAD_from_raw.h5",
}

# -----------------------------------------------------------------------
# 1) Dataset 클래스: 전체 trace, 256-way S-Box 분류
# -----------------------------------------------------------------------
class ASCADDatasetFull(Dataset):
    def __init__(self, h5_path, mode='profiling'):
        hf = h5py.File(h5_path, 'r')
        grp = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        traces = grp['traces'][:]    # (N, L)
        labels = grp['labels'][:]
        hf.close()

        # structured dtype 처리
        if labels.dtype.names:
            labels = labels[labels.dtype.names[0]]

        self.X = torch.from_numpy(traces).float()
        self.y = torch.from_numpy(labels).long()

        # Min–Max 정규화 (trace마다)
        tmin = self.X.min(dim=1, keepdim=True)[0]
        tmax = self.X.max(dim=1, keepdim=True)[0]
        self.X = (self.X - tmin) / (tmax - tmin + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

# -----------------------------------------------------------------------
# 2) CNN 모델 정의
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# 3) 학습/평가 함수
# -----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_corr = total_cnt = 0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_corr += (pred.argmax(1) == y).sum().item()
        total_cnt  += X.size(0)
    return total_loss/total_cnt, total_corr/total_cnt


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = total_corr = total_cnt = 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Eval ", leave=False):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item() * X.size(0)
            total_corr += (pred.argmax(1) == y).sum().item()
            total_cnt  += X.size(0)
    return total_loss/total_cnt, total_corr/total_cnt

# -----------------------------------------------------------------------
# 4) 메인: 모든 데이터셋 순회 실험
# -----------------------------------------------------------------------
def main():
    results = []
    for tag, path in DATASETS.items():
        print(f"\n=== Dataset: {tag} ({os.path.basename(path)}) ===")
        # Profiling_traces/Attack_traces 그룹이 없으면 스킵
        with h5py.File(path, 'r') as hf:
            if 'Profiling_traces' not in hf or 'Attack_traces' not in hf:
                print(f"  ➔ SKIP: '{tag}' 에는 Profiling_traces/Attack_traces 그룹이 없습니다: {list(hf.keys())}")
                continue

        # 데이터 로드
        ds_tr = ASCADDatasetFull(path, 'profiling')
        ds_va = ASCADDatasetFull(path, 'attack')
        L      = ds_tr.X.shape[-1]
        num_cl = int(ds_tr.y.max().item()) + 1

        loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
        loader_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # 모델/최적화/손실
        model = SideChannelCNN(L, num_cl).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR)
        crit  = nn.CrossEntropyLoss()

        # 학습 & 평가
        for ep in range(1, EPOCHS+1):
            loss_tr, acc_tr = train_one_epoch(model, loader_tr, crit, opt)
            loss_va, acc_va = eval_one_epoch(model, loader_va, crit)
            print(f" Epoch {ep:02d}: Train Acc {acc_tr*100:5.2f}%,  Val Acc {acc_va*100:5.2f}%")

        results.append({'dataset': tag, 'val_acc': acc_va * 100})

    # 요약 테이블
    df = pd.DataFrame(results).set_index('dataset')
    print("\n=== Final comparison ===")
    print(df.to_markdown())


if __name__ == '__main__':
    main()
