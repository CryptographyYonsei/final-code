# run_all_hw_poi.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 사용자 설정 섹션
# -----------------------------
dataset_paths = {
    "no_desync":      "./Data/ASCAD.h5",
    "desync_50":      "./Data/ASCAD_desync50.h5",
    "desync_100":     "./Data/ASCAD_desync100.h5",
    "raw_to_ascad":   "./Data/ASCAD_from_raw.h5",
}

# POI 윈도우 (±20 샘플) — SNR 상위 피크 기준
poi_windows = {
    "peak126": (106, 146),
    "peak358": (338, 378),
    "peak507": (487, 527),
    "peak567": (547, 587),
    "peak639": (619, 659),
}

BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3

# --------------------------------------------------------------------
# 1) Dataset 클래스 정의
# --------------------------------------------------------------------
class ASCADDatasetFull(Dataset):
    """전체 트레이스(256-way)"""
    def __init__(self, h5_path, mode='profiling'):
        hf   = h5py.File(h5_path, 'r')
        grp  = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        t    = grp['traces'][:].astype(np.float32)
        y    = grp['labels'][:]
        hf.close()

        if y.dtype.names:
            y = y[y.dtype.names[0]]
        self.traces = torch.from_numpy(t).unsqueeze(1)  # (N,1,L)
        self.labels = torch.from_numpy(y.astype(np.int64))

        mn = self.traces.min(dim=2, keepdim=True)[0]
        mx = self.traces.max(dim=2, keepdim=True)[0]
        self.traces = (self.traces - mn) / (mx - mn + 1e-8)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.traces[i], self.labels[i]


class ASCADDatasetHWPOI(Dataset):
    """HW(9-way) + POI 슬라이스"""
    def __init__(self, h5_path, mode, s, e):
        hf   = h5py.File(h5_path, 'r')
        grp  = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        traces = grp['traces'][:].astype(np.float32)
        y      = grp['labels'][:]
        hf.close()

        if y.dtype.names:
            y = y[y.dtype.names[0]]
        hw = np.array([bin(int(v)).count('1') for v in y], dtype=np.int64)

        L = traces.shape[1]
        start = max(0, min(L-1, s))
        end   = max(start+1, min(L, e))
        poi   = traces[:, start:end] if end-start >= 4 else traces

        t = torch.from_numpy(poi).unsqueeze(1)  # (N,1,poi_len)
        mn = t.min(dim=2, keepdim=True)[0]
        mx = t.max(dim=2, keepdim=True)[0]
        self.traces = (t - mn) / (mx - mn + 1e-8)
        self.labels = torch.from_numpy(hw)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.traces[i], self.labels[i]


# --------------------------------------------------------------------
# 2) 모델 구성
# --------------------------------------------------------------------
def get_cnn(trace_len, num_classes):
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(64 * (trace_len//4), 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

def get_mlp(trace_len, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(trace_len, 512), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

class LSTMNet(nn.Module):
    def __init__(self, trace_len, num_classes, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # x: (batch,1,seq)
        x = x.transpose(1,2)        # (batch,seq,1)
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:])

model_configs = {
    "CNN":  get_cnn,
    "MLP":  get_mlp,
    "LSTM": lambda L,C: LSTMNet(L,C),
}


# --------------------------------------------------------------------
# 3) 학습/평가 루틴
# --------------------------------------------------------------------
def train_epoch(model, loader, crit, opt, device):
    model.train()
    tot_loss = tot_corr = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = crit(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()*x.size(0)
        tot_corr += (out.argmax(1)==y).sum().item()
    return tot_loss/len(loader.dataset), tot_corr/len(loader.dataset)

def eval_epoch(model, loader, crit, device):
    model.eval()
    tot_loss = tot_corr = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss= crit(out, y)
            tot_loss += loss.item()*x.size(0)
            tot_corr += (out.argmax(1)==y).sum().item()
    return tot_loss/len(loader.dataset), tot_corr/len(loader.dataset)


# --------------------------------------------------------------------
# 4) Main: 모든 조합 실행 & 결과 테이블
# --------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = []

    for dname, dpath in dataset_paths.items():
        print(f"\n=== Dataset: {dname} ({os.path.basename(dpath)}) ===")

        # ── Experiment A: Full-trace 256-way
        dsA_tr = ASCADDatasetFull(dpath, 'profiling')
        dsA_va = ASCADDatasetFull(dpath, 'attack')
        L_A    = dsA_tr.traces.shape[-1]
        C_A    = int(dsA_tr.labels.max().item())+1  # should be 256
        trA    = DataLoader(dsA_tr, batch_size=BATCH_SIZE, shuffle=True)
        vaA    = DataLoader(dsA_va, batch_size=BATCH_SIZE, shuffle=False)
        for mname, mfunc in model_configs.items():
            print(f"Training {mname} on FULL-trace...")
            mdl = mfunc(L_A, C_A).to(device)
            opt = optim.Adam(mdl.parameters(), lr=LR)
            crit= nn.CrossEntropyLoss()
            for _ in range(EPOCHS):
                train_epoch(mdl, trA, crit, opt, device)
            _, acc = eval_epoch(mdl, vaA, crit, device)
            records.append({
                'dataset': dname,
                'experiment': 'full_trace',
                'poi': 'full_trace',
                'model': mname,
                'val_acc': acc*100
            })

        # ── Experiment B: HW+POI
        for pname, (s,e) in poi_windows.items():
            print(f"  POI {pname} [{s}:{e}]")
            dsB_tr = ASCADDatasetHWPOI(dpath, 'profiling', s, e)
            dsB_va = ASCADDatasetHWPOI(dpath, 'attack',     s, e)
            L_B    = dsB_tr.traces.shape[-1]
            trB    = DataLoader(dsB_tr, batch_size=BATCH_SIZE, shuffle=True)
            vaB    = DataLoader(dsB_va, batch_size=BATCH_SIZE, shuffle=False)
            for mname, mfunc in model_configs.items():
                mdl = mfunc(L_B, 9).to(device)
                opt = optim.Adam(mdl.parameters(), lr=LR)
                crit= nn.CrossEntropyLoss()
                for _ in range(EPOCHS):
                    train_epoch(mdl, trB, crit, opt, device)
                _, acc = eval_epoch(mdl, vaB, crit, device)
                records.append({
                    'dataset': dname,
                    'experiment': 'hw_poi',
                    'poi': pname,
                    'model': mname,
                    'val_acc': acc*100
                })

    # DataFrame & Pivot & 출력
    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index=['dataset','experiment','poi'],
        columns='model',
        values='val_acc'
    )
    # 1) 텍스트 출력
    print("\n=== Final Results ===")
    print(pivot.round(2).to_markdown())

    # 2) 시각화: pivot을 바로 사용
    ax = pivot.plot(
        kind='bar',
        figsize=(12, 6),
        title='Validation Accuracy by Dataset / Experiment / POI',
    )
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('(dataset, experiment, poi)')
    ax.legend(title='Model')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
