# ascad_model_poi_grid_search.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 사용자 설정 섹션
# -----------------------------
H5_PATH    = r"./Data/ASCAD.h5"  # ASCAD HDF5 파일 경로
BATCH_SIZE = 64                  # 배치 크기
EPOCHS     = 20                  # 학습 에폭 수
LR         = 1e-3                # 학습률

# ────────────────────────────────────
#  auto_poi_snr.py 출력값으로 만든 poi_configs
# ────────────────────────────────────
poi_configs = {
    "547_587": (547, 587),
    "527_607": (527, 607),
    "507_627": (507, 627),
    "619_659": (619, 659),
    "599_679": (599, 699),
    "579_699": (579, 699),

    "106_146": (106, 146),
    "86_166":  (86, 166),
    "66_186":  (66, 186),

    "338_378": (338, 378),
    "318_398": (318, 398),
    "298_418": (298, 418),

    "487_527": (487, 527),
    "467_547": (467, 547),
    "447_567": (447, 567),
}

# -----------------------------
# 모델 구성 함수
# -----------------------------
def get_cnn(trace_len, num_classes):
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(64 * (trace_len // 4), 256), nn.ReLU(),
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
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 1, seq_len) → (batch, seq_len, 1)
        x = x.squeeze(1).unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model_configs = {
    "CNN": get_cnn,
    "MLP": get_mlp,
    "LSTM": lambda L, C: LSTMNet(L, C)
}

# -----------------------------------------------------------------------
# Dataset 클래스 (HW+POI)
# -----------------------------------------------------------------------
class ASCADDatasetHWPOI(Dataset):
    def __init__(self, h5_path, mode, poi_start, poi_end):
        hf = h5py.File(h5_path, 'r')
        grp = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        traces = grp['traces'][:]    # shape (N, L)
        labels = grp['labels'][:]
        hf.close()

        # structured dtype 처리
        if labels.dtype.names:
            labels = labels[labels.dtype.names[0]]
        # Hamming Weight (0~8)
        hw = np.array([bin(int(x)).count('1') for x in labels], dtype=np.int64)

        # POI 슬라이스
        L = traces.shape[1]
        start = min(max(0, poi_start), L-1)
        end   = min(max(start+1, poi_end), L)
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
        # returns (1, seq_len), label
        return self.traces[idx].unsqueeze(0), self.labels[idx]

# -----------------------------------------------------------------------
# 학습/평가 함수
# -----------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_corr = total_cnt = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_corr += (out.argmax(1) == y).sum().item()
        total_cnt  += x.size(0)
    return total_loss/total_cnt, total_corr/total_cnt

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_corr = total_cnt = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_corr += (out.argmax(1) == y).sum().item()
            total_cnt  += x.size(0)
    return total_loss/total_cnt, total_corr/total_cnt

# -----------------------------------------------------------------------
# 메인: 모델 × POI 그리드 서치, 결과 출력 및 시각화
# -----------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = []

    for mname, mfunc in model_configs.items():
        for pname, (s, e) in poi_configs.items():
            print(f">> Model: {mname}, POI: {pname}")
            # Dataset & DataLoader
            ds_tr = ASCADDatasetHWPOI(H5_PATH, 'profiling', s, e)
            ds_va = ASCADDatasetHWPOI(H5_PATH, 'attack',     s, e)
            L = ds_tr.traces.shape[-1]
            loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            loader_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            # Model/Optimizer/Loss
            model = mfunc(L, 9).to(device)
            opt   = optim.Adam(model.parameters(), lr=LR)
            crit  = nn.CrossEntropyLoss()

            # Train & Eval
            for epoch in range(1, EPOCHS+1):
                _, train_acc = train_epoch(model, loader_tr, crit, opt, device)
                _, val_acc   = eval_epoch(model, loader_va, crit, device)

            records.append({
                'model': mname,
                'poi':   pname,
                'val_acc': val_acc * 100
            })

    # DataFrame 생성 및 출력
    df = pd.DataFrame(records)
    df_pivot = df.pivot(index='poi', columns='model', values='val_acc')
    print("\n=== Numeric Results ===")
    print(df_pivot.to_markdown())

    # 시각화
    df_pivot.plot(marker='o', figsize=(10,6))
    plt.title('Validation Accuracy by Model and POI')
    plt.xlabel('POI Window')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Model')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
