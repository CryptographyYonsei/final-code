# ascad_multi_dataset_search.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

# ----------------------------------------
# 사용자 설정
# ----------------------------------------
DATASETS = {
    "no_desync":      "./Data/ASCAD.h5",
    "desync_50":      "./Data/ASCAD_desync50.h5",
    "desync_100":     "./Data/ASCAD_desync100.h5",
    "raw_to_ascad":   "./Data/ASCAD_from_raw.h5",
}
EPOCHS       = 10   # 테스트용, 실제론 20~50 권장
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 그리드
batch_sizes  = [32, 64]
lrs          = [1e-3, 5e-4]
poi_radii    = [20, 40, 60]   # peak ± radius

# ------------------------------------------------------------------
# 모델 팩토리
# ------------------------------------------------------------------
def get_cnn(L, C):
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(64 * (L//4), 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, C)
    )

def get_mlp(L, C):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(L, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, C)
    )

class LSTMNet(nn.Module):
    def __init__(self, L, C, hs=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hs, batch_first=True)
        self.fc   = nn.Linear(hs, C)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(-1)     # (B, L, 1)
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

model_factories = {
    "CNN":  get_cnn,
    "MLP":  get_mlp,
    "LSTM": lambda L, C: LSTMNet(L, C)
}

# ------------------------------------------------------------------
# SNR 계산
# ------------------------------------------------------------------
def compute_snr(traces, labels, num_classes=256):
    class_means = []
    class_vars  = []
    for c in range(num_classes):
        idx = np.where(labels==c)[0]
        t   = traces[idx]
        class_means.append(t.mean(axis=0))
        class_vars .append(t.var(axis=0))
    class_means   = np.stack(class_means)
    class_vars    = np.stack(class_vars)
    overall_mean  = traces.mean(axis=0)
    between_var   = ((class_means - overall_mean)**2).mean(axis=0)
    within_var    = class_vars.mean(axis=0) + 1e-10
    return between_var / within_var

# ------------------------------------------------------------------
# Dataset 클래스 (HW + POI)
# ------------------------------------------------------------------
class ASCAD_HW_POI(Dataset):
    def __init__(self, h5_path, mode, start, end):
        hf    = h5py.File(h5_path,"r")
        grp   = hf['Profiling_traces'] if mode=='profiling' else hf['Attack_traces']
        traces= grp['traces'][:]   # (N, L)
        labels= grp['labels'][:]
        hf.close()

        if labels.dtype.names:
            labels = labels[labels.dtype.names[0]]
        hw = np.array([bin(int(x)).count("1") for x in labels], dtype=np.int64)

        poi = traces[:, start:end]
        t   = torch.from_numpy(poi).float()
        tmin= t.min(dim=1, keepdim=True)[0]
        tmax= t.max(dim=1, keepdim=True)[0]
        self.traces = (t - tmin) / (tmax - tmin + 1e-8)
        self.labels = torch.from_numpy(hw).long()

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.traces[i].unsqueeze(0), self.labels[i]

# ------------------------------------------------------------------
# 학습/평가 함수
# ------------------------------------------------------------------
def train_one_epoch(model, loader, opt, crit):
    model.train()
    corr = loss_sum = n = 0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss= crit(out,y)
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0)
        corr     += (out.argmax(1)==y).sum().item()
        n        += x.size(0)
    return corr/n, loss_sum/n

def eval_one_epoch(model, loader, crit):
    model.eval()
    corr = loss_sum = n = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out= model(x)
            loss= crit(out,y)
            loss_sum += loss.item()*x.size(0)
            corr     += (out.argmax(1)==y).sum().item()
            n        += x.size(0)
    return corr/n, loss_sum/n

# ------------------------------------------------------------------
# 메인: 데이터셋별 그리드 서치
# ------------------------------------------------------------------
def main():
    all_results = []

    for dname, dpath in DATASETS.items():
        # 1) profiling trace 로딩 후 SNR 계산
        with h5py.File(dpath,"r") as hf:
            prof_tr = hf["Profiling_traces"]["traces"][:]
            prof_lb = hf["Profiling_traces"]["labels"][:]
        if prof_lb.dtype.names:
            prof_lb = prof_lb[prof_lb.dtype.names[0]]
        snr = compute_snr(prof_tr, prof_lb, num_classes=256)
        peaks = np.argsort(snr)[-5:][::-1]

        # 2) 각 peak & radius 그리드
        for peak in peaks:
            for r in poi_radii:
                s,e = max(0, peak-r), min(prof_tr.shape[1], peak+r)
                poi_nm = f"peak{peak}_±{r}"

                # 데이터로더 준비
                ds_tr = ASCAD_HW_POI(dpath,"profiling", s,e)
                ds_va = ASCAD_HW_POI(dpath,"attack",     s,e)
                for bs in batch_sizes:
                    lt = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=4)
                    lv = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=4)

                    for lr in lrs:
                        for mname, mf in model_factories.items():
                            model = mf(e-s, 9).to(DEVICE)
                            opt   = optim.Adam(model.parameters(), lr=lr)
                            crit  = nn.CrossEntropyLoss()

                            # 학습 & 검증
                            for _ in range(EPOCHS):
                                _, _ = train_one_epoch(model, lt, opt, crit)
                            val_acc, _ = eval_one_epoch(model, lv, crit)

                            print(f"[{dname} | {poi_nm} | {mname} | bs={bs} | lr={lr}] → ValAcc={val_acc*100:.2f}%")
                            all_results.append({
                                "dataset": dname,
                                "poi": poi_nm,
                                "model": mname,
                                "batch_size": bs,
                                "lr": lr,
                                "val_acc": val_acc*100
                            })

    # 3) 전체 결과 DataFrame
    df = pd.DataFrame(all_results)
    print("\n=== ALL RESULTS ===")
    print(df.to_markdown(index=False))

    # 4) 최적 조건 출력 (상위 10)
    top10 = df.sort_values("val_acc", ascending=False).head(10)
    print("\n=== TOP 10 CONFIGS ===")
    print(top10.to_markdown(index=False))

    # 5) 데이터셋별 베스트 1
    best_per_dataset = df.loc[df.groupby("dataset")["val_acc"].idxmax()]
    print("\n=== BEST PER DATASET ===")
    print(best_per_dataset.to_markdown(index=False))

if __name__ == "__main__":
    main()
