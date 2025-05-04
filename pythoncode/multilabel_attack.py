# multilabel_attack.py

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ASCAD_generate import extract_traces  # ASCAD 제공 스크립트

# 0) 1회만 실행: multilabel HDF5 생성
orig_traces = "./Data/ATMega8515_raw_traces.h5"
profiling_idx = list(range(0,50000))
attack_idx    = list(range(50000,60000))
target_pts    = list(range(45400,46100))
extract_traces(
    orig_traces,
    "./Data/ASCAD_ml.h5",
    profiling_idx, attack_idx,
    target_pts,
    profiling_desync=0, attack_desync=0,
    multilabel=1
)

# 1) Dataset 클래스
class ASCADDatasetMulti(Dataset):
    def __init__(self, h5_path, mode="profiling", byte_idx=0):
        hf = h5py.File(h5_path, "r")
        grp = hf["Profiling_traces"] if mode=="profiling" else hf["Attack_traces"]

        # traces: (N, L)
        traces = grp["traces"][:]
        # labels: structured array with fields:
        #   alpha_mask (1,), beta_mask (1,), sbox_masked (16,), …
        labels = grp["labels"][:]
        hf.close()

        # Squeeze structured fields
        alpha = labels["alpha_mask"].squeeze()         # (N,)
        beta  = labels["beta_mask"].squeeze()          # (N,)
        sbox  = np.vstack(labels["sbox_masked"])       # (N,16)

        # 선택한 바이트만
        y_sbox = sbox[:, byte_idx]                     # (N,)

        # Tensor 변환 + 정규화
        X = torch.from_numpy(traces).float()
        X = (X - X.min(dim=1,keepdim=True)[0]) / (X.max(dim=1,keepdim=True)[0] - X.min(dim=1,keepdim=True)[0] + 1e-8)
        self.X     = X.unsqueeze(1)                    # (N,1,L)
        self.alpha = torch.from_numpy(alpha).long()
        self.beta  = torch.from_numpy(beta).long()
        self.y     = torch.from_numpy(y_sbox).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.alpha[idx], self.y[idx]

# 2) 멀티태스크 모델
class MultiTaskCNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv1d(1,32,3,padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        hidden_dim = 64 * (seq_len//4)
        # α 마스크는 256개 클래스, S-Box 출력도 256개 클래스
        self.head_alpha = nn.Linear(hidden_dim, 256)
        self.head_sbox  = nn.Linear(hidden_dim, 256)

    def forward(self, x):
        z = self.shared(x)       # (batch, hidden_dim)
        a = self.head_alpha(z)   # (batch,256)
        s = self.head_sbox(z)    # (batch,256)
        return a, s

# 3) 학습 루프
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = ASCADDatasetMulti("./Data/ASCAD_ml.h5","profiling", byte_idx=0)
    ds_va = ASCADDatasetMulti("./Data/ASCAD_ml.h5","attack",    byte_idx=0)
    loader_tr = DataLoader(ds_tr, batch_size=128, shuffle=True,  num_workers=4)
    loader_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=4)

    model = MultiTaskCNN(seq_len=ds_tr.X.shape[-1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        # train
        model.train()
        tloss = tacc = tc = 0
        for X, a, y in loader_tr:
            X, a, y = X.to(device), a.to(device), y.to(device)
            out_a, out_s = model(X)
            loss = crit(out_a, a) + crit(out_s, y)
            opt.zero_grad(); loss.backward(); opt.step()

            tloss += loss.item()*X.size(0)
            # s-box 태스크 정확도
            tacc += (out_s.argmax(1)==y).sum().item()
            tc   += X.size(0)
        print(f"Epoch{epoch} Train loss:{tloss/tc:.3f}  S-Box acc:{100*tacc/tc:.2f}%")

        # eval (S-Box 만 확인)
        model.eval()
        vacc = vc = 0
        with torch.no_grad():
            for X, a, y in loader_va:
                X, y = X.to(device), y.to(device)
                _, out_s = model(X)
                vacc += (out_s.argmax(1)==y).sum().item()
                vc   += X.size(0)
        print(f"       Val  S-Box acc:{100*vacc/vc:.2f}%\n")

if __name__=="__main__":
    train()
