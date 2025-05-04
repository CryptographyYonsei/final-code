# synthetic_debug.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— 가짜 데이터셋 ———
class SyntheticTraces(Dataset):
    def __init__(self, n_samples=2000, length=100, n_classes=5):
        # 각 클래스별 평균을 조금씩 다르게 줘서 학습가능하게 만듭니다.
        self.traces = []
        self.labels = []
        for c in range(n_classes):
            # 클래스 c: 평균 = c*0.5, 분산 = 1.0
            ts = np.random.randn(n_samples//n_classes, length) + c*0.5
            self.traces.append(ts)
            self.labels += [c]*(n_samples//n_classes)
        self.traces = np.vstack(self.traces).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        x = torch.from_numpy(self.traces[i]).unsqueeze(0)  # (1, length)
        y = torch.tensor(self.labels[i])
        return x, y

# ——— 아주 단순한 CNN 모델 ———
def get_cnn(L, C):
    return nn.Sequential(
        nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*L, C)
    )

# ——— 학습/평가 루프 ———
def train_one_epoch(model, loader, opt, crit):
    model.train()
    correct = total = 0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = crit(out,y)
        loss.backward(); opt.step()
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total   += y.size(0)
    return correct/total

def eval_one_epoch(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
    return correct/total

# ——— 메인 ———
def main():
    ds = SyntheticTraces(n_samples=2000, length=100, n_classes=5)
    trn, val = torch.utils.data.random_split(ds, [1600, 400])
    lt = DataLoader(trn, batch_size=64, shuffle=True)
    lv = DataLoader(val, batch_size=64)

    model = get_cnn(100, 5).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        tr_acc = train_one_epoch(model, lt, opt, crit)
        va_acc = eval_one_epoch(model, lv)
        print(f"Epoch {epoch:02d} — train: {tr_acc*100:.1f}%  valid: {va_acc*100:.1f}%")

if __name__=="__main__":
    main()
