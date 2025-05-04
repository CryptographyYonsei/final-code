import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Generate large synthetic dataset
X, y = make_classification(
    n_samples=200000,    # 200k samples
    n_features=100,     # 100 features
    n_informative=30,   # 30 informative
    n_redundant=10,     # 10 redundant
    n_classes=2,
    random_state=0
)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3. Visualize via PCA
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_test)
plt.figure(figsize=(6,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=y_test, alpha=0.4)
plt.title('PCA of Synthetic Test Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# 4. Prepare PyTorch datasets/loaders
def get_loader(X, y, batch_size=128, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = get_loader(X_train, y_train)
test_loader  = get_loader(X_test, y_test, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 5. Define models
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.fc = nn.Linear(32 * length, 2)
    def forward(self, x):
        x = x.unsqueeze(1)         # [B,1,F]
        x = self.conv(x)           # [B,32,F]
        x = x.flatten(1)
        return self.fc(x)

class LSTMNet(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 2)
    def forward(self, x):
        x = x.unsqueeze(2)         # [B,F,1]
        out, _ = self.lstm(x)      # [B,F,50]
        return self.fc(out[:, -1, :])

# 6. Instantiate
models = {
    'MLP': MLP(X_train.shape[1]).to(device),
    'CNN1D': CNN1D(X_train.shape[1]).to(device),
    'LSTM': LSTMNet(X_train.shape[1]).to(device)
}

# 7. Training & evaluation loop
def train_eval(model, epochs=10, lr=1e-3):
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_accs = []
    valid_accs = []

    for epoch in range(1, epochs+1):
        # --- train ---
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # --- eval on train set ---
        model.eval()
        with torch.no_grad():
            correct_t, total_t = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct_t += (preds == yb).sum().item()
                total_t   += yb.size(0)
            train_acc = correct_t/total_t*100
            train_accs.append(train_acc)

        # --- eval on test set ---
        with torch.no_grad():
            correct_v, total_v = 0, 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct_v += (preds == yb).sum().item()
                total_v   += yb.size(0)
            valid_acc = correct_v/total_v*100
            valid_accs.append(valid_acc)

        print(f"[{model.__class__.__name__}] Epoch {epoch:02d} — train: {train_acc:.1f}%  valid: {valid_acc:.1f}%")

    return train_accs, valid_accs

# 8. 여러 모델 돌려서 결과 수집
history = {}
for name, mdl in models.items():
    train_accs, valid_accs = train_eval(mdl, epochs=10, lr=1e-3)
    history[name] = (train_accs, valid_accs)

# 9. 꺾은선 그래프로 시각화
plt.figure(figsize=(8,5))
epochs = list(range(1, 11))
for name, (ta, va) in history.items():
    plt.plot(epochs, ta, marker='o', linestyle='-', label=f'{name} Train')
    plt.plot(epochs, va, marker='s', linestyle='--', label=f'{name} Valid')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training & Validation Accuracy per Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Print final accuracies
print('Final Accuracies:')
for name, (train_accs, valid_accs) in history.items():
    print(f"{name:<6} Train: {train_accs[-1]:.2f}%  Valid: {valid_accs[-1]:.2f}%")
