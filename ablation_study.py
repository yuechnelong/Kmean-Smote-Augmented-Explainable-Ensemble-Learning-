# -*- coding: utf-8 -*-
"""
消融实验：CNN-only vs BiLSTM-only vs CNN-BiLSTM
================================================
- 特征：7 个（删 GP），5×7 序列；增强 KMeansSMOTE@0.15
- 三个变体用相同超参（hidden=34, lr=0.00797, batch=32）
- 输出：指标对比 + 逐类 F1 热力图
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)
from imblearn.over_sampling import KMeansSMOTE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data.xlsx")
RING_COL = "Ring No."
LABEL_COL = "Geological Condition"
WINDOW = 5
SPLIT = int(round(0.7 * 1380))
DPI = 1500
RATIO = 0.15
HIDDEN = 34
LR = 0.00797
BATCH = 32
EPOCHS = 60

CLASS_ORDER = [
    "Moderately Weathered Sandstone", "Moderately Weathered Sandstone/Silty Clay",
    "Strongly/Moderately Weathered Sandstone", "Strongly Weathered Sandstone",
    "Silty Clay", "Silty Clay/Strongly Weathered Argillaceous Sandstone",
]
CLASS_ABBR = ["MWS", "MWS/SC", "S/MWS", "SWS", "SC", "SC/SWAS"]
class_to_idx = {n: i for i, n in enumerate(CLASS_ORDER)}

df = pd.read_excel(DATA, sheet_name=0)
df.columns = [c.replace("\n", " ") for c in df.columns]
df = df.sort_values(RING_COL).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in (RING_COL, LABEL_COL) and "Grouting Pressure" not in c]
y_all = df[LABEL_COL].map(class_to_idx).values
X_feat = df[feat_cols].values.astype(np.float64)
N = len(df)
n_feat = len(feat_cols)

Xf, Xs, y, t = [], [], [], []
for i in range(WINDOW - 1, N - 1):
    Xf.append(X_feat[i - WINDOW + 1: i + 1].reshape(-1))
    Xs.append(X_feat[i - WINDOW + 1: i + 1])
    y.append(y_all[i + 1])
    t.append(i + 1)
Xf, Xs, y, t = map(np.array, (Xf, Xs, y, t))
mask = t < SPLIT
Xf_tr, Xf_te = Xf[mask], Xf[~mask]
Xs_tr, Xs_te = Xs[mask], Xs[~mask]
ytr, yte = y[mask], y[~mask]

sc = StandardScaler().fit(Xs_tr.reshape(-1, n_feat))
Xs_tr_s = sc.transform(Xs_tr.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
Xs_te_s = sc.transform(Xs_te.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)

counts = pd.Series(ytr).value_counts()
target = int(round(RATIO * counts.max()))
strat = {c: target for c in counts.index if counts[c] < target}
smp = KMeansSMOTE(sampling_strategy=strat, k_neighbors=2, cluster_balance_threshold=0.0,
                  random_state=SEED)
Xa, ya = smp.fit_resample(Xf_tr, ytr)
Xa_seq = sc.transform(Xa.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)


# ---------------- 三个变体 ----------------
class CNNOnly(nn.Module):
    def __init__(self, in_f=n_feat, hidden=HIDDEN, n_cls=6):
        super().__init__()
        self.conv = nn.Conv1d(in_f, hidden, 3, padding=1)
        self.fc = nn.Linear(hidden, n_cls)
    def forward(self, x):
        x = x.permute(0, 2, 1); x = torch.relu(self.conv(x)); x = x.mean(dim=2)
        return self.fc(x)


class BiLSTMOnly(nn.Module):
    def __init__(self, in_f=n_feat, hidden=HIDDEN, n_cls=6):
        super().__init__()
        self.lstm = nn.LSTM(in_f, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_cls)
    def forward(self, x):
        x, _ = self.lstm(x); x = x.mean(dim=1)
        return self.fc(x)


class CNNBiLSTM(nn.Module):
    def __init__(self, in_f=n_feat, hidden=HIDDEN, n_cls=6):
        super().__init__()
        self.conv = nn.Conv1d(in_f, hidden, 3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_cls)
    def forward(self, x):
        x = x.permute(0, 2, 1); x = torch.relu(self.conv(x)); x = x.permute(0, 2, 1)
        x, _ = self.lstm(x); x = x.mean(dim=1)
        return self.fc(x)


def train_model(model, Xtr, ytr, Xte):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


variants = [
    ("CNN-only", CNNOnly()),
    ("BiLSTM-only", BiLSTMOnly()),
    ("CNN-BiLSTM", CNNBiLSTM()),
]

results = []
per_f1 = {}
for name, model in variants:
    yp = train_model(model, Xa_seq, ya, Xs_te_s)
    acc = accuracy_score(yte, yp)
    bacc = balanced_accuracy_score(yte, yp)
    mf = precision_recall_fscore_support(yte, yp, labels=range(6), average="macro", zero_division=0)[2]
    wf = precision_recall_fscore_support(yte, yp, labels=range(6), average="weighted", zero_division=0)[2]
    f1 = precision_recall_fscore_support(yte, yp, labels=range(6), average=None, zero_division=0)[2]
    results.append({"Model": name, "Accuracy": round(acc, 4), "Balanced_Acc": round(bacc, 4),
                    "Macro_F1": round(mf, 4), "Weighted_F1": round(wf, 4), "Params": n_params(model)})
    per_f1[name] = f1
    print(f"{name:<12} Acc={acc:.4f} BalancedAcc={bacc:.4f} MacroF1={mf:.4f} WeightedF1={wf:.4f} 参数={n_params(model)}")

res = pd.DataFrame(results)
f1_df = pd.DataFrame(per_f1, index=CLASS_ABBR).T
res.to_csv(os.path.join(HERE, "消融实验.csv"), index=False, encoding="utf-8-sig")
f1_df.to_csv(os.path.join(HERE, "消融_逐类F1.csv"), encoding="utf-8-sig")

print("\n===== 消融实验指标 =====")
print(res.to_string(index=False))
print("\n===== 逐类 F1 =====")
print(f1_df.round(4).to_string())

# ---------------- 图1：指标对比 ----------------
order = ["CNN-only", "BiLSTM-only", "CNN-BiLSTM"]
fig, ax = plt.subplots(figsize=(7, 4.6))
x = np.arange(3); w = 0.36
mf = [res[res["Model"] == m]["Macro_F1"].iloc[0] for m in order]
ba = [res[res["Model"] == m]["Balanced_Acc"].iloc[0] for m in order]
ax.bar(x - w/2, mf, w, label="Macro-F1", color="#2E5E8C", edgecolor="white")
ax.bar(x + w/2, ba, w, label="Balanced Accuracy", color="#9BB8D3", edgecolor="white")
for i in range(3):
    ax.text(i - w/2, mf[i] + 0.005, f"{mf[i]:.4f}", ha="center", fontsize=12)
    ax.text(i + w/2, ba[i] + 0.005, f"{ba[i]:.4f}", ha="center", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=14)
ax.set_ylabel("Score", fontsize=16); ax.set_ylim(0.85, 1.0)
ax.legend(frameon=False, fontsize=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, ls="--", axis="y")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ablation_metrics.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 图2：逐类 F1 热力图 ----------------
fig, ax = plt.subplots(figsize=(7, 3.6))
mat = f1_df.reindex(order).values
im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_ABBR, fontsize=13)
ax.set_yticks(range(3)); ax.set_yticklabels(order, fontsize=13)
for i in range(3):
    for j in range(6):
        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=11,
                color="black" if 0.72 < mat[i, j] < 0.95 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("F1", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ablation_per_class_f1.png"), dpi=DPI, facecolor="white")
plt.close(fig)

print("\n输出文件已保存到:", HERE)
