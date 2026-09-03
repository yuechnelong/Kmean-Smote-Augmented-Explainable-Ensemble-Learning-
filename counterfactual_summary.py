# -*- coding: utf-8 -*-
"""反事实分析汇总：3 个错分样本的干预矩阵热力图 + 汇总表。"""
import os
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
HORIZON = 1
SPLIT = int(round(0.7 * 1380))
DPI = 2000
CLASS6 = 5
CLASS2 = 1

CLASS_ORDER = [
    "Moderately Weathered Sandstone", "Moderately Weathered Sandstone/Silty Clay",
    "Strongly/Moderately Weathered Sandstone", "Strongly Weathered Sandstone",
    "Silty Clay", "Silty Clay/Strongly Weathered Argillaceous Sandstone",
]
CLASS_LABELS = ["MWS", "MWS/SC", "S/MWS", "SWS", "SC", "SC/SWAS"]
class_to_idx = {n: i for i, n in enumerate(CLASS_ORDER)}
FEAT_SHORT = ["TTF", "CT", "CS", "CP", "GV", "IA", "AR"]

df = pd.read_excel(DATA, sheet_name=0)
df.columns = [c.replace("\n", " ") for c in df.columns]
df = df.sort_values(RING_COL).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in (RING_COL, LABEL_COL) and "Grouting Pressure" not in c]
y_all = df[LABEL_COL].map(class_to_idx).values
X_feat = df[feat_cols].values.astype(np.float64)
N = len(df)

X, y, t = [], [], []
for i in range(WINDOW - 1, N - HORIZON):
    X.append(X_feat[i - WINDOW + 1: i + 1].reshape(-1))
    y.append(y_all[i + HORIZON])
    t.append(i + HORIZON)
X, y, t = np.array(X), np.array(y), np.array(t)
mask = t < SPLIT
Xtr, ytr = X[mask], y[mask]
Xte, yte = X[~mask], y[~mask]

n_feat = len(feat_cols)
ACTIONABLE_IDX = list(range((WINDOW - 1) * n_feat, WINDOW * n_feat))
feat_std = Xtr.std(axis=0) + 1e-9
action_std = feat_std[ACTIONABLE_IDX]

counts = pd.Series(ytr).value_counts()
target = int(round(0.15 * counts.max()))
strat = {c: target for c in counts.index if counts[c] < target}
smp = KMeansSMOTE(sampling_strategy=strat, k_neighbors=2, cluster_balance_threshold=0.0,
                  random_state=SEED)
Xa, ya = smp.fit_resample(Xtr, ytr)

# 序列标准化 + CNN-BiLSTM（优化后超参）
sc = StandardScaler().fit(Xtr.reshape(-1, n_feat))


class CNNBiLSTM(nn.Module):
    def __init__(self, in_f=n_feat, hidden=34, n_cls=6):
        super().__init__()
        self.conv = nn.Conv1d(in_f, hidden, 3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_cls)
    def forward(self, x):
        x = x.permute(0, 2, 1); x = torch.relu(self.conv(x)); x = x.permute(0, 2, 1)
        x, _ = self.lstm(x); x = x.mean(dim=1); return self.fc(x)


def predict_flat(X_flat):
    X_seq_s = sc.transform(X_flat.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_seq_s, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


model = CNNBiLSTM()
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.00797, weight_decay=1e-4)
Xa_seq = sc.transform(Xa.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
Xt = torch.tensor(Xa_seq, dtype=torch.float32); yt = torch.tensor(ya, dtype=torch.long)
dl = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
model.train()
for _ in range(60):
    for xb, yb in dl:
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

proba = predict_flat(Xte)
yp = proba.argmax(1)

mis_comp = [i for i in np.where(yp != yte)[0] if yte[i] in (CLASS2, CLASS6)]


def best_counterfactual(sample):
    query = Xte[sample]; true_cls = yte[sample]
    donor_idx = np.where(ytr == true_cls)[0]
    rng = np.random.RandomState(SEED)
    donors = Xtr[rng.choice(donor_idx, size=min(150, len(donor_idx)), replace=False)]
    cands = []
    for donor in donors:
        c = query.copy(); c[ACTIONABLE_IDX] = donor[ACTIONABLE_IDX]
        cands.append(c)
        for nc in range(1, 4):
            for sub in combinations(ACTIONABLE_IDX, nc):
                c = query.copy(); c[list(sub)] = donor[list(sub)]
                cands.append(c)
    cands = np.array(cands)
    p = predict_flat(cands)
    flipped = p.argmax(axis=1) == true_cls
    delta = cands[:, ACTIONABLE_IDX] - query[ACTIONABLE_IDX]
    l2 = np.sqrt(((delta / action_std) ** 2).sum(axis=1))
    n_changed = (np.abs(delta) > 1e-8).sum(axis=1)
    valid = np.where(flipped)[0]
    if len(valid) == 0:
        return None
    order = valid[np.lexsort((l2[valid], n_changed[valid]))]
    best_idx = order[0]
    std_delta = delta[best_idx] / action_std
    return {
        "sample": sample,
        "true": true_cls, "wrong": yp[sample],
        "flip_rate": 100 * len(valid) / len(cands),
        "distance": l2[best_idx],
        "n_changed": int(n_changed[best_idx]),
        "std_delta": std_delta,
        "orig": query[ACTIONABLE_IDX],
        "new": cands[best_idx, ACTIONABLE_IDX],
        "p0": proba[sample, true_cls], "p1": p[best_idx, true_cls],
    }


results = [best_counterfactual(s) for s in mis_comp]
results = [r for r in results if r is not None]

# 干预矩阵（3 样本 × 8 特征）
mat = np.array([r["std_delta"] for r in results])
row_labels = [f"S{r['sample']}  {CLASS_LABELS[r['true']]}→{CLASS_LABELS[r['wrong']]}  (d={r['distance']:.2f})"
              for r in results]

# ---------------- 汇总热力图 ----------------
fig, ax = plt.subplots(figsize=(10.8, 3.6))
vmax = np.abs(mat).max()
im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(n_feat)); ax.set_xticklabels(FEAT_SHORT, fontsize=14)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=14)
for i in range(len(row_labels)):
    for j in range(n_feat):
        v = mat[i, j]
        if abs(v) < 1e-9:
            ax.text(j, i, "0", ha="center", va="center", fontsize=13, color="#B0B6BF")
        else:
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=13,
                    color="white" if abs(v) > vmax * 0.45 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
cb.set_label("Standardized intervention  Δ/σ", fontsize=14)
ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", length=0)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_summary_heatmap.png"), dpi=DPI, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

# ---------------- 汇总表 ----------------
rows = []
for r in results:
    rows.append({
        "样本": f"S{r['sample']}",
        "真实→误判": f"{CLASS_LABELS[r['true']]}→{CLASS_LABELS[r['wrong']]}",
        "翻转率(%)": round(r["flip_rate"], 1),
        "最小距离": round(r["distance"], 2),
        "干预特征数": r["n_changed"],
        "P(真实类)原始": round(r["p0"], 3),
        "P(真实类)反事实": round(r["p1"], 3),
        **{FEAT_SHORT[j]: round(r["std_delta"][j], 2) for j in range(n_feat)},
    })
sum_df = pd.DataFrame(rows)
sum_df.to_csv(os.path.join(HERE, "反事实汇总表.csv"), index=False, encoding="utf-8-sig")

print("===== 反事实汇总表 =====")
print(sum_df.to_string(index=False))
print("\n输出文件已保存到:", HERE)
