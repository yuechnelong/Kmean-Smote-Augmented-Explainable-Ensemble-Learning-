# -*- coding: utf-8 -*-
"""
反事实分析（参考因果约束反事实框架，适配 6 类地层识别）
========================================================
- 对全部错分复合地层样本（Class 2/6）逐一分析
- 不可变特征：32 个历史时滞（t-4 ~ t-1）
- 可操作特征：当前环 8 个掘进参数（t-0）
- 每个样本输出：权衡散点（密度热力）、特征变化频率、最优反事实对比
"""
import os
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
DPI = 1500
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
BLUE, RED, GREEN, GOLD, GRAY = "#2E5E8C", "#C0504D", "#4C8B5B", "#D4A017", "#B0B6BF"

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

# 序列标准化 + CNN-BiLSTM（优化后超参 hidden=34, lr=0.00797, batch=32）
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
print(f"错分复合地层样本: {mis_comp}")
for i in mis_comp:
    print(f"  样本{i}: 真实={CLASS_LABELS[yte[i]]}, 误判={CLASS_LABELS[yp[i]]}, "
          f"误判概率={proba[i, yp[i]]:.3f}")


def build_candidates(query, donors, max_changes=3):
    cands, metas = [], []
    for d, donor in enumerate(donors):
        c = query.copy(); c[ACTIONABLE_IDX] = donor[ACTIONABLE_IDX]
        cands.append(c); metas.append(("full", d, list(ACTIONABLE_IDX)))
        for nc in range(1, max_changes + 1):
            for sub in combinations(ACTIONABLE_IDX, nc):
                c = query.copy(); c[list(sub)] = donor[list(sub)]
                cands.append(c); metas.append(("sparse", d, list(sub)))
    return np.array(cands), metas


def analyze_sample(sample):
    query = Xte[sample]
    true_cls = yte[sample]
    wrong_cls = yp[sample]

    donor_idx = np.where(ytr == true_cls)[0]
    rng = np.random.RandomState(SEED)
    max_donors = min(150, len(donor_idx))
    donors = Xtr[rng.choice(donor_idx, size=max_donors, replace=False)]

    cands, metas = build_candidates(query, donors)
    p = predict_flat(cands)
    p_true = p[:, true_cls]
    flipped = p.argmax(axis=1) == true_cls
    other_max = np.max(np.delete(p, true_cls, axis=1), axis=1)
    margin = p_true - other_max

    delta = cands[:, ACTIONABLE_IDX] - query[ACTIONABLE_IDX]
    l2 = np.sqrt(((delta / action_std) ** 2).sum(axis=1))
    n_changed = (np.abs(delta) > 1e-8).sum(axis=1)
    valid = np.where(flipped)[0]

    if len(valid) == 0:
        print(f"  样本{sample} 无有效反事实，跳过")
        return

    order = valid[np.lexsort((l2[valid], n_changed[valid]))]
    best_idx = order[0]
    top_n = min(30, len(order))
    orig_margin = proba[sample, true_cls] - np.max(np.delete(proba[sample], true_cls))

    tag = f"s{sample}_{true_cls+1}{wrong_cls+1}"

    # 图1 权衡散点（密度热力）
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    ax.axhspan(0, margin.max() * 1.15, color="#4C8B5B", alpha=0.05, zorder=0)
    ax.axhline(0, color="#333", lw=1.3, zorder=1)
    ax.text(0.02, 0.97, "flipped region", transform=ax.transAxes, fontsize=9,
            color="#4C8B5B", style="italic", va="top", ha="left")
    hb = ax.hexbin(l2, margin, gridsize=32, cmap="Blues", mincnt=1, zorder=2,
                   linewidths=0.2, norm=LogNorm())
    ax.scatter([l2[best_idx]], [margin[best_idx]], s=230, marker="*", c=RED,
               edgecolors="black", linewidths=0.8, zorder=6, label="Selected counterfactual")
    ax.scatter([0], [orig_margin], s=170, marker="D", c=GOLD, edgecolors="black",
               linewidths=0.8, zorder=6, label="Original sample")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Number of candidates", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("Standardized L2 distance (actionable features)", fontsize=11)
    ax.set_ylabel(f"Decision margin:  P({CLASS_LABELS[true_cls]}) − max P(other)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=2, markerscale=1.1, handletextpad=0.4, columnspacing=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"fig1_tradeoff_{tag}.png"), dpi=DPI, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)

    # 图2 特征变化频率
    top_cands = order[:top_n]
    freq = (np.abs(cands[top_cands][:, ACTIONABLE_IDX] - query[ACTIONABLE_IDX]) > 1e-8).mean(axis=0) * 100
    freq_df = pd.DataFrame({"Feature": FEAT_SHORT, "Frequency": freq})
    freq_df = freq_df[freq_df["Frequency"] > 0].sort_values("Frequency")
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.45 * len(freq_df))))
    bars = ax.barh(freq_df["Feature"], freq_df["Frequency"], height=0.5,
                   color="#5B8DB8", edgecolor="#333", linewidth=0.7, zorder=3)
    for b, v in zip(bars, freq_df["Frequency"]):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=9, color="#333")
    ax.set_xlabel(f"Change frequency among top {top_n} counterfactuals (%)", fontsize=11)
    ax.set_xlim(0, 112)
    ax.axvline(50, color="#999", ls=(0, (3, 3)), lw=0.7, alpha=0.65, zorder=1)
    ax.grid(axis="x", ls=(0, (4, 3)), lw=0.5, color="#C7C7C7", alpha=0.6, zorder=0)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.08)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"fig2_frequency_{tag}.png"), dpi=DPI, facecolor="white")
    plt.close(fig)

    # 图3 最优反事实对比
    best = cands[best_idx]
    delta_best = best[ACTIONABLE_IDX] - query[ACTIONABLE_IDX]
    std_delta = delta_best / action_std
    orig = query[ACTIONABLE_IDX]; new = best[ACTIONABLE_IDX]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ypos = np.arange(n_feat)
    colors = [RED if d < -1e-9 else BLUE if d > 1e-9 else GRAY for d in delta_best]
    ax.barh(ypos, std_delta, color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.axvline(0, color="#444", lw=0.9, zorder=2)
    for i, d in enumerate(delta_best):
        if abs(d) > 1e-9:
            sign = "+" if d > 0 else ""
            lab = f"{orig[i]:.1f}→{new[i]:.1f}"
            xpos = std_delta[i] + (0.12 if std_delta[i] >= 0 else -0.12)
            ha = "left" if std_delta[i] >= 0 else "right"
            ax.text(xpos, i, lab, va="center", ha=ha, fontsize=8.5, color="#333")
    ax.set_yticks(ypos); ax.set_yticklabels(FEAT_SHORT, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized intervention  Δ / σ", fontsize=11)
    ax.set_xlim(min(std_delta) - 1.6, max(std_delta) + 2.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, ls="--", axis="x", lw=0.6)
    ax.text(0.98, 0.06,
            f"P({CLASS_LABELS[true_cls]}):  {proba[sample, true_cls]:.3f}  →  {p_true[best_idx]:.3f}\n"
            f"Prediction:  {CLASS_LABELS[wrong_cls]}  →  {CLASS_LABELS[true_cls]}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F5F7FA", ec="#C9CFD8", lw=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"fig3_best_{tag}.png"), dpi=DPI, facecolor="white")
    plt.close(fig)

    # 汇总
    print(f"\n===== 样本{sample}（{CLASS_LABELS[true_cls]} → {CLASS_LABELS[wrong_cls]}） =====")
    print(f"  候选 {len(cands)}，有效翻转 {len(valid)}（{100*len(valid)/len(cands):.1f}%）")
    print(f"  最优反事实: 改 {int(n_changed[best_idx])} 个特征, 距离={l2[best_idx]:.2f}")
    print(f"  P({CLASS_LABELS[true_cls]}): {proba[sample, true_cls]:.3f} → {p_true[best_idx]:.3f}")
    for i in np.argsort(-np.abs(std_delta)):
        if abs(delta_best[i]) > 1e-9:
            print(f"    {FEAT_SHORT[i]:<18} Δ/σ={std_delta[i]:+.2f}  ({orig[i]:.1f}→{new[i]:.1f})")
    freq_txt = ", ".join(f"{r['Feature']}:{r['Frequency']:.0f}%" for _, r in freq_df.iterrows())
    print(f"  特征变化频率: {freq_txt}")


for sample in mis_comp:
    analyze_sample(sample)

print("\n输出文件已保存到:", HERE)
