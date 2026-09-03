# -*- coding: utf-8 -*-
"""
数据增强前后聚类对比图（t-SNE，同一投影空间）
================================================
- 左：增强前（原始训练样本）
- 右：增强后（原始样本淡显 + KMeansSMOTE 合成样本加粗显示）
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import KMeansSMOTE
warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data.xlsx")
RING_COL = "Ring No."
LABEL_COL = "Geological Condition"
WINDOW = 5
RANDOM_STATE = 42
DPI = 1500
SPLIT = int(round(0.7 * 1380))
BEST_RATIO = 0.2

CLASS_ORDER = [
    "Moderately Weathered Sandstone",                       # Class 1
    "Moderately Weathered Sandstone/Silty Clay",            # Class 2
    "Strongly/Moderately Weathered Sandstone",              # Class 3
    "Strongly Weathered Sandstone",                         # Class 4
    "Silty Clay",                                            # Class 5
    "Silty Clay/Strongly Weathered Argillaceous Sandstone",  # Class 6
]
CLASS_LABELS = ["MWS", "MWS/SC", "S/MWS", "SWS", "SC", "SC/SWAS"]
class_to_idx = {n: i for i, n in enumerate(CLASS_ORDER)}
PALETTE6 = ["#2E5E8C", "#C0504D", "#6B8E23", "#D4A017", "#7B5FA6", "#3B9C9C"]

df = pd.read_excel(DATA, sheet_name=0)
df.columns = [c.replace("\n", " ") for c in df.columns]
df = df.sort_values(RING_COL).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in (RING_COL, LABEL_COL) and "Grouting Pressure" not in c]
y_all = df[LABEL_COL].map(class_to_idx).values
X_feat = df[feat_cols].values.astype(np.float64)
N = len(df)

# 构造 t+1 滑窗
X, y, t = [], [], []
for i in range(WINDOW - 1, N - 1):
    X.append(X_feat[i - WINDOW + 1: i + 1].reshape(-1))
    y.append(y_all[i + 1])
    t.append(i + 1)
X, y, t = np.array(X), np.array(y), np.array(t)
mask = t < SPLIT
Xtr, ytr = X[mask], y[mask]

# KMeansSMOTE 增强（ratio=0.2）
counts = pd.Series(ytr).value_counts()
n_max = counts.max()
strat = {c: int(round(BEST_RATIO * n_max)) for c, n in counts.items()
         if int(round(BEST_RATIO * n_max)) > n}
smp = KMeansSMOTE(sampling_strategy=strat, k_neighbors=2, cluster_balance_threshold=0.0,
                  random_state=RANDOM_STATE)
Xa, ya = smp.fit_resample(Xtr, ytr)          # 原始在前、合成在后
X_syn, y_syn = Xa[len(Xtr):], ya[len(Xtr):]

# 标准化（真实训练集上拟合）
scaler = StandardScaler().fit(Xtr)
Zs = scaler.transform(Xtr)
Zsyn = scaler.transform(X_syn)

# 同一 t-SNE 投影（真实 + 合成合并拟合）
print("计算 t-SNE（真实 + 合成合并）...")
Zall = np.vstack([Zs, Zsyn])
perp = min(30, max(5, int(0.2 * len(Zall))))
tsne = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_STATE, init="pca",
            learning_rate="auto")
Z2 = tsne.fit_transform(Zall)
Z_real = Z2[:len(Zs)]
Z_syn = Z2[len(Zs):]

# ---------------- 绘图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

# 左：增强前
ax = axes[0]
for c in range(6):
    m = ytr == c
    ax.scatter(Z_real[m, 0], Z_real[m, 1], s=26, c=PALETTE6[c], alpha=0.75,
               edgecolors="white", linewidths=0.3, label=CLASS_LABELS[c])
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.03, 0.97, "Before augmentation", transform=ax.transAxes, va="top",
        fontsize=18, fontweight="bold", color="black")
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)

# 右：增强后（原始淡显 + 合成加粗）
ax = axes[1]
for c in range(6):
    m = ytr == c
    ax.scatter(Z_real[m, 0], Z_real[m, 1], s=26, c=PALETTE6[c], alpha=0.25,
               edgecolors="white", linewidths=0.3)
    ms = y_syn == c
    if ms.sum() > 0:
        ax.scatter(Z_syn[ms, 0], Z_syn[ms, 1], s=32, c=PALETTE6[c], alpha=0.9,
                   edgecolors="black", linewidths=0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.03, 0.97, "After KMeansSMOTE augmentation", transform=ax.transAxes, va="top",
        fontsize=18, fontweight="bold", color="black")
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)

handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE6[c], markersize=8,
                      label=CLASS_LABELS[c]) for c in range(6)]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=16,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout()
out = os.path.join(HERE, "fig_cluster_before_after.png")
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("已生成:", out)
print(f"增强前训练样本: {len(Xtr)}，增强后: {len(Xa)}（新增 {len(X_syn)} 个合成样本）")
print("各少数类新增样本数:", {CLASS_LABELS[c]: int((y_syn == c).sum())
                            for c in np.unique(y_syn)})
