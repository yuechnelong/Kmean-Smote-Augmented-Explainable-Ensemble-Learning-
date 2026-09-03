# -*- coding: utf-8 -*-
"""
聚类可视化 + 生成样本定量评估
==============================
- t-SNE 可视化：6 类地层整体结构；少数复合地层 + SMOTE/KMeansSMOTE 合成样本对比
- 定量评估：真实数据聚类质量（Silhouette / DB / CH）
            合成样本质量（保真度 / 污染率 / 多样性，对比 RandomOver/SMOTE/KMeansSMOTE）
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, pairwise_distances)
from imblearn.over_sampling import SMOTE, KMeansSMOTE, RandomOverSampler
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
# 少数复合地层（增强重点）
MINOR_CLASSES = [1, 5]   # Class 2、Class 6

# ---------------- 数据与窗口 ----------------
df = pd.read_excel(DATA, sheet_name=0)
df.columns = [c.replace("\n", " ") for c in df.columns]
df = df.sort_values(RING_COL).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in (RING_COL, LABEL_COL) and "Grouting Pressure" not in c]
y_all = df[LABEL_COL].map(class_to_idx).values
X_feat = df[feat_cols].values.astype(np.float64)
N = len(df)


def build_dataset(horizon=1):
    X, y, t = [], [], []
    for i in range(WINDOW - 1, N - horizon):
        X.append(X_feat[i - WINDOW + 1: i + 1].reshape(-1))
        y.append(y_all[i + horizon])
        t.append(i + horizon)
    return np.array(X), np.array(y), np.array(t)


def get_strategy(y, r):
    counts = pd.Series(y).value_counts()
    n_max = counts.max()
    return {c: int(round(r * n_max)) for c, n in counts.items() if int(round(r * n_max)) > n}


X, y, t = build_dataset(1)
mask = t < SPLIT
Xtr, ytr = X[mask], y[mask]

# 标准化（在真实训练集上拟合）
scaler = StandardScaler().fit(Xtr)
Xs = scaler.transform(Xtr)

# ---------------- 合成样本生成（ratio=0.5，少数类） ----------------
strat = get_strategy(ytr, 0.5)
samplers = {
    "RandomOver": RandomOverSampler(sampling_strategy=strat, random_state=RANDOM_STATE),
    "SMOTE": SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=RANDOM_STATE),
    "KMeansSMOTE": KMeansSMOTE(sampling_strategy=strat, k_neighbors=2,
                               cluster_balance_threshold=0.0, random_state=RANDOM_STATE),
}
syn = {}
for name, smp in samplers.items():
    Xa, ya = smp.fit_resample(Xtr, ytr)
    # 只保留少数类（Class 2、6）的合成样本
    keep = np.isin(ya, MINOR_CLASSES)
    syn[name] = (scaler.transform(Xa[keep]), ya[keep])

# ---------------- 图1：6 类整体 t-SNE ----------------
print("计算 t-SNE（整体）...")
tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
Z = tsne.fit_transform(Xs)

fig, ax = plt.subplots(figsize=(7, 5.6))
for c in range(6):
    m = ytr == c
    # 少数类画大一点、多数类画小一点
    s = 60 if c in MINOR_CLASSES else 28
    ax.scatter(Z[m, 0], Z[m, 1], s=s, c=PALETTE6[c], alpha=0.75, edgecolors="white",
               linewidths=0.4, label=CLASS_LABELS[c])
ax.set_xlabel("t-SNE 1", fontsize=18)
ax.set_ylabel("t-SNE 2", fontsize=18)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(frameon=False, fontsize=16, loc="best", markerscale=1.1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_tsne_all_classes.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 图2：少数类 + 合成样本 t-SNE ----------------
# 取少数类真实样本 + 各方法合成样本，共同降维
real_minor = np.isin(ytr, MINOR_CLASSES)
Xm = np.vstack([Xs[real_minor]] + [syn[n][0] for n in samplers])
labels_m = np.hstack([ytr[real_minor]] + [syn[n][1] for n in samplers])
src = np.hstack([["real"] * real_minor.sum()]
                + [[n] * len(syn[n][0]) for n in samplers])

print("计算 t-SNE（少数类+合成）...")
perp = min(30, max(5, int(0.2 * len(Xm))))
tsne2 = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
Z2 = tsne2.fit_transform(Xm)

fig, ax = plt.subplots(figsize=(7.6, 5.8))
marker_map = {"real": ("o", None), "SMOTE": ("^", "SMOTE"), "KMeansSMOTE": ("D", "KMeansSMOTE")}
for c in MINOR_CLASSES:
    for nm, (mk, _) in marker_map.items():
        m = (labels_m == c) & (src == nm)
        if nm == "real":
            ax.scatter(Z2[m, 0], Z2[m, 1], s=70, c=PALETTE6[c], alpha=0.85,
                       edgecolors="white", linewidths=0.5, marker=mk,
                       label=f"{CLASS_LABELS[c]} (real)")
        else:
            ax.scatter(Z2[m, 0], Z2[m, 1], s=60, facecolors="none", edgecolors=PALETTE6[c],
                       linewidths=1.3, marker=mk, alpha=0.85, label=f"{CLASS_LABELS[c]} ({nm})")
ax.set_xlabel("t-SNE 1", fontsize=18)
ax.set_ylabel("t-SNE 2", fontsize=18)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(frameon=False, fontsize=15, loc="best", markerscale=1.0, ncol=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_tsne_minority_synthetic.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 定量评估 ----------------
def quality(X_real, y_real, X_syn, y_syn):
    """返回 (保真度 fidelity↓, 污染率 contamination↓, 多样性 diversity)。"""
    nn_all = NearestNeighbors(n_neighbors=1).fit(X_real)
    dist_all, idx_all = nn_all.kneighbors(X_syn)
    nearest_class = y_real[idx_all.ravel()]
    contamination = float(np.mean(nearest_class != y_syn))

    same_dists = []
    for c in np.unique(y_syn):
        Xr = X_real[y_real == c]; Xs = X_syn[y_syn == c]
        if len(Xr) == 0 or len(Xs) == 0:
            continue
        d, _ = NearestNeighbors(n_neighbors=1).fit(Xr).kneighbors(Xs)
        same_dists.extend(d.ravel())
    fidelity = float(np.mean(same_dists))

    divs = []
    for c in np.unique(y_syn):
        Xs = X_syn[y_syn == c]
        if len(Xs) > 1:
            D = pairwise_distances(Xs)
            iu = np.triu_indices(len(Xs), k=1)
            divs.extend(D[iu])
    diversity = float(np.mean(divs)) if divs else 0.0
    return fidelity, contamination, diversity

# 真实数据聚类质量（6 类）
print("计算聚类质量指标...")
sil = silhouette_score(Xs, ytr, sample_size=min(2000, len(Xs)), random_state=RANDOM_STATE)
db = davies_bouldin_score(Xs, ytr)
ch = calinski_harabasz_score(Xs, ytr)

rows = []
for name in samplers:
    Xs_syn, ys_syn = syn[name]
    f, c, d = quality(Xs, ytr, Xs_syn, ys_syn)
    rows.append({"Method": name, "Fidelity(↓)": round(f, 4),
                 "Contamination(↓)": round(c, 4), "Diversity": round(d, 4)})
    print(f"{name:<12} 保真度={f:.4f}  污染率={c:.4f}  多样性={d:.4f}")

qual_df = pd.DataFrame(rows)
qual_df.to_csv(os.path.join(HERE, "合成样本定量评估.csv"), index=False, encoding="utf-8-sig")
print(f"\n真实数据聚类质量: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}, "
      f"Calinski-Harabasz={ch:.2f}")

# ---------------- 图3：合成样本质量对比（条形图） ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
met = [("Fidelity(↓)", "Fidelity (lower is better)"),
       ("Contamination(↓)", "Contamination rate (lower is better)"),
       ("Diversity", "Diversity (higher is more varied)")]
for i, (col, ylab) in enumerate(met):
    ax = axes[i]
    vals = qual_df[col].values
    cols = ["#9BB8D3" if col.endswith("(↓)") else "#2E5E8C"] * len(vals)
    ax.bar(qual_df["Method"], vals, color=["#9BB8D3", "#6B8E23", "#2E5E8C"], edgecolor="white")
    for j, v in enumerate(vals):
        ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=16)
    ax.set_ylabel(ylab, fontsize=16)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(alpha=0.25, ls="--", axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_synthetic_quality_metrics.png"), dpi=DPI, facecolor="white")
plt.close(fig)

print("\n输出文件已保存到:", HERE)
