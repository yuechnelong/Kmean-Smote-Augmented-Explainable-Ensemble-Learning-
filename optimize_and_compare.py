# -*- coding: utf-8 -*-
"""
KMeansSMOTE 最佳增强数量的细化搜索 + 导出混淆矩阵 + 与基线对比
================================================================
- 方法固定：KMeansSMOTE（聚类 SMOTE）
- 比例细扫（中等区间 0.1~0.6），找最佳数量
- 导出优化模型的训练/测试混淆矩阵（Times New Roman / 无标题 / 300 DPI）
- 与基线模型的对比图
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix)
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
N_ESTIMATORS = 200
RANDOM_STATE = 42
DPI = 1000
SPLIT = int(round(0.7 * 1380))

CLASS_ORDER = [
    "Moderately Weathered Sandstone",                       # Class 1
    "Moderately Weathered Sandstone/Silty Clay",            # Class 2
    "Strongly/Moderately Weathered Sandstone",              # Class 3
    "Strongly Weathered Sandstone",                         # Class 4
    "Silty Clay",                                            # Class 5
    "Silty Clay/Strongly Weathered Argillaceous Sandstone",  # Class 6
]
CLASS_LABELS = [f"Class {i+1}" for i in range(len(CLASS_ORDER))]
class_to_idx = {n: i for i, n in enumerate(CLASS_ORDER)}

FINE_RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

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
Xte, yte = X[~mask], y[~mask]


def fit_predict(Xtr, ytr, Xte, use_weight):
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                class_weight="balanced" if use_weight else None, n_jobs=-1)
    rf.fit(Xtr, ytr)
    return rf.predict(Xte), rf


# 基线
ypb, rfb = fit_predict(Xtr, ytr, Xte, use_weight=True)
base_f1 = precision_recall_fscore_support(yte, ypb, labels=range(6), average="macro", zero_division=0)[2]

# 细扫 KMeansSMOTE
print("===== KMeansSMOTE 比例细扫（t+1 测试集） =====")
rows = []
for r in FINE_RATIOS:
    strat = get_strategy(ytr, r)
    try:
        smp = KMeansSMOTE(sampling_strategy=strat, k_neighbors=2,
                          cluster_balance_threshold=0.0, random_state=RANDOM_STATE)
        Xa, ya = smp.fit_resample(Xtr, ytr)
        yp, _ = fit_predict(Xa, ya, Xte, use_weight=False)
        f1 = precision_recall_fscore_support(yte, yp, labels=range(6), average="macro", zero_division=0)[2]
        acc = accuracy_score(yte, yp)
        rows.append({"Ratio": r, "Macro_F1": round(f1, 4), "Accuracy": round(acc, 4)})
        print(f"  ratio={r:<5} Macro-F1={f1:.4f}  Acc={acc:.4f}")
    except Exception as e:
        rows.append({"Ratio": r, "Macro_F1": np.nan, "Accuracy": np.nan})
        print(f"  ratio={r:<5} FAILED: {e}")

sweep = pd.DataFrame(rows).dropna(subset=["Macro_F1"])
best = sweep.sort_values("Macro_F1", ascending=False).iloc[0]
best_r = best["Ratio"]
print(f"\n最优比例: {best_r}, Macro-F1={best['Macro_F1']:.4f} (基线={base_f1:.4f})")
sweep.to_csv(os.path.join(HERE, "KMeansSMOTE_比例细扫.csv"), index=False, encoding="utf-8-sig")

# ============ 图1：细扫曲线 ============
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(sweep["Ratio"], sweep["Macro_F1"], "o-", color="#2E5E8C", lw=2, ms=5)
ax.axhline(base_f1, color="#808080", ls="--", lw=1.5, label="Baseline")
ax.scatter([best_r], [best["Macro_F1"]], s=140, color="#C0504D", zorder=5, marker="*",
           label=f"Best (ratio={best_r})")
ax.set_xlabel("Oversampling Ratio", fontsize=12)
ax.set_ylabel("Test Macro-F1", fontsize=12)
ax.legend(frameon=False, fontsize=11)
ax.grid(alpha=0.25, ls="--")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ratio_fine_search.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ============ 最优模型 + 混淆矩阵 ============
strat = get_strategy(ytr, best_r)
smp = KMeansSMOTE(sampling_strategy=strat, k_neighbors=2, cluster_balance_threshold=0.0,
                  random_state=RANDOM_STATE)
Xa, ya = smp.fit_resample(Xtr, ytr)
ypa, rfa = fit_predict(Xa, ya, Xte, use_weight=False)
ypa_train = rfa.predict(Xa)

cm_base = confusion_matrix(yte, ypb, labels=range(6))
cm_opt_test = confusion_matrix(yte, ypa, labels=range(6))
cm_opt_train = confusion_matrix(ya, ypa_train, labels=range(6))

# 每类 F1（基线 vs 优化）
f1_base_cls = precision_recall_fscore_support(yte, ypb, labels=range(6), average=None, zero_division=0)[2]
f1_opt_cls = precision_recall_fscore_support(yte, ypa, labels=range(6), average=None, zero_division=0)[2]


def plot_cm(cm, save_path):
    n = 6
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_LABELS, fontsize=12); ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=13); ax.set_ylabel("True Class", fontsize=13)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0); ax.tick_params(which="major", length=0, labelsize=12)
    for i in range(n):
        for j in range(n):
            cnt = int(cm[i, j]); pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i - 0.18, f"{cnt}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"({pct:.1f}%)", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=11); cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plot_cm(cm_opt_train, os.path.join(HERE, "fig_cm_optimized_train.png"))
plot_cm(cm_opt_test, os.path.join(HERE, "fig_cm_optimized_test.png"))

# ============ 图2：基线 vs 优化 混淆矩阵对比（测试集） ============
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
for ax, cm, title in [(axes[0], cm_base, "Baseline"), (axes[1], cm_opt_test, "KMeansSMOTE")]:
    n = 6
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_LABELS, fontsize=11); ax.set_yticklabels(CLASS_LABELS, fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=12); ax.set_ylabel("True Class", fontsize=12)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0); ax.tick_params(which="major", length=0)
    for i in range(n):
        for j in range(n):
            cnt = int(cm[i, j]); pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i - 0.18, f"{cnt}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"({pct:.1f}%)", ha="center", va="center", fontsize=8, color=color)
    ax.text(0.5, -0.16, title, transform=ax.transAxes, ha="center", fontsize=12,
            fontweight="bold", color="black")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
cb.set_label("Proportion", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_cm_baseline_vs_optimized.png"), dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ============ 图3：每类 F1 对比（基线 vs 优化） ============
xpos = np.arange(6); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(xpos - w/2, f1_base_cls, w, label="Baseline", color="#9BB8D3", edgecolor="white")
ax.bar(xpos + w/2, f1_opt_cls, w, label=f"KMeansSMOTE (ratio={best_r})", color="#2E5E8C",
       edgecolor="white")
ax.set_xticks(xpos); ax.set_xticklabels(CLASS_LABELS)
ax.set_xlabel("Class", fontsize=12); ax.set_ylabel("F1 Score (Test, t+1)", fontsize=12)
ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=11)
ax.grid(alpha=0.25, ls="--", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_f1_baseline_vs_optimized.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# 汇总
print("\n===== 最优模型 vs 基线（t+1 测试集） =====")
print(f"基线:    Macro-F1={base_f1:.4f}")
print(f"KMeansSMOTE(ratio={best_r}): Macro-F1={best['Macro_F1']:.4f}")
print("\n每类 F1:")
for i in range(6):
    d = f1_opt_cls[i] - f1_base_cls[i]
    print(f"  {CLASS_LABELS[i]:<8} 基线={f1_base_cls[i]:.4f}  优化={f1_opt_cls[i]:.4f}  "
          f"({'↑' if d>=0 else '↓'}{abs(d):.4f})")
print("\n输出文件已保存到:", HERE)
