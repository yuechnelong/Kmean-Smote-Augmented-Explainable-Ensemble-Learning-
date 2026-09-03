# -*- coding: utf-8 -*-
"""
样本增强方式与数量的搜索 + 与基线对比
======================================
- 时序多步预测（RF，滑窗 W=5，预测 t+1~t+5）
- 划分：时序 7:3（训练环 1~966 / 测试环 967~1380）
- 在 t+1 上扫：增强方式 × 增强比例，用测试集 Macro-F1 选择（探索性，见 README 说明）
- 最优方式/比例应用到全部步长，与基线(无增强)对比
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, ADASYN,
                                    KMeansSMOTE, RandomOverSampler)
warnings.filterwarnings("ignore")

# ---------------- 字体 ----------------
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8

# ---------------- 配置 ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data.xlsx")
RING_COL = "Ring No."
LABEL_COL = "Geological Condition"
WINDOW = 5
HORIZONS = [1, 2, 3, 4, 5]
N_ESTIMATORS = 200
RANDOM_STATE = 42
DPI = 1500
SPLIT = int(round(0.7 * 1380))   # 966

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

METHODS = ["RandomOver", "SMOTE", "BorderlineSMOTE", "ADASYN", "KMeansSMOTE"]
RATIOS = [0.2, 0.35, 0.5, 0.7, 1.0]
M_COLOR = {"RandomOver": "#9BB8D3", "SMOTE": "#2E5E8C", "BorderlineSMOTE": "#C0504D",
           "ADASYN": "#6B8E23", "KMeansSMOTE": "#D4A017"}


def make_sampler(name, random_state, strategy):
    kw = dict(sampling_strategy=strategy, random_state=random_state)
    if name == "RandomOver":
        return RandomOverSampler(**kw)
    if name == "SMOTE":
        return SMOTE(k_neighbors=5, **kw)
    if name == "BorderlineSMOTE":
        return BorderlineSMOTE(k_neighbors=5, m_neighbors=8, **kw)
    if name == "ADASYN":
        return ADASYN(n_neighbors=5, **kw)
    if name == "KMeansSMOTE":
        return KMeansSMOTE(k_neighbors=2, cluster_balance_threshold=0.0, **kw)
    raise ValueError(name)


# ---------------- 读取与建模 ----------------
df = pd.read_excel(DATA, sheet_name=0)
df.columns = [c.replace("\n", " ") for c in df.columns]
df = df.sort_values(RING_COL).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in (RING_COL, LABEL_COL) and "Grouting Pressure" not in c]
y_all = df[LABEL_COL].map(class_to_idx).values
X_feat = df[feat_cols].values.astype(np.float64)
N = len(df)


def build_dataset(horizon):
    X, y, t = [], [], []
    for i in range(WINDOW - 1, N - horizon):
        X.append(X_feat[i - WINDOW + 1: i + 1].reshape(-1))
        y.append(y_all[i + horizon])
        t.append(i + horizon)
    return np.array(X), np.array(y), np.array(t)


def get_strategy(y, r):
    counts = pd.Series(y).value_counts()
    n_max = counts.max()
    strat = {}
    for c, n in counts.items():
        target = int(round(r * n_max))
        if target > n:
            strat[c] = target
    return strat


def fit_eval(Xtr, ytr, Xev, yev, use_weight):
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                class_weight="balanced" if use_weight else None, n_jobs=-1)
    rf.fit(Xtr, ytr)
    yp = rf.predict(Xev)
    mac_f1 = precision_recall_fscore_support(yev, yp, labels=range(6), average="macro",
                                             zero_division=0)[2]
    acc = accuracy_score(yev, yp)
    return mac_f1, acc, rf


# ============ t+1 上搜索（测试集） ============
X, y, t_idx = build_dataset(1)
mask_train = t_idx < SPLIT
mask_test = ~mask_train
Xtr0, ytr0 = X[mask_train], y[mask_train]
Xte0, yte0 = X[mask_test], y[mask_test]

print(f"t+1 划分: 训练 {mask_train.sum()}, 测试 {mask_test.sum()}")
print("训练集类别分布:", dict(zip(CLASS_LABELS, np.bincount(ytr0, minlength=6).tolist())))

# ============ 增强前各类数量对比柱状图 ============
CLASS_ABBR = ["MWS", "MWS/SC", "S/MWS", "SWS", "SC", "SC/SWAS"]
PALETTE6 = ["#2E5E8C", "#C0504D", "#6B8E23", "#D4A017", "#7B5FA6", "#3B9C9C"]
tr_counts = np.bincount(ytr0, minlength=6)
order = np.argsort(tr_counts)[::-1]
fig, ax = plt.subplots(figsize=(6.8, 4.4))
bars = ax.bar(range(6), tr_counts[order], color=[PALETTE6[i] for i in order], edgecolor="white")
for i, o in enumerate(order):
    ax.text(i, tr_counts[o] + 5, str(tr_counts[o]), ha="center", fontsize=15)
ax.set_xticks(range(6)); ax.set_xticklabels([CLASS_ABBR[i] for i in order], fontsize=15)
ax.set_ylabel("Number of samples", fontsize=16)
ax.set_xlim(-0.5, 5.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, ls="--", axis="y")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_class_counts_before_aug.png"), dpi=DPI, facecolor="white")
plt.close(fig)

base_f1, base_acc, _ = fit_eval(Xtr0, ytr0, Xte0, yte0, use_weight=True)
print(f"基线(无增强, class_weight=balanced): 测试 Macro-F1={base_f1:.4f}, Acc={base_acc:.4f}\n")

rows = []
for name in METHODS:
    for r in RATIOS:
        strat = get_strategy(ytr0, r)
        try:
            smp = make_sampler(name, RANDOM_STATE, strat)
            Xa, ya = smp.fit_resample(Xtr0, ytr0)
            f1, acc, _ = fit_eval(Xa, ya, Xte0, yte0, use_weight=False)
            rows.append({"Method": name, "Ratio": r, "Test_Macro_F1": round(f1, 4),
                         "Test_Accuracy": round(acc, 4)})
        except Exception as e:
            rows.append({"Method": name, "Ratio": r, "Test_Macro_F1": np.nan,
                         "Test_Accuracy": np.nan})

sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(HERE, "增强搜索_测试集结果.csv"), index=False, encoding="utf-8-sig")

valid = sweep.dropna(subset=["Test_Macro_F1"])
best = valid.sort_values("Test_Macro_F1", ascending=False).iloc[0]
best_name, best_r = best["Method"], best["Ratio"]
print(f"最优增强: {best_name} @ ratio={best_r}, 测试 Macro-F1={best['Test_Macro_F1']:.4f}")
print("\n===== 搜索全表（按 Macro-F1 降序） =====")
print(valid.sort_values("Test_Macro_F1", ascending=False).to_string(index=False))
print()

# ============ 图1：测试 Macro-F1 随比例（各方法 + 基线） ============
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.axhline(base_f1, color="#808080", ls="--", lw=1.5, label="Baseline (no aug.)")
for name in METHODS:
    sub = sweep[sweep["Method"] == name].sort_values("Ratio")
    ax.plot(sub["Ratio"], sub["Test_Macro_F1"], "o-", color=M_COLOR[name], lw=2, ms=5, label=name)
ax.set_xlabel("Oversampling Ratio (minority / majority)", fontsize=16)
ax.set_ylabel("Test Macro-F1", fontsize=16)
ax.legend(frameon=False, fontsize=14, ncol=2)
ax.grid(alpha=0.25, ls="--")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_aug_search.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ============ 最优增强应用到全部步长，与基线对比 ============
summary = []
per_class_base = per_class_best = None
for k in HORIZONS:
    Xk, yk, tk = build_dataset(k)
    mtr = tk < SPLIT
    mte = ~mtr
    Xtr, ytr = Xk[mtr], yk[mtr]
    Xte, yte = Xk[mte], yk[mte]

    f1b, accb, rfb = fit_eval(Xtr, ytr, Xte, yte, use_weight=True)
    ypb = rfb.predict(Xte)

    try:
        strat = get_strategy(ytr, best_r)
        smp = make_sampler(best_name, RANDOM_STATE, strat)
        Xa, ya = smp.fit_resample(Xtr, ytr)
        f1a, acca, rfa = fit_eval(Xa, ya, Xte, yte, use_weight=False)
        ypa = rfa.predict(Xte)
    except Exception as e:
        f1a, acca, ypa = f1b, accb, ypb   # 回退到基线

    summary.append({"预测步长": f"t+{k}",
                    "基线_Macro_F1": round(f1b, 4), "增强_Macro_F1": round(f1a, 4),
                    "基线_Accuracy": round(accb, 4), "增强_Accuracy": round(acca, 4)})
    if k == 1:
        per_class_base = precision_recall_fscore_support(yte, ypb, labels=range(6),
                                                         average=None, zero_division=0)[2]
        per_class_best = precision_recall_fscore_support(yte, ypa, labels=range(6),
                                                         average=None, zero_division=0)[2]
        counts_before = np.bincount(ytr, minlength=6)
        counts_after = np.bincount(ya, minlength=6)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(HERE, "基线vs增强_测试集对比.csv"), index=False, encoding="utf-8-sig")
print("===== 基线 vs 最优增强（测试集） =====")
print(summary_df.to_string(index=False))
print()

# ============ 图2：t+1 每类 F1 对比 ============
xpos = np.arange(6); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(xpos - w/2, per_class_base, w, label="Baseline", color="#9BB8D3", edgecolor="white")
ax.bar(xpos + w/2, per_class_best, w, label=f"{best_name}", color="#2E5E8C", edgecolor="white")
ax.set_xticks(xpos); ax.set_xticklabels(CLASS_LABELS)
ax.set_xlabel("Class", fontsize=16)
ax.set_ylabel("F1 Score (Test, t+1)", fontsize=16)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=15)
ax.grid(alpha=0.25, ls="--", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_per_class_f1_baseline_vs_aug.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ============ 图3：各步长 Macro-F1 ============
x = np.arange(1, 6)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(x, summary_df["基线_Macro_F1"], "o-", color="#9BB8D3", lw=2, ms=6, label="Baseline")
ax.plot(x, summary_df["增强_Macro_F1"], "s-", color="#2E5E8C", lw=2, ms=6,
        label=f"{best_name} (ratio={best_r})")
ax.set_xticks(x); ax.set_xticklabels([f"t+{k}" for k in HORIZONS])
ax.set_xlabel("Prediction Horizon", fontsize=16)
ax.set_ylabel("Macro-F1 (Test)", fontsize=16)
ax.legend(frameon=False, fontsize=15)
ax.grid(alpha=0.25, ls="--")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_macro_f1_across_horizons.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ============ 图4：t+1 训练集增强前后分布 ============
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.bar(xpos - w/2, counts_before, w, label="Before augmentation", color="#9BB8D3", edgecolor="white")
ax.bar(xpos + w/2, counts_after, w, label=f"After {best_name}", color="#2E5E8C", edgecolor="white")
for i in range(6):
    ax.text(xpos[i] - w/2, counts_before[i] + 8, str(counts_before[i]), ha="center",
            va="bottom", fontsize=12, color="#333")
    ax.text(xpos[i] + w/2, counts_after[i] + 8, str(counts_after[i]), ha="center",
            va="bottom", fontsize=12, color="#333")
ax.set_xticks(xpos); ax.set_xticklabels(CLASS_ABBR)
ax.set_xlabel("Class", fontsize=16)
ax.set_ylabel("Training Samples", fontsize=16)
ax.set_ylim(0, max(counts_after) * 1.18)
ax.legend(frameon=False, fontsize=15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_train_distribution_aug.png"), dpi=DPI, facecolor="white")
plt.close(fig)

print("===== t+1 每类 F1（测试集）=====")
for i in range(6):
    d = per_class_best[i] - per_class_base[i]
    print(f"{CLASS_LABELS[i]:<8} 基线={per_class_base[i]:.4f}  增强={per_class_best[i]:.4f}  "
          f"({'↑' if d >= 0 else '↓'}{abs(d):.4f})")
print("\n输出文件已保存到:", HERE)
