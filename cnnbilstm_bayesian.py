# -*- coding: utf-8 -*-
"""
CNN-BiLSTM 贝叶斯超参数优化（t+1，k 折交叉验证）+ 全部图表导出
================================================================
- 特征：7 个（删 GP），5×7 序列
- 增强：KMeansSMOTE @ 0.15（每折训练集内）
- 优化：optuna TPE，3 折分层交叉验证，目标 = 平均 Macro-F1
- 导出：收敛曲线、默认vs优化、参数重要性、搜索散点、混淆矩阵、指标表
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from imblearn.over_sampling import KMeansSMOTE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
N_TRIALS = 30
N_FOLDS = 3
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

# 序列标准化
sc = StandardScaler().fit(Xs_tr.reshape(-1, n_feat))
Xs_tr_s = sc.transform(Xs_tr.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
Xs_te_s = sc.transform(Xs_te.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)

# 增强（扁平特征）
def augment(Xf, yf):
    counts = pd.Series(yf).value_counts()
    target = int(round(RATIO * counts.max()))
    strat = {c: target for c in counts.index if counts[c] < target}
    smp = KMeansSMOTE(sampling_strategy=strat, k_neighbors=2,
                      cluster_balance_threshold=0.0, random_state=SEED)
    return smp.fit_resample(Xf, yf)


def macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, labels=range(6),
                                           average="macro", zero_division=0)[2]


class CNNBiLSTM(nn.Module):
    def __init__(self, in_f, hidden, n_cls=6):
        super().__init__()
        self.conv = nn.Conv1d(in_f, hidden, 3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_cls)

    def forward(self, x):
        x = x.permute(0, 2, 1); x = torch.relu(self.conv(x)); x = x.permute(0, 2, 1)
        x, _ = self.lstm(x); x = x.mean(dim=1); return self.fc(x)


def train_cnnbilstm(Xtr, ytr, Xte, hidden, lr, batch_size, epochs=EPOCHS):
    model = CNNBiLSTM(Xtr.shape[2], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()


# k 折 CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(Xf_tr, ytr))


def cv_score(hidden, lr, batch_size):
    scores = []
    for train_idx, val_idx in folds:
        Xa, ya = augment(Xf_tr[train_idx], ytr[train_idx])
        Xa_seq = sc.transform(Xa.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
        Xv_seq = Xs_tr_s[val_idx]
        yp = train_cnnbilstm(Xa_seq, ya, Xv_seq, hidden, lr, batch_size)
        scores.append(macro_f1(ytr[val_idx], yp))
    return float(np.mean(scores))


def objective(trial):
    hidden = trial.suggest_int("hidden", 16, 64)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    return cv_score(hidden, lr, batch_size)


# ---------------- 贝叶斯优化 ----------------
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"优化完成，耗时 {time.time()-t0:.1f}s，{len(study.trials)} 次试验 × {N_FOLDS} 折")
print(f"最优 CV Macro-F1 = {study.best_value:.4f}")
print("最优超参数:")
for k, v in study.best_params.items():
    print(f"  {k} = {v}")

hist = pd.DataFrame([{**t.params, "cv_Macro_F1": t.value} for t in study.trials])
hist.to_csv(os.path.join(HERE, "优化试验历史.csv"), index=False, encoding="utf-8-sig")

# ---------------- 默认 vs 优化（测试集） ----------------
Xa_full, ya_full = augment(Xf_tr, ytr)
Xa_seq_full = sc.transform(Xa_full.reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)


def final_eval(hidden, lr, batch_size, name):
    yp = train_cnnbilstm(Xa_seq_full, ya_full, Xs_te_s, hidden, lr, batch_size, epochs=EPOCHS)
    return {"Model": name,
            "Accuracy": accuracy_score(yte, yp),
            "Balanced_Acc": balanced_accuracy_score(yte, yp),
            "Macro_F1": macro_f1(yte, yp)}


rows = [final_eval(32, 1e-3, 64, "Default CNN-BiLSTM"),
        final_eval(study.best_params["hidden"], study.best_params["lr"],
                   study.best_params["batch_size"], "Optimized CNN-BiLSTM")]
res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "默认vs优化_结果.csv"), index=False, encoding="utf-8-sig")
print("\n===== 默认 vs 优化（测试集） =====")
print(res.round(4).to_string(index=False))

# ---------------- 图1：收敛曲线 ----------------
best_so_far = np.maximum.accumulate([t.value for t in study.trials])
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(range(1, len(best_so_far) + 1), [t.value for t in study.trials], "o",
        color="#9BB8D3", ms=4, alpha=0.6, label="Trial value")
ax.plot(range(1, len(best_so_far) + 1), best_so_far, "-", color="#2E5E8C", lw=2,
        label="Best so far")
ax.set_xlabel("Trial", fontsize=16)
ax.set_ylabel("Cross-validation Macro-F1", fontsize=16)
ax.legend(frameon=False, fontsize=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, ls="--")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_opt_history.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 图2：默认 vs 优化 ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
xpos = np.arange(2); w = 0.3
for j, met in enumerate(["Accuracy", "Macro_F1"]):
    vals = res[met].values
    ax.bar(xpos + (j - 0.5) * w, vals, w, label=met, color=["#9BB8D3", "#2E5E8C"][j],
           edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(xpos[i] + (j - 0.5) * w, v + 0.004, f"{v:.4f}", ha="center", fontsize=12)
ax.set_xticks(xpos); ax.set_xticklabels(res["Model"], fontsize=14)
ax.set_ylabel("Score", fontsize=16); ax.set_ylim(0.88, 1.0)
ax.legend(frameon=False, fontsize=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, ls="--", axis="y")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig2_default_vs_optimized.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 图3：参数重要性 ----------------
try:
    imp = optuna.importance.get_param_importances(study)
    names = list(imp.keys()); vals = list(imp.values())
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh([names[i] for i in order], [vals[i] for i in order], color="#2E5E8C", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=14)
    ax.tick_params(labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, ls="--", axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_param_importance.png"), dpi=DPI, facecolor="white")
    plt.close(fig)
except Exception as e:
    print("参数重要性计算失败:", e)

# ---------------- 图4：搜索散点 ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
pairs = [("hidden", "cv_Macro_F1"), ("lr", "cv_Macro_F1"), ("batch_size", "cv_Macro_F1")]
for ax, (px, py) in zip(axes, pairs):
    ax.scatter(hist[px], hist[py], s=28, c="#2E5E8C", alpha=0.6, edgecolors="white")
    ax.set_xlabel(px, fontsize=14)
    ax.set_ylabel("CV Macro-F1", fontsize=14)
    ax.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, ls="--")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig4_search_scatter.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 最优模型：混淆矩阵 + 逐类指标 ----------------
yp_opt = train_cnnbilstm(Xa_seq_full, ya_full, Xs_te_s,
                         study.best_params["hidden"], study.best_params["lr"],
                         study.best_params["batch_size"], epochs=EPOCHS)
yp_opt_tr = train_cnnbilstm(Xa_seq_full, ya_full, Xs_tr_s,
                            study.best_params["hidden"], study.best_params["lr"],
                            study.best_params["batch_size"], epochs=EPOCHS)

cm_train = confusion_matrix(ytr, yp_opt_tr, labels=range(6))
cm_test = confusion_matrix(yte, yp_opt, labels=range(6))


def plot_cm(cm, path):
    n = 6
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_ABBR, fontsize=16, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_ABBR, fontsize=16)
    ax.set_xlabel("Predicted Class", fontsize=17); ax.set_ylabel("True Class", fontsize=17)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0); ax.tick_params(which="major", length=0, labelsize=14)
    for i in range(n):
        for j in range(n):
            cnt = int(cm[i, j]); pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i - 0.18, f"{cnt}", ha="center", va="center", fontsize=17,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"({pct:.1f}%)", ha="center", va="center", fontsize=12, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=14); cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plot_cm(cm_train, os.path.join(HERE, "fig_cm_train.png"))
plot_cm(cm_test, os.path.join(HERE, "fig_cm_test.png"))

# 逐类指标
p, r, f1, _ = precision_recall_fscore_support(yte, yp_opt, labels=range(6), average=None, zero_division=0)
cls_df = pd.DataFrame({"类别": CLASS_ABBR, "精确率": p.round(4), "召回率": r.round(4), "F1": f1.round(4)})
with pd.ExcelWriter(os.path.join(HERE, "最优CNN-BiLSTM_结果.xlsx")) as w:
    pd.DataFrame([{"超参数": k, "最优值": v} for k, v in study.best_params.items()]).to_excel(
        w, sheet_name="0-最佳超参数", index=False)
    res.round(4).to_excel(w, sheet_name="1-默认vs优化", index=False)
    cls_df.to_excel(w, sheet_name="2-逐类指标", index=False)
    pd.DataFrame(cm_test, index=CLASS_ABBR, columns=CLASS_ABBR).to_excel(w, sheet_name="3-混淆矩阵")

print("\n===== 逐类指标（测试集） =====")
print(cls_df.to_string(index=False))
print("\n输出文件已保存到:", HERE)
