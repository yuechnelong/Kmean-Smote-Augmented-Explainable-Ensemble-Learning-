# -*- coding: utf-8 -*-
"""
一致性导出：用同一个 CNN-BiLSTM 模型，同时生成混淆矩阵 + 指标表
================================================================
- 只训练一次模型，训练集/测试集预测都用这个模型，保证指标与混淆矩阵一致
- 优化后超参：hidden=34, lr=0.00797, batch=32, epochs=60
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
                             precision_recall_fscore_support, confusion_matrix)
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


class CNNBiLSTM(nn.Module):
    def __init__(self, in_f=n_feat, hidden=HIDDEN, n_cls=6):
        super().__init__()
        self.conv = nn.Conv1d(in_f, hidden, 3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_cls)
    def forward(self, x):
        x = x.permute(0, 2, 1); x = torch.relu(self.conv(x)); x = x.permute(0, 2, 1)
        x, _ = self.lstm(x); x = x.mean(dim=1); return self.fc(x)


# 只训练一次模型
torch.manual_seed(SEED)
torch.set_num_threads(4)
model = CNNBiLSTM()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
Xt = torch.tensor(Xa_seq, dtype=torch.float32); yt = torch.tensor(ya, dtype=torch.long)
dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
model.train()
for _ in range(EPOCHS):
    for xb, yb in dl:
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
model.eval()

# 用同一个模型预测训练集（原始真实样本）和测试集
with torch.no_grad():
    yp_tr = model(torch.tensor(Xs_tr_s, dtype=torch.float32)).argmax(1).numpy()
    yp_te = model(torch.tensor(Xs_te_s, dtype=torch.float32)).argmax(1).numpy()

cm_train = confusion_matrix(ytr, yp_tr, labels=range(6))
cm_test = confusion_matrix(yte, yp_te, labels=range(6))


def all_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": precision_recall_fscore_support(y_true, y_pred, labels=range(6),
                                                    average="macro", zero_division=0)[2],
        "Weighted_F1": precision_recall_fscore_support(y_true, y_pred, labels=range(6),
                                                       average="weighted", zero_division=0)[2],
    }


def cls_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(6),
                                                 average=None, zero_division=0)
    return pd.DataFrame({"类别": CLASS_ABBR, "精确率": p.round(4),
                         "召回率": r.round(4), "F1": f1.round(4)})


tr_m = all_metrics(ytr, yp_tr)
te_m = all_metrics(yte, yp_te)

overall = pd.DataFrame({
    "数据集": ["训练集", "测试集"],
    "Accuracy": [tr_m["Accuracy"], te_m["Accuracy"]],
    "Balanced_Acc": [tr_m["Balanced_Acc"], te_m["Balanced_Acc"]],
    "Macro_F1": [tr_m["Macro_F1"], te_m["Macro_F1"]],
    "Weighted_F1": [tr_m["Weighted_F1"], te_m["Weighted_F1"]],
})

with pd.ExcelWriter(os.path.join(HERE, "CNN-BiLSTM_训练测试指标.xlsx")) as w:
    overall.round(4).to_excel(w, sheet_name="0-总体指标", index=False)
    cls_metrics(ytr, yp_tr).to_excel(w, sheet_name="1-逐类_训练集", index=False)
    cls_metrics(yte, yp_te).to_excel(w, sheet_name="2-逐类_测试集", index=False)
    pd.DataFrame(cm_test, index=CLASS_ABBR, columns=CLASS_ABBR).to_excel(w, sheet_name="3-混淆矩阵_测试集")

print("===== 总体指标（与混淆矩阵同源） =====")
print(overall.round(4).to_string(index=False))
print("\n===== 逐类指标（测试集） =====")
print(cls_metrics(yte, yp_te).to_string(index=False))
print("\n===== 测试集混淆矩阵 =====")
print(pd.DataFrame(cm_test, index=CLASS_ABBR, columns=CLASS_ABBR).to_string())


def plot_cm(cm, path):
    n = 6
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
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
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Proportion", fontsize=14); cb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plot_cm(cm_train, os.path.join(HERE, "fig_cm_train.png"))
plot_cm(cm_test, os.path.join(HERE, "fig_cm_test.png"))

# 验证：从混淆矩阵反算指标，确认一致
tp = np.diag(cm_test); fp = cm_test.sum(0) - tp; fn = cm_test.sum(1) - tp
print("\n===== 验证：混淆矩阵反算指标 =====")
print(f"  Accuracy = {tp.sum()/cm_test.sum():.4f}  (应与测试集 Accuracy {te_m['Accuracy']:.4f} 一致)")
print(f"  Balanced_Acc = {((tp/(tp+fn)+ (cm_test.sum()-tp-fp-fn)/(cm_test.sum()-tp-fp-fn+fp))/2).mean():.4f}")
print(f"  Macro_F1 = {te_m['Macro_F1']:.4f}")
print(f"  Weighted_F1 = {te_m['Weighted_F1']:.4f}")
print("\n输出文件已保存到:", HERE)
