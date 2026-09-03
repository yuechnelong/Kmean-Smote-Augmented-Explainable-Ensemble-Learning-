# -*- coding: utf-8 -*-
"""
导出最优 CNN-BiLSTM 的训练集 + 测试集指标（总体 + 逐类）
========================================================
优化后超参：hidden=34, lr=0.00797, batch=32
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)
from imblearn.over_sampling import KMeansSMOTE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
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


def train_predict(Xtr, ytr, Xte):
    model = CNNBiLSTM()
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


# 训练集（原始真实样本）+ 测试集预测
yp_tr = train_predict(Xa_seq, ya, Xs_tr_s)
yp_te = train_predict(Xa_seq, ya, Xs_te_s)


def all_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mf = precision_recall_fscore_support(y_true, y_pred, labels=range(6), average="macro", zero_division=0)[2]
    wf = precision_recall_fscore_support(y_true, y_pred, labels=range(6), average="weighted", zero_division=0)[2]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(6), average=None, zero_division=0)
    return {"Accuracy": acc, "Balanced_Acc": bacc, "Macro_F1": mf, "Weighted_F1": wf,
            "precision": p, "recall": r, "f1": f1}


tr_m = all_metrics(ytr, yp_tr)
te_m = all_metrics(yte, yp_te)

overall = pd.DataFrame({
    "数据集": ["训练集", "测试集"],
    "Accuracy": [tr_m["Accuracy"], te_m["Accuracy"]],
    "Balanced_Acc": [tr_m["Balanced_Acc"], te_m["Balanced_Acc"]],
    "Macro_F1": [tr_m["Macro_F1"], te_m["Macro_F1"]],
    "Weighted_F1": [tr_m["Weighted_F1"], te_m["Weighted_F1"]],
})


def cls_df(m, name):
    return pd.DataFrame({"类别": CLASS_ABBR, "精确率": m["precision"].round(4),
                         "召回率": m["recall"].round(4), "F1": m["f1"].round(4)})


with pd.ExcelWriter(os.path.join(HERE, "CNN-BiLSTM_训练测试指标.xlsx")) as w:
    overall.round(4).to_excel(w, sheet_name="0-总体指标", index=False)
    cls_df(tr_m, "训练集").to_excel(w, sheet_name="1-逐类_训练集", index=False)
    cls_df(te_m, "测试集").to_excel(w, sheet_name="2-逐类_测试集", index=False)

print("===== 总体指标 =====")
print(overall.round(4).to_string(index=False))
print("\n===== 逐类指标（训练集） =====")
print(cls_df(tr_m, "训练集").to_string(index=False))
print("\n===== 逐类指标（测试集） =====")
print(cls_df(te_m, "测试集").to_string(index=False))
print("\n输出文件已保存到:", HERE)
