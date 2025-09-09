
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PRED_PATH = "test_predictions.csv"      # 改成你的檔名；需包含真實標籤與機率
OUT = "monitor/perf_latest.csv"

df = pd.read_csv(PRED_PATH)

# 很寬鬆的欄名偵測（可自行指定）
y_col = next(c for c in df.columns if c.lower() in ["y","label","target","default","is_bad","bad_flag"])
p_col = next(c for c in df.columns if ("prob" in c.lower()) or ("score" in c.lower()) or ("pred" in c.lower()) or (c.lower()=="p1"))

y = df[y_col].astype(int)
p = pd.to_numeric(df[p_col], errors="coerce")

auc = float(roc_auc_score(y, p))
fpr, tpr, _ = roc_curve(y, p)
ks  = float(np.max(tpr - fpr))

prev_auc = prev_ks = None
if os.path.exists(OUT):
    old = pd.read_csv(OUT).tail(1)
    prev_auc = old.get("auc", pd.Series([None])).iloc[-1]
    prev_ks  = old.get("ks",  pd.Series([None])).iloc[-1]

pd.DataFrame([{"auc": auc, "ks": ks, "prev_auc": prev_auc, "prev_ks": prev_ks}]).to_csv(OUT, index=False)
print("wrote", OUT, {"auc": auc, "ks": ks})