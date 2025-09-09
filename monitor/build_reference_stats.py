# build_reference_stats.py
import pandas as pd, numpy as np, pickle, os

TRAIN = "X_train_1.csv"
OUT   = "monitor/reference_stats.pkl"
NUM_FEATURES = [
    "int_rate","dti","fico_mid","installment_to_income",
    "log_loan_amnt","log_annual_inc","credit_history_length"
]
N_BINS = 20  # 夠簡單、穩定

df = pd.read_csv(TRAIN)
ref = {"features": {}}

for col in NUM_FEATURES:
    if col not in df.columns: continue
    s_all = pd.to_numeric(df[col], errors="coerce").replace([np.inf,-np.inf], np.nan)
    s = s_all.dropna()
    if s.empty: continue

    bins = np.histogram_bin_edges(s.values, bins=N_BINS)
    counts, _ = np.histogram(s.values, bins=bins)

    ref["features"][col] = {
        "type": "numeric",
        "count": int(s.size),
        "missing_rate": float(s_all.isna().mean()),
        "min": float(s.min()), "max": float(s.max()),
        "p01": float(s.quantile(0.01)), "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)), "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)), "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "bins": bins.tolist(),
        "counts": counts.tolist(),
    }

os.makedirs("monitor", exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump(ref, f)

print(f"saved -> {OUT}")

