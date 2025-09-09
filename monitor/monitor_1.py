# monitor/monitor.py  — slim + robust
import os, json, pickle, argparse, sys, pathlib
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

print("[DEBUG] Python:", sys.executable)
print("[DEBUG] CWD:", os.getcwd())

CFG_PATH = "monitor/config.yaml"
REF_PATH = "monitor/reference_stats.pkl"
DEFAULT_BATCH = "monitor/X_new.csv"     # ← 與頁面一致
FEAT_OUT = "monitor/metrics_log.csv"
BATCH_OUT = "monitor/batch_metrics_log.csv"
PERF_PATH = "monitor/perf_latest.csv"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)): return None
        return float(x)
    except Exception:
        return None

def write_append_one_row(out_csv, row_dict):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(out_csv):
        df_row.to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_row.to_csv(out_csv, index=False, encoding="utf-8-sig")

def calculate_psi(series, bins, expected_counts):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or bins is None or expected_counts is None:
        return np.nan
    bins = np.asarray(bins, dtype=float)
    exp = np.asarray(expected_counts, dtype=float)
    if exp.size != bins.size - 1:  # 防呆：形狀不對就放棄
        return np.nan
    new_counts, _ = np.histogram(s.values, bins=bins)
    new_pct = new_counts / max(new_counts.sum(), 1)
    exp_pct = exp / max(exp.sum(), 1)
    new_pct = np.where(new_pct == 0, 1e-8, new_pct)
    exp_pct = np.where(exp_pct == 0, 1e-8, exp_pct)
    return float(np.sum((new_pct - exp_pct) * np.log(new_pct / exp_pct)))

def main(batch_csv):
    print(f"[DEBUG] exists({REF_PATH}) =", pathlib.Path(REF_PATH).exists())
    print(f"[DEBUG] exists({batch_csv}) =", pathlib.Path(batch_csv).exists())
    print(f"[DEBUG] exists({CFG_PATH}) =", pathlib.Path(CFG_PATH).exists())

    cfg = load_yaml(CFG_PATH) if os.path.exists(CFG_PATH) else {}
    with open(REF_PATH, "rb") as f:
        ref = pickle.load(f)
    df_new = pd.read_csv(batch_csv).replace([np.inf, -np.inf], np.nan)

    ref_stats = ref["features"] if (isinstance(ref, dict) and "features" in ref) else ref
    common = [c for c in ref_stats.keys() if c in df_new.columns]
    if not common:
        pd.DataFrame([]).to_csv(FEAT_OUT, index=False, encoding="utf-8-sig")
        print(f"[WARN] no common columns; wrote empty {FEAT_OUT}")
        return

    rows = []
    for col in common:
        base = ref_stats.get(col, {})
        base_mean = base.get("mean", np.nan)

        s = pd.to_numeric(df_new[col], errors="coerce")
        miss_rate = float(pd.isna(s).mean())
        mean_diff = float(s.mean() - base_mean) if pd.notna(base_mean) else np.nan

        bins, counts = base.get("bins"), base.get("counts")
        t = str(base.get("type", "")).lower()
        is_num = (t in ("numeric", "numerical")) or (bins is not None and counts is not None)

        psi = calculate_psi(s, bins, counts) if is_num else np.nan
        rows.append({"feature": col, "psi": psi, "missing_rate_new": miss_rate, "mean_diff": mean_diff})

    feat_df = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    os.makedirs(os.path.dirname(FEAT_OUT), exist_ok=True)
    feat_df.to_csv(FEAT_OUT, index=False, encoding="utf-8-sig")
    print(f"[OK] feature-level → {FEAT_OUT} ({len(feat_df)} rows)")

    # batch summary
    feat_df["psi_num"]  = pd.to_numeric(feat_df["psi"], errors="coerce")
    feat_df["miss_num"] = pd.to_numeric(feat_df["missing_rate_new"], errors="coerce")

    psi_max_value = psi_max_feature = None
    if feat_df["psi_num"].notna().any():
        r = feat_df.loc[feat_df["psi_num"].idxmax()]
        psi_max_value, psi_max_feature = float(r["psi_num"]), r["feature"]

    max_missing_rate = max_missing_feature = None
    if feat_df["miss_num"].notna().any():
        r2 = feat_df.loc[feat_df["miss_num"].idxmax()]
        max_missing_rate, max_missing_feature = float(r2["miss_num"]), r2["feature"]

    top3 = (feat_df.dropna(subset=["psi_num"])
                   .sort_values("psi_num", ascending=False)
                   .head(3)[["feature","psi_num"]]
                   .rename(columns={"psi_num":"psi"}))
    top_drift_json = json.dumps(
        [{"feature": a, "psi": float(b), "ref": "ref-dist"} for a, b in top3.values],
        ensure_ascii=False
    )

    auc = ks = auc_drop = ks_drop = None
    if os.path.exists("monitor/perf_latest.csv"):
        perf = pd.read_csv("monitor/perf_latest.csv")
        if not perf.empty:
            last = perf.iloc[-1].to_dict()
            auc, ks = safe_float(last.get("auc")), safe_float(last.get("ks"))
            prev_auc, prev_ks = safe_float(last.get("prev_auc")), safe_float(last.get("prev_ks"))
            auc_drop = (prev_auc - auc) if (auc is not None and prev_auc is not None) else None
            ks_drop  = (prev_ks  - ks ) if (ks  is not None and prev_ks  is not None) else None

    summary = {
        "run_name":      cfg.get("run_name", "daily_batch"),
        "batch_time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_window":   cfg.get("data_window", "N/A"),
        "model_version": cfg.get("model_version", "v1.0"),
        "data_source":   cfg.get("data_source", "unknown / demo"),
        "report_owner":  cfg.get("report_owner", "Christine Ding (Data Scientist)"),
        "auc": auc, "ks": ks, "auc_drop": auc_drop, "ks_drop": ks_drop,
        "psi_max_value": psi_max_value, "psi_max_feature": psi_max_feature,
        "max_missing_rate": max_missing_rate, "max_missing_feature": max_missing_feature,
        "pass_80_rule": cfg.get("pass_80_rule", True),
        "fairness_groups": cfg.get("fairness_groups", "income_group, addr_state"),
        "dq_notes": cfg.get("dq_notes", "No unusual ETL/schema issues observed."),
        "top_drift_json": top_drift_json,
    }
    write_append_one_row(BATCH_OUT, summary)
    print(f"[OK] batch-level → {BATCH_OUT}")
    print(summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_csv", default=DEFAULT_BATCH)
    args = ap.parse_args()
    main(args.batch_csv)

