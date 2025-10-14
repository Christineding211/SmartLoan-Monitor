# smartloan_agent/agent_fs.py (OPTIMIZED VERSION)

import os, json, datetime, yaml
from pathlib import Path
import pandas as pd
import numpy as np
from jinja2 import Template
from sklearn.metrics import roc_auc_score

CFG_PATH = "monitor/config.yaml"
BATCH_OUT = "monitor/batch_metrics_log.csv"
PERF_PATH = "monitor/perf_latest.csv"
TEMPLATE_PATH = "monitor/compliance_report.md.j2"
DEFAULT_OUT_DIR = "monitor/reports"
XNEW_PATH = "monitor/X_new.csv"
METRICS_OUT = "monitor/metrics_log.csv"

# ---------- Helper Functions ----------
def _safe_float(x):
    """Safely convert to float, handling None/NaN/Inf."""
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None

def _compute_auc_ks_from_csv(csv_path, y_col="label", p_col="score", max_rows=10000):
    """
    OPTIMIZED: Compute AUC/KS from CSV with row/column limits.
    Only reads first N rows and required columns for speed.
    """
    if not os.path.exists(csv_path):
        return None, None
    
    try:
        # Read only needed columns + limit rows for speed
        df = pd.read_csv(
            csv_path,
            usecols=[y_col, p_col],
            nrows=max_rows,
            engine="c",  # Faster C parser
            encoding="utf-8-sig"
        )
        
        if y_col not in df.columns or p_col not in df.columns:
            return None, None
        
        # Convert to numeric and filter valid rows
        y = pd.to_numeric(df[y_col], errors="coerce")
        p = pd.to_numeric(df[p_col], errors="coerce")
        mask = (~y.isna()) & (~p.isna())
        y, p = y[mask].values, p[mask].values
        
        if len(y) < 5 or len(np.unique(y)) < 2:
            return None, None
        
        # Compute AUC
        auc = float(roc_auc_score(y, p))
        
        # Compute KS statistic
        tmp = pd.DataFrame({"y": y, "p": p}).sort_values("p")
        denom_good = max(1, (1 - tmp["y"]).sum())
        denom_bad = max(1, tmp["y"].sum())
        tmp["cum_good"] = (1 - tmp["y"]).cumsum() / denom_good
        tmp["cum_bad"] = (tmp["y"]).cumsum() / denom_bad
        ks = float(np.max(np.abs(tmp["cum_bad"] - tmp["cum_good"])))
        
        return auc, ks
        
    except Exception as e:
        print(f"Warning: AUC/KS computation failed - {e}")
        return None, None


class SmartLoanAgentFS:
    """
    OPTIMIZED File System Agent:
    - Reads batch_metrics_log.csv (only last row)
    - Computes missing AUC/KS from X_new.csv (sampled)
    - Applies config.yaml thresholds
    - Generates Markdown governance report
    """
    
    def __init__(self, cfg_path=CFG_PATH, template_path=TEMPLATE_PATH):
        # Load config
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}
        
        self.template_path = template_path
        
        # Setup output directory
        out_dir = (self.cfg.get("report") or {}).get("out_dir") or DEFAULT_OUT_DIR
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _latest_batch_row(self):
        """OPTIMIZED: Read only the last row from batch log."""
        if not os.path.exists(BATCH_OUT):
            raise FileNotFoundError(
                f"{BATCH_OUT} not found. Run New Batch → Run Monitor first."
            )
        
        # Use tail(1) to efficiently get last row
        df = pd.read_csv(
            BATCH_OUT,
            engine="c",
            encoding="utf-8-sig"
        ).tail(1)
        
        if df.empty:
            raise ValueError(f"{BATCH_OUT} is empty.")
        
        return df.iloc[0].to_dict()

    def run(self):
        """Main execution: read metrics, apply thresholds, generate report."""
        cfg = self.cfg
        row = self._latest_batch_row()

        # ----- Extract metrics from batch summary -----
        psi_val = _safe_float(row.get("psi_max_value"))
        auc = _safe_float(row.get("auc"))
        ks = _safe_float(row.get("ks"))
        auc_drop = _safe_float(row.get("auc_drop"))
        ks_drop = _safe_float(row.get("ks_drop"))

        # ----- Fallback: compute AUC/KS from X_new.csv if missing -----
        labels_cfg = cfg.get("labels") or {}
        y_col = labels_cfg.get("y_col", "label")
        p_col = labels_cfg.get("p_col", "score")
        
        if auc is None or ks is None:
            print(f"AUC/KS missing in batch log. Computing from {XNEW_PATH}...")
            a2, k2 = _compute_auc_ks_from_csv(
                XNEW_PATH, 
                y_col=y_col, 
                p_col=p_col, 
                max_rows=10000  # Sample first 10k rows for speed
            )
            if auc is None:
                auc = a2
            if ks is None:
                ks = k2

        # ----- Compute drops from baseline (if available) -----
        base = cfg.get("baseline") or {}
        base_auc = _safe_float(base.get("roc_auc"))
        base_ks = _safe_float(base.get("ks"))
        
        if auc is not None and base_auc is not None:
            auc_drop = max(0.0, base_auc - auc)
        if ks is not None and base_ks is not None:
            ks_drop = max(0.0, base_ks - ks)

        # ----- Thresholds -----
        psi_warn = float(cfg.get("psi_threshold_warn", 0.10))
        psi_alert = float(cfg.get("psi_threshold_alert", 0.20))
        auc_drop_alert = float(cfg.get("auc_drop_alert", 0.05))
        ks_drop_alert = float(cfg.get("ks_drop_alert", 0.10))

        # ----- Level Functions -----
        def lv_psi(v):
            if v is None:
                return "N/A"
            v = float(v)
            if v > psi_alert:
                return "ALERT"
            if v > psi_warn:
                return "WARN"
            return "OK"

        def lv_drop(v, alert_thr):
            if v is None:
                return "N/A"
            return "ALERT" if float(v) >= alert_thr else "OK"

        psi_level = lv_psi(psi_val)
        auc_level = lv_drop(auc_drop, auc_drop_alert)
        ks_level = lv_drop(ks_drop, ks_drop_alert)

        # ----- Fairness -----
        fairness_level = "N/A"
        if "pass_80_rule" in row:
            fairness_level = "OK" if bool(row["pass_80_rule"]) else "ALERT"

        # ----- Overall Status -----
        levels = [psi_level, auc_level, ks_level, fairness_level]
        overall = "OK"
        if "ALERT" in levels:
            overall = "ALERT"
        elif "WARN" in levels:
            overall = "WARN"

        # ----- Build Metrics Dict -----
        metrics = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "psi": psi_val,
            "psi_level": psi_level,
            "auc": auc,
            "auc_drop": auc_drop,
            "auc_level": auc_level,
            "ks": ks,
            "ks_drop": ks_drop,
            "ks_level": ks_level,
            "fairness": {
                "source": "batch_metrics_log.csv",
                "pass_80_rule": row.get("pass_80_rule", None)
            },
            "fairness_level": fairness_level,
            "overall": overall,
            "source_files": {
                "batch": BATCH_OUT,
                "perf": PERF_PATH,
                "xnew": XNEW_PATH
            },
            "labels_used": {
                "y_col": y_col,
                "p_col": p_col
            }
        }

        # ----- Generate Report -----
        with open(self.template_path, "r", encoding="utf-8") as f:
            tpl = Template(f.read())
        
        md = tpl.render(m=metrics, cfg=cfg)
        
        timestamp_safe = metrics['timestamp'].replace(':', '-')
        out_path = self.out_dir / f"governance_{timestamp_safe}.md"
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        return metrics, str(out_path)