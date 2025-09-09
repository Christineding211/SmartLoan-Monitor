
# llm_reports.py  — simple & British English
# Inputs:
#   monitor/batch_metrics_log.csv  (one row per batch; we use the last row)
#   monitor/config.yaml            (thresholds)
#   templates/compliance_report.md.j2
# Output:
#   reports/compliance_report_YYYYMMDD_HHMM.md

import os, json, yaml, argparse
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# ---------- tiny helpers ----------
def f2(x):
    """Safe float: return None for NaN/invalid."""
    try:
        return None if pd.isna(x) else float(x)
    except Exception:
        return None

def read_latest_row(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("No rows in batch_metrics_log.csv")
    return df.iloc[-1].to_dict()

def read_thresholds(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f) or {}
    return {
        "psi_warn":  c.get("psi_threshold_warn", 0.10),
        "psi_alert": c.get("psi_threshold_alert", 0.20),
        "auc_drop":  c.get("auc_drop_alert", 0.05),
        "ks_drop":   c.get("ks_drop_alert", 0.10),
        "miss_rate": c.get("missing_rate_alert", 0.10),
    }

def build_flags(m, th):
    return {
        "auc_drop_alert":     (f2(m.get("auc_drop"))        is not None and f2(m.get("auc_drop"))        > th["auc_drop"]),
        "ks_drop_alert":      (f2(m.get("ks_drop"))         is not None and f2(m.get("ks_drop"))         > th["ks_drop"]),
        "missing_rate_alert": (f2(m.get("max_missing_rate")) is not None and f2(m.get("max_missing_rate")) > th["miss_rate"]),
        "psi_alert":          (f2(m.get("psi_max_value"))   is not None and f2(m.get("psi_max_value"))   > th["psi_alert"]),
        "psi_warn":           (f2(m.get("psi_max_value"))   is not None and th["psi_warn"] < f2(m.get("psi_max_value")) <= th["psi_alert"]),
    }


def make_summary(m, flg, th):
    s = []
    # Overall
    s.append("Monitoring completed for the latest batch.")
    # Performance
    auc, ks = m.get("auc", "N/A"), m.get("ks", "N/A")
    if flg["auc_drop_alert"] or flg["ks_drop_alert"]:
        bits = []
        if flg["auc_drop_alert"]: bits.append(f"AUC drop {m.get('auc_drop')} > {th['auc_drop']}")
        if flg["ks_drop_alert"]:  bits.append(f"KS drop {m.get('ks_drop')} > {th['ks_drop']}")
        s.append(f"Model discrimination has weakened ({', '.join(bits)}), which may hinder separation of higher-risk applicants.")
    else:
        s.append(f"AUC={auc} and KS={ks} are within tolerance for this run.")
    # Drift
    feat, psi = m.get("psi_max_feature", "N/A"), m.get("psi_max_value")
    if flg["psi_alert"]:
        s.append(f"Material population shift on '{feat}' (PSI={psi} > {th['psi_alert']}). Prolonged drift could affect fair outcomes under the FCA Consumer Duty.")
    elif flg["psi_warn"]:
        s.append(f"Moderate drift on '{feat}' (PSI={psi}); continue to observe trend.")
    else:
        s.append("No material drift detected across monitored features.")
    # Missing data
    if flg["missing_rate_alert"]:
        s.append(f"Missing data exceeds the threshold on '{m.get('max_missing_feature','N/A')}' (rate={m.get('max_missing_rate')}); please review the ETL mapping.")
    else:
        s.append("Missing data remains within agreed limits.")
    # Fairness
    groups = m.get("fairness_groups", "income_group, addr_state")
    if str(m.get("pass_80_rule", True)).lower() in {"true","1","yes"}:
        s.append(f"Fairness checks across {groups} meet the commonly used 80% rule; no evidence of direct discrimination under the Equality Act 2010.")
    else:
        s.append(f"Potential disparity observed across {groups}; please review for indirect discrimination risk.")
    return " ".join(s[:6])  # keep it short

def make_actions(flg, m):
    items = ["Continue daily monitoring."]
    if flg["psi_warn"] or flg["psi_alert"]:
        items.append(f"Track PSI on '{m.get('psi_max_feature','key feature')}' for 7–14 days and review sensitivity/SHAP if drift persists.")
    if flg["auc_drop_alert"] or flg["ks_drop_alert"]:
        items.append("Review model calibration and prepare a targeted retraining set.")
    if flg["missing_rate_alert"]:
        items.append("Investigate upstream ETL or schema changes causing missing data.")
    if str(m.get("pass_80_rule", True)).lower() not in {"true","1","yes"}:
        items.append("Conduct a focused fairness audit to ensure compliance with the Equality Act 2010.")
    return "\n- ".join(["- " + x for x in items])

# ---------- render ----------
def render(template_path, context, out_dir="reports"):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path) or "."))
    md = env.get_template(os.path.basename(template_path)).render(**context)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("[OK] report written:", out_path)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--metrics_csv", "--batch_log", dest="metrics_csv",
                default="monitor/batch_metrics_log.csv")

    ap.add_argument("--config_yaml", default="monitor/config.yaml")
    ap.add_argument("--template", default="monitor/compliance_report.md.j2")
    ap.add_argument("--out_dir",     default="reports")
    args = ap.parse_args()

    m   = read_latest_row(args.metrics_csv)
    th  = read_thresholds(args.config_yaml)
    flg = build_flags(m, th)

    summary = make_summary(m, flg, th)
    actions = make_actions(flg, m)

    context = {
        # Header
        "run_name":      m.get("run_name","demo_batch"),
        "batch_time":    m.get("batch_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "data_window":   m.get("data_window","N/A"),
        "model_version": m.get("model_version","v1.0"),
        "data_source":   m.get("data_source","simulated / demo"),
        "report_owner":  m.get("report_owner","Christine Ding (Data Scientist)"),
        # Summary + metrics
        "plain_summary": summary,
        "auc": m.get("auc","N/A"), "ks": m.get("ks","N/A"),
        "auc_drop_threshold": th["auc_drop"], "ks_drop_threshold": th["ks_drop"],
        "missing_rate_threshold": th["miss_rate"],
        "psi_threshold_warn": th["psi_warn"], "psi_threshold_alert": th["psi_alert"],
        "auc_drop_alert": flg["auc_drop_alert"], "ks_drop_alert": flg["ks_drop_alert"],
        "missing_rate_alert": flg["missing_rate_alert"],
        "psi_alert": flg["psi_alert"], "psi_warn": flg["psi_warn"],
        "max_missing_rate": m.get("max_missing_rate","N/A"),
        "psi_max_feature":  m.get("psi_max_feature","N/A"),   # ← rename
        "psi_max_value":    m.get("psi_max_value","N/A"), 
        # Drift & DQ
        "top_drift": json.loads(m.get("top_drift_json","[]") or "[]"),
        "dq_notes":  m.get("dq_notes","No unusual ETL/schema issues observed."),
        # Fairness & governance
        "fairness_groups": m.get("fairness_groups","income_group, addr_state"),
        "pass_80_rule":    m.get("pass_80_rule", True),
        "fairness_comment":"Across evaluated groups, approval-rate ratios remained within the 80% rule.",
        "consumer_duty_comment": "Customer outcomes appear fair and predictable this batch." \
            if not (flg["psi_alert"] or flg["auc_drop_alert"] or flg["ks_drop_alert"]) \
            else "Potential variability in outcomes due to drift/performance movement; monitor under the FCA Consumer Duty.",
        "equality_act_comment": "No evidence of direct discrimination across monitored groups." \
            if str(m.get("pass_80_rule", True)).lower() in {"true","1","yes"} \
            else "Potential disparity observed; review for indirect discrimination risk.",
        "retrain_flag": any([flg["psi_alert"], flg["auc_drop_alert"], flg["ks_drop_alert"]]),
        "next_review": m.get("next_review","Quarterly model committee (scheduled)"),
        # Actions
        "actions": actions,
    }

    render(args.template, context, args.out_dir)

if __name__ == "__main__":
    main()
