
import os
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import yaml
import argparse
from openai import OpenAI
import streamlit as st
import glob
from dotenv import load_dotenv
load_dotenv()  






# Basic settings
MONITOR_DIR = "monitor"
REPORTS_DIR = "reports"
METRICS_CSV = os.path.join(MONITOR_DIR, "batch_metrics_log.csv")
CONFIG_YAML = os.path.join(MONITOR_DIR, "config.yaml")
J2_FILE = os.path.join(MONITOR_DIR, "compliance_report.md.j2")
PROMPT_FILE = os.path.join(MONITOR_DIR, "llm_system_prompt.txt")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Helper functions
def safe_float(value):
    """Convert value to float, return None if it fails"""
    try:
        return None if pd.isna(value) else float(value)
    except Exception:
        return None

def get_latest_metrics(csv_path):
    """Read the latest row from the CSV file"""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The CSV file is empty")
    return df.iloc[-1].to_dict()

def get_thresholds(yaml_path):
    """Read thresholds from the YAML file"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return {
        "psi_warn": config.get("psi_threshold_warn", 0.10),
        "psi_alert": config.get("psi_threshold_alert", 0.20),
        "auc_drop": config.get("auc_drop_alert", 0.05),
        "ks_drop": config.get("ks_drop_alert", 0.10),
        "miss_rate": config.get("missing_rate_alert", 0.10),
    }

# Use else False to set no alert if the value is None (empty), skipping the comparison
def check_flags(metrics, thresholds):
    return {
        "auc_drop_alert": safe_float(metrics.get("auc_drop")) > thresholds["auc_drop"] if metrics.get("auc_drop") is not None else False,
        "ks_drop_alert": safe_float(metrics.get("ks_drop")) > thresholds["ks_drop"] if metrics.get("ks_drop") is not None else False,
        "missing_rate_alert": safe_float(metrics.get("max_missing_rate")) > thresholds["miss_rate"] if metrics.get("max_missing_rate") is not None else False,
        "psi_alert": safe_float(metrics.get("psi_max_value")) > thresholds["psi_alert"] if metrics.get("psi_max_value") is not None else False,
        "psi_warn": thresholds["psi_warn"] < safe_float(metrics.get("psi_max_value")) <= thresholds["psi_alert"] if metrics.get("psi_max_value") is not None else False,
    }

# Rule-based functions
def rule_based_summary(metrics, flags, thresholds):
    """Generate a rule-based summary"""
    summary = ["Monitoring completed for the latest batch."]
    if flags["auc_drop_alert"] or flags["ks_drop_alert"]:
        issues = [f"AUC drop {metrics.get('auc_drop')}" if flags["auc_drop_alert"] else "",
                  f"KS drop {metrics.get('ks_drop')}" if flags["ks_drop_alert"] else ""]
        summary.append(f"Model weakened ({', '.join(filter(None, issues))}).")
    else:
        summary.append(f"AUC={metrics.get('auc', 'N/A')} and KS={metrics.get('ks', 'N/A')} are fine.")
    if flags["psi_alert"]:
        summary.append(f"Significant drift on '{metrics.get('psi_max_feature', 'N/A')}' (PSI={metrics.get('psi_max_value')}).")
    elif flags["psi_warn"]:
        summary.append(f"Moderate drift on '{metrics.get('psi_max_feature', 'N/A')}' (PSI={metrics.get('psi_max_value')}); keep an eye on it.")
    else:
        summary.append("No significant drift detected.")
    if flags["missing_rate_alert"]:
        summary.append(f"Excessive missing data on '{metrics.get('max_missing_feature', 'N/A')}'.")
    else:
        summary.append("Missing data is within acceptable limits.")
    if str(metrics.get("pass_80_rule", True)).lower() in {"true", "1", "yes"}:
        summary.append("Fairness meets the 80% rule.")
    else:
        summary.append(f"Potential unfairness in {metrics.get('fairness_groups', 'N/A')}; please review.")
    return " ".join(summary)

def rule_based_actions(flags, metrics):
    """Generate rule-based action recommendations"""
    actions = ["Continue daily monitoring."]
    if flags["psi_warn"] or flags["psi_alert"]:
        actions.append(f"Track '{metrics.get('psi_max_feature', 'key feature')}' PSI for 7-14 days.")
    if flags["auc_drop_alert"] or flags["ks_drop_alert"]:
        actions.append("Review model calibration and plan retraining.")
    if flags["missing_rate_alert"]:
        actions.append("Investigate ETL processes for missing data.")
    if str(metrics.get("pass_80_rule", True)).lower() not in {"true", "1", "yes"}:
        actions.append("Conduct a fairness audit.")
    return "\n- ".join(["- " + a for a in actions])

def render_report(template_path, context):
    """Render the report using Jinja2 and save it"""
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path) or "."), autoescape=False)
    md = env.get_template(os.path.basename(template_path)).render(**context)
    file_name = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    file_path = os.path.join(REPORTS_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md)
    return file_path, md

def read_file(file_path):
    """Read file content, return error message if it fails"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"(Failed to read: {e})"

def save_file(text, prefix):
    """Save text to a new file with a timestamp"""
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{prefix}{time_str}.md"
    file_path = os.path.join(REPORTS_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

def make_llm_summary(metrics, prompt):
    """Generate an LLM-based summary with fallback to keyword-based summary."""

    api_key = os.getenv("OPENAI_API_KEY")
    try:
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set, using fallback.")

        #  api_key
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(metrics)}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # --- Fallback  ---
        text = str(metrics)
        keywords = ["auc", "psi", "alert", "missing"]
        key_lines = [
            line for line in text.splitlines()
            if any(word in line.lower() for word in keywords)
        ]

        if not key_lines:
            return (
                " LLM unavailable, and no key metrics detected.\n"
                "Please check your input or set OPENAI_API_KEY for richer summaries.\n\n"
                f"Error: {e}"
            )

        # extract points 
        summary = "\n".join(f"- {line}" for line in key_lines[:5])
        return (
            "## Simple Summary (Offline Mode)\n"
            f"{summary}\n\n"
            "_Note: Set OPENAI_API_KEY for better results._"
        )

# UI: Report Generation and Management Centre
st.title("📑 Report Generation and Management Centre")

# 1. 📂 Latest Compliance Report Download (Rule-based)
st.subheader("📂 Latest Compliance Report Download (Rule-based)")
if os.path.exists(METRICS_CSV) and os.path.exists(CONFIG_YAML):
    metrics = get_latest_metrics(METRICS_CSV)
    thresholds = get_thresholds(CONFIG_YAML)
    flags = check_flags(metrics, thresholds)
    summary = rule_based_summary(metrics, flags, thresholds)
    actions = rule_based_actions(flags, metrics)
    if os.path.exists(J2_FILE):
        file_path, rendered_md = render_report(J2_FILE, {
            "run_name": metrics.get("run_name", "demo_batch"),
            "batch_time": metrics.get("batch_time", datetime.now().strftime("%Y-%m-%d %H:%M")),
            "data_window": metrics.get("data_window", "N/A"),
            "model_version": metrics.get("model_version", "v1.0"),
            "data_source": metrics.get("data_source", "simulated / demo"),
            "report_owner": metrics.get("report_owner", "Christine Ding (Data Scientist)"),
            "plain_summary": summary,
            "auc": metrics.get("auc", "N/A"), "ks": metrics.get("ks", "N/A"),
            "auc_drop_threshold": thresholds["auc_drop"], "ks_drop_threshold": thresholds["ks_drop"],
            "missing_rate_threshold": thresholds["miss_rate"],
            "psi_threshold_warn": thresholds["psi_warn"], "psi_threshold_alert": thresholds["psi_alert"],
            "auc_drop_alert": flags["auc_drop_alert"], "ks_drop_alert": flags["ks_drop_alert"],
            "missing_rate_alert": flags["missing_rate_alert"],
            "psi_alert": flags["psi_alert"], "psi_warn": flags["psi_warn"],
            "max_missing_rate": metrics.get("max_missing_rate", "N/A"),
            "psi_max_feature": metrics.get("psi_max_feature", "N/A"),
            "psi_max_value": metrics.get("psi_max_value", "N/A"),
            "top_drift": [],
            "dq_notes": metrics.get("dq_notes", "No unusual ETL/schema issues observed."),
            "fairness_groups": metrics.get("fairness_groups", "income_group, addr_state"),
            "pass_80_rule": metrics.get("pass_80_rule", True),
            "fairness_comment": "Across evaluated groups, approval-rate ratios remained within the 80% rule.",
            "consumer_duty_comment": "Customer outcomes appear fair and predictable this batch." if not any([flags["psi_alert"], flags["auc_drop_alert"], flags["ks_drop_alert"]]) else "Potential variability in outcomes; monitor under FCA Consumer Duty.",
            "equality_act_comment": "No evidence of direct discrimination across monitored groups." if str(metrics.get("pass_80_rule", True)).lower() in {"true", "1", "yes"} else "Potential disparity observed; review for indirect discrimination risk.",
            "retrain_flag": any([flags["psi_alert"], flags["auc_drop_alert"], flags["ks_drop_alert"]]),
            "next_review": metrics.get("next_review", "Quarterly model committee (scheduled)"),
            "actions": actions,
        })
        st.write(f"Latest report: {os.path.basename(file_path)}")
        st.download_button("Download Compliance Report", data=rendered_md, file_name=os.path.basename(file_path), mime="text/markdown")
        with st.expander("Preview Rendered Report"):
            st.markdown(rendered_md)
    else:
        st.warning("Template monitor/compliance_report.md.j2 not found.")
else:
    st.warning("Missing monitor/batch_metrics_log.csv or monitor/config.yaml.")

st.divider()

# 2. ⚙️ Prompt Preview / Switch Mode
st.subheader("⚙️ Prompt Preview / Switch Mode")
prompt_options = {
    "Executive Briefing": "Summarise the report in 5-7 bullet points, focusing on risks and actions this week. ",
    "Regulatory Focus": "Summarise with explicit mapping to FCA Consumer Duty and fairness (80% rule). Highlight breaches, mitigations, and next steps.",
    "Technical Details": "Summarise technical findings: drift (PSI), performance (AUC/KS trends), missing data, and fairness by group. Include concise numbers."
}
mode = st.radio("Choose summary mode:", list(prompt_options.keys()), horizontal=True)
prompt = st.text_area("Edit Prompt", value=prompt_options[mode], height=100)

st.divider()

# 3. 🖱️ Generate LLM Summary
st.subheader("🖱️ Generate LLM Summary (for Senior Management)")
if os.path.exists(METRICS_CSV) and os.path.exists(CONFIG_YAML):
    if st.button("Generate Report"):
        with st.spinner("Generating…"):
            metrics = get_latest_metrics(METRICS_CSV)
            summary = make_llm_summary(metrics, prompt)
            file_path = save_file(summary, "llm_summary_")
            st.success(f"Generated: {os.path.basename(file_path)}")
            st.download_button("Download LLM Summary", data=summary, file_name=os.path.basename(file_path), mime="text/markdown")
            with st.expander("Preview LLM Summary"):
                st.markdown(summary)
else:
    st.warning("Need monitor/batch_metrics_log.csv and monitor/config.yaml to generate summary.")

st.divider()

# 4. 📝 Historical LLM Summaries
st.subheader("📝 Historical LLM Summaries (Last 5)")
llm_files = sorted(glob.glob(os.path.join(REPORTS_DIR, "llm_summary_*.md")), key=os.path.getmtime, reverse=True)[:5]
if llm_files:
    for file_path in llm_files:
        file_name = os.path.basename(file_path)
        with st.expander(f"📄 {file_name}"):
            text = read_file(file_path)
            st.markdown(text)
            st.download_button("Download This File", data=text, file_name=file_name, mime="text/markdown")
else:
    st.info("No LLM summaries yet. Try generating one!")