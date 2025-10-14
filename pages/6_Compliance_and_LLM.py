
import os
import pandas as pd
import numpy as np
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import yaml
import argparse
from openai import OpenAI
import streamlit as st
import glob
from dotenv import load_dotenv

load_dotenv()

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Compliance & LLM | SmartLoan",
    page_icon="📑",
    layout="wide"
)

# ===== BASIC SETTINGS =====
MONITOR_DIR = "monitor"
REPORTS_DIR = "reports"
METRICS_CSV = os.path.join(MONITOR_DIR, "batch_metrics_log.csv")
CONFIG_YAML = os.path.join(MONITOR_DIR, "config.yaml")
J2_FILE = os.path.join(MONITOR_DIR, "compliance_report.md.j2")
PROMPT_FILE = os.path.join(MONITOR_DIR, "llm_system_prompt.txt")

os.makedirs(REPORTS_DIR, exist_ok=True)

# ===== HELPER FUNCTIONS =====
def safe_float(value):
    """Convert value to float, return None if it fails or is NaN/Inf."""
    try:
        if value is None or pd.isna(value):
            return None
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None

def safe_compare_gt(value, threshold):
    """
    Safely compare value > threshold.
    Returns False if either value is None/NaN (no alert triggered).
    """
    val = safe_float(value)
    thr = safe_float(threshold)
    
    if val is None or thr is None:
        return False
    
    return val > thr

def safe_compare_range(value, lower, upper):
    """
    Check if lower < value <= upper.
    Returns False if any value is None/NaN.
    """
    val = safe_float(value)
    low = safe_float(lower)
    up = safe_float(upper)
    
    if val is None or low is None or up is None:
        return False
    
    return low < val <= up

def get_latest_metrics(csv_path):
    """Read the latest row from the CSV file."""
    df = pd.read_csv(csv_path, engine="c", encoding="utf-8-sig")
    if df.empty:
        raise ValueError("The CSV file is empty")
    return df.iloc[-1].to_dict()

def get_thresholds(yaml_path):
    """Read thresholds from the YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return {
        "psi_warn": config.get("psi_threshold_warn", 0.10),
        "psi_alert": config.get("psi_threshold_alert", 0.20),
        "auc_drop": config.get("auc_drop_alert", 0.05),
        "ks_drop": config.get("ks_drop_alert", 0.10),
        "miss_rate": config.get("missing_rate_alert", 0.10),
    }

def check_flags(metrics, thresholds):
    """
    Check if metrics exceed alert thresholds.
    Safe handling of None/NaN values - returns False if data missing.
    """
    return {
        "auc_drop_alert": safe_compare_gt(
            metrics.get("auc_drop"), 
            thresholds["auc_drop"]
        ),
        "ks_drop_alert": safe_compare_gt(
            metrics.get("ks_drop"), 
            thresholds["ks_drop"]
        ),
        "missing_rate_alert": safe_compare_gt(
            metrics.get("max_missing_rate"), 
            thresholds["miss_rate"]
        ),
        "psi_alert": safe_compare_gt(
            metrics.get("psi_max_value"), 
            thresholds["psi_alert"]
        ),
        "psi_warn": safe_compare_range(
            metrics.get("psi_max_value"),
            thresholds["psi_warn"],
            thresholds["psi_alert"]
        ),
    }

# ===== RULE-BASED FUNCTIONS =====
def rule_based_summary(metrics, flags, thresholds):
    """Generate a rule-based summary."""
    summary = ["Monitoring completed for the latest batch."]
    
    # Performance
    if flags["auc_drop_alert"] or flags["ks_drop_alert"]:
        issues = []
        if flags["auc_drop_alert"]:
            auc_drop = safe_float(metrics.get("auc_drop"))
            issues.append(f"AUC drop {auc_drop:.3f}" if auc_drop else "AUC drop detected")
        if flags["ks_drop_alert"]:
            ks_drop = safe_float(metrics.get("ks_drop"))
            issues.append(f"KS drop {ks_drop:.3f}" if ks_drop else "KS drop detected")
        summary.append(f"Model performance weakened ({', '.join(issues)}).")
    else:
        auc = safe_float(metrics.get("auc"))
        ks = safe_float(metrics.get("ks"))
        auc_str = f"{auc:.3f}" if auc else "N/A"
        ks_str = f"{ks:.3f}" if ks else "N/A"
        summary.append(f"AUC={auc_str} and KS={ks_str} are within acceptable range.")
    
    # Drift
    if flags["psi_alert"]:
        psi_val = safe_float(metrics.get("psi_max_value"))
        psi_str = f"{psi_val:.3f}" if psi_val else "N/A"
        summary.append(
            f"Significant drift detected on '{metrics.get('psi_max_feature', 'N/A')}' "
            f"(PSI={psi_str})."
        )
    elif flags["psi_warn"]:
        psi_val = safe_float(metrics.get("psi_max_value"))
        psi_str = f"{psi_val:.3f}" if psi_val else "N/A"
        summary.append(
            f"Moderate drift on '{metrics.get('psi_max_feature', 'N/A')}' "
            f"(PSI={psi_str}); continue monitoring."
        )
    else:
        summary.append("No significant drift detected.")
    
    # Missing data
    if flags["missing_rate_alert"]:
        miss_rate = safe_float(metrics.get("max_missing_rate"))
        miss_str = f"{miss_rate:.1%}" if miss_rate else "N/A"
        summary.append(
            f"Excessive missing data on '{metrics.get('max_missing_feature', 'N/A')}' "
            f"({miss_str})."
        )
    else:
        summary.append("Missing data is within acceptable limits.")
    
    # Fairness
    pass_80 = metrics.get("pass_80_rule")
    if str(pass_80).lower() in {"true", "1", "yes"}:
        summary.append("Fairness metrics meet the 80% rule.")
    else:
        summary.append(
            f"Potential fairness concern in {metrics.get('fairness_groups', 'N/A')}; "
            f"review required."
        )
    
    return " ".join(summary)

def rule_based_actions(flags, metrics):
    """Generate rule-based action recommendations."""
    actions = ["Continue daily monitoring."]
    
    if flags["psi_warn"] or flags["psi_alert"]:
        actions.append(
            f"Track '{metrics.get('psi_max_feature', 'key feature')}' PSI "
            f"for 7-14 days."
        )
    
    if flags["auc_drop_alert"] or flags["ks_drop_alert"]:
        actions.append("Review model calibration and plan retraining.")
    
    if flags["missing_rate_alert"]:
        actions.append("Investigate ETL processes for missing data issues.")
    
    if str(metrics.get("pass_80_rule", True)).lower() not in {"true", "1", "yes"}:
        actions.append("Conduct a fairness audit across protected groups.")
    
    return "\n".join(["- " + a for a in actions])

def render_report(template_path, context):
    """Render the report using Jinja2 and save it."""
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template_path) or "."), 
        autoescape=False
    )
    template = env.get_template(os.path.basename(template_path))
    md = template.render(**context)
    
    file_name = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    file_path = os.path.join(REPORTS_DIR, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    return file_path, md

def read_file(file_path):
    """Read file content, return error message if it fails."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"(Failed to read: {e})"

def save_file(text, prefix):
    """Save text to a new file with a timestamp."""
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{prefix}{time_str}.md"
    file_path = os.path.join(REPORTS_DIR, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return file_path

def safe_format(value, decimals=3):
    """
    Format number with specified decimals, return 'N/A' if None.
    Used to prepare values for Jinja2 templates.
    """
    val = safe_float(value)
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"

def make_llm_summary(metrics, prompt):
    """Generate an LLM-based summary with fallback."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
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
        # Fallback: simple keyword-based summary
        key_lines = [
            line for line in str(metrics).splitlines() 
            if any(word in line.lower() for word in ["auc", "psi", "alert", "missing"])
        ]
        
        if not key_lines:
            return "No key information found for summary."
        
        summary = "\n".join(f"- {line}" for line in key_lines[:5])
        return (
            f"## Simple Summary (Offline Mode)\n\n{summary}\n\n"
            f"_Note: Set OPENAI_API_KEY for AI-enhanced summaries._\n\n"
            f"_Error: {str(e)}_"
        )

# ===== MAIN UI =====
st.title("📑 Compliance Report & LLM Summary Generator")
st.markdown("""
Generate automated compliance reports based on monitoring metrics, 
with optional AI-enhanced summaries for senior management.
""")
st.markdown("---")

# ===== 1. LATEST COMPLIANCE REPORT =====
st.subheader("📂 Latest Compliance Report (Rule-Based)")

if os.path.exists(METRICS_CSV) and os.path.exists(CONFIG_YAML):
    try:
        metrics = get_latest_metrics(METRICS_CSV)
        thresholds = get_thresholds(CONFIG_YAML)
        flags = check_flags(metrics, thresholds)
        summary = rule_based_summary(metrics, flags, thresholds)
        actions = rule_based_actions(flags, metrics)
        
        if os.path.exists(J2_FILE):
            # Prepare context with pre-formatted values (strings)
            context = {
                "run_name": metrics.get("run_name", "demo_batch"),
                "batch_time": metrics.get("batch_time", datetime.now().strftime("%Y-%m-%d %H:%M")),
                "data_window": metrics.get("data_window", "N/A"),
                "model_version": metrics.get("model_version", "v1.0"),
                "data_source": metrics.get("data_source", "simulated / demo"),
                "report_owner": metrics.get("report_owner", "Christine Ding (Data Scientist)"),
                "plain_summary": summary,
                
                # Pre-formatted metric values (already strings, safe for Jinja2)
                "auc": safe_format(metrics.get("auc"), 3),
                "ks": safe_format(metrics.get("ks"), 3),
                "auc_drop": safe_format(metrics.get("auc_drop"), 3),
                "ks_drop": safe_format(metrics.get("ks_drop"), 3),
                "max_missing_rate": safe_format(metrics.get("max_missing_rate"), 1),
                "psi_max_value": safe_format(metrics.get("psi_max_value"), 4),
                
                # Thresholds (numeric - OK for Jinja2 formatting)
                "auc_drop_threshold": thresholds["auc_drop"],
                "ks_drop_threshold": thresholds["ks_drop"],
                "missing_rate_threshold": thresholds["miss_rate"],
                "psi_threshold_warn": thresholds["psi_warn"],
                "psi_threshold_alert": thresholds["psi_alert"],
                
                # Alert flags (boolean)
                "auc_drop_alert": flags["auc_drop_alert"],
                "ks_drop_alert": flags["ks_drop_alert"],
                "missing_rate_alert": flags["missing_rate_alert"],
                "psi_alert": flags["psi_alert"],
                "psi_warn": flags["psi_warn"],
                
                # Other string fields
                "psi_max_feature": metrics.get("psi_max_feature", "N/A"),
                "max_missing_feature": metrics.get("max_missing_feature", "N/A"),
                "top_drift": [],
                "dq_notes": metrics.get("dq_notes", "No unusual ETL/schema issues observed."),
                "fairness_groups": metrics.get("fairness_groups", "income_group, addr_state"),
                "pass_80_rule": metrics.get("pass_80_rule", True),
                "fairness_comment": (
                    "Across evaluated groups, approval-rate ratios remained within the 80% rule."
                ),
                "consumer_duty_comment": (
                    "Customer outcomes appear fair and predictable this batch." 
                    if not any([flags["psi_alert"], flags["auc_drop_alert"], flags["ks_drop_alert"]]) 
                    else "Potential variability in outcomes; monitor under FCA Consumer Duty."
                ),
                "equality_act_comment": (
                    "No evidence of direct discrimination across monitored groups." 
                    if str(metrics.get("pass_80_rule", True)).lower() in {"true", "1", "yes"} 
                    else "Potential disparity observed; review for indirect discrimination risk."
                ),
                "retrain_flag": any([
                    flags["psi_alert"], 
                    flags["auc_drop_alert"], 
                    flags["ks_drop_alert"]
                ]),
                "next_review": metrics.get("next_review", "Quarterly model committee (scheduled)"),
                "actions": actions,
            }
            
            file_path, rendered_md = render_report(J2_FILE, context)
            
            st.success(f"✅ Latest report generated: `{os.path.basename(file_path)}`")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    "📥 Download Report",
                    data=rendered_md,
                    file_name=os.path.basename(file_path),
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with st.expander("👁️ Preview Rendered Report"):
                st.markdown(rendered_md)
        else:
            st.warning(f"⚠️ Template not found: `{J2_FILE}`")
    
    except Exception as e:
        st.error(f"❌ Error generating report: {e}")
        with st.expander("🐛 Debug Info"):
            st.exception(e)
else:
    st.warning(f"⚠️ Missing required files: `{METRICS_CSV}` or `{CONFIG_YAML}`")

st.markdown("---")

# ===== 2. PROMPT CUSTOMIZATION =====
st.subheader("⚙️ Customize AI Summary Prompt")

prompt_options = {
    "Executive Briefing": (
        "Summarise the report in 5-7 bullet points, focusing on risks and "
        "actions required this week. Use business language, not technical jargon."
    ),
    "Regulatory Focus": (
        "Summarise with explicit mapping to FCA Consumer Duty and fairness (80% rule). "
        "Highlight any breaches, mitigations taken, and next steps."
    ),
    "Technical Details": (
        "Summarise technical findings: drift (PSI), performance (AUC/KS trends), "
        "missing data, and fairness by group. Include specific numbers."
    )
}

mode = st.radio(
    "Choose summary style:",
    list(prompt_options.keys()),
    horizontal=True
)

prompt = st.text_area(
    "Edit Prompt (optional):",
    value=prompt_options[mode],
    height=100
)

st.markdown("---")

# ===== 3. GENERATE LLM SUMMARY =====
st.subheader("🤖 Generate AI-Enhanced Summary")

if os.path.exists(METRICS_CSV) and os.path.exists(CONFIG_YAML):
    if st.button("🚀 Generate AI Summary", use_container_width=True):
        with st.spinner("Generating AI summary..."):
            try:
                metrics = get_latest_metrics(METRICS_CSV)
                summary = make_llm_summary(metrics, prompt)
                file_path = save_file(summary, "llm_summary_")
                
                st.success(f"✅ Generated: `{os.path.basename(file_path)}`")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.download_button(
                        "📥 Download Summary",
                        data=summary,
                        file_name=os.path.basename(file_path),
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with st.expander("👁️ Preview AI Summary"):
                    st.markdown(summary)
            
            except Exception as e:
                st.error(f"❌ Error generating AI summary: {e}")
                with st.expander("🐛 Debug Info"):
                    st.exception(e)
else:
    st.warning(f"⚠️ Need `{METRICS_CSV}` and `{CONFIG_YAML}` to generate summary.")

st.markdown("---")

# ===== 4. HISTORICAL SUMMARIES =====
st.subheader("📝 Historical AI Summaries (Last 5)")

llm_files = sorted(
    glob.glob(os.path.join(REPORTS_DIR, "llm_summary_*.md")),
    key=os.path.getmtime,
    reverse=True
)[:5]

if llm_files:
    for file_path in llm_files:
        file_name = os.path.basename(file_path)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")
        
        with st.expander(f"📄 {file_name} (Created: {file_time})"):
            text = read_file(file_path)
            st.markdown(text)
            
            st.download_button(
                "📥 Download",
                data=text,
                file_name=file_name,
                mime="text/markdown",
                key=f"download_{file_name}"
            )
else:
    st.info("ℹ️ No AI summaries yet. Generate one using the button above!")

st.markdown("---")
st.caption("📑 **SmartLoan Compliance & LLM** | Automated Reporting with AI | UK FCA Standards")