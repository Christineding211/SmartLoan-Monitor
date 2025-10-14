
import streamlit as st
import os

st.title("📊 SmartLoan Monitor Dashboard")
st.write("Model Governance & Monitoring System")

st.subheader("File Health Check")
st.json({
    "Reference stats (pkl)": os.path.exists("monitor/reference_stats.pkl"),
    "Metrics log (csv)": os.path.exists("monitor/metrics_log.csv"),
    "Batch metrics (csv)": os.path.exists("monitor/batch_metrics.csv"),
    "Reports folder": os.path.isdir("reports"),
    "Any SHAP image": any(f.lower().startswith("shap") for f in os.listdir("reports"))
})
