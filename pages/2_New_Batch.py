# pages/2_New_Batch.py

import os
import sys
import subprocess
import pandas as pd
import streamlit as st

st.title("New Batch / Monitoring")

BATCH_PATH = "monitor/X_new.csv"
os.makedirs("monitor", exist_ok=True)

# Upload CSV -> save as monitor/X_new.csv
up = st.file_uploader("Upload CSV", type="csv")
if up:
    pd.read_csv(up).to_csv(BATCH_PATH, index=False)
    st.success(f"Saved → {BATCH_PATH}")

# Run monitor
if st.button("Run Monitor"):
    if not os.path.exists(BATCH_PATH):
        st.error("monitor/X_new.csv not found – upload a batch first.")
    else:
        with st.spinner("Running monitor…"):
            res = subprocess.run(
                [sys.executable, "monitor/monitor_1.py", "--batch_csv", BATCH_PATH],
                capture_output=True,
                text=True,
            )
            st.code(res.stdout or "(no stdout)")
            if res.returncode == 0:
                st.success("Monitor executed")
                st.rerun()  # refresh to show latest logs
            else:
                st.error(res.stderr)

# Show latest batch summary
batch_log = "monitor/batch_metrics_log.csv"
st.subheader("Batch Summary")
if os.path.exists(batch_log):
    try:
        df = pd.read_csv(batch_log)
        if not df.empty:
            # Display the most recent batch (assuming sorted by timestamp or index)
            st.write("Latest Batch Metrics:")
            st.dataframe(df.tail(1))  # Show only the latest row
            # Optionally, display all metrics in a table
            st.write("All Batch Metrics History:")
            st.dataframe(df)
        else:
            st.warning("Batch metrics log is empty.")
    except Exception as e:
        st.error(f"Error reading batch metrics log: {e}")
else:
    st.info("No batch metrics log found. Run the monitor to generate metrics.")