python# pages/2_New_Batch.py
import os
import sys
import subprocess
import pandas as pd
import streamlit as st

st.title("New Batch / Monitoring")

# Define paths at the top
BATCH_PATH = "monitor/X_new.csv"
BATCH_LOG = "monitor/batch_metrics_log.csv"
os.makedirs("monitor", exist_ok=True)

# Auto-generate demo metrics on first load if needed
if os.path.exists(BATCH_PATH) and not os.path.exists(BATCH_LOG):
    with st.spinner("Initializing demo data..."):
        result = subprocess.run(
            [sys.executable, "monitor/monitor_1.py", "--batch_csv", BATCH_PATH],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            st.success("✓ Demo data ready!")
        else:
            st.warning("Note: Auto-initialization encountered an issue, but you can still run manually below.")

# Show status
if os.path.exists(BATCH_PATH):
    st.success(f"✓ Demo batch file loaded: {BATCH_PATH}")
else:
    st.info("No default batch file found. Please upload a CSV file.")

# Upload CSV (optional override)
up = st.file_uploader("Upload New CSV (optional - overrides demo file)", type="csv")
if up:
    pd.read_csv(up).to_csv(BATCH_PATH, index=False)
    st.success(f"Saved → {BATCH_PATH}")

# Run monitor button
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
                st.success("✓ Monitor executed successfully")
                st.rerun()
            else:
                st.error(res.stderr)

# Show latest batch summary
st.subheader("Batch Summary")

if os.path.exists(BATCH_LOG):
    try:
        df = pd.read_csv(BATCH_LOG)
        if not df.empty:
            st.write("**Latest Batch Metrics:**")
            st.dataframe(df.tail(1), use_container_width=True)
            
            with st.expander("📊 View All Batch History"):
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("Batch metrics log is empty.")
    except Exception as e:
        st.error(f"Error reading batch metrics log: {e}")
else:
    st.info("No batch metrics log found. Run the monitor to generate metrics.")
```
