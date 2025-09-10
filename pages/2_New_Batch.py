# pages/2_New_Batch.py
import os, sys, subprocess
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
                capture_output=True, text=True
            )
        st.code(res.stdout or "(no stdout)")
        if res.returncode == 0:
            st.success("Monitor executed")
            st.rerun()  # refresh to show latest logs
        else:
            st.error(res.stderr)

# Show latest batch summary
batch_log = "monitor/batch_metrics_log.csv"
if os.path.exists(batch_log) and os.path.getsize(batch_log) > 0:
    df = pd.read_csv(batch_log)
    last = df.tail(1).copy()

    # 統一所有數值欄位小數點 3 位
    last = last.round(3)

    # 輸出成 JSON（易讀）
    last_rec = last.to_dict(orient="records")[0]
    st.subheader("Latest batch summary:")
    st.json(last_rec)
else:
    st.info("Run the monitor to generate batch_metrics_log.csv.")
