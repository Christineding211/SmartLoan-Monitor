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

    # 依欄位型態/名稱設定小數位
    num_cols = last.select_dtypes(include="number").columns.tolist()

    # 預設 3 位小數
    last[num_cols] = last[num_cols].round(3)

    # 例外：非常小的比率/缺失率保留 4 位；PSI 可 3 位
    col_4dp = [c for c in num_cols if "rate" in c.lower()]  # e.g. max_missing_rate
    if col_4dp:
        last[col_4dp] = last[col_4dp].round(4)

    # 若有明確欄位名也可指定
    for c in ["psi_max_value"]:
        if c in last.columns:
            last[c] = last[c].round(3)

    # 轉 dict 並用 st.json 輸出（更易讀）
    last_rec = last.to_dict(orient="records")[0]
    st.subheader("Latest batch summary:")
    st.json(last_rec)
else:
    st.info("Run the monitor to generate batch_metrics_log.csv.")

