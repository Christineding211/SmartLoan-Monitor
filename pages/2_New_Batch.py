# pages/2_New_Batch.py
import os, sys, subprocess
import pandas as pd
import streamlit as st
import json

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

# 顯示最新批次摘要
if os.path.exists(batch_log) and os.path.getsize(batch_log) > 0:
    df = pd.read_csv(batch_log)
    last = df.tail(1).copy()

    # 依欄位型態/名稱設定小數位
    num_cols = last.select_dtypes(include="number").columns.tolist()
    # 預設 3 位小數
    last[num_cols] = last[num_cols].round(3)
    # 例外：非常小的比率/缺失率保留 4 位
    col_4dp = [c for c in num_cols if "rate" in c.lower()]  # e.g. max_missing_rate
    if col_4dp:
        last[col_4dp] = last[col_4dp].round(4)
    # 若有明確欄位名也可指定
    for c in ["psi_max_value"]:
        if c in last.columns:
            last[c] = last[c].round(3)

    # 解析 top_drift_json 並限制 psi 為 3 位小數
    if "top_drift_json" in last.columns:
        try:
            # 將字符串解析為 JSON 列表
            drift_data = json.loads(last["top_drift_json"].iloc[0])
            # 對 PSI 值進行 3 位小數處理
            for item in drift_data:
                if "psi" in item:
                    item["psi"] = round(float(item["psi"]), 3)
            last["top_drift_json"] = [drift_data]  # 存為列表
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse top_drift_json: {e}")
            last["top_drift_json"] = last["top_drift_json"]  # 保留原始字符串

    # 轉為字典並格式化數值為 3 位小數
    last_rec = last.to_dict(orient="records")[0]
    for key, value in last_rec.items():
        if isinstance(value, (int, float)):
            last_rec[key] = round(value, 3)
        elif key == "top_drift_json" and isinstance(value, list):
            for item in value[0]:  # 因為 value 是 [drift_data]
                if "psi" in item:
                    item["psi"] = round(float(item["psi"]), 3)

    # 顯示摘要
    st.subheader("Latest batch summary:")
    st.json(last_rec)
else:
    st.info("Run the monitor to generate batch_metrics_log.csv.")
