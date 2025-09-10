# pages/2_New_Batch.py

import os
import pandas as pd
import streamlit as st
import subprocess
import sys
import json

# 設定頁面標題
st.title("New Batch Processing")

# 定義路徑
BATCH_PATH = "monitor/X_new.csv"
MONITOR_LOG = "monitor/batch_metrics_log.csv"
os.makedirs("monitor", exist_ok=True)

# 上傳 CSV -> 保存為 monitor/X_new.csv
st.header("Upload New Batch")
up = st.file_uploader("Upload a new batch CSV file", type="csv", key="batch_uploader")
if up:
    try:
        df = pd.read_csv(up)
        df.to_csv(BATCH_PATH, index=False)
        st.success(f"Saved → {BATCH_PATH}")
    except Exception as e:
        st.error(f"Failed to save CSV: {e}")

# 運行 monitor
st.header("Run Monitor")
if st.button("Run Monitor"):
    if not os.path.exists(BATCH_PATH):
        st.error("monitor/X_new.csv not found – please upload a batch first.")
    else:
        with st.spinner("Running monitor…"):
            res = subprocess.run(
                [sys.executable, "monitor/monitor_1.py", "--batch_csv", BATCH_PATH],
                capture_output=True, text=True
            )
            st.code(res.stdout or "(no stdout)")
            if res.returncode == 0:
                st.success("Monitor executed successfully")
                st.rerun()  # 刷新以顯示最新結果
            else:
                st.error(f"Monitor failed: {res.stderr}")

# 顯示最新批次摘要
st.header("Latest Batch Summary")
if os.path.exists(MONITOR_LOG) and os.path.getsize(MONITOR_LOG) > 0:
    df = pd.read_csv(MONITOR_LOG, encoding="utf-8-sig")  # 處理可能的 BOM
    last = df.tail(1).copy()

    # 依欄位型態/名稱設定小數位 (統一為 3 位，除非特別指定)
    num_cols = last.select_dtypes(include="number").columns.tolist()
    last[num_cols] = last[num_cols].round(3)
    col_4dp = [c for c in num_cols if "rate" in c.lower()]  # e.g. max_missing_rate
    if col_4dp:
        last[col_4dp] = last[col_4dp].round(4)  # 保留 4 位給比率/缺失率
    for c in ["psi_max_value"]:
        if c in last.columns:
            last[c] = last[c].round(3)

    # 解析 top_drift_json 並限制 psi 為 3 位小數
    if "top_drift_json" in last.columns:
        try:
            raw_json = last["top_drift_json"].iloc[0]
            st.write(f"Debug: Raw top_drift_json = {raw_json}")  # 除錯用
            drift_data = json.loads(raw_json)
            if not isinstance(drift_data, list):
                st.warning("top_drift_json is not a list. Using empty list.")
                drift_data = []
            for item in drift_data:
                if isinstance(item, dict):  # 確保 item 是字典
                    st.write(f"Debug: Processing item = {item}")  # 除錯用
                    if "psi" in item and item["psi"] is not None:
                        try:
                            psi_value = float(item["psi"])  # 嘗試轉換為浮點數
                            item["psi"] = round(psi_value, 3)  # 限制為 3 位小數
                        except (ValueError, TypeError) as e:
                            st.warning(f"Invalid psi value in {item}: {e}. Setting to 0.0.")
                            item["psi"] = 0.0  # 後備值
                    else:
                        st.warning(f"Missing or null psi in {item}. Setting to 0.0.")
                        item["psi"] = 0.0  # 後備值
                else:
                    st.warning(f"Invalid item type in drift_data: {item}. Skipping.")
            last["top_drift_json"] = [drift_data]
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in top_drift_json: {e}")
            last["top_drift_json"] = [[]]

    # 轉為字典並格式化數值為 3 位小數
    last_rec = last.to_dict(orient="records")[0]
    for key, value in last_rec.items():
        if isinstance(value, (int, float)):
            last_rec[key] = round(value, 3)
        elif key == "top_drift_json" and isinstance(value, list):
            for item in value[0]:
                if "psi" in item:
                    item["psi"] = round(float(item["psi"]), 3)

    # 顯示摘要
    st.subheader("Latest Batch Details:")
    st.json(last_rec)
else:
    st.info("Run the monitor to generate batch_metrics_log.csv or ensure the file exists.")

# 可選：顯示上傳的原始數據（除錯用）
if os.path.exists(BATCH_PATH):
    st.subheader("Uploaded Batch Data:")
    uploaded_df = pd.read_csv(BATCH_PATH)
    st.dataframe(uploaded_df)
