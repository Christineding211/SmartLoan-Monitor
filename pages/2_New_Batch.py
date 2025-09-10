# pages/2_New_Batch.py
import os, sys, subprocess, json
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

    # 1) 所有數值欄位統一 3 位小數
    last = last.round(3)

    # 2) 安全處理 top_drift_json（把 psi 改為 3 位小數）
    def _safe_float(x):
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        try:
            return float(str(x).strip())
        except Exception:
            return None

    def _round_psi_in_drift(value):
        """value 可能是字串(JSON)或 list[dict]，將其中 'psi' 轉為 3 位小數"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value

        # 轉成 Python 物件
        obj = value
        if isinstance(value, str):
            try:
                obj = json.loads(value)
            except Exception:
                return value  # 不是合法 JSON，維持原樣

        # 只處理 list[dict] 形態
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict) and "psi" in it:
                    f = _safe_float(it["psi"])
                    if f is not None:
                        it["psi"] = round(f, 3)
            return obj  # 保持為物件，st.json 會比較好讀
        return value

    if "top_drift_json" in last.columns:
        last.loc[last.index[-1], "top_drift_json"] = _round_psi_in_drift(
            last.loc[last.index[-1], "top_drift_json"]
        )

    # 3) 輸出
    last_rec = last.to_dict(orient="records")[0]
    st.subheader("Latest batch summary:")
    st.json(last_rec)
else:
    st.info("Run the monitor to generate batch_metrics_log.csv.")
