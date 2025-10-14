import os, glob
import streamlit as st

st.title("Explainability")

# --- newest compliance report (by file mtime) ---
mds = sorted(glob.glob("reports/compliance_report_*.md"), key=os.path.getmtime, reverse=True)
if mds:
    latest = mds[0]
    st.caption(f"Showing: {os.path.basename(latest)}")
    with open(latest, "r", encoding="utf-8") as f:
        st.markdown(f.read())
else:
    st.info("No compliance report found in /reports.")

st.divider()

# --- SHAP images in a sensible order ---
order = ["bar", "waterfall", "beeswarm"]
imgs = [p for p in glob.glob("reports/*shap*.png")]

if imgs:
    for kw in order:
        for fp in imgs:
            if kw in os.path.basename(fp).lower():
                # 用 columns() 置中
                col1, col2, col3 = st.columns([1,2,1])  # 左中右欄位
                with col2:  # 把圖放在中間欄位
                    st.image(fp, caption=os.path.basename(fp), width=600)  # 控制寬度
else:
    st.info("No SHAP images found.")
