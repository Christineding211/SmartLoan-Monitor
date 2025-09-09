
# pages/3_Metrics_and_Drift.py
import os, yaml
import pandas as pd
import streamlit as st
import altair as alt

st.title("Metrics & Drift")

# --- 讀取門檻（若無 config 就用預設） ---
cfg_path = "monitor/config.yaml"
psi_warn, psi_alert = 0.10, 0.20
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
        psi_warn = float(cfg.get("psi_threshold_warn", psi_warn))
        psi_alert = float(cfg.get("psi_threshold_alert", psi_alert))
except Exception:
    pass

p = "monitor/metrics_log.csv"
if not os.path.exists(p):
    st.warning("metrics_log.csv not found. Please run a batch first.")
    st.stop()

df = pd.read_csv(p)

# 只處理有 PSI 的行（通常是數值特徵）；其餘留給 Missing 檢視
df["psi"] = pd.to_numeric(df.get("psi"), errors="coerce")
df_num = df[df["psi"].notna()].copy()

# Level 分級
def label_level(x: float) -> str:
    if pd.isna(x): return ""
    if x <= psi_warn: return "OK"
    if x <= psi_alert: return "WARN"
    return "ALERT"

df_num["level"] = df_num["psi"].apply(label_level)

# --- Summary 卡片 ---
ok_n = int((df_num["level"] == "OK").sum())
warn_n = int((df_num["level"] == "WARN").sum())
alert_n = int((df_num["level"] == "ALERT").sum())

c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Features (with PSI)", len(df_num))
c2.metric("OK", ok_n)
c3.metric("WARN", warn_n)
c4.metric("ALERT", alert_n)

st.caption(f"Rules (from config): OK ≤ {psi_warn:.2f}, WARN {psi_warn:.2f}–{psi_alert:.2f}, ALERT > {psi_alert:.2f}")

# --- Tabs：Top 漂移圖 / 明細表 / 遺漏率 ---
tab1, tab2, tab3 = st.tabs(["Top drift (chart)", "Details", "Missingness"])


with tab1:
    n_feats = int(len(df_num))
    if n_feats == 0:
        st.info("No numerical features with PSI available in this batch.")
        st.stop()

    # 預設值可從 config 取，沒有就用 10，再夾在 [1, n_feats]
    top_k_default = int(cfg.get("ui", {}).get("top_k_default", 10)) if "cfg" in locals() else 10
    topk = st.slider(
        "Top K by PSI",
        min_value=1,
        max_value=n_feats,
        value=max(1, min(top_k_default, n_feats)),
        step=1
    )

    top_df = (df_num.sort_values("psi", ascending=False)
                    .head(topk)[["feature","psi","level"]])

    # 顏色分級：OK=綠、WARN=橙、ALERT=紅
    colour_scale = alt.Scale(
        domain=["OK", "WARN", "ALERT"],
        range=["#7ddf82", "#f4b96d", "#f57f7f"]
    )

    base = alt.Chart(top_df).encode(
        x=alt.X("feature:N", sort="-y", title=None),
        y=alt.Y("psi:Q", title="PSI"),
        color=alt.Color("level:N", scale=colour_scale, legend=alt.Legend(title="Level"))
    )

    bars = base.mark_bar()
    warn_rule = alt.Chart(pd.DataFrame({"y": [psi_warn]})).mark_rule(strokeDash=[4,4], color="#ff8f00").encode(y="y:Q")
    alert_rule = alt.Chart(pd.DataFrame({"y": [psi_alert]})).mark_rule(strokeDash=[4,4], color="#c62828").encode(y="y:Q")

    st.altair_chart((bars + warn_rule + alert_rule).properties(height=320), use_container_width=True)

    st.dataframe(top_df, use_container_width=True)
    st.caption(f"(PSI has been calculated for {n_feats} features in total)")




with tab2:
    show_cols = ["feature","psi","level","missing_rate_new","mean_diff"]
    st.dataframe(
        df_num.sort_values("psi", ascending=False)[show_cols],
        use_container_width=True
    )

with tab3:
    # 針對所有特徵（含無 PSI 的）看缺失
    mcol = "missing_rate_new"
    if mcol in df.columns:
        m = df[[ "feature", mcol ]].copy()
        m[mcol] = pd.to_numeric(m[mcol], errors="coerce").fillna(0.0)
        m = m.sort_values(mcol, ascending=False)
        st.dataframe(m.head(30), use_container_width=True)
    else:
        st.info("No missing_rate_new column found in metrics_log.csv")

# 下載原檔
st.download_button("Download metrics_log.csv", data=open(p, "rb"), file_name="metrics_log.csv")

