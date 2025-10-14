import pandas as pd
import numpy as np
import streamlit as st

# DEBUG: Check for NaN in data
st.set_page_config(
    page_title="Fairness Analysis | SmartLoan",
    page_icon="⚖️",
    layout="wide"
)



# ---------- Helper Functions ----------
def recover_annual_inc(s_log: pd.Series) -> pd.Series:
    """Recover annual income from log-transformed values."""
    rec = np.expm1(s_log)
    return np.exp(s_log) if (rec < 0).any() else rec

def make_state_group(s: pd.Series, min_count: int = 200) -> pd.Series:
    """Group low-frequency states into 'OTHER' category."""
    s = s.astype(str).fillna("UNKNOWN")
    vc = s.value_counts()
    keep = set(vc[vc >= min_count].index)
    return s.where(s.isin(keep), "OTHER")

def simple_fairness(df, y_true, y_score, group_col, target_approval=0.40):
    """
    Compute fairness metrics using a GLOBAL cutoff (same for all groups).
    Returns per-group stats and summary with 80% rule compliance.
    """
    d = df[[y_true, y_score, group_col]].dropna().copy()
    
    # Global cutoff: approve top X% based on lowest risk scores
    cutoff = float(np.quantile(d[y_score].values, target_approval))
    d["approve"] = (d[y_score] <= cutoff).astype(int)

    rows = []
    for g, gdf in d.groupby(group_col):
        n = len(gdf)
        ar = gdf["approve"].mean() if n else np.nan
        
        # TPR for "good" customers (y_true == 0)
        good = (gdf[y_true] == 0)
        tprg = gdf.loc[good, "approve"].mean() if good.sum() > 0 else np.nan
        
        rows.append({
            "group": g, 
            "n": n, 
            "approve_rate": ar, 
            "TPR_good": tprg
        })
    
    per = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    # Disparate Impact Ratio (DIR) vs majority group
    base = per.loc[per["n"].idxmax(), "approve_rate"]
    per["DIR_vs_majority"] = per["approve_rate"] / base if base > 0 else np.nan

    # 80% rule: min/max approval rate ratio
    dir_min_over_max = per["approve_rate"].min() / per["approve_rate"].max()
    
    # TPR range (fairness across groups)
    tpr_range = (
        (per["TPR_good"].max() - per["TPR_good"].min()) 
        if per["TPR_good"].notna().any() 
        else np.nan
    )

    summary = {
        "used_cutoff": cutoff,
        "DIR_min_over_max": float(dir_min_over_max),
        "TPR_good_range": float(tpr_range) if pd.notna(tpr_range) else np.nan,
        "pass_80_rule": bool(dir_min_over_max >= 0.8),
    }
    return per, summary

def group_cutoff_fairness(df, y_true, y_score, group_col, target_approval=0.40):
    """
    Compute fairness metrics using GROUP-SPECIFIC cutoffs (equalizing approval rates).
    Returns per-group stats and summary with 80% rule compliance.
    """
    d = df[[y_true, y_score, group_col]].dropna().copy()
    
    # Calculate group-specific cutoffs to achieve target approval rate in each group
    cuts = d.groupby(group_col)[y_score].quantile(target_approval).to_dict()

    d["approve_fair"] = 0
    for g, c in cuts.items():
        d.loc[(d[group_col] == g) & (d[y_score] <= c), "approve_fair"] = 1

    rows = []
    for g, gdf in d.groupby(group_col):
        n = len(gdf)
        ar = gdf["approve_fair"].mean() if n else np.nan
        
        good = (gdf[y_true] == 0)
        tprg = gdf.loc[good, "approve_fair"].mean() if good.sum() > 0 else np.nan
        
        rows.append({
            "group": g, 
            "n": n, 
            "approve_rate": ar, 
            "TPR_good": tprg
        })
    
    per = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    base = per.loc[per["n"].idxmax(), "approve_rate"]
    per["DIR_vs_majority"] = per["approve_rate"] / base if base > 0 else np.nan

    dir_min_over_max = per["approve_rate"].min() / per["approve_rate"].max()
    tpr_range = (
        (per["TPR_good"].max() - per["TPR_good"].min()) 
        if per["TPR_good"].notna().any() 
        else np.nan
    )

    summary = {
        "used_cutoffs": "group-specific",
        "DIR_min_over_max": float(dir_min_over_max),
        "TPR_good_range": float(tpr_range) if pd.notna(tpr_range) else np.nan,
        "pass_80_rule": bool(dir_min_over_max >= 0.8),
    }
    return per, summary

def kpi_label(ok: bool) -> str:
    """Format pass/fail status with emoji."""
    return "✅ PASS" if ok else "❌ FAIL"

# ===== LOAD DATA WITH CACHING =====
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_fairness_data():
    """Load fairness test data with caching."""
    try:
        df = pd.read_csv("monitor/test_fairness_features.csv", index_col=0)
        return df
    except FileNotFoundError:
        st.error("❌ File not found: `monitor/test_fairness_features.csv`")
        st.info("Please ensure the fairness test data is available.")
        st.stop()

# ===== MAIN UI =====
st.title("⚖️ Fairness Analysis (Before vs After)")
st.markdown("""
Compare fairness metrics between:
- **Before**: Global cutoff policy (same threshold for all groups)
- **After**: Group-specific cutoffs (equalizing approval rates)

This demonstrates compliance with the **80% rule** (Disparate Impact Ratio ≥ 0.80).
""")
st.markdown("---")

# Load data
df = load_fairness_data()

# ===== CONTROLS =====
col_control1, col_control2 = st.columns([2, 1])

with col_control1:
    group_options = [c for c in ["income_group_auth", "state_group"] if c in df.columns]
    
    if not group_options:
        st.error("❌ No group columns found in data. Expected: 'income_group_auth' or 'state_group'")
        st.stop()
    
    group_col = st.selectbox(
        "📊 Group attribute",
        group_options,
        format_func=lambda x: "Income Group" if x == "income_group_auth" else "State (Grouped)"
    )

with col_control2:
    target = st.slider(
        "🎯 Target approval rate",
        min_value=5,
        max_value=95,
        value=40,
        step=1,
        format="%d%%"   # ✅ valid
    ) / 100

st.caption(f"**Policy comparison**: Global cutoff vs Group-specific cutoffs | Target approval = {target:.0%}")

# ===== COMPUTE FAIRNESS METRICS =====
with st.spinner("Computing fairness metrics..."):
    per_b, sum_b = simple_fairness(df, "loan_status", "pd_score", group_col, target)
    per_a, sum_a = group_cutoff_fairness(df, "loan_status", "pd_score", group_col, target)

# ===== DATA QUALITY WARNING =====
small_groups = per_b[per_b["n"] < 30]["group"].tolist()
if small_groups:
    st.warning(f"""
    ⚠️ **Small Sample Warning**: The following groups have fewer than 30 samples: 
    **{', '.join(str(g) for g in small_groups)}**
    
    Fairness metrics for these groups may not be statistically reliable. 
    Consider collecting more data or grouping categories.
    """)
    
missing_data_groups = per_b[per_b["TPR_good"].isna()]["group"].tolist()
if missing_data_groups:
    st.info(f"""
    ℹ️ **Missing Data**: Some metrics show "N/A" for groups: 
    **{', '.join(str(g) for g in missing_data_groups)}**
    
    This typically means insufficient data for that calculation.
    """)

# ===== KPI CARDS =====
st.subheader("📊 Key Fairness Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "80% Rule (Before)",
        kpi_label(sum_b["pass_80_rule"]),
        f"DIR={sum_b['DIR_min_over_max']:.3f}",
        delta_color="off"
    )

with col2:
    st.metric(
        "80% Rule (After)",
        kpi_label(sum_a["pass_80_rule"]),
        f"DIR={sum_a['DIR_min_over_max']:.3f}",
        delta_color="off"
    )

with col3:
    st.metric(
        "TPR Range (Before)",
        f"{sum_b['TPR_good_range']:.3f}",
        help="Range of True Positive Rates for 'good' customers across groups"
    )

with col4:
    st.metric(
        "TPR Range (After)",
        f"{sum_a['TPR_good_range']:.3f}",
        delta=f"{sum_a['TPR_good_range'] - sum_b['TPR_good_range']:.3f}",
        delta_color="inverse",
        help="Lower is better (more consistent treatment)"
    )

st.markdown("---")

# ===== PER-GROUP DETAILS =====
st.subheader("📋 Per-Group Breakdown")

tab1, tab2 = st.tabs(["🔴 Before (Global Cutoff)", "🟢 After (Group Cutoffs)"])

with tab1:
    st.caption(f"**Policy**: Single global cutoff at score ≤ {sum_b['used_cutoff']:.4f}")
    
    # Safe formatting: handle NaN/None values
    display_per_b = per_b.copy()
    for col in ["approve_rate", "TPR_good"]:
        if col in display_per_b.columns:
            display_per_b[col] = display_per_b[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
    if "DIR_vs_majority" in display_per_b.columns:
        display_per_b["DIR_vs_majority"] = display_per_b["DIR_vs_majority"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    
    st.dataframe(display_per_b, use_container_width=True)

with tab2:
    st.caption("**Policy**: Group-specific cutoffs to equalize approval rates")
    
    # Safe formatting: handle NaN/None values
    display_per_a = per_a.copy()
    for col in ["approve_rate", "TPR_good"]:
        if col in display_per_a.columns:
            display_per_a[col] = display_per_a[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
    if "DIR_vs_majority" in display_per_a.columns:
        display_per_a["DIR_vs_majority"] = display_per_a["DIR_vs_majority"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    
    st.dataframe(display_per_a, use_container_width=True)

# ===== VISUALIZATION =====
st.subheader("📊 Approval Rate Comparison")

chart_df = pd.concat(
    [
        per_b[["group", "approve_rate"]].assign(Stage="Before"),
        per_a[["group", "approve_rate"]].assign(Stage="After"),
    ],
    ignore_index=True,
)

# Convert to percentage for better readability
chart_df["approve_rate"] = chart_df["approve_rate"] * 100

st.bar_chart(
    chart_df.pivot(index="group", columns="Stage", values="approve_rate"),
    height=400
)
st.caption("Y-axis: Approval rate (%)")

# ===== SUMMARY TABLE =====
st.markdown("---")
st.subheader("📄 Fairness Summary Report")

summary_df = pd.DataFrame([
    {
        "stage": "Before",
        "policy": "global_cutoff",
        "target_approval": target,
        "pass_80_rule": sum_b["pass_80_rule"],
        "DIR_min_over_max": round(sum_b["DIR_min_over_max"], 6),
        "TPR_good_range": round(sum_b["TPR_good_range"], 6)
    },
    {
        "stage": "After",
        "policy": "group_cutoffs",
        "target_approval": target,
        "pass_80_rule": sum_a["pass_80_rule"],
        "DIR_min_over_max": round(sum_a["DIR_min_over_max"], 6),
        "TPR_good_range": round(sum_a["TPR_good_range"], 6)
    },
])

st.dataframe(summary_df, use_container_width=True)

# ===== DOWNLOAD BUTTON =====
col_dl1, col_dl2 = st.columns([1, 4])
with col_dl1:
    st.download_button(
        "💾 Download Summary CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        "fairness_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

# ===== INTERPRETATION GUIDE =====
with st.expander("📖 How to Interpret These Metrics"):
    st.markdown("""
    ### Key Metrics Explained
    
    **80% Rule (Disparate Impact Ratio)**
    - Measures if approval rates are similar across groups
    - **Pass**: DIR ≥ 0.80 (min approval rate ≥ 80% of max approval rate)
    - **Fail**: DIR < 0.80 (indicates potential adverse impact)
    
    **TPR_good Range**
    - True Positive Rate for "good" (non-default) customers
    - Lower range = more consistent treatment across groups
    - Measures quality of service fairness
    
    **DIR_vs_majority**
    - Each group's approval rate compared to the largest group
    - Values < 0.80 indicate potential discrimination
    
    ### Regulatory Context (UK FCA)
    - The 80% rule comes from US EEOC guidance but is widely used
    - FCA expects firms to monitor for **adverse outcomes** across protected characteristics
    - Group-specific cutoffs can help achieve fairness but require justification
    """)

st.markdown("---")
st.caption("⚖️ **SmartLoan Fairness Monitor** | Built with Streamlit | UK FCA Compliance")