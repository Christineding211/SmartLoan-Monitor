import pandas as pd
import numpy as np
import streamlit as st

# ---------- helpers ----------
def recover_annual_inc(s_log: pd.Series) -> pd.Series:
    rec = np.expm1(s_log)
    return np.exp(s_log) if (rec < 0).any() else rec

def make_state_group(s: pd.Series, min_count: int = 200) -> pd.Series:
    s = s.astype(str).fillna("UNKNOWN")
    vc = s.value_counts()
    keep = set(vc[vc >= min_count].index)
    return s.where(s.isin(keep), "OTHER")

def simple_fairness(df, y_true, y_score, group_col, target_approval=0.40):
    d = df[[y_true, y_score, group_col]].dropna().copy()
    cutoff = float(np.quantile(d[y_score].values, target_approval))
    d["approve"] = (d[y_score] <= cutoff).astype(int)

    rows = []
    for g, gdf in d.groupby(group_col):
        n = len(gdf)
        ar = gdf["approve"].mean() if n else np.nan
        good = (gdf[y_true] == 0)
        tprg = gdf.loc[good, "approve"].mean() if good.sum() > 0 else np.nan
        rows.append({"group": g, "n": n, "approve_rate": ar, "TPR_good": tprg})
    per = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    base = per.loc[per["n"].idxmax(), "approve_rate"]
    per["DIR_vs_majority"] = per["approve_rate"] / base if base > 0 else np.nan

    dir_min_over_max = per["approve_rate"].min() / per["approve_rate"].max()
    tpr_range = (per["TPR_good"].max() - per["TPR_good"].min()) if per["TPR_good"].notna().any() else np.nan

    summary = {
        "used_cutoff": cutoff,
        "DIR_min_over_max": float(dir_min_over_max),
        "TPR_good_range": float(tpr_range) if pd.notna(tpr_range) else np.nan,
        "pass_80_rule": bool(dir_min_over_max >= 0.8),
    }
    return per, summary

def group_cutoff_fairness(df, y_true, y_score, group_col, target_approval=0.40):
    d = df[[y_true, y_score, group_col]].dropna().copy()
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
        rows.append({"group": g, "n": n, "approve_rate": ar, "TPR_good": tprg})
    per = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    base = per.loc[per["n"].idxmax(), "approve_rate"]
    per["DIR_vs_majority"] = per["approve_rate"] / base if base > 0 else np.nan

    dir_min_over_max = per["approve_rate"].min() / per["approve_rate"].max()
    tpr_range = (per["TPR_good"].max() - per["TPR_good"].min()) if per["TPR_good"].notna().any() else np.nan

    summary = {
        "used_cutoffs": "group-specific",
        "DIR_min_over_max": float(dir_min_over_max),
        "TPR_good_range": float(tpr_range) if pd.notna(tpr_range) else np.nan,
        "pass_80_rule": bool(dir_min_over_max >= 0.8),
    }
    return per, summary

def kpi_label(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"

# ---------- UI ----------
st.title("Fairness Summary (Before vs After)")

df = pd.read_csv("monitor/test_fairness_features.csv", index_col=0)

group_options = [c for c in ["income_group_auth", "state_group"] if c in df.columns]
group_col = st.selectbox("Group attribute", group_options,
                         format_func=lambda x: "Income group" if x=="income_group_auth" else "State (grouped)")

target = st.slider("Target approval rate", 0.05, 0.95, 0.40, 0.01)

# ---------- compute ----------
per_b, sum_b = simple_fairness(df, "loan_status", "pd_score", group_col, target)
per_a, sum_a = group_cutoff_fairness(df, "loan_status", "pd_score", group_col, target)

# ---------- KPI cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("80% rule (Before)", kpi_label(sum_b["pass_80_rule"]), f"DIR={sum_b['DIR_min_over_max']:.3f}")
c2.metric("80% rule (After)",  kpi_label(sum_a["pass_80_rule"]), f"DIR={sum_a['DIR_min_over_max']:.3f}")
c3.metric("TPR_good range (Before)", f"{sum_b['TPR_good_range']:.3f}")
c4.metric("TPR_good range (After)",  f"{sum_a['TPR_good_range']:.3f}")

st.caption(f"Policy: Global cutoff vs Group-specific cutoffs | Target approval = {target:.0%}")

# ---------- per-group tables ----------
tab1, tab2 = st.tabs(["Before (global cutoff)", "After (group cutoffs)"])
with tab1:
    st.dataframe(per_b, use_container_width=True)
with tab2:
    st.dataframe(per_a, use_container_width=True)

# ---------- one key chart: Approval rate ----------
st.subheader("Approval rate by group (Before vs After)")
chart_df = pd.concat(
    [
        per_b[["group", "approve_rate"]].assign(Stage="Before"),
        per_a[["group", "approve_rate"]].assign(Stage="After"),
    ],
    ignore_index=True,
)
st.bar_chart(chart_df.pivot(index="group", columns="Stage", values="approve_rate"))

# ---------- summary table + download ----------
summary_df = pd.DataFrame([
    {"stage": "Before", "policy": "global_cutoff",
     "target_approval": target,
     "pass_80_rule": sum_b["pass_80_rule"],
     "DIR_min_over_max": round(sum_b["DIR_min_over_max"], 6),
     "TPR_good_range": round(sum_b["TPR_good_range"], 6)},
    {"stage": "After", "policy": "group_cutoffs",
     "target_approval": target,
     "pass_80_rule": sum_a["pass_80_rule"],
     "DIR_min_over_max": round(sum_a["DIR_min_over_max"], 6),
     "TPR_good_range": round(sum_a["TPR_good_range"], 6)},
])
st.subheader("Fairness summary table")
st.dataframe(summary_df, use_container_width=True)
st.download_button(
    "Download fairness_summary.csv",
    summary_df.to_csv(index=False).encode("utf-8"),
    "fairness_summary.csv",
)
