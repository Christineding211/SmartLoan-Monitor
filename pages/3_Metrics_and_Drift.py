# pages/3_Metrics_and_Drift.py

# pages/3_Metrics_and_Drift.py

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import smartloan_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from smartloan_agent.agent_fs import SmartLoanAgentFS

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Metrics & Drift | SmartLoan",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Metrics & Drift Monitor")
st.markdown("---")

# ===== CACHED DATA LOADING =====
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_latest_metrics():
    """Load and compute latest governance metrics."""
    agent = SmartLoanAgentFS()
    return agent.run()

# ===== MAIN CONTENT =====
try:
    with st.spinner("Loading latest metrics..."):
        metrics, report_path = load_latest_metrics()
    
    # Overall Status Banner
    status = metrics['overall']
    status_colors = {
        "OK": "🟢",
        "WARN": "🟡", 
        "ALERT": "🔴"
    }
    st.success(f"{status_colors.get(status, '⚪')} **Overall Status: {status}**")
    
    # ===== METRICS GRID =====
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="PSI (Max Feature)",
            value=f"{metrics['psi']:.4f}" if metrics['psi'] is not None else "N/A",
            delta=metrics['psi_level'],
            delta_color="off"
        )
    
    with col2:
        st.metric(
            label="AUC",
            value=f"{metrics['auc']:.3f}" if metrics['auc'] is not None else "N/A",
            delta=f"Drop: {metrics['auc_drop']:.3f}" if metrics['auc_drop'] is not None else "N/A",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="KS Statistic",
            value=f"{metrics['ks']:.3f}" if metrics['ks'] is not None else "N/A",
            delta=f"Drop: {metrics['ks_drop']:.3f}" if metrics['ks_drop'] is not None else "N/A",
            delta_color="inverse"
        )
    
    with col4:
        fairness_pass = metrics['fairness'].get('pass_80_rule')
        fairness_status = "✅ Pass" if fairness_pass else "❌ Fail" if fairness_pass is False else "N/A"
        st.metric(
            label="Fairness (80% Rule)",
            value=fairness_status,
            delta=metrics['fairness_level'],
            delta_color="off"
        )
    
    st.markdown("---")
    
    # ===== DETAILED STATUS =====
    st.subheader("📋 Detailed Status")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Drift & Performance")
        
        # PSI Status
        psi_color = {"OK": "🟢", "WARN": "🟡", "ALERT": "🔴"}.get(metrics['psi_level'], "⚪")
        st.markdown(f"**PSI**: {psi_color} {metrics['psi_level']}")
        if metrics['psi'] is not None:
            st.caption(f"Value: {metrics['psi']:.4f}")
        
        # AUC Status
        auc_color = {"OK": "🟢", "ALERT": "🔴"}.get(metrics['auc_level'], "⚪")
        st.markdown(f"**AUC**: {auc_color} {metrics['auc_level']}")
        if metrics['auc'] is not None:
            st.caption(f"Current: {metrics['auc']:.3f} | Drop: {metrics['auc_drop']:.3f}" if metrics['auc_drop'] else f"Current: {metrics['auc']:.3f}")
        
        # KS Status
        ks_color = {"OK": "🟢", "ALERT": "🔴"}.get(metrics['ks_level'], "⚪")
        st.markdown(f"**KS**: {ks_color} {metrics['ks_level']}")
        if metrics['ks'] is not None:
            st.caption(f"Current: {metrics['ks']:.3f} | Drop: {metrics['ks_drop']:.3f}" if metrics['ks_drop'] else f"Current: {metrics['ks']:.3f}")
    
    with col_b:
        st.markdown("### Fairness & Governance")
        
        # Fairness Status
        fairness_color = {"OK": "🟢", "ALERT": "🔴", "N/A": "⚪"}.get(metrics['fairness_level'], "⚪")
        st.markdown(f"**Fairness (80% Rule)**: {fairness_color} {metrics['fairness_level']}")
        
        # Report Info
        st.markdown("**Generated Report**")
        st.caption(f"📄 `{report_path}`")
        st.caption(f"🕒 {metrics['timestamp']}")
    
    st.markdown("---")
    
    # ===== SOURCE FILES INFO =====
    with st.expander("🔍 Source Files & Data"):
        st.json(metrics['source_files'])
    
    # ===== REFRESH BUTTON =====
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

except FileNotFoundError as e:
    st.error(f"❌ **File Not Found**: {e}")
    st.info("💡 Please run **New Batch → Run Monitor** first to generate metrics.")

except Exception as e:
    st.error(f"❌ **Error Loading Metrics**: {e}")
    st.exception(e)  # Show full traceback for debugging
    
    with st.expander("🐛 Debug Info"):
        st.write("**Python Path:**")
        st.code("\n".join(sys.path))
        st.write("**Current Working Directory:**")
        st.code(str(Path.cwd()))