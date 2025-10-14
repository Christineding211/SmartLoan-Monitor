

"""
Agent tools - wraps existing functionality for LangChain
"""
from langchain_core.tools import tool
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@tool
def run_new_batch_processing(csv_path: str) -> dict:
    """
    Process a new batch of data through the monitoring pipeline.
    Calculates PSI, performance metrics, and logs results.
    """
    # TODO: Extract logic from your 2_New_Batch.py
    # For now, return placeholder
    return {"status": "processed", "message": f"Processed {csv_path}"}


@tool
def check_drift_metrics() -> dict:
    """
    Check current drift metrics (PSI) for all features.
    Returns top drifted features and severity.
    """
    # TODO: Extract logic from your 3_Metrics_and_Drift.py
    return {"status": "checked", "drifted_features": 2}


@tool
def run_fairness_audit(protected_attr: str = "gender") -> dict:
    """
    Run fairness analysis on specified protected attribute.
    Calculates disparate impact, 80% rule compliance.
    """
    # TODO: Extract logic from your 5_Faireness_1.py
    return {"status": "audited", "protected_attr": protected_attr}


@tool
def generate_compliance_report() -> dict:
    """
    Generate comprehensive compliance report.
    """
    from smartloan_agent.agent_fs import SmartLoanAgentFS
    agent = SmartLoanAgentFS()
    metrics, report_path = agent.run()
    return {"metrics": metrics, "report_path": report_path}


@tool
def get_current_metrics() -> dict:
    """
    Get current model performance metrics (AUC, KS, PSI).
    """
    try:
        df = pd.read_csv("monitor/batch_metrics_log.csv").tail(1)
        return df.iloc[0].to_dict()
    except Exception as e:
        return {"error": str(e)}