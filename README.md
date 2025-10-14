# SmartLoan – Credit-Risk Prediction, Governance & AI-Powered Monitoring Dashboard

A lightweight, end-to-end credit-risk prediction and **autonomous monitoring system** built with Streamlit, simulating how UK lenders manage **IFRS 9 / PRA SS1/23 / FCA Consumer Duty** model governance with AI-driven automation.
It demonstrates not only predictive modelling (Logistic Regression & XGBoost) but also **responsible AI practices** enhanced by an **intelligent monitoring agent**: automated drift tracking (PSI), fairness testing, explainability (SHAP), real-time alerting, and compliance-style reporting for executives.


## 🚀Demo: Try live demo here: [SmartLoan Dashboard](https://smartloan-dashboard.streamlit.app)

## ❓ What problem does this solve?
Financial institutions face both decision-making and regulatory risks if their credit-risk models are **not accurate, monitored, explainable, and fair**. 

- Predictions can silently degrade when **data drifts** or key inputs go missing — the AI Agent now **detects and alerts** these issues automatically.  
- Lack of transparent reporting to management and regulators — the Agent **auto-generates compliance reports** aligned to FCA/PRA expectations.  
- Risk of **unfair lending outcomes** across customer groups (e.g., income segments, geography) — the Agent continuously **audits fairness** using the 80% rule.  
- Teams often operate in isolation, lacking a central governance process — the Agent provides **centralised, automated oversight** and alerting.

## 👉 How this aligns with real credit-risk workflows
This project mirrors how **modern, automated credit-risk teams** operate in financial institutions:  

- **Model Development:** Default prediction using Logistic Regression (scorecard-style) and XGBoost (non-linear benchmark), tracked with MLflow for version control.  
- **Automated Monitoring:** The AI Agent performs **continuous PSI-based drift detection**, anomaly monitoring, and model performance tracking,generating monthly governance packs without manual effort.  
- **Fairness & Consumer Duty Compliance:** **80%-rule fairness checks** assess equal outcomes across customer segments and flag potential Consumer Duty breaches.  
- **Explainability & Validation:** SHAP-based feature interpretation with agent-generated summaries supports internal model validation and audits.  
- **Regulatory Reporting:** Agent automatically produces **FCA/PRA-style compliance documentation** covering drift, fairness, and performance metrics for governance reviews.  



## Key features
### 1. New Batch / Monitoring
Upload a new CSV batch and instantly compute metrics (AUC/KS), drift (PSI), and missingness. Provides a one-stop health check for model monitoring.

<img width="707" height="478" alt="image" src="https://github.com/user-attachments/assets/fafc67d3-2b73-4481-819f-c9e0f7885f3d" />


### 2.Metrics & Drift
Visualises top-K drifted features with PSI, using OK/WARN/ALERT thresholds from config.yaml. Helps spot early warning signs before model degradation.

<img width="652" height="657" alt="image" src="https://github.com/user-attachments/assets/d753ac82-3005-4fd7-919d-a3a1051358c5" />



### 3. Explainability
Generates SHAP plots (bar, beeswarm, and waterfall ) for both Logistic Regression and XGBoost models. Shows which features drive credit-risk predictions.

<img width="547" height="572" alt="image" src="https://github.com/user-attachments/assets/d096e3ee-2162-4890-975d-6e49fc131008" />


### 4. Fairness
Tests across income groups (income_group_auth) and regions (addr_state), applying the 80% rule before and after group-specific cut-offs. Shows approval rates and TPR_good snapshots to highlight fairness gaps.

<img width="509" height="682" alt="image" src="https://github.com/user-attachments/assets/136e4b30-2b97-41cd-81db-f4d8342f87c4" />


### 5. Compliance & LLM
Automates compliance reports using rule-based thresholds (PSI, AUC/KS drops, fairness checks). Can also produce concise LLM one-pagers tailored for executives, summarising risks, governance issues, and overall model health.

<img width="532" height="656" alt="image" src="https://github.com/user-attachments/assets/4c0c08a9-5205-415a-9597-a3f6cc20bb09" />


### Role Relevance
This project simulates the practical workflow of a **Credit Risk Analyst / Risk Modeller**:
- Developing and validating scorecard-style models (PD / default prediction).  
- Monitoring model stability and documenting governance metrics (PSI, KS, fairness).  
- Producing regulator-friendly compliance summaries for management review.  
- Applying **UK regulatory standards** (IFRS 9, PRA SS1/23, FCA Consumer Duty) to model lifecycle management.  



### Tech stack section
Streamlit · Python (pandas, numpy, scikit-learn) · SHAP · Jinja2 · MLflow 

