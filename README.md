# SmartLoan – Credit-Risk Prediction, Governance & Monitoring Dashboard

A lightweight, end-to-end credit-risk prediction and monitoring app built with Streamlit, 
simulating how UK lenders manage **IFRS 9 / PRA SS1/23 / FCA Consumer Duty** model governance.  
It demonstrates not only predictive modelling (Logistic Regression & XGBoost) but also **responsible AI practices**: drift tracking (PSI), fairness testing, explainability (SHAP), and compliance-style reporting for executives.


## 🚀Demo: Try live demo here: [SmartLoan Dashboard](https://smartloan-dashboard.streamlit.app)

## What problem does this solve?
Financial institutions face both decision-making risks and regulatory/reputational risks if their credit-risk models are **not accurate, monitored, explainable, and fair**.
- Unreliable predictions may lead to poor lending decisions and financial losses.
- Models can silently break when data drifts or key inputs go missing.
- Lack of transparent reporting to senior management and regulators.
- Risk of unfair lending outcomes across customer groups (e.g., income segments, geography).
- Teams operate in isolation, lacking a single governance framework.

## 👉 How this project aligns with real credit-risk workflows
This project mirrors how real credit-risk teams operate in financial institutions:

- **Model Development:** Default prediction using Logistic Regression (scorecard-style) and XGBoost (non-linear benchmark).  
- **Model Monitoring:** PSI-based drift alerts and performance tracking as part of monthly governance packs.  
- **Fairness & Consumer Duty:** 80% rule check to assess equal outcomes across customer segments.  
- **Explainability & Validation:** SHAP-based feature interpretation to support internal model validation and audit.  
- **Regulatory Reporting:** Auto-generated summaries mimicking documentation required under FCA/PRA expectations.  


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

