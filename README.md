# SmartLoan – Credit-Risk Prediction, Governance & Monitoring Dashboard

 A lightweight, end-to-end credit-risk prediction and monitoring app built with Streamlit.It generates loan default predictions (Logistic Regression & XGBoost), and demonstrates practical model governance: drift tracking (PSI), performance metrics (AUC/KS), explainability (SHAP), fairness testing (80% rule with group cut-offs), and compliance-style reporting (rule-based + optional LLM summary for executives).

## What problem does this solve?
Financial institutions face both decision-making risks and regulatory/reputational risks if their credit-risk models are **not accurate, monitored, explainable, and fair**.
- Unreliable predictions may lead to poor lending decisions and financial losses.
- Models can silently break when data drifts or key inputs go missing.
- Lack of transparent reporting to senior management and regulators.
- Risk of unfair lending outcomes across customer groups (e.g., income segments, geography).
- Teams operate in isolation, lacking a single governance framework.

## This project addresses those problems by providing a central monitoring dashboard that:
- Produces **loan default predictions** and evaluates them with **AUC/KS**.  
- Tracks **model performance** and alerts when thresholds are breached.  
- Detects **data drift** via PSI and flags high-risk features.
- Produces **explainability outputs (SHAP** bar, beeswarm, waterfall plots).
- Runs **fairness checks** (80% rule before vs after group cut-offs).
- Uses **MLflow to track experiments**, ensuring reproducible and optimised model development.  
- Generates **compliance reports** in plain English, with optional **LLM executive summaries**.

## Key features
### 1. New Batch / Monitoring
Upload a new CSV batch and instantly compute metrics (AUC/KS), drift (PSI), and missingness. Provides a one-stop health check for model monitoring.

<img width="599" height="866" alt="image" src="https://github.com/user-attachments/assets/e0e0893e-db0f-42f7-a979-50abf389d48a" />

### 2.Metrics & Drift
Visualises top-K drifted features with PSI, using OK/WARN/ALERT thresholds from config.yaml. Helps spot early warning signs before model degradation.

<img width="671" height="869" alt="image" src="https://github.com/user-attachments/assets/bcde2455-9105-48a3-a2d6-24b72c6607f6" />


### 3. Explainability
Generates SHAP plots (bar, beeswarm, and waterfall ) for both Logistic Regression and XGBoost models. Shows which features drive credit-risk predictions.

### 4. Fairness
Tests across income groups (income_group_auth) and regions (addr_state), applying the 80% rule before and after group-specific cut-offs. Shows approval rates and TPR_good snapshots to highlight fairness gaps.

### 5. Compliance & LLM
Automates compliance reports using rule-based thresholds (PSI, AUC/KS drops, fairness checks). Can also produce concise LLM one-pagers tailored for executives, summarising risks, governance issues, and overall model health.



### Tech stack section
Streamlit · Python (pandas, numpy, scikit-learn) · SHAP · Jinja2 · MLflow 

