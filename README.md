# SmartLoan – Credit-Risk Prediction, Governance & AI-Powered Monitoring Dashboard
Lightweight, end-to-end demo showing how lenders embed **SS1/23, IFRS 9**, and **Consumer Duty** requirements: **PD models** (Logistic Regression, XGBoost) with **audit-ready MI**—PSI drift monitoring, fairness checks, **SHAP explainability**, automated alerting, and compliance-style monthly reports.


## 🚀Demo: Try live demo here: [SmartLoan Dashboard](https://smartloan-dashboard.streamlit.app)

## 🎯  What problem does this solve?
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
Autonomous monitoring agent that tracks model drift (PSI), performance metrics (AUC/KS), and fairness compliance, generating automated alerts and governance reports

<img width="1025" height="587" alt="image" src="https://github.com/user-attachments/assets/d50b1e4f-7df7-44d3-b03a-20e90daa0269" />



### 3. Explainability
Generates SHAP plots (bar, beeswarm, and waterfall ) for both Logistic Regression and XGBoost models. Shows which features drive credit-risk predictions.

<img width="547" height="572" alt="image" src="https://github.com/user-attachments/assets/d096e3ee-2162-4890-975d-6e49fc131008" />


### 4. Fairness
Interactive fairness audit dashboard that compares global vs group-specific thresholds, measures 80% rule (Disparate Impact Ratio) and TPR parity across income groups, ensuring equitable lending outcomes aligned with FCA Consumer Duty standards.

<img width="1116" height="664" alt="image" src="https://github.com/user-attachments/assets/fe2d5bef-df85-409d-b74f-5155a1d89091" />


### 5. Compliance & LLM
Automates compliance reports using rule-based thresholds (PSI, AUC/KS drops, fairness checks). Can also produce concise LLM one-pagers tailored for executives, summarising risks, governance issues, and overall model health.

<img width="1104" height="560" alt="image" src="https://github.com/user-attachments/assets/f046d743-7ec3-485c-8794-424a3d134279" />


### 6. AI Agent Chat
Conversational AI agent that monitors model health, automatically detects drift and performance issues, prioritises risks, and delivers actionable governance recommendations through natural language interaction.

<img width="1251" height="776" alt="image" src="https://github.com/user-attachments/assets/738f7d34-82af-4061-8d28-0c991b4c2b2b" />



### Role Relevance
This project simulates the practical workflow of a **Credit Risk Analyst / Risk Modeller:**

- Developing and validating scorecard-style models (PD / default prediction).
- Monitoring model stability and documenting governance metrics (PSI, KS, fairness).
- **Deploying AI-powered automation** for continuous model monitoring and intelligent alerting.
- Producing regulator-friendly compliance summaries for management review.
- Applying **regulatory standards** (IFRS 9, PRA SS1/23, FCA Consumer Duty) to model lifecycle management. 



### Tech stack section
Streamlit · Python (pandas, numpy, scikit-learn) · SHAP · Jinja2 · MLflow · OpenAI GPT-4o mini · LangChain

