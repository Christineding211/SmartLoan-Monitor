

# Daily Model Monitoring Report (Demo)

**Run name:** daily_batch  
**Batch time:** 2025-08-30 23:23:56  
**Data window:** nan  
**Model version:** v1.0  
**Data source:** {'daily_file_pattern': 'data/new_batch_*.csv'}  
**Report owner:** Christine Ding (Data Scientist)

---

## 1) Executive Summary
Monitoring completed for the latest batch. AUC=0.692 and KS=0.285 are fine. Significant drift on 'fico_mid' (PSI=0.262). Missing data is within acceptable limits. Fairness meets the 80% rule.

---

## 2) Key Metrics vs Thresholds
- **AUC:** 0.692  ✅
- **KS:** 0.285  ✅
- **Missing rate (max feature):** 0.0004199  ✅
- **PSI (max feature):** N/A = N/A  🔴 (> 0.2)

> > *Tip:* AUC/KS indicate the model’s discriminatory power; PSI reflects shifts in data distribution; Missing rate indicates data quality.


---

## 3) Drift & Data Quality
Top drifted features:

- No material drift detected.


Notes: No unusual ETL/schema issues observed.

---

## 4) Fairness Snapshot
Groups evaluated: income_group, addr_state  
80% rule pass: True

Across evaluated groups, approval-rate ratios remained within the 80% rule.

---

## 5) Compliance & Governance
- **FCA Consumer Duty:** Potential variability in outcomes; monitor under FCA Consumer Duty.
- **Equality Act 2010:** No evidence of direct discrimination across monitored groups.
- **Model governance:** Retrain required? **True**  
- **Next review:** Quarterly model committee (scheduled)

---

## 6) Recommended Actions
- Continue daily monitoring.
- - Track 'fico_mid' PSI for 7-14 days.

---

**References (UK context)**
- FCA Consumer Duty (fair value, outcomes): https://www.fca.org.uk/firms/consumer-duty  
- UK Equality Act 2010 (anti-discrimination): https://www.legislation.gov.uk/ukpga/2010/15/contents  
- NIST AI Risk Management Framework (model risk & monitoring): https://www.nist.gov/publications/ai-risk-management-framework