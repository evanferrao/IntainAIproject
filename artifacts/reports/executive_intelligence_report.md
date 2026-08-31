# Loan Performance Intelligence Engine — Executive Intelligence Report

**Generated**: `2026-08-31 14:11:16 UTC`  
**Project**: Intain Campus FinTech Challenge 2026 AI Track Prototype  
**Dataset Provenance Notice**: *Prototype implementation utilizes publicly available Lending Club historical data and relationship-driven derived temporal performance data because the organizer-provided competition dataset was unavailable at implementation time.*

---

## 1. Executive Summary & Portfolio Health

- **Total Evaluated Records**: `25,000`
- **Overall Dataset Quality Score**: `99.89/100` (`EXCELLENT`)
- **Train/Test Drift Status**: `ALERT` (5 high-drift features detected)
- **High-Risk Exception Triage**: `0` records require manual underwriter audit or source reconciliation.

---

## 2. Predictive Machine Learning Performance

Summary of non-LLM machine learning models evaluated on out-of-time chronological validation data:

| Predictive Task | Model Type | ROC-AUC | PR-AUC | F1-Score | Brier Score | Calibration ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Delinquency (Next 3M)** | Improved LightGBM | `0.911` | `0.6807` | `0.6541` | `0.0385` | `0.0994` |
| **Default (Next 12M)** | Improved LightGBM | `0.9931` | `0.7445` | `0.7083` | `0.01` | `0.153` |
| **Prepayment (Next 12M)** | Improved LightGBM | `0.9235` | `0.4887` | `0.0` | `0.0309` | `0.3328` |

### Multiclass Next-State Model
- **Next-State Macro F1**: `0.7046`
- **Weighted F1**: `0.9012`

---

## 3. Macroeconomic Scenario Stress Simulation

*Disclaimer: Scenario outputs are model-based stress projections, not guaranteed causal forecasts.*

| Scenario Name | Macroeconomic Assumption | Projected Delinquency | Projected Default | Projected Prepayment | Expected Loss Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BASE** | Prevailing market baseline | `8.30%` | `2.87%` | `3.89%` | `1.65%` |
| **ADVERSE_CREDIT** | +300 bps unemp, -40 FICO | `10.64%` | `5.37%` | `1.66%` | `3.04%` |
| **HIGH_PREPAYMENT** | -150 bps rate drop, refi surge | `7.67%` | `2.07%` | `7.31%` | `1.19%` |

---

## 4. Anomaly & Exception Detection Triage

- **Deterministic Rule Violations**: `0` records
- **Source Reconciliation Conflicts**: `32` records
- **Multivariate ML Outliers**: `108` records

---

## 5. Governance and LLM Grounding Notice

All LLM Copilot recommendations and reviewer notes are strictly grounded in:
1. Canonical field definitions (`data_dictionary.md`)
2. Codified business rules (`validation_rules.json`)
3. True computed model outputs and anomaly evidence

**Recommendation — requires human review.**
