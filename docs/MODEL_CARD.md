# Model Card — Loan Performance Intelligence Engine

## 1. Model Details
- **System Name**: Loan Performance Intelligence Engine
- **Target Application**: Intain Campus FinTech Challenge 2026 (AI Track)
- **Version**: 1.0.0
- **Model Categories**:
  - **Delinquency Risk Model**: Calibrated LightGBM Classifier (3-Month Horizon, 30+ DPD)
  - **Default Risk Model**: Calibrated LightGBM Classifier (12-Month Horizon, Charge-Off)
  - **Prepayment Propensity Model**: Calibrated LightGBM Classifier (12-Month Horizon, Voluntary Full Payoff)
  - **Next-State Transition Model**: Multiclass LightGBM Classifier (Monthly Migration across 7 states)
  - **Markov Transition Matrix**: Multi-Period Empirical Matrix Chain ($P(S_{t+h} | S_t) = T^h$)
  - **Multivariate Anomaly Detector**: Unsupervised Isolation Forest + Deterministic Business Logic Rule Engine
- **Underlying Predictive ML Stack**: Python 3.12, Scikit-Learn, LightGBM, Imbalanced-Learn, Scipy, Lifelines.

---

## 2. Intended Use & Governance
- **Primary Objective**: Provide institutional loan servicers, asset managers, and credit underwriters with continuous risk forecasting, anomaly detection, multi-source reconciliation, and scenario stress testing on retail loan portfolios.
- **Intended Users**: Risk Analysts, Portfolio Managers, Underwriters, Internal Audit Teams, and Credit Committee Reviewers.
- **Out-of-Scope Use Cases**:
  - Fully autonomous, unmonitored loan rejection or foreclosure execution without human review.
  - Applying retail consumer credit risk models directly to commercial real estate or corporate syndications.
- **Governance Requirement**: All outputs and LLM copilot summaries carry the mandatory disclaimer: **"Recommendation — requires human review"**.

---

## 3. Dataset Provenance & Limitations
- **Source Provenance**: Publicly available **Lending Club** historical loan data (`wordsforthewise/lending-club`).
- **Reason for Prototype Dataset**: The official organizer-provided competition dataset was unavailable at implementation time.
- **Adapter Architecture**: All source inputs pass through the `LendingClubAdapter` to conform strictly to the `CanonicalStaticLoan` and `CanonicalMonthlyPerformance` schemas, ensuring seamless zero-code replacement when the organizer dataset is released.
- **Derived Data Transparency**:
  - Static borrower attributes (FICO, DTI, balance, income, interest rate) reflect real empirical distributions.
  - Multi-period monthly performance panels and secondary-source servicer updates are relationship-driven derived prototype data engineered specifically to demonstrate dynamic Markov transitions, roll-rates, and data tape reconciliation logic.

---

## 4. Feature Engineering & Target Definitions

### Target Definitions
1. `next_3m_delinquency_flag`: Binary indicator (1 if borrower enters $\ge 30$ DPD within $t+1 \dots t+3$, 0 otherwise).
2. `next_12m_default_flag`: Binary indicator (1 if loan charges off / defaults within $t+1 \dots t+12$, 0 otherwise).
3. `next_12m_prepayment_flag`: Binary indicator (1 if loan is fully paid early within $t+1 \dots t+12$, 0 otherwise).
4. `next_state`: Multiclass target ($S_{t+1} \in \{\text{CURRENT}, \text{DELINQUENT\_30}, \text{DELINQUENT\_60}, \text{DELINQUENT\_90}, \text{DEFAULT}, \text{PREPAID}, \text{CLOSED}\}$).

### Leakage Prevention Controls
- **Temporal Isolation**: Features at observation time $T$ are computed strictly using data available at or prior to $T$.
- **No Target Infiltration**: Final loan outcomes or post-observation repayment statuses are never included in feature vectors.
- **Fitted Transformer Persistence**: All imputation, scaling, and categorical encoders are fit exclusively on the chronological training split.

---

## 5. Validation Methodology
- **Time-Aware Chronological Splitting**:
  - Chronological cutoff partitioning ensures all training observations precede validation observations.
  - Loan ID grouping prevents observations from the same loan from appearing across both training and test sets.
- **Class Imbalance Treatment**: Class weighting (`class_weight='balanced'`) and sample weighting applied during training without ungrounded random oversampling of temporal series.
- **Probability Calibration**: Platt scaling (sigmoid) and Isotonic regression applied via 3-Fold Cross-Validation (`CalibratedClassifierCV`) to ensure predicted probabilities match empirical observation frequencies.

---

## 6. Evaluation Metrics
*Computed directly on out-of-time test partitions (never hardcoded):*
- **Delinquency Model**: ROC-AUC, PR-AUC (Average Precision), F1-Score, Brier Score, Expected Calibration Error (ECE).
- **Default Model**: ROC-AUC, PR-AUC, Precision, Recall, F1-Score, Brier Score, ECE.
- **Prepayment Model**: ROC-AUC, PR-AUC, F1-Score, Brier Score.
- **Next-State Model**: Multiclass Macro F1, Weighted F1, Per-Class Precision & Recall, Confusion Matrix.

---

## 7. Known Failure Modes & Edge Cases
1. **Low Liquidity Shocks in High FICO Borrowers**: False Negatives can occur when high-FICO borrowers experience sudden exogenous shocks (e.g. abrupt layoff, uninsured medical catastrophe).
2. **High DTI Borrowers with External Wealth**: False Positives occur when high-DTI borrowers maintain substantial uncaptured liquid reserves or family balance sheets.
3. **Macroeconomic Regime Shifts**: Projections under severe stress assume structural stability in underlying transition matrices and should be interpreted as model-based stress projections, not guaranteed causal forecasts.

---

## 8. LLM Copilot & AI Governance
- **Role of LLM**: The LLM functions purely as an assistant for natural language explanations, rule lookup, exception summaries, and underwriter notes.
- **Predictive Independence**: The LLM does NOT generate core predictions or risk probabilities.
- **Audit Logging**: All queries, retrieved context chunks, generated text, model metadata, and reviewer acceptance statuses are logged to `artifacts/logs/copilot_audit_log.jsonl`.
- **Zero API-Key Dependency**: When external LLM API credentials are not supplied, the system seamlessly operates in deterministic grounded mode without crashing.
