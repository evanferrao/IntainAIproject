# Canonical Loan Data Dictionary

This document defines the canonical schema for the **Loan Performance Intelligence Engine** (Intain Campus FinTech Challenge 2026 AI Track).

> **Important Provenance Note:**
> All source columns mapped from raw Lending Club data are clearly documented below. Derived temporal metrics and secondary-source servicer update fields are explicitly labeled as **[DERIVED PROTOTYPE]** to ensure complete transparency.

---

## 1. Static Loan Master (`loan_static_attributes.csv`)

| Field Name | Type | Description | Allowed / Valid Range | Source Lineage |
| :--- | :--- | :--- | :--- | :--- |
| `loan_id` | `VARCHAR(64)` | Unique loan identifier. Primary Key. | Non-empty string | Source: `id` |
| `origination_month` | `VARCHAR(7)` | Month of origination (`YYYY-MM`). | 2007-01 to 2026-12 | Source: `issue_d` |
| `original_balance` | `FLOAT` | Total initial principal balance ($). | `> 0` ($1,000 – $100,000) | Source: `loan_amnt` |
| `funded_balance` | `FLOAT` | Amount funded by investors ($). | `> 0` | Source: `funded_amnt` |
| `interest_rate` | `FLOAT` | Annualized nominal interest rate (%). | `1.0% – 40.0%` | Source: `int_rate` |
| `original_term` | `INTEGER` | Loan repayment term in months. | `36` or `60` | Source: `term` |
| `installment` | `FLOAT` | Monthly contractual payment amount ($). | `> 0` | Source: `installment` |
| `credit_score` | `FLOAT` | Borrower FICO credit score at origination. | `300 – 850` | Source: `fico_range_low`/`high` |
| `credit_score_band` | `VARCHAR(16)` | Risk tier category (Super Prime, Prime, Near Prime, Subprime, Deep Subprime). | Tier enum | Derived from `credit_score` |
| `dti` | `FLOAT` | Debt-to-income ratio (excluding mortgage). | `0.0% – 100.0%` | Source: `dti` |
| `dti_band` | `VARCHAR(16)` | Debt-to-income band (`<20%`, `20-35%`, `35-50%`, `>50%`). | Band enum | Derived from `dti` |
| `annual_income` | `FLOAT` | Borrower self-reported annual income ($). | `> 0` | Source: `annual_inc` |
| `employment_length` | `INTEGER` | Years of employment at origination. | `0 – 10` (10 means 10+) | Source: `emp_length` |
| `home_ownership` | `VARCHAR(16)` | Housing occupancy status (`RENT`, `OWN`, `MORTGAGE`, `OTHER`). | Categorical | Source: `home_ownership` |
| `loan_purpose` | `VARCHAR(32)` | Stated purpose of loan (`debt_consolidation`, `credit_card`, `home_improvement`, etc.). | Categorical | Source: `purpose` |
| `state` | `VARCHAR(2)` | Borrower residence 2-letter US state code. | Valid US State | Source: `addr_state` |
| `revolving_utilization` | `FLOAT` | Total revolving line credit utilization rate (%). | `0.0% – 150.0%` | Source: `revol_util` |
| `delinquencies_2yrs` | `INTEGER` | Number of 30+ DPD incidences in past 2 years. | `0 – 50` | Source: `delinq_2yrs` |
| `inquiries_6m` | `INTEGER` | Number of credit inquiries in past 6 months. | `0 – 30` | Source: `inq_last_6mths` |
| `total_accounts` | `INTEGER` | Total number of credit lines on credit file. | `1 – 200` | Source: `total_acc` |
| `servicer_name` | `VARCHAR(32)` | Loan servicing institution entity name. | Non-empty string | [DERIVED PROTOTYPE] |
| `document_status` | `VARCHAR(16)` | Verification status of borrower documents (`VERIFIED`, `SOURCE_VERIFIED`, `UNVERIFIED`). | Enum | Source: `verification_status` |

---

## 2. Monthly Performance Panel (`loan_monthly_performance_*.csv`)

| Field Name | Type | Description | Valid Range | Source Lineage |
| :--- | :--- | :--- | :--- | :--- |
| `loan_id` | `VARCHAR(64)` | Foreign key to `loan_static_attributes`. | Non-empty string | Linked |
| `reporting_month` | `VARCHAR(7)` | Panel observation date (`YYYY-MM`). | `YYYY-MM >= origination_month` | [DERIVED PROTOTYPE] |
| `month_index` | `INTEGER` | Sequential integer month of observation. | `0, 1, 2, ...` | [DERIVED PROTOTYPE] |
| `loan_age_months` | `INTEGER` | Elapsed months since origination. | `0 <= age <= original_term` | [DERIVED PROTOTYPE] |
| `remaining_term_months` | `INTEGER` | Contractual months remaining. | `0 <= remaining <= original_term` | [DERIVED PROTOTYPE] |
| `current_balance` | `FLOAT` | Outstanding principal balance at reporting month. | `0.0 <= balance <= original_balance` | [DERIVED PROTOTYPE] |
| `current_status` | `VARCHAR(16)` | Current servicing status (`CURRENT`, `DELINQUENT_30`, `DELINQUENT_60`, `DELINQUENT_90`, `DEFAULT`, `PREPAID`, `CLOSED`). | Enum | [DERIVED PROTOTYPE] |
| `days_past_due` | `INTEGER` | Days past due on scheduled payment. | `0 – 360` | [DERIVED PROTOTYPE] |
| `modification_flag` | `BOOLEAN` | Whether loan terms have undergone formal modification. | `0` or `1` | [DERIVED PROTOTYPE] |
| `prepayment_flag` | `BOOLEAN` | Indicates full early prepayment termination in reporting month. | `0` or `1` | [DERIVED PROTOTYPE] |
| `default_flag` | `BOOLEAN` | Indicates default / charge-off event in reporting month. | `0` or `1` | [DERIVED PROTOTYPE] |
| `next_state` | `VARCHAR(16)` | Status at $t+1$ (Target for next-state multi-class ML). | Enum | [DERIVED PROTOTYPE] |
| `next_3m_delinquency_flag` | `BOOLEAN` | Whether loan enters $\ge 30$ DPD within next 3 months. | `0` or `1` | [DERIVED PROTOTYPE] |
| `next_12m_default_flag` | `BOOLEAN` | Whether loan defaults within next 12 months. | `0` or `1` | [DERIVED PROTOTYPE] |
| `next_12m_prepayment_flag` | `BOOLEAN` | Whether loan prepays within next 12 months. | `0` or `1` | [DERIVED PROTOTYPE] |

---

## 3. Servicer Secondary Updates (`servicer_updates.csv`)

| Field Name | Type | Description | Source Lineage |
| :--- | :--- | :--- | :--- |
| `update_id` | `VARCHAR(64)` | Unique identifier for update payload. | [DERIVED PROTOTYPE] |
| `loan_id` | `VARCHAR(64)` | Target loan reference ID. | Linked |
| `servicer_name` | `VARCHAR(32)` | Reporting sub-servicer or data tape entity. | [DERIVED PROTOTYPE] |
| `reported_status` | `VARCHAR(16)` | Status recorded in sub-servicer update tape. | [DERIVED PROTOTYPE] |
| `reported_balance` | `FLOAT` | Reported balance in sub-servicer update. | [DERIVED PROTOTYPE] |
| `reported_dpd` | `INTEGER` | Reported days past due. | [DERIVED PROTOTYPE] |
| `document_status` | `VARCHAR(16)` | Servicer recorded document audit status. | [DERIVED PROTOTYPE] |
| `last_updated_at` | `DATETIME` | Timestamp of servicer tape transmission. | [DERIVED PROTOTYPE] |
| `source_system` | `VARCHAR(32)` | Subservicer system name (e.g. `SERVICER_CORE_B`, `TAPE_EXTRACT_X`). | [DERIVED PROTOTYPE] |

---

## 4. Macroeconomic Scenario Factors (`macro_scenarios.csv`)

| Scenario Name | Unemployment Shock (bps) | Benchmark Rate Shift (bps) | HPI Shift (%) | FICO Drift (pts) | Prepayment Multiplier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BASE` | `0` | `0` | `+2.0%` | `0` | `1.00` | Baseline prevailing macro environment. |
| `ADVERSE_CREDIT` | `+300` | `+100` | `-10.0%` | `-40` | `0.45` | Severe stagflationary recession, elevated default risk. |
| `HIGH_PREPAYMENT` | `-50` | `-150` | `+5.0%` | `+10` | `1.85` | Rate drop refinance wave, accelerated prepayment. |
