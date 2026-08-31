"""Derived Monthly Performance Panel Generator for Temporal Modeling.

PROVENANCE:
This module creates a clearly labeled DERIVED monthly performance panel from real Lending Club
loan characteristics and risk profiles to support temporal modeling, transition matrices,
survival curves, and scenario stress testing for the Intain Challenge prototype.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.config.settings import VALID_STATUSES, TERMINAL_STATUSES, RANDOM_SEED
from src.utils.logger import logger


class MonthlyPanelGenerator:
    """Generates relationship-driven monthly performance trajectories for static loans."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed

    def generate_monthly_panel(
        self,
        static_df: pd.DataFrame,
        max_horizon_months: int = 24,
        max_loans: Optional[int] = None
    ) -> pd.DataFrame:
        """Simulates realistic monthly servicing performance panels from static loan records.
        
        Transition dynamics are driven by:
        - Baseline credit risk (credit_score, dti, interest_rate)
        - Seasonality & loan aging hazard curve (peak default hazard at months 9-18)
        - Prepayment incentive (interest rate spread, prime credit)
        - Historical state friction (e.g. 30 DPD more likely to cure or migrate to 60 DPD)
        """
        np.random.seed(self.seed)
        
        loans = static_df.copy()
        if max_loans and len(loans) > max_loans:
            loans = loans.sample(n=max_loans, random_state=self.seed).reset_index(drop=True)

        logger.info(f"Generating derived monthly panel for {len(loans):,} loans over {max_horizon_months} months...")

        panel_records: List[Dict] = []

        for idx, row in loans.iterrows():
            loan_id = str(row["loan_id"])
            orig_month = str(row["origination_month"])
            orig_bal = float(row["original_balance"])
            term = int(row["original_term"])
            rate = float(row["interest_rate"])
            installment = float(row["installment"])
            credit_score = float(row["credit_score"])
            dti = float(row["dti"])

            # Compute intrinsic borrower hazard multiplier
            # Low FICO, high DTI, high rate -> higher default hazard
            fico_factor = np.exp((720.0 - credit_score) / 60.0)
            dti_factor = 1.0 + (dti / 30.0) * 0.5
            rate_factor = 1.0 + (rate / 15.0) * 0.4
            borrower_risk = np.clip(fico_factor * dti_factor * rate_factor, 0.2, 8.0)

            # Prepayment propensity (higher for high FICO, high balance, low/medium rate)
            prepay_propensity = np.clip(
                (credit_score / 700.0) ** 2.0 * (orig_bal / 15000.0) ** 0.5 * 0.012,
                0.002, 0.045
            )

            # Parse start date
            try:
                start_dt = pd.to_datetime(f"{orig_month}-01")
            except Exception:
                start_dt = pd.to_datetime("2018-01-01")

            current_status = "CURRENT"
            current_bal = orig_bal
            dpd = 0
            is_modified = 0

            loan_panel = []

            for m in range(max_horizon_months):
                rep_month = (start_dt + pd.DateOffset(months=m)).strftime("%Y-%m")
                age = m
                rem_term = max(0, term - age)

                # Prior state snapshot
                prev_status = current_status

                # State transition determination
                if prev_status in TERMINAL_STATUSES:
                    # Stays terminal
                    current_bal = 0.0
                    dpd = 0
                elif prev_status == "CURRENT":
                    # Aging default hazard multiplier (humped curve peaking around month 12)
                    age_hazard = 1.2 * np.exp(-((age - 12.0) ** 2.0) / 72.0) + 0.3
                    p_to_delinq30 = np.clip(0.018 * borrower_risk * age_hazard, 0.003, 0.18)
                    p_to_prepay = np.clip(prepay_propensity * (1.0 + age * 0.02), 0.001, 0.06)

                    u = np.random.rand()
                    if u < p_to_delinq30:
                        current_status = "DELINQUENT_30"
                        dpd = 30
                    elif u < p_to_delinq30 + p_to_prepay:
                        current_status = "PREPAID"
                        current_bal = 0.0
                        dpd = 0
                    elif rem_term <= 0:
                        current_status = "CLOSED"
                        current_bal = 0.0
                        dpd = 0
                    else:
                        current_status = "CURRENT"
                        # Normal principal amortization
                        interest_part = current_bal * (rate / 100.0 / 12.0)
                        principal_part = max(0.0, installment - interest_part)
                        current_bal = max(0.0, current_bal - principal_part)
                        dpd = 0

                elif prev_status == "DELINQUENT_30":
                    # Cure probability vs Roll forward vs Prepay
                    p_cure = np.clip(0.55 / np.sqrt(borrower_risk), 0.15, 0.80)
                    p_roll = np.clip(0.35 * np.sqrt(borrower_risk), 0.10, 0.70)
                    
                    u = np.random.rand()
                    if u < p_cure:
                        current_status = "CURRENT"
                        dpd = 0
                    elif u < p_cure + p_roll:
                        current_status = "DELINQUENT_60"
                        dpd = 60
                    else:
                        current_status = "DELINQUENT_30"
                        dpd = 30

                elif prev_status == "DELINQUENT_60":
                    p_cure = np.clip(0.25 / np.sqrt(borrower_risk), 0.05, 0.50)
                    p_roll = np.clip(0.60 * np.sqrt(borrower_risk), 0.35, 0.85)

                    u = np.random.rand()
                    if u < p_cure:
                        current_status = "CURRENT"
                        dpd = 0
                    elif u < p_cure + p_roll:
                        current_status = "DELINQUENT_90"
                        dpd = 90
                    else:
                        current_status = "DELINQUENT_60"
                        dpd = 60

                elif prev_status == "DELINQUENT_90":
                    p_cure = 0.10
                    p_default = 0.75

                    u = np.random.rand()
                    if u < p_cure:
                        current_status = "DELINQUENT_30"
                        dpd = 30
                    elif u < p_cure + p_default:
                        current_status = "DEFAULT"
                        current_bal = 0.0
                        dpd = 120
                    else:
                        current_status = "DELINQUENT_90"
                        dpd = 90

                # Check modification simulation
                if current_status in ["DELINQUENT_60", "DELINQUENT_90"] and is_modified == 0:
                    if np.random.rand() < 0.08:
                        is_modified = 1

                record = {
                    "loan_id": loan_id,
                    "reporting_month": rep_month,
                    "month_index": m,
                    "loan_age_months": age,
                    "remaining_term_months": rem_term,
                    "current_balance": round(current_bal, 2),
                    "current_status": current_status,
                    "days_past_due": dpd,
                    "modification_flag": is_modified,
                    "prepayment_flag": 1 if current_status == "PREPAID" and prev_status != "PREPAID" else 0,
                    "default_flag": 1 if current_status == "DEFAULT" and prev_status != "DEFAULT" else 0
                }
                loan_panel.append(record)

                if current_status in TERMINAL_STATUSES and m >= 1:
                    # Can terminate early after recording terminal month
                    break

            # Now, calculate future forward-looking targets STRICTLY from future records (No target leakage!)
            n_obs = len(loan_panel)
            for i in range(n_obs):
                # Next-state target (t+1)
                if i + 1 < n_obs:
                    loan_panel[i]["next_state"] = loan_panel[i+1]["current_status"]
                else:
                    loan_panel[i]["next_state"] = loan_panel[i]["current_status"]

                # Next 3 months delinquency (t+1 to t+3)
                future_3m_statuses = [loan_panel[k]["current_status"] for k in range(i+1, min(i+4, n_obs))]
                loan_panel[i]["next_3m_delinquency_flag"] = 1 if any("DELINQUENT" in s or s == "DEFAULT" for s in future_3m_statuses) else 0

                # Next 12 months default (t+1 to t+12)
                future_12m_statuses = [loan_panel[k]["current_status"] for k in range(i+1, min(i+13, n_obs))]
                loan_panel[i]["next_12m_default_flag"] = 1 if any(s == "DEFAULT" for s in future_12m_statuses) else 0

                # Next 12 months prepayment (t+1 to t+12)
                loan_panel[i]["next_12m_prepayment_flag"] = 1 if any(s == "PREPAID" for s in future_12m_statuses) else 0

            panel_records.extend(loan_panel)

        panel_df = pd.DataFrame(panel_records)
        logger.info(f"Generated monthly performance panel with {len(panel_df):,} total monthly records.")
        return panel_df
