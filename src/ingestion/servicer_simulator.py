"""Derived Secondary-Source Servicer Updates Simulator.

PROVENANCE:
This module creates a clearly labeled DERIVED secondary-source servicer update dataset
to simulate realistic tape extracts, partial updates, stale timestamps, and source reconciliation
conflicts (status mismatch, balance drift, document status conflicts) for the Intain Challenge prototype.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from src.config.settings import RANDOM_SEED
from src.utils.logger import logger


class ServicerUpdateSimulator:
    """Simulates secondary-source subservicer updates and data tape transmissions with realistic anomalies."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed

    def generate_servicer_updates(
        self,
        static_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        conflict_rate: float = 0.08,
        stale_rate: float = 0.05
    ) -> pd.DataFrame:
        """Generates secondary-source servicer updates mirroring the canonical monthly panel with controlled anomalies."""
        np.random.seed(self.seed)
        
        logger.info(f"Generating derived servicer updates (conflict rate: {conflict_rate:.1%}, stale rate: {stale_rate:.1%})...")
        
        # Sample latest monthly state per loan
        latest_monthly = monthly_df.sort_values("month_index").groupby("loan_id").last().reset_index()
        merged = latest_monthly.merge(static_df, on="loan_id", how="inner")
        
        subservicer_systems = ["SERVICER_CORE_B", "TAPE_EXTRACT_EAST", "SUB_SERVICER_TITAN", "LEGACY_MAINFRAME_4"]
        
        updates: List[Dict] = []
        
        for idx, row in merged.iterrows():
            loan_id = str(row["loan_id"])
            servicer_name = str(row.get("servicer_name", "Apex_Loan_Servicing"))
            true_status = str(row["current_status"])
            true_balance = float(row["current_balance"])
            true_dpd = int(row["days_past_due"])
            true_doc = str(row.get("document_status", "VERIFIED"))
            rep_month = str(row["reporting_month"])
            
            # Defaults to matching master record
            reported_status = true_status
            reported_balance = true_balance
            reported_dpd = true_dpd
            reported_doc = true_doc
            source_sys = np.random.choice(subservicer_systems)
            
            # Timestamp generation (usually recent)
            try:
                base_dt = pd.to_datetime(f"{rep_month}-28 14:00:00")
            except Exception:
                base_dt = pd.to_datetime("2018-12-28 14:00:00")
                
            # Stale timestamp injection
            if np.random.rand() < stale_rate:
                # 90 to 180 days stale
                stale_days = np.random.randint(90, 180)
                update_ts = (base_dt - pd.Timedelta(days=stale_days)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                update_ts = (base_dt - pd.Timedelta(hours=np.random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S")
                
            # Conflict injection (status, balance, or document status)
            if np.random.rand() < conflict_rate:
                conflict_type = np.random.choice(["status_mismatch", "balance_mismatch", "doc_conflict"])
                if conflict_type == "status_mismatch":
                    if true_status == "CURRENT":
                        reported_status = "DELINQUENT_30"
                        reported_dpd = 30
                    elif "DELINQUENT" in true_status:
                        reported_status = "CURRENT"
                        reported_dpd = 0
                elif conflict_type == "balance_mismatch":
                    # Balance drift of 10% - 30%
                    drift = np.random.choice([-1, 1]) * (true_balance * np.random.uniform(0.10, 0.30) + 500.0)
                    reported_balance = max(0.0, true_balance + drift)
                elif conflict_type == "doc_conflict":
                    reported_doc = "UNVERIFIED" if true_doc == "VERIFIED" else "VERIFIED"
                    
            update_record = {
                "update_id": f"UPD_{idx+1000000:07d}",
                "loan_id": loan_id,
                "servicer_name": servicer_name,
                "reported_status": reported_status,
                "reported_balance": round(reported_balance, 2),
                "reported_dpd": reported_dpd,
                "document_status": reported_doc,
                "last_updated_at": update_ts,
                "source_system": source_sys
            }
            updates.append(update_record)
            
        updates_df = pd.DataFrame(updates)
        logger.info(f"Generated {len(updates_df):,} servicer secondary update records.")
        return updates_df
