"""Servicer Updates and Multi-Source Reconciliation Conflict Engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import logger


class SourceReconciliationEngine:
    """Reconciles canonical master records against secondary-source servicer updates to flag discrepancies."""

    def __init__(self, balance_tolerance_pct: float = 0.05, staleness_days_threshold: int = 60):
        self.balance_tolerance_pct = balance_tolerance_pct
        self.staleness_days_threshold = staleness_days_threshold

    def reconcile(
        self,
        canonical_master_df: pd.DataFrame,
        servicer_updates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compares canonical records against servicer updates and identifies conflicts."""
        logger.info(f"Reconciling {len(canonical_master_df):,} master records against {len(servicer_updates_df):,} servicer updates...")
        
        merged = canonical_master_df.merge(
            servicer_updates_df,
            on="loan_id",
            how="left",
            suffixes=("", "_servicer")
        )

        n_records = len(merged)
        has_status_conflict = []
        has_balance_conflict = []
        has_doc_conflict = []
        is_stale_update = []
        conflict_reasons = []

        now_dt = pd.to_datetime("2019-01-01 00:00:00")

        for idx, row in merged.iterrows():
            reasons = []
            
            # Status check
            master_status = str(row.get("current_status", "CURRENT"))
            servicer_status = str(row.get("reported_status", master_status))
            status_mismatch = (master_status != servicer_status) and (servicer_status != "nan")
            has_status_conflict.append(status_mismatch)
            if status_mismatch:
                reasons.append(f"Status mismatch (Master: {master_status} vs Servicer: {servicer_status})")

            # Balance check
            master_bal = float(row.get("current_balance", 0.0))
            servicer_bal = float(row.get("reported_balance", master_bal))
            bal_diff = abs(master_bal - servicer_bal)
            bal_mismatch = (bal_diff > 100.0) and (bal_diff / max(1.0, master_bal) > self.balance_tolerance_pct)
            has_balance_conflict.append(bal_mismatch)
            if bal_mismatch:
                reasons.append(f"Balance drift (${master_bal:,.2f} vs ${servicer_bal:,.2f}, diff: ${bal_diff:,.2f})")

            # Document audit check
            master_doc = str(row.get("document_status", "VERIFIED"))
            servicer_doc = str(row.get("document_status_servicer", master_doc))
            doc_mismatch = (master_doc != servicer_doc) and (servicer_doc != "nan")
            has_doc_conflict.append(doc_mismatch)
            if doc_mismatch:
                reasons.append(f"Doc status conflict (Master: {master_doc} vs Servicer: {servicer_doc})")

            # Staleness check
            update_ts_str = str(row.get("last_updated_at", ""))
            try:
                update_dt = pd.to_datetime(update_ts_str)
                age_days = (now_dt - update_dt).days
                stale = age_days > self.staleness_days_threshold
            except Exception:
                stale = False
            is_stale_update.append(stale)
            if stale:
                reasons.append(f"Stale servicer tape ({age_days} days old)")

            conflict_reasons.append("; ".join(reasons) if reasons else "Reconciled with zero source conflicts")

        merged["has_status_conflict"] = has_status_conflict
        merged["has_balance_conflict"] = has_balance_conflict
        merged["has_doc_conflict"] = has_doc_conflict
        merged["is_stale_update"] = is_stale_update
        merged["has_reconciliation_conflict"] = (
            merged["has_status_conflict"] |
            merged["has_balance_conflict"] |
            merged["has_doc_conflict"] |
            merged["is_stale_update"]
        )
        merged["reconciliation_notes"] = conflict_reasons

        n_conflicts = merged["has_reconciliation_conflict"].sum()
        logger.info(f"Reconciliation completed: {n_conflicts:,} records ({n_conflicts/max(1, n_records):.1%}) flagged with source conflicts.")
        return merged
