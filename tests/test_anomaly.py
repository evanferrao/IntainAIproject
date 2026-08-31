"""Tests for Anomaly Detection, Rule Engine, and Reconciliation."""

import pandas as pd
import numpy as np
from src.anomaly.rule_engine import DeterministicRuleEngine
from src.anomaly.reconciliation import SourceReconciliationEngine
from src.anomaly.composite_scorer import CompositeAnomalyScorer


def test_deterministic_rule_engine():
    static_df = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"],
        "original_balance": [10000.0, -500.0, 15000.0],  # L2 invalid balance
        "interest_rate": [12.0, 15.0, 65.0],              # L3 invalid rate
        "credit_score": [700.0, 710.0, 720.0],
        "dti": [20.0, 15.0, 10.0],
        "annual_income": [60000.0, 50000.0, 80000.0],
        "original_term": [36, 36, 36],
        "document_status": ["VERIFIED", "VERIFIED", "VERIFIED"],
        "state": ["CA", "NY", "FL"]
    })

    engine = DeterministicRuleEngine()
    evaluated = engine.evaluate_static_records(static_df)

    assert evaluated.loc[0, "has_deterministic_violation"] == False
    assert evaluated.loc[1, "has_deterministic_violation"] == True  # Negative balance
    assert evaluated.loc[2, "has_deterministic_violation"] == True  # Rate > 40%


def test_source_reconciliation_engine():
    master_df = pd.DataFrame({
        "loan_id": ["L1", "L2"],
        "current_status": ["CURRENT", "CURRENT"],
        "current_balance": [10000.0, 5000.0],
        "document_status": ["VERIFIED", "VERIFIED"]
    })

    servicer_df = pd.DataFrame({
        "loan_id": ["L1", "L2"],
        "reported_status": ["CURRENT", "DELINQUENT_30"],  # L2 status mismatch
        "reported_balance": [10000.0, 8500.0],           # L2 balance drift
        "document_status_servicer": ["VERIFIED", "VERIFIED"],
        "last_updated_at": ["2018-12-20", "2018-12-20"]
    })

    recon_engine = SourceReconciliationEngine()
    reconciled = recon_engine.reconcile(master_df, servicer_df)

    assert reconciled.loc[0, "has_reconciliation_conflict"] == False
    assert reconciled.loc[1, "has_reconciliation_conflict"] == True
    assert reconciled.loc[1, "has_status_conflict"] == True
