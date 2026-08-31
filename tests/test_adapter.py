"""Tests for Lending Club Adapter."""

import pandas as pd
from src.adapters.lending_club_adapter import LendingClubAdapter


def test_lending_club_adapter_transformation():
    raw_df = pd.DataFrame({
        "id": ["1001", "1002"],
        "issue_d": ["Dec-2015", "Jan-18"],
        "loan_amnt": ["10000", 25000],
        "funded_amnt": [10000, 25000],
        "term": [" 36 months", " 60 months"],
        "int_rate": ["12.5%", "8.25%"],
        "installment": [334.5, 509.2],
        "fico_range_low": [680, 720],
        "fico_range_high": [684, 724],
        "dti": [14.2, 22.5],
        "annual_inc": [75000, 110000],
        "emp_length": ["10+ years", "< 1 year"],
        "home_ownership": ["MORTGAGE", "RENT"],
        "purpose": ["debt_consolidation", "credit_card"],
        "addr_state": ["CA", "TX"],
        "revol_util": ["55.4%", "22.1%"],
        "delinq_2yrs": [0, 1],
        "inq_last_6mths": [1, 0],
        "total_acc": [18, 25],
        "verification_status": ["Source Verified", "Not Verified"]
    })

    adapter = LendingClubAdapter()
    canonical_df = adapter.transform(raw_df)

    assert len(canonical_df) == 2
    assert "origination_month" in canonical_df.columns
    assert canonical_df.loc[0, "origination_month"] == "2015-12"
    assert canonical_df.loc[1, "origination_month"] == "2018-01"
    assert canonical_df.loc[0, "original_term"] == 36
    assert canonical_df.loc[1, "original_term"] == 60
    assert canonical_df.loc[0, "interest_rate"] == 12.5
    assert canonical_df.loc[0, "credit_score"] == 682.0
    assert canonical_df.loc[0, "employment_length"] == 10
    assert canonical_df.loc[1, "employment_length"] == 0
    assert canonical_df.loc[0, "document_status"] == "SOURCE_VERIFIED"
    assert canonical_df.loc[1, "document_status"] == "UNVERIFIED"
