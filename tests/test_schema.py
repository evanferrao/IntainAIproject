"""Tests for canonical schema validation and Pydantic models."""

import pytest
from pydantic import ValidationError
from src.config.schema import CanonicalStaticLoan, CanonicalMonthlyPerformance, SubmissionRecord


def test_canonical_static_loan_valid():
    valid_data = {
        "loan_id": "LC_00001",
        "origination_month": "2018-06",
        "original_balance": 15000.0,
        "funded_balance": 15000.0,
        "interest_rate": 12.5,
        "original_term": 36,
        "installment": 501.76,
        "credit_score": 710.0,
        "credit_score_band": "Prime",
        "dti": 18.5,
        "dti_band": "20-35%",
        "annual_income": 85000.0,
        "employment_length": 5,
        "home_ownership": "MORTGAGE",
        "loan_purpose": "debt_consolidation",
        "state": "CA",
        "revolving_utilization": 45.0,
        "delinquencies_2yrs": 0,
        "inquiries_6m": 1,
        "total_accounts": 20,
        "servicer_name": "Apex_Loan_Servicing",
        "document_status": "VERIFIED"
    }
    loan = CanonicalStaticLoan(**valid_data)
    assert loan.loan_id == "LC_00001"
    assert loan.credit_score == 710.0


def test_canonical_static_loan_invalid_score():
    invalid_data = {
        "loan_id": "LC_00002",
        "origination_month": "2018-06",
        "original_balance": 10000.0,
        "funded_balance": 10000.0,
        "interest_rate": 10.0,
        "original_term": 36,
        "installment": 300.0,
        "credit_score": 250.0,  # Invalid: below 300
        "credit_score_band": "Deep Subprime",
        "dti": 15.0,
        "dti_band": "<20%",
        "annual_income": 60000.0,
        "employment_length": 2,
        "home_ownership": "RENT",
        "loan_purpose": "credit_card",
        "state": "NY",
        "revolving_utilization": 30.0,
        "delinquencies_2yrs": 0,
        "inquiries_6m": 0,
        "total_accounts": 10,
        "servicer_name": "Apex_Loan_Servicing",
        "document_status": "VERIFIED"
    }
    with pytest.raises(ValidationError):
        CanonicalStaticLoan(**invalid_data)


def test_submission_record_validation():
    valid_sub = {
        "loan_id": "LC_1001",
        "delinquency_probability": 0.052,
        "default_probability": 0.018,
        "prepayment_probability": 0.095,
        "next_state": "CURRENT",
        "exception_probability": 0.02,
        "exception_type": "NONE",
        "anomaly_score": 12.5,
        "top_drivers": "prime_fico_low_dti",
        "reviewer_action": "AUTO_APPROVE",
        "confidence": "HIGH"
    }
    sub = SubmissionRecord(**valid_sub)
    assert sub.confidence == "HIGH"
    assert sub.default_probability == 0.018
