"""Tests for Grounded LLM Copilot and Governance."""

from src.copilot.grounded_copilot import GroundedReviewerCopilot
from src.copilot.evaluation_suite import get_llm_evaluation_benchmarks


def test_copilot_grounded_loan_review_note():
    copilot = GroundedReviewerCopilot()
    static = {
        "original_balance": 15000.0,
        "credit_score": 710,
        "dti": 18.2,
        "interest_rate": 11.5,
        "loan_purpose": "debt_consolidation",
        "document_status": "VERIFIED"
    }
    preds = {
        "default_probability": 0.024,
        "delinquency_probability": 0.045,
        "prepayment_probability": 0.082,
        "next_state": "CURRENT"
    }
    anomaly = {
        "anomaly_score": 12.0,
        "anomaly_severity": "LOW",
        "reviewer_action": "AUTO_APPROVE",
        "top_drivers": "prime_credit",
        "reasons": "No discrepancies found."
    }

    result = copilot.generate_loan_reviewer_note(
        loan_id="LC_TEST_01",
        static_record=static,
        model_predictions=preds,
        anomaly_evidence=anomaly
    )

    assert "LC_TEST_01" in result["loan_id"]
    assert "Recommendation — requires human review" in result["response"]
    assert "AUTO_APPROVE" in result["response"]


def test_copilot_evaluation_benchmarks():
    benchmarks = get_llm_evaluation_benchmarks()
    assert len(benchmarks) >= 3
    categories = [b["failure_category"] for b in benchmarks]
    assert any("Vague" in c for c in categories)
    assert any("Unsupported" in c for c in categories)
    assert any("Overconfident" in c for c in categories)
