"""Tests for Groq Client, Grounded Reviewer Copilot, and Audit Governance."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.copilot.groq_client import GroqClient
from src.copilot.grounded_copilot import GroundedReviewerCopilot
from src.copilot.audit_logger import CopilotAuditLogger


def test_groq_client_missing_api_key_graceful():
    """Verifies that missing GROQ_API_KEY does not crash and returns clear error."""
    client = GroqClient(api_key="", model="llama-3.3-70b-versatile")
    assert not client.is_available()

    res = client.generate_response("System prompt", "User prompt")
    assert res["success"] is False
    assert res["response"] is None
    assert "GROQ_API_KEY is not configured" in res["error"]


def test_groq_client_mocked_api_call():
    """Verifies Groq client handles API responses at the mock boundary."""
    client = GroqClient(api_key="fake_key_for_test", model="llama-3.3-70b-versatile")
    assert client.is_available()

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Grounded reviewer response."))]

    with patch.object(client, "_get_client") as mock_get:
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.return_value = mock_completion
        mock_get.return_value = mock_groq_instance

        res = client.generate_response("System prompt", "User question")
        assert res["success"] is True
        assert res["response"] == "Grounded reviewer response."
        assert res["latency_seconds"] >= 0.0
        assert res["error"] is None


def test_copilot_context_assembly():
    """Verifies structured context includes all required grounded sections."""
    copilot = GroundedReviewerCopilot()
    static = {
        "loan_id": "LC_TEST_01",
        "original_balance": 15000.0,
        "credit_score": 710,
        "dti": 18.2,
        "interest_rate": 11.5,
        "loan_purpose": "debt_consolidation"
    }
    preds = {
        "default_probability": 0.024,
        "delinquency_probability": 0.045,
        "prepayment_probability": 0.082,
        "next_state": "CURRENT",
        "confidence": "HIGH"
    }
    anomaly = {
        "anomaly_score": 12.0,
        "anomaly_severity": "LOW",
        "reviewer_action": "AUTO_APPROVE",
        "top_drivers": "standard_performing_profile"
    }

    context = copilot.build_structured_context(
        loan_id="LC_TEST_01",
        static_record=static,
        model_predictions=preds,
        anomaly_evidence=anomaly
    )

    assert "loan_information" in context
    assert "model_predictions" in context
    assert "anomaly_evidence" in context
    assert "reviewer_triage" in context
    assert context["loan_information"]["loan_id"] == "LC_TEST_01"
    assert context["model_predictions"]["default_probability"] == 0.024


def test_copilot_audit_logging_and_context_hash(tmp_path):
    """Verifies audit logging writes JSONL with SHA-256 context hash and latency."""
    log_file = tmp_path / "test_audit.jsonl"
    logger = CopilotAuditLogger(log_path=log_file)

    context = {"loan_id": "LC_001", "default_prob": 0.03}
    record = logger.log_interaction(
        loan_id="LC_001",
        user_question="Why was this flagged?",
        model="llama-3.3-70b-versatile",
        structured_context=context,
        response="Grounded analysis text.",
        latency=0.45,
        success=True
    )

    assert record["context_hash"] != "unknown_hash"
    assert len(record["context_hash"]) == 16
    assert record["latency"] == 0.45

    # Check file contents
    assert log_file.exists()
    with open(log_file, "r") as f:
        line = f.readline()
        saved = json.loads(line)
        assert saved["loan_id"] == "LC_001"
        assert saved["context_hash"] == record["context_hash"]
        assert "api_key" not in saved
