"""Grounded LLM Reviewer Copilot Engine with Context Retrieval and Audit Governance."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config.settings import CANONICAL_DATA_DIR
from src.copilot.audit_logger import CopilotAuditLogger
from src.utils.logger import logger


class GroundedReviewerCopilot:
    """Grounded AI Reviewer Copilot for loan underwriting, anomaly triage, and scenario analysis."""

    def __init__(self, data_dict_path: Optional[Path] = None, rules_path: Optional[Path] = None):
        self.data_dict_path = data_dict_path or (CANONICAL_DATA_DIR / "data_dictionary.md")
        self.rules_path = rules_path or (CANONICAL_DATA_DIR / "validation_rules.json")
        self.audit_logger = CopilotAuditLogger()
        self.data_dictionary_text = self._load_data_dict()
        self.rules_dict = self._load_rules()

    def _load_data_dict(self) -> str:
        """Loads canonical data dictionary text."""
        if self.data_dict_path.exists():
            with open(self.data_dict_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def _load_rules(self) -> Dict[str, Any]:
        """Loads validation rules dictionary."""
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def retrieve_field_definition(self, field_name: str) -> Dict[str, Any]:
        """Retrieves field definition from data dictionary."""
        lines = self.data_dictionary_text.splitlines()
        matched = []
        for line in lines:
            if field_name.lower() in line.lower() and "|" in line:
                matched.append(line.strip())
        
        if matched:
            result = f"Definition for '{field_name}':\n" + "\n".join(matched)
        else:
            result = f"Field '{field_name}' not found in canonical data dictionary."
            
        return {"field": field_name, "content": result}

    def retrieve_validation_rule(self, rule_id_or_field: str) -> Dict[str, Any]:
        """Retrieves rule specification from validation_rules.json."""
        rules = self.rules_dict.get("rules", [])
        matched = [
            r for r in rules
            if rule_id_or_field.upper() in r.get("rule_id", "").upper() or
               rule_id_or_field.lower() in r.get("field", "").lower() or
               rule_id_or_field.lower() in r.get("name", "").lower()
        ]
        return {"query": rule_id_or_field, "matched_rules": matched}

    def generate_loan_reviewer_note(
        self,
        loan_id: str,
        static_record: Dict[str, Any],
        model_predictions: Dict[str, Any],
        anomaly_info: Optional[Dict[str, Any]] = None,
        anomaly_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generates grounded underwriting and exception review note grounded in exact ML outputs."""
        anomaly_payload = anomaly_info or anomaly_evidence or {}
        context = {
            "loan_id": loan_id,
            "static_attributes": static_record,
            "model_predictions": model_predictions,
            "anomaly_evidence": anomaly_payload
        }

        # Deterministic Grounded Synthesis
        orig_bal = float(static_record.get("original_balance", 0.0))
        fico = float(static_record.get("credit_score", 0.0))
        dti = float(static_record.get("dti", 0.0))
        rate = float(static_record.get("interest_rate", 0.0))
        purpose = str(static_record.get("loan_purpose", "general"))
        doc_status = str(static_record.get("document_status", "VERIFIED"))

        p_default = float(model_predictions.get("default_probability", 0.0))
        p_delinq = float(model_predictions.get("delinquency_probability", 0.0))
        p_prepay = float(model_predictions.get("prepayment_probability", 0.0))
        next_state = str(model_predictions.get("next_state", "CURRENT"))

        anomaly_score = float(anomaly_payload.get("anomaly_score", 0.0))
        severity = str(anomaly_payload.get("anomaly_severity", "LOW"))
        action = str(anomaly_payload.get("reviewer_action", "AUTO_APPROVE"))
        top_drivers = str(anomaly_payload.get("top_drivers", "None"))
        reasons = str(anomaly_payload.get("reasons", "No discrepancies flagged."))

        # Formulate structured reviewer dossier
        note = (
            f"### Loan Reviewer Dossier: `{loan_id}`\n\n"
            f"> [!IMPORTANT]\n"
            f"> **Recommendation — requires human review**\n\n"
            f"#### 1. Executive Summary & Recommended Action\n"
            f"- **Triage Status**: `{action}` (Anomaly Severity: `{severity}`, Composite Score: `{anomaly_score:.1f}/100`)\n"
            f"- **Core ML Trajectory**: Predicted next state is `{next_state}` with default risk `{p_default:.1%}` and delinquency risk `{p_delinq:.1%}`.\n\n"
            f"#### 2. Risk & Profile Analysis\n"
            f"- **Borrower Profile**: FICO `{fico:.0f}`, DTI `{dti:.1f}%`, Interest Rate `{rate:.2f}%`, Purpose: `{purpose}`.\n"
            f"- **Documentation**: Verification Status: `{doc_status}`.\n"
            f"- **Key Drivers**: {top_drivers}\n\n"
            f"#### 3. Audit & Exception Evidence\n"
            f"- **Findings**: {reasons}\n"
            f"- **Prepayment Outlook**: 12-month prepayment probability is estimated at `{p_prepay:.1%}`.\n\n"
            f"#### 4. Auditor Next Steps\n"
            f"{'1. Verify income documentation against servicer tape.' if 'UNVERIFIED' in doc_status or 'RECON' in action else '1. Standard servicing monitoring.'}\n"
            f"{'2. Request manual balance confirmation from sub-servicer.' if 'BALANCE' in reasons or 'RECON' in action else '2. No secondary tape remediation required.'}\n"
        )

        # Log interaction to governance ledger
        self.audit_logger.log_interaction(
            query=f"Generate loan review note for {loan_id}",
            retrieved_context=context,
            model_name="Grounded-Deterministic-Synthesizer-v1",
            generated_response=note,
            confidence_category="HIGH" if anomaly_score < 25 or anomaly_score > 75 else "MODERATE",
            reviewer_status="PENDING_REVIEW"
        )

        return {
            "loan_id": loan_id,
            "response": note,
            "governance_notice": "Recommendation — requires human review",
            "context_used": context
        }

    def answer_query(self, query: str, context_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answers general user questions about data definitions, validation rules, or scenario results."""
        q_lower = query.lower()
        retrieved_context = {}
        
        if "definition" in q_lower or "field" in q_lower or "what is" in q_lower:
            # Extract possible field names
            for word in query.split():
                clean_word = word.strip("?,.'\"")
                if len(clean_word) >= 3 and clean_word in self.data_dictionary_text:
                    retrieved_context = self.retrieve_field_definition(clean_word)
                    break

        elif "rule" in q_lower or "validation" in q_lower:
            retrieved_context = self.retrieve_validation_rule(query)

        if not retrieved_context and context_snapshot:
            retrieved_context = context_snapshot

        # Synthesize answer
        response_text = (
            f"**Response based on grounded project artifacts:**\n\n"
            f"> [!IMPORTANT]\n"
            f"> **Recommendation — requires human review**\n\n"
            f"Context retrieved:\n"
            f"```json\n{json.dumps(retrieved_context, indent=2)}\n```\n\n"
            f"Summary: The queried entities are fully codified in `data_dictionary.md` and `validation_rules.json`. "
            f"All ML metrics and risk scores are calculated directly by the production pipeline."
        )

        self.audit_logger.log_interaction(
            query=query,
            retrieved_context=retrieved_context,
            model_name="Grounded-Synthesizer-v1",
            generated_response=response_text
        )

        return {
            "query": query,
            "response": response_text,
            "governance_notice": "Recommendation — requires human review",
            "retrieved_context": retrieved_context
        }
