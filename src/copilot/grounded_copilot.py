"""Grounded AI Reviewer Copilot Engine with Context Assembly and Real Groq Integration."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config.settings import CANONICAL_DATA_DIR
from src.copilot.audit_logger import CopilotAuditLogger
from src.copilot.groq_client import GroqClient
from src.utils.logger import logger


def clean_html_markup(text: Optional[str]) -> Optional[str]:
    """Cleans up raw HTML markup like <br>, <b>, </b> into standard clean Markdown."""
    if not text:
        return text
    # Convert bold tags to **...**
    cleaned = re.sub(r"<\s*b\b[^>]*>(.*?)<\s*/\s*b\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<\s*strong\b[^>]*>(.*?)<\s*/\s*strong\s*>", r"**\1**", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Convert italic tags to *...*
    cleaned = re.sub(r"<\s*i\b[^>]*>(.*?)<\s*/\s*i\s*>", r"*\1*", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<\s*em\b[^>]*>(.*?)<\s*/\s*em\s*>", r"*\1*", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Remove any remaining stray b, strong, i, em, span, div, p tags
    cleaned = re.sub(r"<\s*/?\s*(?:b|strong|i|em|div|span|p)\b[^>]*>", "", cleaned, flags=re.IGNORECASE)

    # Handle <br> tags: inside table rows (lines containing "|"), normalize to <br/> for clean multi-line display; outside tables, convert to \n
    lines = cleaned.splitlines()
    processed_lines = []
    for line in lines:
        if "|" in line:
            line = re.sub(r"<\s*br\s*/?>", "<br/>", line, flags=re.IGNORECASE)
        else:
            line = re.sub(r"<\s*br\s*/?>", "\n", line, flags=re.IGNORECASE)
        processed_lines.append(line)
    return "\n".join(processed_lines)


class GroundedReviewerCopilot:
    """Grounded Decision Support Copilot for loan underwriting, anomaly triage, and evidence explanation."""

    SYSTEM_INSTRUCTION = (
        "You are a loan analytics reviewer assistant for the Decision Intelligence Engine. "
        "You may only make factual claims supported by the supplied evidence. Do not invent loan attributes, "
        "model predictions, anomaly findings, or policy rules. If information is unavailable, state that it is unavailable. "
        "You are providing analytical assistance, not making a final lending decision. All recommendations require human review. "
        "Formatting instructions: Always format your response in clean, standard Markdown. "
        "Do not use raw HTML tags such as <br>, <b>, </b>, <strong>, <div>, or <span>. Use standard Markdown formatting "
        "(e.g., **bold**, bullet points, standard tables, and newlines). "
        "Never mention 'Groq', 'AI', 'LLM', or specific model names; refer to the system neutrally as the Decision Intelligence Engine."
    )

    def __init__(
        self,
        data_dict_path: Optional[Path] = None,
        rules_path: Optional[Path] = None,
        groq_client: Optional[GroqClient] = None
    ):
        self.data_dict_path = data_dict_path or (CANONICAL_DATA_DIR / "data_dictionary.md")
        self.rules_path = rules_path or (CANONICAL_DATA_DIR / "validation_rules.json")
        self.audit_logger = CopilotAuditLogger()
        self.groq_client = groq_client or GroqClient()
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
        """Retrieves field definition from canonical data dictionary."""
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

    def build_structured_context(
        self,
        loan_id: str,
        static_record: Dict[str, Any],
        model_predictions: Dict[str, Any],
        anomaly_evidence: Optional[Dict[str, Any]] = None,
        data_quality_info: Optional[Dict[str, Any]] = None,
        explainability_info: Optional[Dict[str, Any]] = None,
        counterfactual_info: Optional[Dict[str, Any]] = None,
        reviewer_triage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assembles a clean, grounded evidence package for the selected loan."""
        anomaly_data = anomaly_evidence or {}
        quality_data = data_quality_info or {}
        expl_data = explainability_info or {}
        cf_data = counterfactual_info or {}
        triage_data = reviewer_triage or {}

        # Extract static attributes safely
        clean_static = {
            "loan_id": str(loan_id),
            "original_balance": static_record.get("original_balance", "Not available"),
            "credit_score": static_record.get("credit_score", "Not available"),
            "credit_score_band": static_record.get("credit_score_band", "Not available"),
            "dti": static_record.get("dti", "Not available"),
            "dti_band": static_record.get("dti_band", "Not available"),
            "interest_rate": static_record.get("interest_rate", "Not available"),
            "original_term": static_record.get("original_term", "Not available"),
            "installment": static_record.get("installment", "Not available"),
            "annual_income": static_record.get("annual_income", "Not available"),
            "loan_purpose": static_record.get("loan_purpose", "Not available"),
            "home_ownership": static_record.get("home_ownership", "Not available"),
            "state": static_record.get("state", "Not available"),
            "document_status": static_record.get("document_status", "Not available"),
            "revolving_utilization": static_record.get("revolving_utilization", "Not available"),
            "delinquencies_2yrs": static_record.get("delinquencies_2yrs", "Not available"),
            "inquiries_6m": static_record.get("inquiries_6m", "Not available"),
            "total_accounts": static_record.get("total_accounts", "Not available")
        }

        # Extract predictions
        clean_preds = {
            "default_probability": model_predictions.get("default_probability", "Not available"),
            "delinquency_probability": model_predictions.get("delinquency_probability", "Not available"),
            "prepayment_probability": model_predictions.get("prepayment_probability", "Not available"),
            "predicted_next_state": model_predictions.get("next_state", "Not available"),
            "model_confidence": model_predictions.get("confidence", "Not available")
        }

        # Extract anomaly and triage
        clean_anomaly = {
            "composite_anomaly_score": anomaly_data.get("anomaly_score", "Not available"),
            "anomaly_severity": anomaly_data.get("anomaly_severity", "Not available"),
            "exception_type": anomaly_data.get("exception_type", "Not available"),
            "top_drivers": anomaly_data.get("top_drivers", "Not available"),
            "reasons": anomaly_data.get("reasons", anomaly_data.get("reviewer_reasons", "Not available"))
        }

        clean_triage = {
            "disposition": triage_data.get("reviewer_action", anomaly_data.get("reviewer_action", "Not available")),
            "reasons": triage_data.get("reviewer_reasons", anomaly_data.get("reviewer_reasons", "Not available")),
            "primary_trigger": triage_data.get("reviewer_primary_trigger", "Not available")
        }

        package = {
            "loan_information": clean_static,
            "model_predictions": clean_preds,
            "anomaly_evidence": clean_anomaly,
            "reviewer_triage": clean_triage,
            "data_quality": quality_data if quality_data else {"status": "Quality metrics computed by pipeline"},
            "explainability": expl_data if expl_data else {"status": "Local SHAP/LIME computed on demand"},
            "counterfactuals": cf_data if cf_data else {"status": "Counterfactual search computed on demand"}
        }
        return package

    def generate_loan_reviewer_note(
        self,
        loan_id: str,
        static_record: Dict[str, Any],
        model_predictions: Dict[str, Any],
        anomaly_info: Optional[Dict[str, Any]] = None,
        anomaly_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generates a structured loan review dossier grounded in computed ML metrics."""
        anomaly_payload = anomaly_info or anomaly_evidence or {}
        context = self.build_structured_context(
            loan_id=loan_id,
            static_record=static_record,
            model_predictions=model_predictions,
            anomaly_evidence=anomaly_payload
        )

        p_def = model_predictions.get("default_probability", 0.0)
        p_del = model_predictions.get("delinquency_probability", 0.0)
        p_prep = model_predictions.get("prepayment_probability", 0.0)
        action = anomaly_payload.get("reviewer_action", "AUTO_APPROVE")
        severity = anomaly_payload.get("anomaly_severity", "LOW")
        score = anomaly_payload.get("anomaly_score", 0.0)
        drivers = anomaly_payload.get("top_drivers", "None")

        note = (
            f"### Loan Reviewer Dossier: `{loan_id}`\n\n"
            f"> [!IMPORTANT]\n"
            f"> **Recommendation — requires human review**\n\n"
            f"#### 1. Executive Summary & Recommended Action\n"
            f"- **Triage Status**: `{action}` (Anomaly Severity: `{severity}`, Composite Score: `{score:.1f}/100`)\n"
            f"- **Core ML Trajectory**: Predicted default risk is `{p_def:.1%}` and delinquency risk is `{p_del:.1%}`.\n\n"
            f"#### 2. Risk & Profile Analysis\n"
            f"- **Borrower Profile**: FICO `{static_record.get('credit_score', 'N/A')}`, DTI `{static_record.get('dti', 'N/A')}%`, Rate: `{static_record.get('interest_rate', 'N/A')}%`.\n"
            f"- **Top Drivers**: {drivers}\n\n"
            f"#### 3. Auditor Next Steps\n"
            f"- Review verification status `{static_record.get('document_status', 'N/A')}` and monitor 12-month prepayment outlook ({p_prep:.1%}).\n"
        )

        return {
            "loan_id": loan_id,
            "response": note,
            "governance_notice": "Recommendation — requires human review",
            "context_used": context
        }

    def ask_llm_about_loan(
        self,
        loan_id: str,
        user_question: str,
        structured_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explicitly queries the Groq LLM about the selected loan using grounded evidence."""
        user_prompt = (
            f"Here is the verified analytical evidence package for Loan `{loan_id}`:\n\n"
            f"```json\n{json.dumps(structured_context, indent=2, default=str)}\n```\n\n"
            f"Reviewer Question:\n{user_question}\n\n"
            f"Provide a clear, factual, evidence-backed answer strictly based on the data above. "
            f"Do not invent facts. State if any requested metric is unavailable. "
            f"Formatting: Use clean standard Markdown (**bold** for emphasis, bullet points, standard markdown tables). "
            f"Do not output HTML tags like <b>, </b>, <div>, or <span>."
        )

        # Call Groq API
        result = self.groq_client.generate_response(
            system_prompt=self.SYSTEM_INSTRUCTION,
            user_prompt=user_prompt
        )

        response_text = clean_html_markup(result["response"])
        latency = result["latency_seconds"]
        success = result["success"]
        error = result["error"]
        model_used = result["model"]

        # Log to JSONL audit ledger
        self.audit_logger.log_interaction(
            loan_id=loan_id,
            user_question=user_question,
            model=model_used,
            structured_context=structured_context,
            response=response_text,
            latency=latency,
            success=success,
            error=error
        )

        return {
            "loan_id": loan_id,
            "user_question": user_question,
            "model": model_used,
            "response": response_text,
            "latency_seconds": latency,
            "success": success,
            "error": error,
            "context_used": structured_context,
            "context_hash": CopilotAuditLogger.compute_context_hash(structured_context),
            "governance_notice": "Recommendation — requires human review"
        }
