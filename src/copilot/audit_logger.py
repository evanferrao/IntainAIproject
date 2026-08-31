"""LLM Governance and Audit Logging."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.settings import LOGS_DIR
from src.utils.logger import logger


class CopilotAuditLogger:
    """Logs prompts, retrieved context, generated LLM text, confidence metrics, and reviewer decisions."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (LOGS_DIR / "copilot_audit_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        query: str,
        retrieved_context: Dict[str, Any],
        model_name: str,
        generated_response: str,
        confidence_category: str = "HIGH",
        reviewer_status: str = "PENDING_REVIEW"
    ) -> Dict[str, Any]:
        """Appends an interaction record to the JSONL audit ledger."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "model_name": model_name,
            "retrieved_context": retrieved_context,
            "generated_response": generated_response,
            "confidence_category": confidence_category,
            "reviewer_status": reviewer_status,
            "governance_notice": "Recommendation — requires human review"
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed writing copilot audit log: {e}")

        return record

    def get_recent_audit_logs(self, limit: int = 50) -> list:
        """Reads recent audit log entries."""
        if not self.log_path.exists():
            return []
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line.strip()))
            return entries[-limit:]
        except Exception as e:
            logger.error(f"Failed reading copilot audit logs: {e}")
            return []
