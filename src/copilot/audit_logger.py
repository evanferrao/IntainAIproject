"""LLM Governance and Audit Logging."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config.settings import LOGS_DIR
from src.utils.logger import logger


class CopilotAuditLogger:
    """Logs prompts, retrieved context hashes, generated LLM text, latency, and reviewer interactions."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (LOGS_DIR / "copilot_audit_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_context_hash(context: Dict[str, Any]) -> str:
        """Computes a deterministic SHA-256 hash of the evidence context package."""
        try:
            serialized = json.dumps(context, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "unknown_hash"

    def log_interaction(
        self,
        loan_id: str,
        user_question: str,
        model: str,
        structured_context: Dict[str, Any],
        response: Optional[str],
        latency: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Appends an interaction record to the JSONL audit ledger."""
        context_hash = self.compute_context_hash(structured_context)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loan_id": str(loan_id),
            "model": str(model),
            "user_question": str(user_question),
            "context_hash": context_hash,
            "structured_context": structured_context,
            "response": response if response is not None else "Not available",
            "latency": round(float(latency), 3),
            "success": bool(success),
            "error": error,
            "governance_notice": "Recommendation — requires human review"
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed writing copilot audit log: {e}")

        return record

    def get_recent_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
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
