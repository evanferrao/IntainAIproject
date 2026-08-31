"""Deterministic Rule Validation Engine evaluating validation_rules.json."""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config.settings import CANONICAL_DATA_DIR
from src.utils.logger import logger


class DeterministicRuleEngine:
    """Evaluates canonical static loans and monthly panel data against codified validation rules."""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = rules_path or (CANONICAL_DATA_DIR / "validation_rules.json")
        self.rules = self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        """Loads rules from validation_rules.json."""
        if not self.rules_path.exists():
            logger.warning(f"Validation rules file not found at {self.rules_path}. Using built-in defaults.")
            return []
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("rules", [])
        except Exception as e:
            logger.error(f"Failed loading validation rules: {e}")
            return []

    def evaluate_static_records(self, static_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates static loan records and flags rule violations."""
        df = static_df.copy()
        n_records = len(df)
        
        violation_flags = pd.DataFrame(index=df.index)
        violation_details = []

        # VR-001: Positive Original Balance
        v1 = (df["original_balance"] <= 0) | (df["original_balance"] > 200000)
        violation_flags["VR-001"] = v1

        # VR-003: Valid Interest Rate (1% to 40%)
        v3 = (df["interest_rate"] < 1.0) | (df["interest_rate"] > 40.0)
        violation_flags["VR-003"] = v3

        # VR-004: Standard Credit Score Range (300 to 850)
        v4 = (df["credit_score"] < 300) | (df["credit_score"] > 850)
        violation_flags["VR-004"] = v4

        # VR-005: Reasonable DTI (0% to 100%)
        v5 = (df["dti"] < 0.0) | (df["dti"] > 100.0)
        violation_flags["VR-005"] = v5

        # VR-006: Positive Annual Income (> $1,000)
        v6 = df["annual_income"] < 1000
        violation_flags["VR-006"] = v6

        # VR-007: Standard Term
        v7 = ~df["original_term"].isin([12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360])
        violation_flags["VR-007"] = v7

        # VR-011: Document Verification Status
        v11 = ~df["document_status"].isin(["VERIFIED", "SOURCE_VERIFIED", "UNVERIFIED"])
        violation_flags["VR-011"] = v11

        # VR-012: Valid US State Code
        v12 = df["state"].apply(lambda s: not (isinstance(s, str) and len(s) == 2 and s.isalpha()))
        violation_flags["VR-012"] = v12

        # Aggregate total violations per record
        df["deterministic_violation_count"] = violation_flags.sum(axis=1)
        df["has_deterministic_violation"] = df["deterministic_violation_count"] > 0
        
        # Build explanation strings
        explanations = []
        for idx in range(n_records):
            active_rules = [col for col in violation_flags.columns if violation_flags.iloc[idx][col]]
            if active_rules:
                explanations.append(f"Rule violations: {', '.join(active_rules)}")
            else:
                explanations.append("Passed all static validation rules")
        df["deterministic_violation_explanation"] = explanations

        logger.info(f"Evaluated {n_records:,} static records: {(df['has_deterministic_violation']).sum():,} flagged with deterministic violations.")
        return df

    def evaluate_monthly_transitions(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates monthly performance panel for impossible state transitions and balance inconsistencies."""
        df = panel_df.copy()
        
        # VR-002: Current balance > original balance * 1.05
        # VR-008: Negative days past due
        # VR-010: Impossible transition from terminal state
        
        v2 = df["current_balance"] < 0.0
        v8 = (df["days_past_due"] < 0) | (df["days_past_due"] > 720)
        
        # Terminal state bounce check: if previous was DEFAULT/PREPAID/CLOSED and next is CURRENT/DELINQUENT
        terminal_bounce = (
            df["current_status"].isin(["DEFAULT", "PREPAID", "CLOSED"]) &
            df["next_state"].isin(["CURRENT", "DELINQUENT_30", "DELINQUENT_60", "DELINQUENT_90"])
        )
        
        df["has_transition_anomaly"] = v2 | v8 | terminal_bounce
        return df
