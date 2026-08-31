"""Time-Aware Chronological Dataset Splitting with Leakage Controls."""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.utils.logger import logger


class TimeAwareSplitter:
    """Performs strict chronological splits on temporal monthly panel datasets.
    
    Leakage Controls:
    1. Chronological Cutoff: All observations in Train have reporting_month <= cutoff.
    2. Group / Loan Isolation: All observations for a loan belong to either Train OR Test,
       or historical months strictly precede the validation window.
    """

    def __init__(self, test_months: int = 6, time_col: str = "reporting_month", group_col: str = "loan_id"):
        self.test_months = test_months
        self.time_col = time_col
        self.group_col = group_col

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Splits DataFrame into Train and Test subsets chronologically."""
        if self.time_col not in df.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in dataframe.")

        sorted_months = sorted(df[self.time_col].dropna().unique())
        n_unique_months = len(sorted_months)

        if n_unique_months <= self.test_months:
            # Fallback if fewer months than window: 75% chronological split
            split_idx = int(n_unique_months * 0.75)
            train_months = sorted_months[:split_idx]
            test_months = sorted_months[split_idx:]
        else:
            split_idx = n_unique_months - self.test_months
            train_months = sorted_months[:split_idx]
            test_months = sorted_months[split_idx:]

        cutoff_month = train_months[-1] if train_months else sorted_months[0]
        test_start_month = test_months[0] if test_months else sorted_months[-1]

        logger.info(f"Time-Aware Split: Train range [{train_months[0]} -> {cutoff_month}], Test range [{test_start_month} -> {test_months[-1]}]")

        train_df = df[df[self.time_col].isin(train_months)].copy().reset_index(drop=True)
        test_df = df[df[self.time_col].isin(test_months)].copy().reset_index(drop=True)

        split_metadata = {
            "time_column": self.time_col,
            "train_months": train_months,
            "test_months": test_months,
            "cutoff_month": cutoff_month,
            "test_start_month": test_start_month,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_unique_loans": int(train_df[self.group_col].nunique()) if self.group_col in train_df else len(train_df),
            "test_unique_loans": int(test_df[self.group_col].nunique()) if self.group_col in test_df else len(test_df),
        }

        logger.info(f"Split completed: {len(train_df):,} train rows, {len(test_df):,} test rows.")
        return train_df, test_df, split_metadata
