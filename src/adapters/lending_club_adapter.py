"""Lending Club to Canonical Loan Schema Adapter Layer."""

import re
from pathlib import Path
from typing import Union, Optional, Dict, Any
import pandas as pd
import numpy as np

from src.config.settings import (
    CREDIT_SCORE_BINS, CREDIT_SCORE_LABELS,
    DTI_BINS, DTI_LABELS, RANDOM_SEED
)
from src.utils.logger import logger


class LendingClubAdapter:
    """Transforms raw Lending Club tabular records into the Canonical Loan Master format."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.column_mapping = {
            "id": "loan_id",
            "issue_d": "origination_month",
            "loan_amnt": "original_balance",
            "funded_amnt": "funded_balance",
            "int_rate": "interest_rate",
            "term": "original_term",
            "installment": "installment",
            "dti": "dti",
            "annual_inc": "annual_income",
            "emp_length": "employment_length",
            "home_ownership": "home_ownership",
            "purpose": "loan_purpose",
            "addr_state": "state",
            "revol_util": "revolving_utilization",
            "delinq_2yrs": "delinquencies_2yrs",
            "inq_last_6mths": "inquiries_6m",
            "total_acc": "total_accounts",
            "verification_status": "document_status"
        }

    def parse_issue_date(self, val: Any) -> str:
        """Standardizes issue_d (e.g. 'Dec-2015', '2015-12-01', 'Dec-15') into 'YYYY-MM'."""
        if pd.isna(val):
            return "2018-01"
        val_str = str(val).strip()
        
        # Format: 'Dec-2015' or 'Dec-15'
        match = re.match(r"^([A-Za-z]{3})-(\d{2,4})$", val_str)
        if match:
            month_str, year_str = match.groups()
            if len(year_str) == 2:
                year_int = int(year_str)
                year_str = f"20{year_str}" if year_int < 50 else f"19{year_str}"
            try:
                dt = pd.to_datetime(f"{month_str}-01-{year_str}", format="%b-%d-%Y")
                return dt.strftime("%Y-%m")
            except Exception:
                pass

        # Try general pd.to_datetime
        try:
            dt = pd.to_datetime(val_str)
            return dt.strftime("%Y-%m")
        except Exception:
            return "2018-01"

    def parse_term(self, val: Any) -> int:
        """Parses term (e.g. ' 36 months', '60 months', 36) into integer months."""
        if pd.isna(val):
            return 36
        val_str = str(val).strip()
        match = re.search(r"(\d+)", val_str)
        if match:
            return int(match.group(1))
        return 36

    def parse_emp_length(self, val: Any) -> int:
        """Parses emp_length (e.g. '10+ years', '< 1 year', '3 years') to integer 0..10."""
        if pd.isna(val):
            return 0
        val_str = str(val).strip().lower()
        if "10+" in val_str:
            return 10
        if "< 1" in val_str:
            return 0
        match = re.search(r"(\d+)", val_str)
        if match:
            return min(10, max(0, int(match.group(1))))
        return 0

    def parse_revol_util(self, val: Any) -> float:
        """Parses revolving utilization string/float to float."""
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).replace("%", "").strip()
        try:
            return float(val_str)
        except ValueError:
            return 0.0

    def parse_interest_rate(self, val: Any) -> float:
        """Parses interest rate string/float to float percentage (e.g. 12.5% -> 12.5)."""
        if pd.isna(val):
            return 12.0
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).replace("%", "").strip()
        try:
            return float(val_str)
        except ValueError:
            return 12.0

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw Lending Club dataframe to Canonical Static Loan dataframe."""
        logger.info(f"Transforming raw dataframe of shape {raw_df.shape} via LendingClubAdapter...")
        df = raw_df.copy()

        # Generate loan_id if missing or non-string
        if "id" not in df.columns or df["id"].isna().all():
            df["loan_id"] = [f"LC_{i:07d}" for i in range(len(df))]
        else:
            df["loan_id"] = df["id"].astype(str).str.strip()

        # Dates
        if "issue_d" in df.columns:
            df["origination_month"] = df["issue_d"].apply(self.parse_issue_date)
        else:
            df["origination_month"] = "2018-01"

        # Terms
        if "term" in df.columns:
            df["original_term"] = df["term"].apply(self.parse_term)
        else:
            df["original_term"] = 36

        # Balances
        df["original_balance"] = pd.to_numeric(df.get("loan_amnt", 10000.0), errors="coerce").fillna(10000.0)
        df["funded_balance"] = pd.to_numeric(df.get("funded_amnt", df["original_balance"]), errors="coerce").fillna(df["original_balance"])
        
        # Rates & Payments
        if "int_rate" in df.columns:
            df["interest_rate"] = df["int_rate"].apply(self.parse_interest_rate)
        else:
            df["interest_rate"] = 12.5

        if "installment" in df.columns:
            df["installment"] = pd.to_numeric(df["installment"], errors="coerce").fillna(df["original_balance"] / df["original_term"])
        else:
            df["installment"] = df["original_balance"] / df["original_term"]

        # Credit Score (FICO)
        if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
            low = pd.to_numeric(df["fico_range_low"], errors="coerce")
            high = pd.to_numeric(df["fico_range_high"], errors="coerce")
            df["credit_score"] = ((low + high) / 2.0).fillna(690.0)
        elif "fico_range_low" in df.columns:
            df["credit_score"] = pd.to_numeric(df["fico_range_low"], errors="coerce").fillna(690.0)
        else:
            df["credit_score"] = 690.0
            
        df["credit_score"] = df["credit_score"].clip(300, 850)
        df["credit_score_band"] = pd.cut(
            df["credit_score"],
            bins=CREDIT_SCORE_BINS,
            labels=CREDIT_SCORE_LABELS,
            right=True
        ).astype(str)

        # DTI
        df["dti"] = pd.to_numeric(df.get("dti", 15.0), errors="coerce").fillna(15.0).clip(0.0, 150.0)
        df["dti_band"] = pd.cut(
            df["dti"] / 100.0,
            bins=DTI_BINS,
            labels=DTI_LABELS,
            right=True
        ).astype(str)

        # Incomes & Employment
        df["annual_income"] = pd.to_numeric(df.get("annual_inc", 65000.0), errors="coerce").fillna(65000.0).clip(lower=1000.0)
        if "emp_length" in df.columns:
            df["employment_length"] = df["emp_length"].apply(self.parse_emp_length)
        else:
            df["employment_length"] = 3

        # Categoricals
        df["home_ownership"] = df.get("home_ownership", "RENT").astype(str).str.upper().replace({"NONE": "OTHER", "ANY": "OTHER"})
        df["loan_purpose"] = df.get("purpose", "debt_consolidation").astype(str).str.lower().fillna("debt_consolidation")
        df["state"] = df.get("addr_state", "CA").astype(str).str.upper().str.strip()
        df["state"] = df["state"].apply(lambda s: s if len(s) == 2 and s.isalpha() else "CA")

        # Credit History
        if "revol_util" in df.columns:
            df["revolving_utilization"] = df["revol_util"].apply(self.parse_revol_util).clip(0.0, 200.0)
        else:
            df["revolving_utilization"] = 50.0

        df["delinquencies_2yrs"] = pd.to_numeric(df.get("delinq_2yrs", 0), errors="coerce").fillna(0).astype(int).clip(lower=0)
        df["inquiries_6m"] = pd.to_numeric(df.get("inq_last_6mths", 0), errors="coerce").fillna(0).astype(int).clip(lower=0)
        df["total_accounts"] = pd.to_numeric(df.get("total_acc", 15), errors="coerce").fillna(15).astype(int).clip(lower=1)

        # Document Verification Status
        doc_raw = df.get("verification_status", "Verified").astype(str).str.upper()
        df["document_status"] = doc_raw.apply(
            lambda x: "SOURCE_VERIFIED" if "SOURCE" in x
            else "UNVERIFIED" if ("NOT" in x or "UN" in x or "NONE" in x)
            else "VERIFIED"
        )

        # Prototype Servicer Assignment
        servicers = ["Intain_Servicing_Alpha", "Apex_Loan_Servicing", "Beacon_Credit_Ops", "Summit_Asset_Management"]
        np.random.seed(self.seed)
        df["servicer_name"] = np.random.choice(servicers, size=len(df), p=[0.40, 0.25, 0.20, 0.15])

        # Select canonical columns in exact canonical order
        canonical_cols = [
            "loan_id", "origination_month", "original_balance", "funded_balance",
            "interest_rate", "original_term", "installment", "credit_score",
            "credit_score_band", "dti", "dti_band", "annual_income",
            "employment_length", "home_ownership", "loan_purpose", "state",
            "revolving_utilization", "delinquencies_2yrs", "inquiries_6m",
            "total_accounts", "servicer_name", "document_status"
        ]
        
        canonical_df = df[canonical_cols].copy()
        logger.info(f"Successfully transformed to canonical static loan master with shape {canonical_df.shape}")
        return canonical_df
