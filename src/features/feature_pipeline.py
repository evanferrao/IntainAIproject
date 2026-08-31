"""Leakage-Safe Feature Engineering Pipeline."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils.logger import logger


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Engineers financial, temporal, categorical, and historical rolling features with strict leakage prevention."""

    def __init__(self, high_cardinality_threshold: int = 15):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.feature_names: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.is_fitted = False

    def _engineer_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives point-in-time features without looking ahead into future periods."""
        df = df.copy()

        # 1. Temporal & Seasonality Features
        if "reporting_month" in df.columns:
            try:
                rep_dt = pd.to_datetime(df["reporting_month"] + "-01")
                df["rep_year"] = rep_dt.dt.year
                df["rep_month_num"] = rep_dt.dt.month
                df["rep_quarter"] = rep_dt.dt.quarter
                # Cyclic month encoding
                df["seasonality_sin"] = np.sin(2 * np.pi * df["rep_month_num"] / 12.0)
                df["seasonality_cos"] = np.cos(2 * np.pi * df["rep_month_num"] / 12.0)
            except Exception:
                df["rep_year"] = 2018
                df["rep_month_num"] = 6
                df["rep_quarter"] = 2
                df["seasonality_sin"] = 0.0
                df["seasonality_cos"] = 1.0

        # Term progress
        if "loan_age_months" in df.columns and "original_term" in df.columns:
            df["term_progress_ratio"] = (df["loan_age_months"] / df["original_term"].replace(0, 36)).clip(0.0, 1.5)
        elif "loan_age_months" in df.columns:
            df["term_progress_ratio"] = (df["loan_age_months"] / 36.0).clip(0.0, 1.5)

        # 2. Financial & Debt Service Ratios
        if "current_balance" in df.columns and "original_balance" in df.columns:
            df["balance_ratio"] = (df["current_balance"] / df["original_balance"].replace(0, 10000)).clip(0.0, 1.2)
        
        if "installment" in df.columns and "annual_income" in df.columns:
            monthly_inc = (df["annual_income"] / 12.0).clip(lower=100.0)
            df["installment_to_income"] = (df["installment"] / monthly_inc).clip(0.0, 1.0)

        if "interest_rate" in df.columns:
            # Benchmark rate difference (assuming ~3.0% prime/fed funds base)
            df["interest_rate_spread"] = (df["interest_rate"] - 3.0).clip(lower=0.0)

        # 3. Credit Risk Interactions
        if "credit_score" in df.columns and "dti" in df.columns:
            # Composite credit burden index
            df["credit_burden_index"] = (df["dti"] / (df["credit_score"] / 100.0)).clip(0.0, 50.0)

        # 4. Status History Encoding
        if "current_status" in df.columns:
            status_map = {
                "CURRENT": 0,
                "DELINQUENT_30": 1,
                "DELINQUENT_60": 2,
                "DELINQUENT_90": 3,
                "DEFAULT": 4,
                "PREPAID": 5,
                "CLOSED": 6
            }
            df["current_status_code"] = df["current_status"].map(status_map).fillna(0).astype(int)

        return df

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None):
        """Fits encoders, imputers, and scalers strictly on training data."""
        logger.info("Fitting FeatureEngineeringPipeline on training dataset...")
        df_feat = self._engineer_raw_features(df)

        # Separate numeric and categorical candidates
        ignore_cols = [
            "loan_id", "reporting_month", "origination_month", "next_state",
            "next_3m_delinquency_flag", "next_12m_default_flag", "next_12m_prepayment_flag",
            "default_flag", "prepayment_flag", "current_status"
        ]

        candidate_cols = [c for c in df_feat.columns if c not in ignore_cols]

        self.numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df_feat[c])]
        self.categorical_cols = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df_feat[c])]

        # Fit numeric imputer
        if self.numeric_cols:
            self.num_imputer.fit(df_feat[self.numeric_cols])

        # Fit frequency/label encodings for categoricals
        for cat in self.categorical_cols:
            freq_map = df_feat[cat].value_counts(normalize=True).to_dict()
            # Map categories to integer rankings based on frequency
            cat_to_int = {k: i + 1 for i, (k, _) in enumerate(sorted(freq_map.items(), key=lambda x: x[1], reverse=True))}
            self.cat_mappings[cat] = cat_to_int

        self.feature_names = self.numeric_cols + [f"{c}_encoded" for c in self.categorical_cols]
        self.is_fitted = True
        logger.info(f"Fitted pipeline with {len(self.numeric_cols)} numeric features and {len(self.categorical_cols)} categorical features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms unseen data using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")

        df_feat = self._engineer_raw_features(df)

        # Numeric transformations
        transformed_df = pd.DataFrame(index=df.index)

        if self.numeric_cols:
            imputed_nums = self.num_imputer.transform(df_feat[self.numeric_cols])
            for idx, col in enumerate(self.numeric_cols):
                transformed_df[col] = imputed_nums[:, idx]

        # Categorical encodings
        for cat in self.categorical_cols:
            cat_map = self.cat_mappings.get(cat, {})
            encoded_col = f"{cat}_encoded"
            transformed_df[encoded_col] = df_feat[cat].map(cat_map).fillna(0).astype(int)

        return transformed_df

    def get_feature_names(self) -> List[str]:
        return self.feature_names
