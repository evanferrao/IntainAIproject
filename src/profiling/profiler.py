"""Data Intelligence and Profiling Engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import RANDOM_SEED
from src.utils.logger import logger


class DataProfiler:
    """Performs deep profiling on canonical datasets: statistics, missingness, outliers, and relationships."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed

    def profile_dataset(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """Generates comprehensive profiling metadata for a given dataframe."""
        logger.info(f"Running comprehensive data profiling on {dataset_name} ({len(df):,} rows, {len(df.columns)} cols)...")
        
        n_rows, n_cols = df.shape
        col_profiles = {}
        missing_by_col = {}
        missing_pct_by_col = {}
        
        for col in df.columns:
            series = df[col]
            n_missing = int(series.isna().sum())
            missing_pct = float(n_missing / n_rows) if n_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype)
            
            missing_by_col[col] = n_missing
            missing_pct_by_col[col] = round(missing_pct * 100.0, 2)
            
            col_info: Dict[str, Any] = {
                "dtype": dtype_str,
                "n_missing": n_missing,
                "missing_pct": round(missing_pct * 100.0, 2),
                "n_unique": n_unique,
            }
            
            if pd.api.types.is_numeric_dtype(series):
                clean_num = series.dropna()
                if len(clean_num) > 0:
                    q25, q50, q75 = np.percentile(clean_num, [25, 50, 75])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    n_outliers = int(((clean_num < lower_bound) | (clean_num > upper_bound)).sum())
                    
                    col_info.update({
                        "mean": round(float(clean_num.mean()), 4),
                        "std": round(float(clean_num.std()), 4),
                        "min": round(float(clean_num.min()), 4),
                        "q25": round(float(q25), 4),
                        "median": round(float(q50), 4),
                        "q75": round(float(q75), 4),
                        "max": round(float(clean_num.max()), 4),
                        "skewness": round(float(stats.skew(clean_num)), 4) if len(clean_num) > 2 else 0.0,
                        "kurtosis": round(float(stats.kurtosis(clean_num)), 4) if len(clean_num) > 2 else 0.0,
                        "iqr": round(float(iqr), 4),
                        "n_iqr_outliers": n_outliers,
                        "outlier_pct": round(float(n_outliers / len(clean_num) * 100.0), 2)
                    })
            else:
                top_cats = series.value_counts(dropna=False).head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in top_cats.items()}
                
            col_profiles[col] = col_info

        # Row-level missingness distribution
        row_missing = df.isna().sum(axis=1)
        row_missing_dist = {
            "0_missing_rows_pct": round(float((row_missing == 0).mean() * 100.0), 2),
            "1_to_2_missing_rows_pct": round(float(((row_missing >= 1) & (row_missing <= 2)).mean() * 100.0), 2),
            "3plus_missing_rows_pct": round(float((row_missing >= 3).mean() * 100.0), 2),
            "max_missing_in_single_row": int(row_missing.max())
        }

        # Numeric Correlation Analysis
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = {}
        high_correlations = []
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr()
            corr_matrix = corr_df.round(3).to_dict()
            
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_df.loc[c1, c2]
                    if not pd.isna(val) and abs(val) >= 0.65:
                        high_correlations.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": round(float(val), 4)
                        })

        profile_result = {
            "dataset_name": dataset_name,
            "row_count": n_rows,
            "column_count": n_cols,
            "columns": col_profiles,
            "missingness": {
                "column_missing_counts": missing_by_col,
                "column_missing_pct": missing_pct_by_col,
                "row_missing_distribution": row_missing_dist
            },
            "high_correlations": high_correlations,
            "numeric_correlations": corr_matrix
        }
        
        return profile_result

    def validate_dates_and_integrity(
        self,
        static_df: pd.DataFrame,
        monthly_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Validates date integrity, monotonic ordering, and impossible values."""
        issues = []
        
        # Static checks
        if "origination_month" in static_df.columns:
            invalid_orig = static_df["origination_month"].isna().sum()
            if invalid_orig > 0:
                issues.append(f"{invalid_orig} static records have missing origination_month.")
                
        if "original_balance" in static_df.columns:
            neg_bal = (static_df["original_balance"] <= 0).sum()
            if neg_bal > 0:
                issues.append(f"{neg_bal} static records have non-positive original_balance.")
                
        if "credit_score" in static_df.columns:
            oob_fico = ((static_df["credit_score"] < 300) | (static_df["credit_score"] > 850)).sum()
            if oob_fico > 0:
                issues.append(f"{oob_fico} static records have out-of-bounds FICO score (not in 300..850).")

        # Monthly checks
        if monthly_df is not None:
            if "reporting_month" in monthly_df.columns and "loan_id" in monthly_df.columns:
                merged = monthly_df[["loan_id", "reporting_month", "current_balance", "days_past_due"]].merge(
                    static_df[["loan_id", "origination_month", "original_balance"]],
                    on="loan_id",
                    how="inner"
                )
                
                # Check reporting month < origination month
                date_violations = (merged["reporting_month"] < merged["origination_month"]).sum()
                if date_violations > 0:
                    issues.append(f"{date_violations} monthly records report dates before origination_month.")
                    
                # Check current balance > original balance * 1.05
                bal_violations = (merged["current_balance"] > merged["original_balance"] * 1.05).sum()
                if bal_violations > 0:
                    issues.append(f"{bal_violations} monthly records report current_balance > 105% of original_balance.")

        return {
            "total_integrity_issues_found": len(issues),
            "is_valid": len(issues) == 0,
            "issues": issues
        }
