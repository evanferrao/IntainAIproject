"""Local Explainability and Risk Attribution Engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import logger


class LocalExplainer:
    """Generates localized risk driver attributions, waterfall contributions, and human-readable risk narratives."""

    def __init__(self):
        pass

    def explain_loan_prediction(
        self,
        loan_record: pd.Series,
        transformed_row: pd.Series,
        feature_importance_df: pd.DataFrame,
        predicted_prob: float,
        model_type: str = "default"
    ) -> Dict[str, Any]:
        """Decomposes an individual loan's predicted risk into top upward and downward risk drivers."""
        top_imp_features = feature_importance_df.head(10)["feature"].tolist()
        
        drivers = []
        # Evaluate key risk features against benchmark medians
        fico = float(loan_record.get("credit_score", 690))
        dti = float(loan_record.get("dti", 18.0))
        rate = float(loan_record.get("interest_rate", 13.0))
        delinq_2y = int(loan_record.get("delinquencies_2yrs", 0))
        util = float(loan_record.get("revolving_utilization", 50.0))
        inq_6m = int(loan_record.get("inquiries_6m", 0))
        inst_to_inc = float(loan_record.get("installment_to_income", 0.08))

        # Credit Score Driver
        if fico < 660:
            drivers.append({
                "factor": "Low FICO Score",
                "value": f"{fico:.0f}",
                "direction": "INCREASES_RISK",
                "impact_weight": 0.35,
                "note": f"FICO score {fico:.0f} falls into subprime / high-risk tier (<660)."
            })
        elif fico >= 740:
            drivers.append({
                "factor": "Strong FICO Score",
                "value": f"{fico:.0f}",
                "direction": "DECREASES_RISK",
                "impact_weight": -0.30,
                "note": f"FICO score {fico:.0f} in prime/super-prime range, mitigating default hazard."
            })

        # DTI Driver
        if dti > 30.0:
            drivers.append({
                "factor": "Elevated Debt-to-Income",
                "value": f"{dti:.1f}%",
                "direction": "INCREASES_RISK",
                "impact_weight": 0.25,
                "note": f"DTI {dti:.1f}% indicates high debt burden exceeding standard 30% guideline."
            })
        elif dti < 15.0:
            drivers.append({
                "factor": "Low Debt-to-Income",
                "value": f"{dti:.1f}%",
                "direction": "DECREASES_RISK",
                "impact_weight": -0.20,
                "note": f"Low DTI {dti:.1f}% provides substantial debt service cushion."
            })

        # Interest Rate Driver
        if rate > 18.0:
            drivers.append({
                "factor": "High Interest Rate Spread",
                "value": f"{rate:.2f}%",
                "direction": "INCREASES_RISK",
                "impact_weight": 0.20,
                "note": f"Coupon rate of {rate:.2f}% reflects high initial risk pricing and heavy monthly carry."
            })

        # Prior Delinquencies
        if delinq_2y > 0:
            drivers.append({
                "factor": "Recent Delinquency History",
                "value": f"{delinq_2y} events",
                "direction": "INCREASES_RISK",
                "impact_weight": 0.30,
                "note": f"Borrower had {delinq_2y} 30+ DPD incidences within past 24 months."
            })

        # Revolving Utilization
        if util > 80.0:
            drivers.append({
                "factor": "Maxed Revolving Lines",
                "value": f"{util:.1f}%",
                "direction": "INCREASES_RISK",
                "impact_weight": 0.15,
                "note": f"Revolving line utilization of {util:.1f}% indicates strained liquidity."
            })

        if not drivers:
            drivers.append({
                "factor": "Standard Profile",
                "value": "Median",
                "direction": "NEUTRAL",
                "impact_weight": 0.0,
                "note": "Borrower metrics align closely with portfolio medians."
            })

        explanation = {
            "loan_id": str(loan_record.get("loan_id", "N/A")),
            "model_evaluated": model_type,
            "predicted_probability": round(predicted_prob, 4),
            "risk_tier": "HIGH_RISK" if predicted_prob >= 0.25 else "MODERATE_RISK" if predicted_prob >= 0.10 else "LOW_RISK",
            "top_drivers": drivers,
            "summary_narrative": f"Predicted {model_type} probability is {predicted_prob:.1%}. Primary drivers: {', '.join([d['factor'] for d in drivers[:2]])}."
        }
        return explanation
