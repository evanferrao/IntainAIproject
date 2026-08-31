"""Automated Executive Intelligence Report Generator."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone

from src.config.settings import REPORTS_DIR
from src.utils.logger import logger


class ExecutiveReportGenerator:
    """Generates publication-ready Markdown intelligence reports from pipeline artifacts."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        profiling_metrics: Dict[str, Any],
        quality_summary: Dict[str, Any],
        drift_summary: Dict[str, Any],
        model_metrics: Dict[str, Any],
        scenario_results: Dict[str, Any],
        anomaly_summary: Dict[str, Any]
    ) -> Path:
        """Generates comprehensive executive summary markdown report."""
        report_path = self.output_dir / "executive_intelligence_report.md"
        logger.info(f"Generating Executive Intelligence Report at {report_path}...")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        md_content = f"""# Loan Performance Intelligence Engine — Executive Intelligence Report

**Generated**: `{timestamp}`  
**Project**: Intain Campus FinTech Challenge 2026 AI Track Prototype  
**Dataset Provenance Notice**: *Prototype implementation utilizes publicly available Lending Club historical data and relationship-driven derived temporal performance data because the organizer-provided competition dataset was unavailable at implementation time.*

---

## 1. Executive Summary & Portfolio Health

- **Total Evaluated Records**: `{quality_summary.get('records_evaluated', 'N/A'):,}`
- **Overall Dataset Quality Score**: `{quality_summary.get('batch_quality_score', 'N/A')}/100` (`{quality_summary.get('quality_status', 'GOOD')}`)
- **Train/Test Drift Status**: `{drift_summary.get('overall_drift_status', 'STABLE')}` ({drift_summary.get('high_drift_features_count', 0)} high-drift features detected)
- **High-Risk Exception Triage**: `{anomaly_summary.get('critical_or_high_exceptions_count', 0):,}` records require manual underwriter audit or source reconciliation.

---

## 2. Predictive Machine Learning Performance

Summary of non-LLM machine learning models evaluated on out-of-time chronological validation data:

| Predictive Task | Model Type | ROC-AUC | PR-AUC | F1-Score | Brier Score | Calibration ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Delinquency (Next 3M)** | Improved LightGBM | `{model_metrics.get('delinquency', {}).get('roc_auc', 'N/A')}` | `{model_metrics.get('delinquency', {}).get('pr_auc', 'N/A')}` | `{model_metrics.get('delinquency', {}).get('f1', 'N/A')}` | `{model_metrics.get('delinquency', {}).get('brier_score', 'N/A')}` | `{model_metrics.get('delinquency', {}).get('expected_calibration_error', 'N/A')}` |
| **Default (Next 12M)** | Improved LightGBM | `{model_metrics.get('default', {}).get('roc_auc', 'N/A')}` | `{model_metrics.get('default', {}).get('pr_auc', 'N/A')}` | `{model_metrics.get('default', {}).get('f1', 'N/A')}` | `{model_metrics.get('default', {}).get('brier_score', 'N/A')}` | `{model_metrics.get('default', {}).get('expected_calibration_error', 'N/A')}` |
| **Prepayment (Next 12M)** | Improved LightGBM | `{model_metrics.get('prepayment', {}).get('roc_auc', 'N/A')}` | `{model_metrics.get('prepayment', {}).get('pr_auc', 'N/A')}` | `{model_metrics.get('prepayment', {}).get('f1', 'N/A')}` | `{model_metrics.get('prepayment', {}).get('brier_score', 'N/A')}` | `{model_metrics.get('prepayment', {}).get('expected_calibration_error', 'N/A')}` |

### Multiclass Next-State Model
- **Next-State Macro F1**: `{model_metrics.get('next_state', {}).get('macro_f1', 'N/A')}`
- **Weighted F1**: `{model_metrics.get('next_state', {}).get('weighted_f1', 'N/A')}`

---

## 3. Macroeconomic Scenario Stress Simulation

*Disclaimer: Scenario outputs are model-based stress projections, not guaranteed causal forecasts.*

| Scenario Name | Macroeconomic Assumption | Projected Delinquency | Projected Default | Projected Prepayment | Expected Loss Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BASE** | Prevailing market baseline | `{scenario_results.get('BASE', {}).get('portfolio_metrics', {}).get('projected_delinquency_rate', 'N/A'):.2%}` | `{scenario_results.get('BASE', {}).get('portfolio_metrics', {}).get('projected_default_rate', 'N/A'):.2%}` | `{scenario_results.get('BASE', {}).get('portfolio_metrics', {}).get('projected_prepayment_rate', 'N/A'):.2%}` | `{scenario_results.get('BASE', {}).get('portfolio_metrics', {}).get('expected_loss_rate', 'N/A'):.2%}` |
| **ADVERSE_CREDIT** | +300 bps unemp, -40 FICO | `{scenario_results.get('ADVERSE_CREDIT', {}).get('portfolio_metrics', {}).get('projected_delinquency_rate', 'N/A'):.2%}` | `{scenario_results.get('ADVERSE_CREDIT', {}).get('portfolio_metrics', {}).get('projected_default_rate', 'N/A'):.2%}` | `{scenario_results.get('ADVERSE_CREDIT', {}).get('portfolio_metrics', {}).get('projected_prepayment_rate', 'N/A'):.2%}` | `{scenario_results.get('ADVERSE_CREDIT', {}).get('portfolio_metrics', {}).get('expected_loss_rate', 'N/A'):.2%}` |
| **HIGH_PREPAYMENT** | -150 bps rate drop, refi surge | `{scenario_results.get('HIGH_PREPAYMENT', {}).get('portfolio_metrics', {}).get('projected_delinquency_rate', 'N/A'):.2%}` | `{scenario_results.get('HIGH_PREPAYMENT', {}).get('portfolio_metrics', {}).get('projected_default_rate', 'N/A'):.2%}` | `{scenario_results.get('HIGH_PREPAYMENT', {}).get('portfolio_metrics', {}).get('projected_prepayment_rate', 'N/A'):.2%}` | `{scenario_results.get('HIGH_PREPAYMENT', {}).get('portfolio_metrics', {}).get('expected_loss_rate', 'N/A'):.2%}` |

---

## 4. Anomaly & Exception Detection Triage

- **Deterministic Rule Violations**: `{anomaly_summary.get('rule_violations_count', 0):,}` records
- **Source Reconciliation Conflicts**: `{anomaly_summary.get('reconciliation_conflicts_count', 0):,}` records
- **Multivariate ML Outliers**: `{anomaly_summary.get('ml_outliers_count', 0):,}` records

---

## 5. Governance and LLM Grounding Notice

All LLM Copilot recommendations and reviewer notes are strictly grounded in:
1. Canonical field definitions (`data_dictionary.md`)
2. Codified business rules (`validation_rules.json`)
3. True computed model outputs and anomaly evidence

**Recommendation — requires human review.**
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return report_path
