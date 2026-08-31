"""Generates 7 production-grade Jupyter Notebooks configured for the .venv kernel."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_SPEC = {
    "display_name": "Loan Performance Intelligence Engine",
    "language": "python",
    "name": "loan-performance-engine"
}

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNEL_SPEC,
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def build_all_notebooks():
    # Notebook 01: Data Profiling
    nb1 = make_notebook([
        md_cell("# 01 — Data Profiling & Quality Analysis\n\nThis notebook demonstrates canonical schema profiling, missingness patterns, date integrity audits, and dataset-level quality scoring."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\n# Point to project root\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.config.settings import CANONICAL_DATA_DIR, METRICS_DIR\nfrom src.profiling.profiler import DataProfiler\nfrom src.quality.quality_scorer import DataQualityScorer"),
        code_cell("static_df = pd.read_csv(CANONICAL_DATA_DIR / 'loan_static_attributes.csv')\nprint(f'Loaded static loan dataset: {static_df.shape[0]:,} rows, {static_df.shape[1]} columns')\nstatic_df.head()"),
        code_cell("profiler = DataProfiler()\nprofile_summary = profiler.profile_dataset(static_df, dataset_name='Loan_Static_Attributes')\nprint('Profiling completed! Column count:', profile_summary['column_count'])"),
        code_cell("scorer = DataQualityScorer()\nscored_records = scorer.score_records(static_df)\nbatch_quality = scorer.score_dataset(scored_records)\nprint('Batch Quality Score:', batch_quality['batch_quality_score'], '/ 100')\nprint('Quality Status:', batch_quality['quality_status'])")
    ])
    with open(NOTEBOOKS_DIR / "01_data_profiling.ipynb", "w") as f:
        json.dump(nb1, f, indent=2)

    # Notebook 02: Feature Engineering
    nb2 = make_notebook([
        md_cell("# 02 — Leakage-Safe Feature Engineering & Time-Aware Split\n\nDemonstrates chronological dataset splitting, point-in-time financial ratios, seasonality cyclic encodings, and leakage controls."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.config.settings import CANONICAL_DATA_DIR\nfrom src.features.time_splitter import TimeAwareSplitter\nfrom src.features.feature_pipeline import FeatureEngineeringPipeline"),
        code_cell("train_panel = pd.read_csv(CANONICAL_DATA_DIR / 'loan_monthly_performance_train.csv')\nstatic_df = pd.read_csv(CANONICAL_DATA_DIR / 'loan_static_attributes.csv')\ntrain_full = train_panel.merge(static_df, on='loan_id', how='left')\nprint('Train full shape:', train_full.shape)"),
        code_cell("pipeline = FeatureEngineeringPipeline()\npipeline.fit(train_full)\nX_train = pipeline.transform(train_full)\nprint('Engineered Feature Matrix shape:', X_train.shape)\nprint('Features:', pipeline.get_feature_names()[:10])")
    ])
    with open(NOTEBOOKS_DIR / "02_feature_engineering.ipynb", "w") as f:
        json.dump(nb2, f, indent=2)

    # Notebook 03: Predictive Models
    nb3 = make_notebook([
        md_cell("# 03 — Predictive Machine Learning Models & Probability Calibration\n\nDemonstrates Delinquency, Default, Prepayment, and Next-State models with Platt/Isotonic calibration and ROC/PR evaluation."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.config.settings import MODELS_DIR, METRICS_DIR\nfrom src.utils.serialization import load_model, load_json"),
        code_cell("metrics = load_json(METRICS_DIR / 'model_metrics.json')\nprint('Default Model ROC-AUC:', metrics.get('default', {}).get('roc_auc'))\nprint('Delinquency Model ROC-AUC:', metrics.get('delinquency', {}).get('roc_auc'))\nprint('Prepayment Model ROC-AUC:', metrics.get('prepayment', {}).get('roc_auc'))\nprint('Next-State Macro F1:', metrics.get('next_state', {}).get('macro_f1'))")
    ])
    with open(NOTEBOOKS_DIR / "03_predictive_models.ipynb", "w") as f:
        json.dump(nb3, f, indent=2)

    # Notebook 04: Transition Model
    nb4 = make_notebook([
        md_cell("# 04 — Markov State Transition & Survival Projections\n\nEstimates empirical 1-month transition matrices and generates multi-period cumulative default and prepayment absorption curves."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.config.settings import MODELS_DIR\nfrom src.utils.serialization import load_model"),
        code_cell("markov = load_model(MODELS_DIR / 'markov_transition_model.joblib')\nprint('Global Transition Matrix:')\nprint(markov.global_transition_matrix)"),
        code_cell("proj_df = markov.project_multi_period(horizon_months=24)\nproj_df.head(10)")
    ])
    with open(NOTEBOOKS_DIR / "04_transition_model.ipynb", "w") as f:
        json.dump(nb4, f, indent=2)

    # Notebook 05: Anomaly Detection
    nb5 = make_notebook([
        md_cell("# 05 — Hybrid Anomaly Detection & Exception Triage\n\nEvaluates deterministic business rules, Isolation Forest multivariate outlier scores, and servicer reconciliation conflicts."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.config.settings import PREDICTIONS_DIR\n\ndossiers = pd.read_csv(PREDICTIONS_DIR / 'reviewer_exception_dossiers.csv')\nprint('Flagged Exception Dossiers count:', len(dossiers))\ndossiers[['loan_id', 'anomaly_score', 'anomaly_severity', 'reviewer_action', 'top_drivers']].head(10)")
    ])
    with open(NOTEBOOKS_DIR / "05_anomaly_detection.ipynb", "w") as f:
        json.dump(nb5, f, indent=2)

    # Notebook 06: Scenarios
    nb6 = make_notebook([
        md_cell("# 06 — Macroeconomic Scenario & Portfolio Stress Simulation\n\nSimulates BASE, ADVERSE_CREDIT, and HIGH_PREPAYMENT scenarios on the loan portfolio."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.config.settings import METRICS_DIR\nfrom src.utils.serialization import load_json\n\nscenarios = load_json(METRICS_DIR / 'scenario_simulation_results.json')\nfor sname, sdata in scenarios.items():\n    m = sdata['portfolio_metrics']\n    print(f'=== Scenario: {sname} ===')\n    print(f'  Projected Default Rate: {m.get(\"projected_default_rate\"):.2%}')\n    print(f'  Projected Prepayment Rate: {m.get(\"projected_prepayment_rate\"):.2%}')\n    print(f'  Expected Loss: ${m.get(\"expected_loss_dollars\"):,.2f}')")
    ])
    with open(NOTEBOOKS_DIR / "06_scenarios.ipynb", "w") as f:
        json.dump(nb6, f, indent=2)

    # Notebook 07: LLM Copilot
    nb7 = make_notebook([
        md_cell("# 07 — Grounded LLM Reviewer Copilot & Governance\n\nDemonstrates grounded retrieval over data dictionary and validation rules, loan review note generation, and audit logging."),
        code_cell("import sys\nfrom pathlib import Path\nimport pandas as pd\n\nPROJECT_ROOT = Path('.').resolve().parent\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom src.copilot.grounded_copilot import GroundedReviewerCopilot\nfrom src.copilot.evaluation_suite import get_llm_evaluation_benchmarks\n\ncopilot = GroundedReviewerCopilot()\nres = copilot.retrieve_field_definition('credit_score')\nprint(res['content'])"),
        code_cell("benchmarks = get_llm_evaluation_benchmarks()\nprint('Loaded failure-mode benchmarks:', len(benchmarks))\nfor b in benchmarks:\n    print(f'[{b[\"benchmark_id\"]}] {b[\"failure_category\"]}')")
    ])
    with open(NOTEBOOKS_DIR / "07_llm_copilot.ipynb", "w") as f:
        json.dump(nb7, f, indent=2)

    print("Successfully built all 7 Jupyter Notebooks in notebooks/ directory.")

if __name__ == "__main__":
    build_all_notebooks()
