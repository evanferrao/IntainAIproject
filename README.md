# Loan Performance Intelligence Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Challenge](https://img.shields.io/badge/Intain_FinTech_Challenge-2026_AI_Track-orange.svg)]()

> **Prototype Implementation for the Intain Campus FinTech Challenge 2026 (AI Track)**  
> An enterprise-grade, non-LLM machine learning and data intelligence system for loan data profiling, quality scoring, drift monitoring, delinquency/default/prepayment/next-state modeling, multi-period Markov transition forecasting, hybrid anomaly triage, macroeconomic stress simulation, and grounded LLM reviewer copilot assistance.

---

## Dataset Provenance Notice

> **"Prototype implementation uses publicly available Lending Club historical loan data because the organizer-provided competition dataset was unavailable at implementation time."**

The architecture implements a strict **Adapter Pattern** (`LendingClubAdapter`), isolating all raw dataset specifics from canonical internal schemas (`CanonicalStaticLoan`, `CanonicalMonthlyPerformance`). When the official competition dataset is released, replacing the data source requires updating only the adapter layer without modifying any downstream ML models, profiling pipelines, anomaly engines, or dashboard interfaces.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Canonical Adapter"]
        RawLC["Raw Lending Club / External CSV"] --> Adapter["LendingClubAdapter"]
        Adapter --> CanonicalStatic["Canonical Static Loan Master (loan_static_attributes.csv)"]
        Adapter --> MonthlyGen["Derived Monthly Panel Generator"]
        MonthlyGen --> CanonicalMonthly["Canonical Monthly Panel (train / test)"]
        Adapter --> ServicerSim["Derived Servicer Update Simulator"]
        ServicerSim --> ServicerData["Servicer Secondary Updates (servicer_updates.csv)"]
    end

    subgraph Intelligence ["2. Profiling, Quality & Drift Engine"]
        CanonicalStatic & CanonicalMonthly --> Profiling["Data Profiler (Missingness, Outliers, Integrity)"]
        CanonicalMonthly --> DriftEngine["Train/Test Drift Engine (PSI & KS-Test)"]
        Profiling & DriftEngine --> QualityScore["Quality Scoring (Record & Batch Level)"]
    end

    subgraph ML_Pipeline ["3. ML Predictive & Transition Engine"]
        CanonicalMonthly --> FeatureEng["Leakage-Safe Feature Pipeline"]
        FeatureEng --> TimeSplit["Time-Aware Chronological Split"]
        TimeSplit --> DelinquencyModel["Delinquency Model (LightGBM + Calib)"]
        TimeSplit --> DefaultModel["Default Model (LightGBM + Calib)"]
        TimeSplit --> PrepaymentModel["Prepayment Model (LightGBM + Calib)"]
        TimeSplit --> NextStateModel["Next-State Model (Multiclass LightGBM)"]
        TimeSplit --> TransitionModel["Markov Transition Engine (T^h Roll-Forward)"]
    end

    subgraph Downstream ["4. Simulation, Explainability & Reviewer Copilot"]
        DelinquencyModel & DefaultModel & TransitionModel --> ScenarioEngine["Scenario & Stress Simulation (Base, Adverse, Prepay)"]
        DelinquencyModel & DefaultModel --> ExplainEngine["Explainability (Feature Importance + Local Attribution)"]
        ScenarioEngine & ExplainEngine --> GroundedCopilot["Grounded LLM Reviewer Copilot (Governance Audit Log)"]
    end

    subgraph Deliverables ["5. Application, Reports & Deliverables"]
        ML_Pipeline & Downstream --> StreamlitApp["Interactive Streamlit Dashboard (app/dashboard.py)"]
        ML_Pipeline --> SubGen["Submission Generator (submission.csv)"]
        ML_Pipeline --> Notebooks["7 Jupyter Notebooks"]
        ML_Pipeline --> Docs["Model Card & AI Development Log"]
    end
```

---

## Project Directory Structure

```text
loan-performance-intelligence-engine/
│
├── .venv/                              # Python 3.12 project virtual environment
│
├── data/
│   ├── external/                       # Raw downloaded datasets (Lending Club)
│   ├── raw/
│   ├── canonical/                      # Clean canonical schema files
│   │   ├── loan_static_attributes.csv
│   │   ├── loan_monthly_performance_train.csv
│   │   ├── loan_monthly_performance_test.csv
│   │   ├── servicer_updates.csv
│   │   ├── macro_scenarios.csv
│   │   ├── validation_rules.json
│   │   ├── data_dictionary.md
│   │   └── submission_template.csv
│   └── processed/
│
├── src/
│   ├── config/                         # Settings and Pydantic schemas
│   ├── ingestion/                      # Data acquisition and monthly panel generator
│   ├── adapters/                       # Lending Club to Canonical schema adapter
│   ├── profiling/                      # Missingness, outliers, integrity, PSI drift
│   ├── quality/                        # Record-level and batch-level quality scoring
│   ├── features/                       # Leakage-safe feature engineering & time splitter
│   ├── models/                         # Delinquency, Default, Prepayment, Next-State models
│   ├── evaluation/                     # Metric evaluation suite (ROC, PR, Brier, Calibration)
│   ├── transitions/                    # Markov state transition matrix & multi-period projection
│   ├── anomaly/                        # Rules engine, Isolation Forest, reconciliation triage
│   ├── scenarios/                      # Macroeconomic stress testing engine
│   ├── explainability/                 # Global importance, local attribution, uncertainty
│   ├── copilot/                        # Grounded LLM reviewer copilot & audit logging
│   ├── reporting/                      # Executive intelligence report generator
│   └── utils/                          # Structured logging & serialization
│
├── scripts/
│   ├── download_data.py                # Dataset acquisition script
│   ├── prepare_data.py                 # Ingestion and canonical dataset builder
│   ├── train_models.py                 # Model training and calibration script
│   ├── run_scenarios.py                # Scenario stress testing script
│   ├── generate_submission.py          # Submission generator & validator
│   ├── generate_notebooks.py           # Notebook builder
│   └── run_pipeline.py                 # Unified end-to-end orchestrator
│
├── app/
│   └── dashboard.py                    # 10-module interactive Streamlit dashboard
│
├── notebooks/                          # 7 Production Jupyter Notebooks
│   ├── 01_data_profiling.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_predictive_models.ipynb
│   ├── 04_transition_model.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_scenarios.ipynb
│   └── 07_llm_copilot.ipynb
│
├── artifacts/
│   ├── models/                         # Serialized trained models (*.joblib)
│   ├── metrics/                        # JSON performance metrics & drift reports
│   ├── predictions/                    # submission.csv & exception dossiers
│   ├── plots/
│   ├── reports/                        # Executive Intelligence Report (Markdown)
│   └── logs/                           # Pipeline & Copilot audit JSONL logs
│
├── docs/
│   ├── MODEL_CARD.md                   # Comprehensive Model Card
│   ├── AI_DEVELOPMENT_LOG.md           # Transparent AI development log
│   └── ARCHITECTURE.md                 # System architecture documentation
│
├── tests/                              # Pytest test suite (10 test modules)
├── requirements.txt                    # Pinned Python 3.12 dependencies
├── pyproject.toml
└── README.md
```

---

## Environment Setup (Mandatory Python 3.12)

```bash
# 1. Verify Python 3.12 is installed
python3.12 --version

# 2. Create and activate local virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify environment
which python
# Must output: .../loan-performance-intelligence-engine/.venv/bin/python
```

---

## Jupyter Environment Configuration

The repository registers a dedicated project kernel mapped directly to `.venv`:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name loan-performance-engine --display-name "Loan Performance Intelligence Engine"
```

To launch Jupyter Lab / Notebooks:
```bash
jupyter lab notebooks/
```

---

## Quickstart: Running the End-to-End Pipeline

Execute the unified pipeline orchestrator:

```bash
source .venv/bin/activate

# Run the complete end-to-end pipeline
python scripts/run_pipeline.py
```

Or execute modular pipeline stages sequentially:

```bash
# Stage 1: Acquire Lending Club data (or synthetic prototype fallback)
python scripts/download_data.py

# Stage 2: Transform raw records, generate monthly panels & servicer updates
python scripts/prepare_data.py

# Stage 3: Train and calibrate all ML predictive and transition models
python scripts/train_models.py

# Stage 4: Run macroeconomic stress testing scenarios
python scripts/run_scenarios.py

# Stage 5: Validate and generate submission.csv
python scripts/generate_submission.py
```

---

## Launching the Interactive Streamlit Dashboard

Launch the 10-module rich interactive dashboard:

```bash
source .venv/bin/activate
streamlit run app/dashboard.py
```

### Dashboard Sections:
1. **Portfolio Overview**: Executive KPIs, balance totals, FICO distributions, purpose breakdowns.
2. **Data Profiling**: Column profiles, missingness analysis, Pearson feature correlation matrix.
3. **Data Quality & Drift Monitor**: Dataset quality scores, PSI, and Kolmogorov-Smirnov drift tables.
4. **Predictive Models & Calibration**: ROC-AUC, PR-AUC, F1, Brier score, and Confusion Matrices.
5. **Transition & Survival Modeling**: 1-month transition matrices and 36-month absorption curves.
6. **Anomalies & Exception Dossiers**: Multi-class reviewer triage queue with evidence-driven dispositions.
7. **Scenario Simulation**: Stress testing across `BASE`, `ADVERSE_CREDIT`, and `HIGH_PREPAYMENT`.
8. **Model Explainability**: Local & global SHAP, LIME local linear surrogates, Partial Dependence Plots (PDP), and Ceteris Paribus (ICE) profiles.
9. **Counterfactual Analysis ("What Would Lower Risk?")**: Real model-driven search for sparse, feasible feature modifications that reduce predicted risk.
10. **Grounded AI Reviewer Copilot**: Interactive Groq LLM chat grounded on actual loan evidence, failure-mode benchmarks, and session audit ledger.
11. **Interactive New Loan Simulator**: On-the-fly model inference, quality scoring, anomaly checks, and SHAP attributions for user-entered loans.
12. **Model Card & Governance**: System lineage, assumptions, and governance documentation.

---

## Groq LLM Integration & Grounding Architecture

### Strict Architectural Separation
There is a strict architectural boundary between the quantitative ML system and the conversational LLM:
- **Groq is invoked ONLY** when the reviewer explicitly submits a natural-language query in the AI Reviewer Copilot interface.
- Groq is **NEVER** called to generate predictions, calculate anomaly scores, compute SHAP/LIME, evaluate PDPs, or search for counterfactuals. All quantitative analytics run locally on the trained models.
- If `GROQ_API_KEY` is not provided, the entire ML and explainability pipeline remains 100% operational; only live LLM chat displays an informational notice.

### Configuration (`.env`)
Create a `.env` file in the project root (see `.env.example`):
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### Grounding & Context Construction
When a reviewer asks a question about a selected loan, the engine constructs a structured JSON evidence package containing:
- **Loan Information**: Original balance, FICO, DTI, coupon rate, purpose, state, term.
- **Model Outputs**: Calibrated default probability, delinquency probability, prepayment probability, next state, confidence rating.
- **Anomaly Evidence**: Composite anomaly score, severity, top drivers, deterministic rule flags, servicer tape reconciliation flags.
- **Explainability & Counterfactuals**: Top SHAP drivers, LIME rules, and candidate risk-reducing counterfactuals.
- **Reviewer Triage**: Assigned disposition and deterministic reason bullets.

The LLM receives this evidence alongside a strict system prompt prohibiting hallucinations and ungrounded claims.

### Audit Governance Ledger
Every interaction is appended to `artifacts/logs/copilot_audit_log.jsonl` with:
- `timestamp`: UTC ISO-8601
- `loan_id`: Evaluated loan identifier
- `model`: Configured Groq model name
- `user_question`: Question submitted by user
- `context_hash`: Deterministic SHA-256 digest of the structured evidence package
- `response`: Text returned by Groq
- `latency`: Response time in seconds
- `success`: Boolean success indicator
- `error`: Error description if call failed
- **Security**: API keys and secrets are never logged.

---

## Reviewer Disposition Policy Engine

Reviewer dispositions are deterministically computed by `ReviewerPolicyEngine` (`src/anomaly/reviewer_policy.py`) based on documented thresholds in `src/config/reviewer_policy.py`:
- `AUTO_APPROVE`: Clean data profile, low predicted default risk (<1.0%), no material anomalies or tape conflicts.
- `FLAG_FOR_REVIEW`: Moderate predicted risk (1.0%–5.0%), multivariate statistical anomaly (Isolation Forest flag or score ≥ 45.0), low model confidence, or moderate data quality score (70–85).
- `RECONCILE_SOURCE_CONFLICT`: Source reconciliation conflict detected against servicer update tape.
- `MANUAL_AUDIT_REQUIRED`: Deterministic validation rule violation (e.g. invalid balance, rate > 40%) or severe data quality deficiency (< 70).
- `HIGH_RISK_REVIEW`: High predicted default probability (≥ 5.0%) or high delinquency hazard (≥ 2.0%).

Every loan disposition includes documented, factual reason bullets citing exact computed metrics.

---

## Model Explainability & Counterfactual Analysis

1. **SHAP (SHapley Additive exPlanations)**:
   - Uses `shap.TreeExplainer` on the underlying LightGBM models (`default`, `delinquency`, `prepayment`).
   - Computes local waterfall/bar risk attributions and global mean |SHAP| values.
2. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Uses `lime.lime_tabular.LimeTabularExplainer` against the model's calibrated `predict_proba`.
   - Explains individual predictions via local linear surrogates with condition rules and weights.
3. **Partial Dependence Plots (PDP)**:
   - Evaluates marginal feature effect using `sklearn.inspection.partial_dependence` directly from fitted models.
   - Supports continuous features: `credit_score`, `dti`, `interest_rate`, `installment`, `revolving_utilization`, `annual_income`.
4. **Ceteris Paribus / ICE Profiles**:
   - Evaluates model response curves by varying a single feature across a valid domain while holding all other features fixed.
   - Mandatory notice: *"Model response under controlled feature variation — not a causal estimate."*
5. **Model-Driven Counterfactual Search**:
   - Real optimization search (`CounterfactualSearchEngine`) over mutable features (`credit_score`, `dti`, `interest_rate`, `revolving_utilization`, `annual_income`).
   - Strictly preserves immutable features (`loan_id`, `origination_month`, historical status, etc.).
   - Evaluates actual model `predict_proba` for candidate perturbations.
   - Ranks candidates preferring sparsity (1 feature changed before 2), then risk reduction.
   - Mandatory disclaimer: *"Counterfactuals represent model-implied changes that reduce predicted risk under controlled feature modifications. They are not causal estimates and do not guarantee approval."*

---

## Running Automated Tests

Execute the comprehensive test suite with coverage:

```bash
source .venv/bin/activate
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Limitations & Honest Provenance Disclaimers

1. **Lending Club Provenance**: Lending Club data reflects US unsecured retail installment loans and differs in structure from specific mortgage or auto loan data packs that the competition organizer may provide.
2. **Derived Temporal Panel**: The monthly performance panel is derived from static characteristics and empirical transition hazards to enable temporal modeling. It is explicitly labeled as prototype data.
3. **Derived Secondary Updates**: Servicer updates and conflict tapes are derived to demonstrate multi-source reconciliation.
4. **Scenario Projections**: Scenario simulation outputs represent model-based sensitivity stress projections, not guaranteed causal macroeconomic forecasts.
5. **Counterfactual Non-Causality**: Counterfactual explanations evaluate model response under controlled perturbations and do not constitute causal underwriting guarantees.
6. **Human Review Requirement**: All AI Copilot recommendations carry the mandatory label: **"Recommendation — requires human review"**.

