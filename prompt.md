You are responsible for building the COMPLETE, WORKING, END-TO-END project described below.

This is NOT a request for:

- a mock project
- a UI prototype with fake charts
- placeholder functions
- pseudocode
- hardcoded metrics
- fabricated ML results
- an LLM wrapper pretending to be an ML system

Build a genuinely functioning ML/data science application.

Work autonomously and systematically. Create the environment, install dependencies, implement the code, run tests, execute the pipeline where possible, and fix errors you encounter.

Do not stop after scaffolding.

======================================================================
PROJECT
======================================================================

Project name:

Loan Performance Intelligence Engine

This is a prototype for the Intain Campus FinTech Challenge 2026 AI Track.

The intended system must perform:

1. Loan data profiling
2. Data quality analysis
3. Missingness analysis
4. Outlier detection
5. Relationship validation
6. Train/test drift detection
7. Feature engineering
8. Delinquency prediction
9. Default prediction
10. Prepayment prediction
11. Next-state prediction
12. Time-aware validation
13. Class imbalance handling
14. Probability calibration
15. Survival, hazard, or monthly transition modeling
16. Anomaly and exception detection
17. Scenario and stress simulation
18. Global explainability
19. Local explainability
20. Error analysis
21. Model confidence/uncertainty
22. Grounded LLM-assisted reviewer functionality
23. Prompt logging and LLM governance
24. Reproducible reporting
25. Submission generation
26. Interactive dashboard
27. Model card
28. AI Development Log

The core predictive work MUST use real non-LLM machine learning.

The LLM is only an assistant for grounded explanations, retrieval, summaries, reviewer notes, and natural-language analysis.

======================================================================
CRITICAL DATASET CONTEXT
======================================================================

The competition organizer has provided the problem statement and the expected data-pack structure, but the actual organizer-provided dataset is currently unavailable.

Do NOT wait for the organizer dataset.

Use the real Lending Club Kaggle dataset as the primary prototype dataset.

Dataset:

wordsforthewise/lending-club

IMPORTANT:

Do not claim that Lending Club data is organizer-provided competition data.

The README and dashboard must clearly state:

"Prototype implementation uses publicly available Lending Club historical loan data because the organizer-provided competition dataset was unavailable at implementation time."

The system architecture must make it easy to replace Lending Club with the official organizer dataset later.

======================================================================
DATASET DOWNLOAD
======================================================================

FIRST TRY THIS METHOD:

Python:

import kagglehub

path = kagglehub.dataset_download("wordsforthewise/lending-club")

print("Path to dataset files:", path)

Install kagglehub if necessary.

If KaggleHub cannot download the dataset, use this fallback:

#!/bin/bash

curl -L -o ~/Downloads/lending-club.zip \
https://www.kaggle.com/api/v1/datasets/download/wordsforthewise/lending-club

Then extract the archive.

DO NOT ask me to manually download the dataset unless both automated approaches fail.

If authentication is required, clearly report exactly what is required, but continue implementing everything else that does not depend on the download.

======================================================================
PYTHON ENVIRONMENT — MANDATORY
======================================================================

Use Python 3.12.

The command is:

python3.12

Create a project-local virtual environment:

.venv

Create it with:

python3.12 -m venv .venv

Activate it.

After activation:

python

must resolve to:

.venv/bin/python

Verify:

python --version
which python

Python must be version 3.12.x.

Install all dependencies inside this environment.

Create:

requirements.txt

The following workflow must work:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

======================================================================
JUPYTER ENVIRONMENT
======================================================================

Install inside .venv:

jupyter
ipykernel

Register the virtual environment as a project-specific kernel:

Loan Performance Intelligence Engine

or a similarly clear kernel name.

Ensure the notebooks in this repository are configured to use the .venv kernel.

After setup, this must show the correct environment:

import sys
print(sys.executable)

It must point to the project .venv Python.

Do not rely on the system Python Jupyter kernel.

======================================================================
PROJECT STRUCTURE
======================================================================

Create a clean professional architecture.

Use approximately:

loan-performance-intelligence-engine/
│
├── .venv/
│
├── data/
│   ├── external/
│   │   └── lending_club/
│   │
│   ├── raw/
│   │
│   ├── canonical/
│   │   ├── loan_static_attributes.csv
│   │   ├── loan_monthly_performance_train.csv
│   │   ├── loan_monthly_performance_test.csv
│   │   ├── servicer_updates.csv
│   │   ├── macro_scenarios.csv
│   │   ├── validation_rules.json
│   │   ├── data_dictionary.md
│   │   └── submission_template.csv
│   │
│   └── processed/
│
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── adapters/
│   ├── profiling/
│   ├── quality/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── transitions/
│   ├── anomaly/
│   ├── scenarios/
│   ├── explainability/
│   ├── copilot/
│   ├── reporting/
│   └── utils/
│
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── run_pipeline.py
│   ├── train_models.py
│   ├── run_scenarios.py
│   └── generate_submission.py
│
├── app/
│   └── dashboard.py
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_predictive_models.ipynb
│   ├── 04_transition_model.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_scenarios.ipynb
│   └── 07_llm_copilot.ipynb
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── predictions/
│   ├── plots/
│   └── reports/
│
├── docs/
│   ├── MODEL_CARD.md
│   ├── AI_DEVELOPMENT_LOG.md
│   └── ARCHITECTURE.md
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore

You may improve this structure, but preserve strong separation of responsibilities.

======================================================================
DATA INGESTION AND ADAPTER ARCHITECTURE
======================================================================

The Lending Club dataset does NOT necessarily match the competition schema exactly.

Therefore, build a proper adapter layer.

Architecture:

RAW DATA
    ↓
Dataset Adapter
    ↓
Canonical Loan Schema
    ↓
ML / Profiling / Anomaly / Scenario Pipeline

Do not tightly couple the entire application to Lending Club column names.

Create a configurable schema mapping.

For example:

Lending Club raw field
        ↓
Canonical field

id
        → loan_id

issue_d
        → origination_month

loan_amnt
        → original_balance

funded_amnt
        → funded_balance

int_rate
        → interest_rate

fico_range_low / fico_range_high
        → credit_score

loan_status
        → loan_status / outcome

purpose
        → loan_purpose

dti
        → dti

addr_state
        → state

home_ownership
        → occupancy/home ownership category

term
        → original_term

Adapt according to the actual available columns.

Do not assume columns exist without inspecting the downloaded data.

======================================================================
CANONICAL DATA MODEL
======================================================================

Create a canonical loan schema capable of supporting the competition requirements.

Examples:

loan_id
reporting_month
origination_month
month_index
loan_age_months
remaining_term_months
original_balance
current_balance
interest_rate
credit_score
credit_score_band
ltv_band
dti
dti_band
state
loan_purpose
occupancy_type
property_type
servicer_name
current_status
days_past_due
modification_flag
prepayment_flag
default_flag
last_updated_at
source_system
document_status

Not every Lending Club field will exist.

For unavailable fields:

1. Preserve the distinction between real source data and derived fields.
2. Do not pretend unavailable source fields were directly observed.
3. Derived fields must be documented.

======================================================================
MONTHLY PANEL / TEMPORAL MODELING
======================================================================

The competition expects loan-month panel data.

Lending Club may not provide the exact monthly servicing panel required.

Handle this honestly.

Do NOT falsely claim that derived monthly records are original Lending Club observations.

Build a:

Monthly Panel Generator / Transition Simulation Layer

It must create a clearly labeled DERIVED monthly performance panel from real Lending Club loan characteristics and outcomes.

The purpose is to allow the prototype to demonstrate:

- monthly state transitions
- transition modeling
- delinquency trajectories
- survival/transition analysis
- scenario simulation

Clearly document:

"Derived monthly performance panel generated for prototype temporal modeling."

The generation process must be probabilistic and relationship-driven, not random meaningless data.

Use real Lending Club loan characteristics as inputs.

For example, transition probabilities may depend on:

- credit score
- DTI
- interest rate
- loan purpose
- loan age
- historical state
- loan characteristics

Create realistic conceptual states such as:

CURRENT
DELINQUENT_30
DELINQUENT_60
DELINQUENT_90
DEFAULT
PREPAID
CLOSED

Default, prepaid, and closed should generally behave as terminal states.

Do not create impossible transitions unless intentionally injected as anomalies.

======================================================================
TARGET CREATION
======================================================================

Create real, reproducible targets.

Where actual Lending Club outcome fields exist, use them appropriately.

For the derived temporal panel, generate future targets strictly from FUTURE temporal states.

Examples:

next_3m_delinquency_flag
next_6m_delinquency_flag
next_12m_default_flag
next_12m_prepayment_flag
next_state

CRITICAL:

Prevent target leakage.

Features at time T must not include information from time T+1 or later.

Do not use final loan outcome directly as a feature for predicting itself.

Document the leakage controls.

======================================================================
TASK 1 — DATA PROFILING
======================================================================

Implement a complete data intelligence module.

For every dataset and canonical table calculate:

- row count
- column count
- data types
- missing counts
- missing percentages
- unique counts
- numeric statistics
- categorical distributions
- date ranges

Implement:

MISSINGNESS ANALYSIS

- column-level missingness
- row-level missingness
- missingness patterns

OUTLIER ANALYSIS

Use appropriate methods:

- IQR
- robust statistics
- Isolation Forest where appropriate

Do not blindly use one method for every field.

DATE VALIDATION

Detect:

- invalid date parsing
- reporting dates before origination
- impossible loan ages
- invalid timestamps

RELATIONSHIP ANALYSIS

Implement:

- numeric correlations
- highly dependent features
- categorical associations where practical
- relationship checks

TRAIN/TEST DRIFT

Implement:

- PSI where appropriate
- KS statistics for numerical features
- categorical distribution drift

DATA QUALITY SCORING

Generate:

1. Record-level quality score
2. Dataset/batch-level quality score

Factors should include:

- missingness
- invalid values
- relationship violations
- outliers
- stale records
- source conflicts

Persist profiling outputs.

Generate plots and a report.

======================================================================
TASK 2 — FEATURE ENGINEERING
======================================================================

Build a reproducible feature engineering pipeline.

Include appropriate:

TEMPORAL FEATURES

- loan age
- origination year/month
- vintage
- seasonality
- remaining term

FINANCIAL FEATURES

- balance ratios
- interest rate features
- DTI
- credit features

HISTORICAL FEATURES

For the monthly panel:

- previous state
- rolling delinquency history
- rolling indicators
- trends
- changes

Prevent future leakage.

CATEGORICAL PROCESSING

Use robust preprocessing.

Possible approaches:

- OneHotEncoder
- native categorical handling where supported
- carefully designed ordinal mappings

Ensure train/test consistency.

Persist preprocessing objects.

======================================================================
TASK 3 — PREDICTIVE MODELING
======================================================================

Implement REAL non-LLM machine-learning models.

Train models for:

1. Delinquency prediction
2. Default prediction
3. Prepayment prediction
4. Next-state prediction

Use a TIME-AWARE split.

DO NOT use a random row-level train/test split.

Training data must precede validation data chronologically.

Ensure the same loan's future observations do not leak into earlier validation.

Implement:

BASELINE MODELS

Examples:

- Logistic Regression
- Multinomial Logistic Regression

IMPROVED MODELS

Choose compatible high-quality tabular ML models.

Possible options:

- LightGBM
- XGBoost
- CatBoost
- HistGradientBoosting

Check Python 3.12 compatibility before installation.

Do not install incompatible packages.

CLASS IMBALANCE

Handle using appropriate techniques:

- class weights
- sample weights

Do not blindly perform random oversampling on temporal data.

CALIBRATION

Evaluate probability calibration.

Use:

- calibration curves
- Brier score

Implement calibration where beneficial.

Possible methods:

- sigmoid/Platt scaling
- isotonic regression

METRICS

Binary models:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Recall at fixed precision
- Brier score

Multiclass:

- Macro F1
- per-class precision
- per-class recall
- confusion matrix

All metrics must be computed from actual model predictions.

Never hardcode metric values.

Persist metrics as JSON.

======================================================================
TASK 4 — TRANSITION / SURVIVAL MODEL
======================================================================

Implement a serious monthly transition model.

Create:

STATE TRANSITION MATRIX

Estimate:

P(state_t+1 | state_t, relevant segment)

Where practical, calculate:

- global transition probabilities
- segmented transition probabilities

Implement multi-period projections.

Generate:

- transition matrices
- state probability curves
- cumulative default probability
- cumulative prepayment probability

Compare against a simple baseline.

Document:

- state definitions
- terminal states
- censoring treatment
- transition assumptions

======================================================================
TASK 5 — ANOMALY AND EXCEPTION DETECTION
======================================================================

Implement a hybrid anomaly system.

DETERMINISTIC RULES

Create:

validation_rules.json

Include checks such as:

- negative or impossible balances
- invalid dates
- inconsistent statuses
- impossible transitions
- missing required documentation
- suspicious values

ML ANOMALY DETECTION

Implement:

Isolation Forest

Optionally compare with:

Local Outlier Factor

COMBINED ANOMALY SCORE

Create a normalized score combining:

- deterministic violations
- statistical outliers
- ML anomaly scores
- source conflicts
- stale records

Generate:

- anomaly score
- anomaly severity
- anomaly explanation
- recommended reviewer action

Provide at least 20 reviewer-ready examples if sufficient records exist.

Do not fabricate examples.

They must come from actual records flagged by the implemented system.

======================================================================
SERVICER UPDATES / SOURCE CONFLICTS
======================================================================

Because Lending Club may not provide a separate servicer_updates dataset, create a clearly documented DERIVED prototype secondary-source dataset.

It should simulate realistic partial updates and conflicts based on canonical records.

Examples:

- stale timestamps
- conflicting status
- inconsistent balance
- conflicting document status

Clearly label this as:

"Derived secondary-source prototype data."

Do not claim it is original Lending Club data.

Use it to genuinely implement:

- source conflict detection
- stale record detection
- reconciliation logic

======================================================================
TASK 6 — SCENARIO AND STRESS SIMULATION
======================================================================

Implement:

1. BASE
2. ADVERSE CREDIT
3. HIGH PREPAYMENT

Create:

macro_scenarios.csv

with explicit documented assumptions.

Do not hide assumptions inside the code.

The scenario engine must:

1. Load baseline portfolio
2. Apply scenario transformations
3. Modify relevant risk/transition drivers
4. Generate predictions or transition projections
5. Aggregate portfolio outcomes
6. Compare scenarios

Produce:

- projected delinquency rate
- projected default rate
- projected prepayment rate

Generate segment analysis by available dimensions:

- vintage
- credit score band
- state
- loan purpose
- other meaningful segments

Clearly state:

Scenario outputs are model-based stress projections, not guaranteed causal forecasts.

======================================================================
TASK 7 — EXPLAINABILITY
======================================================================

Implement GLOBAL explainability.

Use:

- feature importance
- permutation importance
- SHAP where computationally feasible

Implement LOCAL explainability.

For an individual loan explain:

- delinquency risk drivers
- default risk drivers
- prepayment drivers
- anomaly drivers

Implement confidence information.

Examples:

- predicted probability
- calibration
- confidence category
- low-confidence warnings

ERROR ANALYSIS

Analyze:

- false positives
- false negatives
- important failure cases

Generate explainability reports.

======================================================================
TASK 8 — GROUNDED LLM REVIEWER COPILOT
======================================================================

The LLM must NOT perform the core predictions.

Implement a reviewer copilot capable of:

- explaining anomaly findings
- summarizing loan risk
- generating reviewer notes
- answering questions about computed results
- retrieving field definitions
- retrieving validation rules
- summarizing scenario outputs

GROUNDING SOURCES:

- data_dictionary.md
- validation_rules.json
- actual computed model outputs
- anomaly evidence
- scenario results

Implement retrieval.

The LLM must not invent metrics.

The LLM must not invent predictions.

Clearly label all output:

"Recommendation — requires human review"

LOGGING

Log:

- prompt
- model
- timestamp
- retrieved context
- output
- acceptance/rejection status

Create an LLM evaluation/demo showing examples of:

- vague output
- unsupported output
- overconfident output

These must be clearly labeled as evaluation examples.

Do not falsely claim they occurred naturally.

The entire application must work without LLM API credentials.

If credentials are unavailable:

- disable live LLM calls gracefully
- keep all ML functionality working
- provide a clear configuration message

======================================================================
TASK 9 — MODEL CARD
======================================================================

Create:

docs/MODEL_CARD.md

Include:

- objective
- intended use
- prototype dataset
- Lending Club provenance
- derived temporal data limitations
- features
- target definitions
- model types
- validation methodology
- time-aware splitting
- metrics
- calibration
- leakage controls
- limitations
- known failure modes
- scenario limitations
- LLM governance

Never invent metrics.

======================================================================
TASK 10 — AI DEVELOPMENT LOG
======================================================================

Create:

docs/AI_DEVELOPMENT_LOG.md

Document honestly:

- AI tools used
- representative prompts
- implementation workflow
- accepted suggestions
- rejected suggestions
- human review requirements
- approximate AI-generated code share
- lessons learned

Do not falsely claim human actions that did not happen.

Clearly distinguish:

AI proposal
Human decision
Accepted implementation
Rejected implementation

======================================================================
SUBMISSION GENERATION
======================================================================

Create:

submission_template.csv

based on the expected prototype output structure.

Possible fields:

- loan_id
- delinquency_probability
- default_probability
- prepayment_probability
- next_state
- exception_probability
- exception_type
- anomaly_score
- top_drivers
- reviewer_action
- confidence

Implement:

scripts/generate_submission.py

The script must:

1. Load actual model outputs.
2. Validate required columns.
3. Validate row counts.
4. Validate value ranges.
5. Write submission.csv.

Do not hardcode fake predictions.

======================================================================
STREAMLIT DASHBOARD
======================================================================

Build a real interactive dashboard.

Use Streamlit unless another local framework is clearly better.

Command:

streamlit run app/dashboard.py

Dashboard sections:

1. Portfolio Overview
2. Data Profiling
3. Data Quality
4. Predictive Models
5. Transition / Survival Analysis
6. Anomalies & Exceptions
7. Scenario Simulation
8. Explainability
9. LLM Reviewer Copilot
10. Model Metadata

CRITICAL:

Dashboard charts must use actual pipeline artifacts.

No hardcoded fake charts.

If the pipeline has not been executed, show instructions.

Clearly display a banner:

"Prototype using Lending Club historical data and derived temporal performance data. Organizer dataset unavailable at implementation time."

======================================================================
NOTEBOOKS
======================================================================

Create useful notebooks.

They must not duplicate the entire pipeline unnecessarily.

Use notebooks for:

01_data_profiling.ipynb
02_feature_engineering.ipynb
03_predictive_models.ipynb
04_transition_model.ipynb
05_anomaly_detection.ipynb
06_scenarios.ipynb
07_llm_copilot.ipynb

The actual reusable logic belongs in src/.

Notebooks should import and demonstrate the production pipeline.

======================================================================
TESTING
======================================================================

Implement pytest tests.

At minimum:

- schema validation
- adapter mapping
- date validation
- feature generation
- leakage prevention
- time-aware split
- validation rules
- anomaly scoring
- scenario transformations
- submission validation

Tests should run without requiring the full dataset.

Use small fixtures.

Run:

python -m pytest

Fix failures.

======================================================================
PIPELINE
======================================================================

Create a reproducible end-to-end workflow.

Expected commands:

source .venv/bin/activate

python scripts/download_data.py

python scripts/prepare_data.py

python scripts/run_pipeline.py

python scripts/generate_submission.py

streamlit run app/dashboard.py

If commands need adjustment, document the exact commands in README.

======================================================================
PIPELINE EXECUTION
======================================================================

scripts/run_pipeline.py should orchestrate:

1. Data loading
2. Schema validation
3. Data profiling
4. Data quality analysis
5. Canonical transformation
6. Feature engineering
7. Time-aware split
8. Baseline training
9. Improved model training
10. Calibration
11. Evaluation
12. Transition modeling
13. Anomaly detection
14. Scenario simulation
15. Explainability
16. Report generation
17. Artifact persistence

Use logging.

Persist artifacts.

Use a reproducible random seed.

======================================================================
PERFORMANCE
======================================================================

The Lending Club dataset may be large.

Do not blindly load multiple copies into memory.

Use:

- efficient dtypes
- selective columns
- sampling for expensive visualizations
- configurable dataset limits
- chunking where useful

However, do not make the project fake-small.

The architecture should scale beyond the prototype.

======================================================================
README
======================================================================

Create a professional README.

Include:

PROJECT OVERVIEW

Explain the Loan Performance Intelligence Engine.

DATASET

Clearly explain:

- Lending Club is used for the prototype.
- It is not the organizer dataset.
- Organizer data was unavailable.
- A schema adapter supports future replacement.

SETUP

Exact Python 3.12 instructions.

DATA DOWNLOAD

Explain KaggleHub and curl fallback.

ARCHITECTURE

Include a Mermaid architecture diagram.

PIPELINE

Explain every major stage.

RUNNING

Provide exact commands.

LIMITATIONS

Explicitly explain:

- Lending Club is not identical to the intended organizer dataset.
- Monthly panel data is derived for prototype temporal modeling.
- Derived secondary-source updates are prototype data.
- Scenario results are model-based simulations.
- Results are not competition benchmark results.

======================================================================
IMPORTANT HONESTY RULE
======================================================================

Never present:

- Lending Club results
- derived monthly-panel results
- synthetic secondary-source conflicts
- scenario projections

as organizer-provided competition results.

Clearly distinguish:

REAL SOURCE DATA
from
DERIVED PROTOTYPE DATA

======================================================================
IMPLEMENTATION ORDER
======================================================================

PHASE 1

Environment:

1. Verify python3.12.
2. Create .venv.
3. Activate it.
4. Install dependencies.
5. Verify python points to .venv.
6. Install Jupyter/ipykernel.
7. Register kernel.

PHASE 2

Foundation:

1. Project structure.
2. Configuration.
3. Logging.
4. Requirements.
5. Tests foundation.

PHASE 3

Dataset:

1. Download Lending Club.
2. Inspect files and columns.
3. Implement adapter.
4. Build canonical schema.

PHASE 4

Data Intelligence:

1. Profiling.
2. Missingness.
3. Outliers.
4. Validation.
5. Drift.
6. Quality scoring.

PHASE 5

Feature Engineering and ML:

1. Leakage-safe features.
2. Time-aware split.
3. Baselines.
4. Improved models.
5. Calibration.
6. Evaluation.

PHASE 6

Temporal Modeling:

1. Derived monthly panel.
2. State transitions.
3. Multi-period projections.

PHASE 7

Anomalies:

1. Rules.
2. Isolation Forest.
3. Combined score.
4. Reviewer examples.

PHASE 8

Scenarios:

1. Base.
2. Adverse credit.
3. High prepayment.

PHASE 9

Explainability:

1. Global.
2. Local.
3. Error analysis.

PHASE 10

Copilot:

1. Retrieval.
2. Grounded prompts.
3. Logging.
4. Governance.

PHASE 11

Application:

1. Dashboard.
2. Reports.
3. Submission.

PHASE 12

Validation:

1. Run tests.
2. Run pipeline.
3. Check artifacts.
4. Check notebook environment.
5. Fix errors.

======================================================================
FINAL ACCEPTANCE REQUIREMENTS
======================================================================

Before declaring the project complete, verify:

ENVIRONMENT

[ ] Python 3.12
[ ] .venv exists
[ ] python resolves to .venv
[ ] requirements.txt works
[ ] Jupyter kernel uses .venv

DATA

[ ] Lending Club downloaded or download process implemented
[ ] Raw columns inspected
[ ] Adapter implemented
[ ] Canonical schema implemented
[ ] Real vs derived data clearly documented

PROFILING

[ ] Missingness
[ ] Outliers
[ ] Relationships
[ ] Validation
[ ] Drift
[ ] Data-quality scoring

ML

[ ] Delinquency model
[ ] Default model
[ ] Prepayment model
[ ] Next-state model
[ ] Time-aware validation
[ ] Baseline
[ ] Improved model
[ ] Imbalance handling
[ ] Calibration
[ ] Real metrics

TEMPORAL

[ ] Derived monthly panel
[ ] Transition model
[ ] Curves/projections
[ ] Baseline comparison

ANOMALIES

[ ] Rule detection
[ ] ML detection
[ ] Combined anomaly score
[ ] Explanations
[ ] Reviewer examples

SCENARIOS

[ ] Base
[ ] Adverse credit
[ ] High prepayment
[ ] Segment analysis

EXPLAINABILITY

[ ] Global importance
[ ] Local explanations
[ ] Confidence
[ ] Error analysis

LLM

[ ] Grounding
[ ] Retrieval
[ ] Prompt logging
[ ] Recommendation-only labeling
[ ] Graceful operation without API key

DELIVERABLES

[ ] Dashboard
[ ] Reports
[ ] Model Card
[ ] AI Development Log
[ ] README
[ ] Tests
[ ] submission.csv generation

======================================================================
FINAL INSTRUCTION
======================================================================

DO NOT stop at scaffolding.

Actually build the project.

Actually create the environment.

Actually install dependencies.

Actually inspect the dataset.

Actually implement the models.

Actually run the tests.

Actually run the pipeline where possible.

Fix errors encountered.

When complete, provide a concise final report containing:

1. Environment status.
2. Dataset download status.
3. Actual dataset files discovered.
4. Lending Club → canonical schema mapping.
5. Components implemented.
6. Models trained.
7. Tests executed and results.
8. Pipeline execution status.
9. Commands to run the project.
10. Known limitations caused by the missing organizer dataset.

Do not claim completion unless the implementation actually exists and has been validated.
