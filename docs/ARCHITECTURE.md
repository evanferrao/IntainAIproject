# System Architecture — Loan Performance Intelligence Engine

## 1. High-Level Architectural Topology

```mermaid
flowchart TD
    subgraph DataIngestion ["1. Ingestion & Canonical Adapter"]
        RawLC["Raw Lending Club / External CSV"] --> LCAdapter["LendingClubAdapter"]
        LCAdapter --> CanonicalMaster["Canonical Static Master (loan_static_attributes.csv)"]
        LCAdapter --> MonthlyGen["Derived Monthly Panel Generator"]
        MonthlyGen --> MonthlyTrain["Monthly Performance Train Panel"]
        MonthlyGen --> MonthlyTest["Monthly Performance Test Panel"]
        LCAdapter --> ServicerSim["Derived Servicer Update Simulator"]
        ServicerSim --> ServicerData["Servicer Secondary Updates (servicer_updates.csv)"]
    end

    subgraph DataIntelligence ["2. Profiling, Quality & Drift Engine"]
        CanonicalMaster & MonthlyTrain --> Profiler["DataProfiler (Missingness, Stats, Integrity)"]
        MonthlyTrain & MonthlyTest --> DriftEngine["DriftDetector (PSI & KS-Test)"]
        Profiler & DriftEngine --> QualityScorer["DataQualityScorer (Record & Batch Level)"]
    end

    subgraph FeaturePipeline ["3. Feature Engineering & Split"]
        MonthlyTrain --> FeatPipe["FeatureEngineeringPipeline (Point-in-Time & Cyclic)"]
        FeatPipe --> TrainMatrix["X_train (Train Matrix)"]
        MonthlyTest --> FeatPipe
        FeatPipe --> TestMatrix["X_test (Out-of-Time Test Matrix)"]
    end

    subgraph PredictiveModeling ["4. Predictive & Transition ML Models"]
        TrainMatrix --> DelinqModel["Delinquency Model (LightGBM + Platt/Isotonic Calib)"]
        TrainMatrix --> DefaultModel["Default Model (LightGBM + Platt/Isotonic Calib)"]
        TrainMatrix --> PrepayModel["Prepayment Model (LightGBM + Platt/Isotonic Calib)"]
        TrainMatrix --> NextStateModel["Next-State Model (Multiclass LightGBM)"]
        MonthlyTrain & MonthlyTest --> MarkovModel["Markov State Transition Engine (T^h Roll-Forward)"]
    end

    subgraph AnomalyEngine ["5. Hybrid Anomaly & Reconciliation Triage"]
        CanonicalMaster --> RuleEngine["Deterministic Rule Engine (validation_rules.json)"]
        TrainMatrix --> IsoForest["Isolation Forest (Unsupervised Outlier)"]
        CanonicalMaster & ServicerData --> ReconEngine["Source Reconciliation Engine (Tape Conflicts)"]
        RuleEngine & IsoForest & ReconEngine --> CompositeScorer["Composite Anomaly Scorer (0-100 & Severity Tiers)"]
    end

    subgraph DownstreamEngine ["6. Simulation, Explainability & Reviewer Copilot"]
        DefaultModel & PrepayModel --> ScenarioEngine["Scenario & Stress Simulation (Base, Adverse, Prepay)"]
        DefaultModel --> Explainer["Global & Local Explainability (Feature Imp + Attribution)"]
        CompositeScorer & ScenarioEngine & Explainer --> Copilot["Grounded LLM Reviewer Copilot (Governance Audit Log)"]
    end

    subgraph Presentation ["7. UI & Deliverables"]
        PredictiveModeling & DownstreamEngine & AnomalyEngine --> StreamlitDashboard["Streamlit Interactive Dashboard (app/dashboard.py)"]
        PredictiveModeling & AnomalyEngine --> SubmissionGen["Submission Generator (submission.csv)"]
        DataIntelligence & PredictiveModeling --> ExecReport["Executive Intelligence Report (Markdown)"]
    end
```

---

## 2. Core Architectural Principles
1. **Strict Canonical Schema Decoupling**: Upstream datasets (Lending Club prototype, synthetic data, or future official organizer datasets) are decoupled via `LendingClubAdapter`. Core ML and downstream engines only interact with canonical schemas.
2. **Zero-Leakage Temporal Modeling**: Feature engineering enforces strict point-in-time calculation. Future targets are derived purely from $t+1 \dots t+h$ future panels and never leak into feature vectors.
3. **Calibrated Tabular ML**: Tree-based gradient boosters are combined with Isotonic regression and Platt scaling to ensure well-calibrated loss estimates.
4. **Hybrid Anomaly Triangulation**: Merges deterministic business logic, unsupervised ML (Isolation Forest), and subservicer tape reconciliation.
5. **Grounded AI Governance**: LLM copilot is grounded in codified data dictionaries and validation rules, logs all interactions to JSONL audit ledgers, and operates 100% offline without external API dependencies.
