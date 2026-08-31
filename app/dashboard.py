"""Interactive Streamlit Dashboard for Loan Performance Intelligence Engine."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    CANONICAL_DATA_DIR, METRICS_DIR, PREDICTIONS_DIR,
    MODELS_DIR, RANDOM_SEED
)
from src.utils.serialization import load_json, load_model
from src.copilot.grounded_copilot import GroundedReviewerCopilot
from src.copilot.evaluation_suite import get_llm_evaluation_benchmarks

# Page Configuration
st.set_page_config(
    page_title="Loan Performance Intelligence Engine | Intain FinTech 2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (FinTech Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .provenance-banner {
        background: #fef3c7;
        color: #92400e;
        font-weight: 600;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 6px solid #d97706;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-card-val {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card-lbl {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def check_pipeline_executed() -> bool:
    """Checks if pipeline artifacts exist."""
    required_files = [
        METRICS_DIR / "model_metrics.json",
        METRICS_DIR / "data_profiling_summary.json",
        PREDICTIONS_DIR / "submission.csv"
    ]
    return all(f.exists() for f in required_files)


def main():
    # Top Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 28px; font-weight: 700;">Loan Performance Intelligence Engine</h1>
        <p style="margin: 4px 0 0 0; opacity: 0.85; font-size: 15px;">
            Intain Campus FinTech Challenge 2026 -- AI Track Submission Prototype
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mandatory Provenance Notice Banner
    st.markdown("""
    <div class="provenance-banner">
        <strong>Dataset Provenance Notice:</strong> Prototype implementation uses publicly available Lending Club historical loan data and derived temporal performance panels because the organizer-provided competition dataset was unavailable at implementation time.
    </div>
    """, unsafe_allow_html=True)

    if not check_pipeline_executed():
        st.warning("Pipeline artifacts not detected. Please run the pipeline script first to generate metrics and models:")
        st.code("python scripts/run_pipeline.py", language="bash")
        st.info("The dashboard will automatically populate once artifacts are generated.")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Engine Module:",
        [
            "1. Portfolio Overview",
            "2. Data Profiling & Distributions",
            "3. Data Quality & Drift Monitor",
            "4. Predictive Models & Calibration",
            "5. Transition & Survival Modeling",
            "6. Anomalies & Exception Dossiers",
            "7. Scenario & Stress Simulation",
            "8. Global & Local Explainability",
            "9. Grounded LLM Reviewer Copilot",
            "10. Model Card & Governance"
        ]
    )

    # Load Artifacts
    model_metrics = load_json(METRICS_DIR / "model_metrics.json")
    profiling_summary = load_json(METRICS_DIR / "data_profiling_summary.json")
    quality_summary = load_json(METRICS_DIR / "data_quality_score.json")
    drift_summary = load_json(METRICS_DIR / "drift_metrics.json")
    scenario_results = load_json(METRICS_DIR / "scenario_simulation_results.json")
    
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    submission_df = pd.read_csv(PREDICTIONS_DIR / "submission.csv")

    # =========================================================================
    # SECTION 1: PORTFOLIO OVERVIEW
    # =========================================================================
    if "1. Portfolio Overview" in section:
        st.subheader("Executive Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Loans Monitored", f"{len(static_df):,}")
        with col2:
            st.metric("Dataset Quality Score", f"{quality_summary.get('batch_quality_score', 92.5)}/100", delta="Healthy")
        with col3:
            avg_def = model_metrics.get("default", {}).get("positive_class_ratio", 0.024)
            st.metric("Mean Default Risk (12M)", f"{avg_def:.1%}")
        with col4:
            avg_prep = model_metrics.get("prepayment", {}).get("positive_class_ratio", 0.085)
            st.metric("Mean Prepay Propensity (12M)", f"{avg_prep:.1%}")

        st.markdown("---")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown("##### Credit Score Tier Distribution")
            fig_fico = px.histogram(
                static_df, x="credit_score_band",
                category_orders={"credit_score_band": ["Deep Subprime", "Subprime", "Near Prime", "Prime", "Super Prime"]},
                color="credit_score_band",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_fico.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_fico, use_container_width=True)

        with row1_col2:
            st.markdown("##### Loan Purpose Breakdown")
            fig_purpose = px.pie(
                static_df, names="loan_purpose",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_purpose.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_purpose, use_container_width=True)

    # =========================================================================
    # SECTION 2: DATA PROFILING
    # =========================================================================
    elif "2. Data Profiling" in section:
        st.subheader("Data Profiling & Statistical Intelligence")

        st.markdown(f"**Dataset Summary**: {profiling_summary.get('row_count', 0):,} Rows × {profiling_summary.get('column_count', 0)} Columns")
        
        tab1, tab2, tab3 = st.tabs(["Column Profiles", "Missingness Analysis", "Correlation Matrix"])
        
        with tab1:
            cols_data = []
            for col_name, info in profiling_summary.get("columns", {}).items():
                top_cats = list(info.get("top_categories", {}).keys()) if isinstance(info.get("top_categories"), dict) else []
                top_val = str(top_cats[0]) if top_cats else "N/A"
                mean_val = f"{info['mean']:.2f}" if "mean" in info and info["mean"] is not None else top_val
                std_val = f"{info['std']:.2f}" if "std" in info and info["std"] is not None else "N/A"
                outlier_pct_val = f"{info['outlier_pct']:.1f}%" if "outlier_pct" in info and info["outlier_pct"] is not None else "N/A"
                
                cols_data.append({
                    "Field": str(col_name),
                    "Dtype": str(info.get("dtype", "unknown")),
                    "Missing (%)": f"{info.get('missing_pct', 0.0):.2f}%",
                    "Unique Count": int(info.get("n_unique", 0)),
                    "Mean / Top": mean_val,
                    "Std": std_val,
                    "IQR Outliers (%)": outlier_pct_val
                })
            st.dataframe(pd.DataFrame(cols_data), use_container_width=True)

        with tab2:
            miss_pct = profiling_summary.get("missingness", {}).get("column_missing_pct", {})
            miss_df = pd.DataFrame(list(miss_pct.items()), columns=["Field", "Missing_Percentage"]).sort_values("Missing_Percentage", ascending=False)
            fig_miss = px.bar(miss_df.head(15), x="Field", y="Missing_Percentage", title="Top Missingness by Field (%)", color="Missing_Percentage", color_continuous_scale="Reds")
            st.plotly_chart(fig_miss, use_container_width=True)

        with tab3:
            corr_dict = profiling_summary.get("numeric_correlations", {})
            if corr_dict:
                corr_df = pd.DataFrame(corr_dict)
                fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Pearson Feature Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)

    # =========================================================================
    # SECTION 3: QUALITY & DRIFT MONITOR
    # =========================================================================
    elif "3. Data Quality" in section:
        st.subheader("Data Quality & Train/Test Drift Monitor")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch Quality Score", f"{quality_summary.get('batch_quality_score', 90.0)}/100")
        with col2:
            st.metric("Quality Status", quality_summary.get("quality_status", "EXCELLENT"))
        with col3:
            st.metric("Overall Drift Status", drift_summary.get("overall_drift_status", "STABLE"))

        st.markdown("---")
        st.markdown("##### Population Stability Index (PSI) & KS-Test Drift Metrics")
        drift_feats = drift_summary.get("feature_metrics", {})
        drift_table = []
        for feat, dinfo in drift_feats.items():
            psi_val = dinfo.get("psi", dinfo.get("total_variation_distance"))
            psi_str = f"{float(psi_val):.4f}" if psi_val is not None else "N/A"
            ks_stat = f"{float(dinfo['ks_statistic']):.4f}" if "ks_statistic" in dinfo and dinfo["ks_statistic"] is not None and dinfo["ks_statistic"] != "N/A" else "N/A"
            ks_pval = f"{float(dinfo['ks_p_value']):.4e}" if "ks_p_value" in dinfo and dinfo["ks_p_value"] is not None and dinfo["ks_p_value"] != "N/A" else "N/A"
            drift_table.append({
                "Feature": str(feat),
                "Type": str(dinfo.get("type", "")),
                "PSI / TVD": psi_str,
                "KS-Statistic": ks_stat,
                "KS p-value": ks_pval,
                "Drift Status": str(dinfo.get("drift_status", "UNKNOWN"))
            })
        st.dataframe(pd.DataFrame(drift_table), use_container_width=True)

    # =========================================================================
    # SECTION 4: PREDICTIVE MODELS
    # =========================================================================
    elif "4. Predictive Models" in section:
        st.subheader("Machine Learning Predictive Models & Calibration")

        task = st.selectbox("Select Predictive Modeling Task:", ["Default (Next 12M)", "Delinquency (Next 3M)", "Prepayment (Next 12M)", "Next-State (Multiclass)"])

        if "Default" in task:
            m = model_metrics.get("default", {})
            b = model_metrics.get("default_baseline", {})
        elif "Delinquency" in task:
            m = model_metrics.get("delinquency", {})
            b = model_metrics.get("delinquency_baseline", {})
        elif "Prepayment" in task:
            m = model_metrics.get("prepayment", {})
            b = model_metrics.get("prepayment_baseline", {})
        else:
            m = model_metrics.get("next_state", {})
            b = model_metrics.get("next_state_baseline", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC", f"{m.get('roc_auc', 'N/A')}", delta=f"vs Base: {b.get('roc_auc', 'N/A')}" if "roc_auc" in b else None)
        with col2:
            st.metric("PR-AUC (Avg Precision)", f"{m.get('pr_auc', 'N/A')}", delta=f"vs Base: {b.get('pr_auc', 'N/A')}" if "pr_auc" in b else None)
        with col3:
            st.metric("F1-Score", f"{m.get('f1', m.get('macro_f1', 'N/A'))}")
        with col4:
            st.metric("Brier Score", f"{m.get('brier_score', 'N/A')}", help="Lower is better calibrated.")

        if "confusion_matrix" in m:
            st.markdown("##### Confusion Matrix")
            cm = np.array(m["confusion_matrix"])
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix: {task}")
            st.plotly_chart(fig_cm, use_container_width=True)

    # =========================================================================
    # SECTION 5: TRANSITION & SURVIVAL
    # =========================================================================
    elif "5. Transition & Survival" in section:
        st.subheader("Monthly State Transition & Multi-Period Projections")

        trans_data = model_metrics.get("transition_model", {})
        global_mat = trans_data.get("global_matrix", {})

        if global_mat:
            st.markdown("##### Empirical 1-Month Transition Matrix P(S_{t+1} | S_t)")
            mat_df = pd.DataFrame(global_mat)
            fig_mat = px.imshow(mat_df, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Global Monthly State Transition Probabilities")
            st.plotly_chart(fig_mat, use_container_width=True)

            markov = load_model(MODELS_DIR / "markov_transition_model.joblib")
            proj_df = markov.project_multi_period(horizon_months=36)

            st.markdown("##### Multi-Period Absorption Curves (36 Months)")
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=proj_df["month"], y=proj_df["cumulative_default"], name="Cumulative Default Rate", line=dict(color="#ef4444", width=3)))
            fig_proj.add_trace(go.Scatter(x=proj_df["month"], y=proj_df["cumulative_prepayment"], name="Cumulative Prepayment Rate", line=dict(color="#3b82f6", width=3)))
            fig_proj.add_trace(go.Scatter(x=proj_df["month"], y=proj_df.get("CURRENT", np.zeros(len(proj_df))), name="Remaining Performing (Current)", line=dict(color="#10b981", dash="dash")))
            fig_proj.update_layout(xaxis_title="Horizon (Months)", yaxis_title="Probability", hovermode="x unified")
            st.plotly_chart(fig_proj, use_container_width=True)

    # =========================================================================
    # SECTION 6: ANOMALIES & EXCEPTIONS
    # =========================================================================
    elif "6. Anomalies & Exceptions" in section:
        st.subheader("Hybrid Anomaly Detection & Exception Dossiers")

        dossier_path = PREDICTIONS_DIR / "reviewer_exception_dossiers.csv"
        if dossier_path.exists():
            dossiers_df = pd.read_csv(dossier_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Critical / High Severity Flags", len(dossiers_df))
            with col2:
                n_rec = (dossiers_df["reviewer_action"] == "RECONCILE_SOURCE_CONFLICT").sum()
                st.metric("Reconciliation Tape Conflicts", n_rec)
            with col3:
                n_audit = (dossiers_df["reviewer_action"] == "MANUAL_AUDIT_REQUIRED").sum()
                st.metric("Manual Audit Required", n_audit)

            st.markdown("##### Flagged Reviewer Priority Exception Triage Queue")
            st.dataframe(
                dossiers_df[[
                    "loan_id", "current_status", "original_balance", "credit_score",
                    "anomaly_score", "anomaly_severity", "exception_type", "reviewer_action", "top_drivers"
                ]].head(25),
                use_container_width=True
            )

    # =========================================================================
    # SECTION 7: SCENARIOS & STRESS
    # =========================================================================
    elif "7. Scenario & Stress" in section:
        st.subheader("Macroeconomic Scenario & Portfolio Stress Simulation")
        st.info("Disclaimer: Scenario outputs are model-based stress projections, not guaranteed causal forecasts.")

        scen_names = list(scenario_results.keys())
        selected_scen = st.selectbox("Select Scenario to Inspect:", scen_names)
        res = scenario_results.get(selected_scen, {})

        p_metrics = res.get("portfolio_metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Projected Default Rate", f"{p_metrics.get('projected_default_rate', 0):.2%}")
        with col2:
            st.metric("Projected Prepayment Rate", f"{p_metrics.get('projected_prepayment_rate', 0):.2%}")
        with col3:
            st.metric("Projected Delinquency Rate", f"{p_metrics.get('projected_delinquency_rate', 0):.2%}")
        with col4:
            st.metric("Expected Portfolio Loss", f"${p_metrics.get('expected_loss_dollars', 0):,.2f}")

        comp_data = []
        for sname, sinfo in scenario_results.items():
            sm = sinfo.get("portfolio_metrics", {})
            comp_data.append({
                "Scenario": sname,
                "Default Rate (%)": sm.get("projected_default_rate", 0) * 100.0,
                "Prepayment Rate (%)": sm.get("projected_prepayment_rate", 0) * 100.0,
                "Loss Rate (%)": sm.get("expected_loss_rate", 0) * 100.0
            })
        fig_comp = px.bar(pd.DataFrame(comp_data), x="Scenario", y=["Default Rate (%)", "Prepayment Rate (%)", "Loss Rate (%)"], barmode="group", title="Scenario Comparison")
        st.plotly_chart(fig_comp, use_container_width=True)

    # =========================================================================
    # SECTION 8: EXPLAINABILITY
    # =========================================================================
    elif "8. Global & Local Explainability" in section:
        st.subheader("Global & Local Model Explainability")

        tab1, tab2 = st.tabs(["Global Feature Importance", "Individual Loan Inspector"])

        with tab1:
            feat_imp_path = METRICS_DIR / "feature_importances.json"
            if feat_imp_path.exists():
                feat_imp = pd.DataFrame(load_json(feat_imp_path))
                fig_imp = px.bar(feat_imp.head(15), x="importance_pct", y="feature", orientation="h", title="Top 15 Feature Importances (%)", color="importance_pct", color_continuous_scale="Teal")
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)

        with tab2:
            st.markdown("##### Interactive Loan Attribution Inspector")
            sample_ids = submission_df["loan_id"].head(20).tolist()
            selected_loan_id = st.selectbox("Select Loan ID to Inspect:", sample_ids)
            
            loan_sub = submission_df[submission_df["loan_id"] == selected_loan_id].iloc[0]
            st.write(loan_sub.to_dict())

    # =========================================================================
    # SECTION 9: LLM REVIEWER COPILOT
    # =========================================================================
    elif "9. Grounded LLM Reviewer Copilot" in section:
        st.subheader("Grounded AI Reviewer Copilot & Governance")
        st.markdown("> **Recommendation -- requires human review**")

        copilot = GroundedReviewerCopilot()
        
        tab1, tab2, tab3 = st.tabs(["Interactive Reviewer Note Generator", "Evaluation Benchmarks", "Audit Governance Ledger"])

        with tab1:
            selected_loan = st.selectbox("Select Loan for Review Note:", submission_df["loan_id"].head(10).tolist())
            if st.button("Generate Grounded Reviewer Dossier"):
                static_rec = static_df[static_df["loan_id"] == selected_loan].iloc[0].to_dict()
                sub_rec = submission_df[submission_df["loan_id"] == selected_loan].iloc[0].to_dict()
                
                note_res = copilot.generate_loan_reviewer_note(
                    loan_id=selected_loan,
                    static_record=static_rec,
                    model_predictions=sub_rec,
                    anomaly_evidence=sub_rec
                )
                st.markdown(note_res["response"])

        with tab2:
            st.markdown("##### Codified LLM Failure-Mode Evaluation Benchmarks")
            benchmarks = get_llm_evaluation_benchmarks()
            for b in benchmarks:
                with st.expander(f"Benchmark: {b['failure_category']} ({b['benchmark_id']})"):
                    st.markdown(f"**Prompt**: `{b['prompt']}`")
                    st.error(f"**Deficient (Unacceptable) Output**:\n{b['unacceptable_llm_response']}")
                    st.warning(f"**Deficiency Explanation**: {b['deficiency_explanation']}")
                    st.success(f"**Grounded Acceptable Output**:\n{b['grounded_acceptable_response']}")

        with tab3:
            st.markdown("##### Recent Copilot Audit Log Records (copilot_audit_log.jsonl)")
            logs = copilot.audit_logger.get_recent_audit_logs(limit=10)
            if logs:
                st.json(logs)
            else:
                st.info("No audit interactions recorded yet.")

    # =========================================================================
    # SECTION 10: MODEL CARD & GOVERNANCE
    # =========================================================================
    elif "10. Model Card" in section:
        st.subheader("Model Card & System Lineage")

        card_path = PROJECT_ROOT / "docs" / "MODEL_CARD.md"
        if card_path.exists():
            with open(card_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Model Card document is located at docs/MODEL_CARD.md.")


if __name__ == "__main__":
    main()
