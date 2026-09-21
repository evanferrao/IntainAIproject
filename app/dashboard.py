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
from src.copilot.groq_client import GroqClient
from src.copilot.evaluation_suite import get_llm_evaluation_benchmarks
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LimeExplainer
from src.explainability.pdp_explainer import PartialDependenceExplainer
from src.explainability.ceteris_paribus import CeterisParibusExplainer
from src.explainability.counterfactual_engine import CounterfactualSearchEngine
from src.anomaly.rule_engine import DeterministicRuleEngine
from src.anomaly.reviewer_policy import ReviewerPolicyEngine
import re
from src.config.schema import CanonicalStaticLoan


def clean_markdown_output(text: str) -> str:
    """Sanitizes raw HTML tags from model responses into clean standard Markdown."""
    if not text:
        return ""
    # Convert bold tags to **...**
    text = re.sub(r"<\s*b\b[^>]*>(.*?)<\s*/\s*b\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*strong\b[^>]*>(.*?)<\s*/\s*strong\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    # Convert italic tags to *...*
    text = re.sub(r"<\s*i\b[^>]*>(.*?)<\s*/\s*i\s*>", r"*\1*", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*em\b[^>]*>(.*?)<\s*/\s*em\s*>", r"*\1*", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove any remaining stray b, strong, i, em, span, div, p tags
    text = re.sub(r"<\s*/?\s*(?:b|strong|i|em|div|span|p)\b[^>]*>", "", text, flags=re.IGNORECASE)

    # Handle <br> tags: inside table rows (lines containing "|"), normalize to <br/> for clean multi-line display; outside tables, convert to \n
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        if "|" in line:
            line = re.sub(r"<\s*br\s*/?>", "<br/>", line, flags=re.IGNORECASE)
        else:
            line = re.sub(r"<\s*br\s*/?>", "\n", line, flags=re.IGNORECASE)
        processed_lines.append(line)
    return "\n".join(processed_lines)


def apply_light_theme(fig):
    """Applies crisp light theme to Plotly figures for presentation."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter, sans-serif")
    )
    return fig


# Page Configuration
st.set_page_config(
    page_title="Loan Performance Intelligence Engine | Intain FinTech 2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (FinTech Light Presentation Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 40%, #1e293b 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(15, 118, 110, 0.15);
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
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .disposition-badge {
        display: inline-block;
        padding: 6px 14px;
        font-size: 14px;
        font-weight: 700;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .badge-auto-approve { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .badge-flag-review { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .badge-reconcile { background: #ffedd5; color: #9a3412; border: 1px solid #fdba74; }
    .badge-manual-audit { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-high-risk { background: #fce7f3; color: #9d174d; border: 1px solid #f472b6; }

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


@st.cache_resource
def load_cached_models():
    """Caches ML models and pipelines in memory."""
    models = {
        "feature_pipeline": load_model(MODELS_DIR / "feature_pipeline.joblib"),
        "default": load_model(MODELS_DIR / "default_model.joblib"),
        "delinquency": load_model(MODELS_DIR / "delinquency_model.joblib"),
        "prepayment": load_model(MODELS_DIR / "prepayment_model.joblib"),
        "next_state": load_model(MODELS_DIR / "next_state_model.joblib"),
        "isolation_forest": load_model(MODELS_DIR / "isolation_forest.joblib"),
        "markov": load_model(MODELS_DIR / "markov_transition_model.joblib")
    }
    return models


@st.cache_resource
def load_cached_explainers(_models, _eval_df):
    """Caches explainers and background data."""
    pipe = _models["feature_pipeline"]
    X_sample = pipe.transform(_eval_df.head(150))
    shap_exp = ShapExplainer()
    lime_exp = LimeExplainer(background_data=X_sample)
    pdp_exp = PartialDependenceExplainer()
    cp_exp = CeterisParibusExplainer()
    cf_engine = CounterfactualSearchEngine()
    return shap_exp, lime_exp, pdp_exp, cp_exp, cf_engine, X_sample


def check_pipeline_executed() -> bool:
    """Checks if pipeline artifacts exist."""
    required_files = [
        METRICS_DIR / "model_metrics.json",
        METRICS_DIR / "data_profiling_summary.json",
        PREDICTIONS_DIR / "submission.csv"
    ]
    return all(f.exists() for f in required_files)


def render_disposition_badge(disposition: str):
    """Renders styled HTML badge for reviewer disposition."""
    cls_map = {
        "AUTO_APPROVE": "badge-auto-approve",
        "FLAG_FOR_REVIEW": "badge-flag-review",
        "RECONCILE_SOURCE_CONFLICT": "badge-reconcile",
        "MANUAL_AUDIT_REQUIRED": "badge-manual-audit",
        "HIGH_RISK_REVIEW": "badge-high-risk"
    }
    badge_cls = cls_map.get(disposition, "badge-flag-review")
    st.markdown(f'<div class="disposition-badge {badge_cls}">Reviewer Disposition: {disposition}</div>', unsafe_allow_html=True)


def main():
    if not check_pipeline_executed():
        st.warning("Pipeline artifacts not detected. Please run the pipeline script first to generate metrics and models:")
        st.code("python scripts/run_pipeline.py", language="bash")
        st.info("The dashboard will automatically populate once artifacts are generated.")
        return

    # Load Artifacts
    model_metrics = load_json(METRICS_DIR / "model_metrics.json")
    profiling_summary = load_json(METRICS_DIR / "data_profiling_summary.json")
    quality_summary = load_json(METRICS_DIR / "data_quality_score.json")
    drift_summary = load_json(METRICS_DIR / "drift_metrics.json")
    scenario_results = load_json(METRICS_DIR / "scenario_simulation_results.json")
    
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")
    submission_df = pd.read_csv(PREDICTIONS_DIR / "submission.csv")

    # Latest observation per test loan merged with static attributes
    latest_test = test_panel.sort_values("month_index").groupby("loan_id").last().reset_index()
    eval_df = latest_test.merge(static_df, on="loan_id", how="left")

    # Load Models and Explainers
    models = load_cached_models()
    shap_exp, lime_exp, pdp_exp, cp_exp, cf_engine, X_sample = load_cached_explainers(models, eval_df)

    # Decision Support Copilot status
    groq_client = GroqClient()
    copilot = GroundedReviewerCopilot(groq_client=groq_client)

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
            "8. Model Explainability (SHAP/LIME/PDP/CP)",
            "9. Counterfactual Analysis ('What Would Lower Risk?')",
            "10. Reviewer Decision Intelligence",
            "11. Interactive New Loan Simulator",
            "12. Model Card & Governance"
        ]
    )

    # Sidebar Loan Selector (Global session state)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interactive Loan Selector")
    
    # Filter by disposition
    disp_options = ["ALL"] + sorted(submission_df["reviewer_action"].unique().tolist())
    sel_disp = st.sidebar.selectbox("Filter Loans by Disposition:", disp_options)
    if sel_disp != "ALL":
        filtered_sub = submission_df[submission_df["reviewer_action"] == sel_disp]
    else:
        filtered_sub = submission_df

    available_loan_ids = filtered_sub["loan_id"].astype(str).tolist()

    if "selected_loan_id" not in st.session_state or st.session_state["selected_loan_id"] not in available_loan_ids:
        st.session_state["selected_loan_id"] = available_loan_ids[0] if available_loan_ids else "N/A"

    current_loan_id = st.sidebar.selectbox(
        "Selected Loan ID:",
        available_loan_ids,
        index=available_loan_ids.index(st.session_state["selected_loan_id"]) if st.session_state["selected_loan_id"] in available_loan_ids else 0,
        key="sidebar_loan_id_select"
    )
    st.session_state["selected_loan_id"] = current_loan_id

    # Sidebar Status Card
    if current_loan_id != "N/A" and current_loan_id in submission_df["loan_id"].values:
        sub_row = submission_df[submission_df["loan_id"] == current_loan_id].iloc[0]
        st.sidebar.markdown(f"**Action**: `{sub_row['reviewer_action']}`")
        st.sidebar.markdown(f"**Default Risk**: `{sub_row['default_probability']:.1%}`")
        st.sidebar.markdown(f"**Anomaly Score**: `{sub_row['anomaly_score']:.1f}/100`")
        st.sidebar.markdown(f"**Confidence**: `{sub_row['confidence']}`")

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
            st.plotly_chart(apply_light_theme(fig_fico), use_container_width=True)

        with row1_col2:
            st.markdown("##### Loan Purpose Breakdown")
            fig_purpose = px.pie(
                static_df, names="loan_purpose",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_purpose.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(apply_light_theme(fig_purpose), use_container_width=True)

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
            st.plotly_chart(apply_light_theme(fig_miss), use_container_width=True)

        with tab3:
            corr_dict = profiling_summary.get("numeric_correlations", {})
            if corr_dict:
                corr_df = pd.DataFrame(corr_dict)
                fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Pearson Feature Correlation Matrix")
                st.plotly_chart(apply_light_theme(fig_corr), use_container_width=True)

    # =========================================================================
    # SECTION 3: DATA QUALITY & DRIFT MONITOR
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
    # SECTION 4: PREDICTIVE MODELS & CALIBRATION
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
            st.plotly_chart(apply_light_theme(fig_cm), use_container_width=True)

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
            st.plotly_chart(apply_light_theme(fig_mat), use_container_width=True)

            markov = models["markov"]
            proj_df = markov.project_multi_period(horizon_months=36)

            st.markdown("##### Multi-Period Absorption Curves (36 Months)")
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=proj_df["month"], y=proj_df["cumulative_default"], name="Cumulative Default Rate", line=dict(color="#ef4444", width=3)))
            fig_proj.add_trace(go.Scatter(x=proj_df["month"], y=proj_df["cumulative_prepayment"], name="Cumulative Prepayment Rate", line=dict(color="#3b82f6", width=3)))
            fig_proj.add_trace(go.Scatter(x=proj_df["month"], y=proj_df.get("CURRENT", np.zeros(len(proj_df))), name="Remaining Performing (Current)", line=dict(color="#10b981", dash="dash")))
            fig_proj.update_layout(xaxis_title="Horizon (Months)", yaxis_title="Probability", hovermode="x unified")
            st.plotly_chart(apply_light_theme(fig_proj), use_container_width=True)

    # =========================================================================
    # SECTION 6: ANOMALIES & EXCEPTION DOSSIERS
    # =========================================================================
    elif "6. Anomalies & Exceptions" in section:
        st.subheader("Hybrid Anomaly Detection & Exception Dossiers")

        dossier_path = PREDICTIONS_DIR / "reviewer_exception_dossiers.csv"
        dossiers_df = pd.read_csv(dossier_path) if dossier_path.exists() else submission_df

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Evaluation Loans", len(submission_df))
        with col2:
            n_flagged = (submission_df["reviewer_action"] != "AUTO_APPROVE").sum()
            st.metric("Flagged for Human Review", n_flagged)
        with col3:
            n_rec = (submission_df["reviewer_action"] == "RECONCILE_SOURCE_CONFLICT").sum()
            st.metric("Tape Reconciliation Conflicts", n_rec)
        with col4:
            n_high = (submission_df["reviewer_action"] == "HIGH_RISK_REVIEW").sum()
            st.metric("High-Risk Review", n_high)

        st.markdown("##### Reviewer Action Distribution Across Portfolio")
        act_counts = submission_df["reviewer_action"].value_counts().reset_index()
        act_counts.columns = ["Reviewer Action", "Count"]
        fig_act = px.bar(
            act_counts, x="Reviewer Action", y="Count", color="Reviewer Action",
            color_discrete_map={
                "AUTO_APPROVE": "#10b981",
                "FLAG_FOR_REVIEW": "#f59e0b",
                "RECONCILE_SOURCE_CONFLICT": "#f97316",
                "MANUAL_AUDIT_REQUIRED": "#ef4444",
                "HIGH_RISK_REVIEW": "#ec4899"
            }
        )
        st.plotly_chart(apply_light_theme(fig_act), use_container_width=True)

        st.markdown("##### Priority Flagged Exception Queue")
        display_cols = [c for c in ["loan_id", "delinquency_probability", "default_probability", "anomaly_score", "exception_type", "reviewer_action", "top_drivers"] if c in submission_df.columns]
        st.dataframe(submission_df[submission_df["reviewer_action"] != "AUTO_APPROVE"][display_cols].head(30), use_container_width=True)

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
        st.plotly_chart(apply_light_theme(fig_comp), use_container_width=True)

    # =========================================================================
    # SECTION 8: MODEL EXPLAINABILITY (SHAP / LIME / PDP / CETERIS PARIBUS)
    # =========================================================================
    elif "8. Model Explainability" in section:
        st.subheader("Local & Global Model Explainability")
        st.markdown(f"**Inspecting Loan**: `{current_loan_id}`")

        # Select target model
        target_model_name = st.selectbox(
            "Select Target Prediction Model to Explain:",
            ["Default (Next 12M)", "Delinquency (Next 3M)", "Prepayment (Next 12M)"]
        )
        model_key = "default" if "Default" in target_model_name else "delinquency" if "Delinquency" in target_model_name else "prepayment"
        target_model = models[model_key]

        # Extract selected loan row
        loan_raw_matches = eval_df[eval_df["loan_id"] == current_loan_id]
        if len(loan_raw_matches) == 0:
            st.error(f"Loan ID `{current_loan_id}` not found in test evaluation dataset.")
            return

        loan_raw_record = loan_raw_matches.iloc[0]
        X_loan = models["feature_pipeline"].transform(pd.DataFrame([loan_raw_record.to_dict()]))
        current_pred_prob = float(target_model.predict_proba(X_loan)[0])

        st.metric(f"Current Predicted {target_model_name} Probability", f"{current_pred_prob:.2%}")

        tab_shap, tab_lime, tab_pdp, tab_cp = st.tabs([
            "1. SHAP Attributions",
            "2. LIME Local Surrogate",
            "3. Partial Dependence (PDP)",
            "4. Ceteris Paribus Profiles"
        ])

        with tab_shap:
            st.markdown("##### Local SHAP Feature Contributions")
            shap_res = shap_exp.explain_instance(
                model_key=model_key,
                model=target_model,
                transformed_row=X_loan,
                raw_row=loan_raw_record,
                top_k=12
            )

            col_s1, col_s2 = st.columns([3, 2])
            with col_s1:
                df_top = pd.DataFrame(shap_res["top_drivers"])
                fig_shap = px.bar(
                    df_top,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="direction",
                    color_discrete_map={"INCREASES_RISK": "#ef4444", "DECREASES_RISK": "#10b981"},
                    title=f"Top SHAP Risk Drivers for `{current_loan_id}` (Base Value: {shap_res['base_value']:.3f})"
                )
                fig_shap.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(apply_light_theme(fig_shap), use_container_width=True)

            with col_s2:
                st.markdown("###### Top Upward Drivers (Increases Risk)")
                st.dataframe(pd.DataFrame(shap_res["top_positive_risk_drivers"])[["feature", "feature_value", "shap_value"]], use_container_width=True)
                st.markdown("###### Top Downward Drivers (Decreases Risk)")
                st.dataframe(pd.DataFrame(shap_res["top_negative_risk_drivers"])[["feature", "feature_value", "shap_value"]], use_container_width=True)

            with st.expander("Global SHAP Feature Summary"):
                global_shap = shap_exp.compute_global_summary(model_key=model_key, model=target_model, X_sample=X_sample)
                fig_gshap = px.bar(global_shap, x="mean_abs_shap", y="feature", orientation="h", title="Global Mean |SHAP| Value", color="mean_abs_shap", color_continuous_scale="Teal")
                fig_gshap.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(apply_light_theme(fig_gshap), use_container_width=True)

        with tab_lime:
            st.markdown("##### LIME Local Surrogate Explanation")
            def predict_fn(x_arr):
                target_est = target_model.improved_model if hasattr(target_model, "improved_model") else target_model
                return target_est.predict_proba(x_arr)

            lime_res = lime_exp.explain_instance(
                predict_fn=predict_fn,
                transformed_row=X_loan.iloc[0],
                num_features=8
            )

            df_lime = pd.DataFrame(lime_res["attributions"])
            fig_lime = px.bar(
                df_lime,
                x="weight",
                y="rule",
                orientation="h",
                color="direction",
                color_discrete_map={"INCREASES_RISK": "#ef4444", "DECREASES_RISK": "#10b981"},
                title=f"LIME Rule Attributions for `{current_loan_id}` (Local Intercept: {lime_res['intercept']:.3f})"
            )
            fig_lime.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(apply_light_theme(fig_lime), use_container_width=True)
            st.dataframe(df_lime[["rule", "weight", "direction"]], use_container_width=True)

        with tab_pdp:
            st.markdown("##### Partial Dependence Plots (PDP)")
            pdp_features = [f for f in PartialDependenceExplainer.RECOMMENDED_PDP_FEATURES if f in X_sample.columns]
            selected_pdp_feat = st.selectbox("Select Feature for PDP:", pdp_features, index=0)

            pdp_result = pdp_exp.compute_pdp(model=target_model, X_sample=X_sample, feature_name=selected_pdp_feat)
            if pdp_result["is_appropriate"]:
                fig_pdp = px.line(
                    x=pdp_result["grid_values"],
                    y=pdp_result["average_predictions"],
                    title=f"Partial Dependence of {target_model_name} on `{selected_pdp_feat}`",
                    labels={"x": selected_pdp_feat, "y": f"Average Predicted {target_model_name} Risk"}
                )
                fig_pdp.update_traces(line=dict(color="#0f766e", width=3))
                st.plotly_chart(apply_light_theme(fig_pdp), use_container_width=True)
                st.caption(f"Feature Spread: {pdp_result['spread']:.4f} (Min: {pdp_result['min_response']:.4f}, Max: {pdp_result['max_response']:.4f})")
            else:
                st.warning(f"PDP Inappropriate for `{selected_pdp_feat}`: {pdp_result.get('reason', 'Unknown reason')}")

        with tab_cp:
            st.markdown("##### Ceteris Paribus / Individual Conditional Expectation (ICE)")
            st.info("Ceteris Paribus analysis evaluates how the model's prediction changes as a single feature varies, holding all other features fixed.")
            
            cp_features = [f for f in ["credit_score", "dti", "interest_rate", "original_balance", "installment", "revolving_utilization", "annual_income"] if f in loan_raw_record.index]
            selected_cp_feat = st.selectbox("Select Feature to Vary:", cp_features, index=0)

            cp_result = cp_exp.compute_profile(
                loan_raw_record=loan_raw_record,
                feature_pipeline=models["feature_pipeline"],
                model=target_model,
                feature_name=selected_cp_feat
            )

            if cp_result["success"]:
                fig_cp = go.Figure()
                fig_cp.add_trace(go.Scatter(
                    x=cp_result["grid_values"],
                    y=cp_result["predicted_probabilities"],
                    mode="lines+markers",
                    name="Model Response Curve",
                    line=dict(color="#2563eb", width=3)
                ))
                fig_cp.add_trace(go.Scatter(
                    x=[cp_result["current_value"]],
                    y=[cp_result["current_prediction"]],
                    mode="markers",
                    name=f"Current Loan (`{cp_result['current_value']}`)",
                    marker=dict(color="#ef4444", size=14, symbol="diamond")
                ))
                fig_cp.update_layout(
                    title=f"Ceteris Paribus Curve: {target_model_name} vs. `{selected_cp_feat}`",
                    xaxis_title=selected_cp_feat,
                    yaxis_title="Predicted Probability",
                    hovermode="x unified"
                )
                st.plotly_chart(apply_light_theme(fig_cp), use_container_width=True)
                st.caption(f"> **Notice**: {cp_result['disclaimer']}")

    # =========================================================================
    # SECTION 9: COUNTERFACTUAL ANALYSIS ("WHAT WOULD LOWER THE RISK?")
    # =========================================================================
    elif "9. Counterfactual Analysis" in section:
        st.subheader("Counterfactual Analysis & Model-Implied Risk Reduction")
        st.markdown(f"**Selected Loan**: `{current_loan_id}`")

        loan_raw_matches = eval_df[eval_df["loan_id"] == current_loan_id]
        if len(loan_raw_matches) == 0:
            st.error(f"Loan ID `{current_loan_id}` not found in test evaluation dataset.")
            return

        loan_raw_record = loan_raw_matches.iloc[0]
        def_model = models["default"]

        cf_results = cf_engine.search_counterfactuals(
            loan_raw_record=loan_raw_record,
            feature_pipeline=models["feature_pipeline"],
            model=def_model,
            model_name="Improved_Default_LightGBM",
            max_results=5,
            min_risk_reduction=0.002
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Default Risk", f"{cf_results['current_risk']:.2%}")
        with col2:
            st.metric("Candidates Evaluated", cf_results["total_candidates_evaluated"])
        with col3:
            st.metric("Viable Counterfactuals Found", len(cf_results["top_counterfactuals"]))

        st.markdown(f"> [!NOTE]\n> **Non-Causal Disclaimer**: {cf_results['disclaimer']}")

        top_cfs = cf_results["top_counterfactuals"]
        if top_cfs:
            st.markdown("##### Candidate Counterfactual Scenarios (Ranked by Sparsity & Risk Reduction)")
            cf_table = []
            for idx, c in enumerate(top_cfs, 1):
                changes_str = "; ".join([
                    f"{feat}: {info['current']} -> {info['counterfactual']} (Δ {info['delta']:+})"
                    for feat, info in c["features_changed"].items()
                ])
                cf_table.append({
                    "Scenario": f"Candidate #{idx}",
                    "Features Changed": changes_str,
                    "Sparsity": c["num_features_changed"],
                    "Counterfactual Risk": f"{c['counterfactual_risk']:.2%}",
                    "Risk Reduction (pp)": f"{c['risk_reduction'] * 100:.2f} pp",
                    "Relative Drop (%)": f"{c['relative_reduction'] * 100:.1f}%",
                    "Distance Cost": c["distance_cost"]
                })
            st.dataframe(pd.DataFrame(cf_table), use_container_width=True)

            # Interactive Explanation of Counterfactuals
            st.markdown("---")
            st.markdown("##### Generate Risk Reduction & Counterfactual Synthesis")
            if st.button("Generate Analytical Risk Explanation"):
                if not groq_client.is_available():
                    st.warning("Decision Support Engine key is not configured in .env. Analytical synthesis is unavailable.")
                else:
                    context = copilot.build_structured_context(
                        loan_id=current_loan_id,
                        static_record=loan_raw_record.to_dict(),
                        model_predictions={"default_probability": cf_results["current_risk"]},
                        counterfactual_info=cf_results
                    )
                    with st.spinner("Generating analytical explanation from grounded evidence..."):
                        ai_res = copilot.ask_llm_about_loan(
                            loan_id=current_loan_id,
                            user_question="Explain these counterfactual results in plain English. What specific changes reduce the predicted default risk most effectively and what should a human underwriter verify?",
                            structured_context=context
                        )
                    if ai_res["success"]:
                        st.success(f"**Analytical Risk Explanation ({ai_res['latency_seconds']}s)**:")
                        st.markdown(clean_markdown_output(ai_res["response"]), unsafe_allow_html=True)
                    else:
                        st.error(f"Decision Support Engine Error: {ai_res['error']}")
        else:
            st.info("No viable counterfactuals found with material risk reduction for this loan (loan is already at minimum risk).")

    # =========================================================================
    # SECTION 10: REVIEWER DECISION INTELLIGENCE
    # =========================================================================
    elif "10. Reviewer Decision Intelligence" in section:
        st.subheader("Reviewer Decision Intelligence & Governance")
        st.markdown(f"**Selected Loan**: `{current_loan_id}`")

        loan_raw_matches = eval_df[eval_df["loan_id"] == current_loan_id]
        if len(loan_raw_matches) == 0:
            st.error(f"Loan ID `{current_loan_id}` not found in test evaluation dataset.")
            return

        loan_raw_record = loan_raw_matches.iloc[0]
        sub_record = submission_df[submission_df["loan_id"] == current_loan_id].iloc[0].to_dict()

        # Display Reviewer Disposition & Evidence
        render_disposition_badge(sub_record.get("reviewer_action", "AUTO_APPROVE"))
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"**Default Probability**: `{sub_record.get('default_probability', 0.0):.2%}`")
            st.markdown(f"**Delinquency Probability**: `{sub_record.get('delinquency_probability', 0.0):.2%}`")
            st.markdown(f"**Anomaly Score**: `{sub_record.get('anomaly_score', 0.0):.1f}/100`")
        with col_r2:
            st.markdown(f"**Model Confidence**: `{sub_record.get('confidence', 'HIGH')}`")
            st.markdown(f"**Exception Type**: `{sub_record.get('exception_type', 'NONE')}`")
            st.markdown(f"**Top Drivers**: `{sub_record.get('top_drivers', 'standard_performing_profile')}`")

        tab_chat, tab_bench, tab_audit = st.tabs(["Interactive Reviewer Inquiry", "Quality & Consistency Benchmarks", "Audit Governance Ledger"])

        with tab_chat:
            st.markdown("##### Loan Decision Intelligence Inquiry")
            
            # Show Decision Support Status
            if groq_client.is_available():
                st.success("Decision Support Engine: Active & Connected")
            else:
                st.warning("Decision Support Engine: Key not configured in `.env`. Automated synthesis is paused; deterministic policies and ML risk models remain active.")

            def _set_suggested_query(q_text: str):
                st.session_state["reviewer_user_query_input"] = q_text
                st.session_state["trigger_reviewer_synthesis"] = True

            # Suggested prompt buttons
            st.markdown("###### Suggested Reviewer Questions:")
            suggested_cols = st.columns(3)
            with suggested_cols[0]:
                st.button(
                    "Why was this loan flagged?",
                    key="q1",
                    on_click=_set_suggested_query,
                    args=("Why was this loan assigned its reviewer disposition, and what specific evidence triggered it?",)
                )
                st.button(
                    "What drives default risk?",
                    key="q2",
                    on_click=_set_suggested_query,
                    args=("What are the strongest drivers of default risk for this loan?",)
                )
            with suggested_cols[1]:
                st.button(
                    "Explain anomaly findings",
                    key="q3",
                    on_click=_set_suggested_query,
                    args=("Explain the anomaly findings and reconciliation status for this loan in plain English.",)
                )
                st.button(
                    "What to verify in audit?",
                    key="q4",
                    on_click=_set_suggested_query,
                    args=("What specific borrower documents or data attributes should a human auditor verify?",)
                )
            with suggested_cols[2]:
                st.button(
                    "Summarize for reviewer",
                    key="q5",
                    on_click=_set_suggested_query,
                    args=("Summarize this loan's risk profile, model predictions, and audit recommendation for a human reviewer.",)
                )
                st.button(
                    "Compare with portfolio",
                    key="q6",
                    on_click=_set_suggested_query,
                    args=("How does this loan's risk and credit score compare with the broader portfolio?",)
                )

            if "reviewer_user_query_input" not in st.session_state:
                st.session_state["reviewer_user_query_input"] = ""

            user_question = st.text_area(
                "Enter question about this loan's analytical evidence:",
                placeholder="Ask any question about this loan's computed metrics, anomaly status, or risk profile...",
                key="reviewer_user_query_input"
            )

            btn_clicked = st.button("Generate Reviewer Synthesis", type="primary")
            auto_trigger = st.session_state.pop("trigger_reviewer_synthesis", False)

            if btn_clicked or auto_trigger:
                if not user_question.strip():
                    st.warning("Please enter a question or click a suggested prompt above.")
                else:
                    context = copilot.build_structured_context(
                        loan_id=current_loan_id,
                        static_record=loan_raw_record.to_dict(),
                        model_predictions=sub_record,
                        anomaly_evidence=sub_record,
                        reviewer_triage=sub_record
                    )

                    with st.spinner("Analyzing grounded loan evidence..."):
                        res = copilot.ask_llm_about_loan(
                            loan_id=current_loan_id,
                            user_question=user_question,
                            structured_context=context
                        )

                    if res["success"]:
                        st.markdown(f"#### Analytical Reviewer Synthesis ({res['latency_seconds']}s)")
                        st.markdown(clean_markdown_output(res["response"]), unsafe_allow_html=True)
                        st.caption(f"Context Hash: `{res['context_hash']}` | Governance: {res['governance_notice']}")
                        with st.expander("Inspect Grounded Evidence Context"):
                            st.json(res["context_used"])
                    else:
                        st.error(f"Synthesis Engine Error: {res['error']}")
                        with st.expander("Inspect Grounded Context That Was Prepared"):
                            st.json(context)

        with tab_bench:
            st.markdown("##### Analytical Quality & Consistency Benchmarks")
            benchmarks = get_llm_evaluation_benchmarks()
            for b in benchmarks:
                with st.expander(f"Benchmark: {b['failure_category']} ({b['benchmark_id']})"):
                    st.markdown(f"**Prompt**: `{b['prompt']}`")
                    st.error(f"**Deficient (Unacceptable) Output**:\n{clean_markdown_output(b['unacceptable_llm_response'])}")
                    st.warning(f"**Deficiency Explanation**: {b['deficiency_explanation']}")
                    st.success(f"**Grounded Acceptable Output**:\n{clean_markdown_output(b['grounded_acceptable_response'])}")

        with tab_audit:
            st.markdown("##### Recent Decision Support Audit Log Records")
            logs = copilot.audit_logger.get_recent_audit_logs(limit=15)
            if logs:
                for entry in reversed(logs):
                    with st.expander(f"{entry.get('timestamp', 'N/A')} | Loan: {entry.get('loan_id', 'N/A')} | Question: {entry.get('user_question', '')[:40]}..."):
                        st.markdown(f"**Engine**: `Decision Intelligence Engine` | **Latency**: `{entry.get('latency', 0.0)}s` | **Success**: `{entry.get('success', False)}`")
                        st.markdown(f"**Question**: {entry.get('user_question', '')}")
                        st.markdown(f"**Response**:\n{clean_markdown_output(entry.get('response', ''))}", unsafe_allow_html=True)
                        if entry.get("error"):
                            st.error(f"Error: {entry['error']}")
                        st.caption(f"Context Hash: `{entry.get('context_hash', 'N/A')}`")
            else:
                st.info("No audit interactions recorded yet.")

    # =========================================================================
    # SECTION 11: INTERACTIVE NEW LOAN SIMULATOR
    # =========================================================================
    elif "11. Interactive New Loan Simulator" in section:
        st.subheader("Interactive New Loan Simulator (Production Inference)")
        st.info("Interactive model inference -- not an actual underwriting decision. Enter new loan attributes to generate live predictions, anomaly checks, explainability, and triage recommendations.")

        with st.form("new_loan_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                f_score = st.slider("Credit Score (FICO)", 300, 850, 680)
                f_dti = st.number_input("Debt-to-Income (DTI %)", min_value=0.0, max_value=100.0, value=22.5, step=0.5)
                f_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=14.5, step=0.25)
                f_term = st.selectbox("Original Term (Months)", [36, 60], index=0)
            with col_f2:
                f_bal = st.number_input("Original Balance ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=500.0)
                f_income = st.number_input("Annual Income ($)", min_value=5000.0, max_value=1000000.0, value=75000.0, step=1000.0)
                f_purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "other"])
                f_home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
            with col_f3:
                f_state = st.selectbox("State", ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "MI", "GA", "NC"])
                f_emp = st.slider("Employment Length (Years)", 0, 10, 5)
                f_doc = st.selectbox("Document Verification Status", ["VERIFIED", "SOURCE_VERIFIED", "UNVERIFIED"])
                f_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0)

            submitted = st.form_submit_button("Run Live Model Inference")

        if submitted:
            # Construct raw record
            monthly_rate = (f_rate / 100.0) / 12.0
            installment = f_bal * (monthly_rate * (1 + monthly_rate)**f_term) / ((1 + monthly_rate)**f_term - 1)

            new_record = {
                "loan_id": "SIM_NEW_001",
                "origination_month": "2026-01",
                "original_balance": float(f_bal),
                "funded_balance": float(f_bal),
                "interest_rate": float(f_rate),
                "original_term": int(f_term),
                "installment": round(installment, 2),
                "credit_score": float(f_score),
                "credit_score_band": "Prime" if f_score >= 720 else "Near Prime" if f_score >= 660 else "Subprime",
                "dti": float(f_dti),
                "dti_band": "<20%" if f_dti < 20 else "20-35%" if f_dti < 35 else ">35%",
                "annual_income": float(f_income),
                "employment_length": int(f_emp),
                "home_ownership": f_home,
                "loan_purpose": f_purpose,
                "state": f_state,
                "revolving_utilization": float(f_util),
                "delinquencies_2yrs": 0,
                "inquiries_6m": 1,
                "total_accounts": 15,
                "servicer_name": "Beacon_Credit_Ops",
                "document_status": f_doc,
                "reporting_month": "2026-01",
                "month_index": 1,
                "loan_age_months": 1,
                "remaining_term_months": f_term - 1,
                "current_balance": float(f_bal),
                "current_status": "CURRENT",
                "days_past_due": 0
            }

            raw_s = pd.Series(new_record)
            raw_df = pd.DataFrame([new_record])

            # 1. Feature pipeline transform
            X_new = models["feature_pipeline"].transform(raw_df)

            # 2. Run models
            p_def = float(models["default"].predict_proba(X_new)[0])
            p_del = float(models["delinquency"].predict_proba(X_new)[0])
            p_prep = float(models["prepayment"].predict_proba(X_new)[0])
            pred_next = str(models["next_state"].predict(X_new)[0])

            # 3. Anomaly & Rule evaluation
            rule_eng = DeterministicRuleEngine()
            rule_eval = rule_eng.evaluate_static_records(raw_df)
            has_rule_violation = bool(rule_eval["has_deterministic_violation"].iloc[0])

            ml_scores, is_ano = models["isolation_forest"].score_samples(X_new)
            ano_score = float(ml_scores[0])

            # 4. Reviewer Triage
            policy_eng = ReviewerPolicyEngine()
            triage_res = policy_eng.evaluate_loan(
                loan_id="SIM_NEW_001",
                default_prob=p_def,
                delinquency_prob=p_del,
                prepayment_prob=p_prep,
                anomaly_score=ano_score,
                is_anomaly=bool(is_ano[0]),
                data_quality_score=98.0,
                has_deterministic_violation=has_rule_violation,
                has_reconciliation_conflict=False
            )

            st.markdown("### Inference Results")
            render_disposition_badge(triage_res["disposition"])

            st.markdown("##### Reasons for Triage Disposition:")
            for r in triage_res["reasons"]:
                st.markdown(f"- {r}")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("12M Default Risk", f"{p_def:.2%}")
            with col_m2:
                st.metric("3M Delinquency Risk", f"{p_del:.2%}")
            with col_m3:
                st.metric("12M Prepayment Risk", f"{p_prep:.2%}")
            with col_m4:
                st.metric("Predicted Next State", pred_next)

            # Show SHAP for new loan
            st.markdown("##### Local SHAP Attribution for New Loan")
            shap_res = shap_exp.explain_instance("default", models["default"], X_new, top_k=8)
            df_new_shap = pd.DataFrame(shap_res["top_drivers"])
            fig_nshap = px.bar(df_new_shap, x="shap_value", y="feature", orientation="h", color="direction", color_discrete_map={"INCREASES_RISK": "#ef4444", "DECREASES_RISK": "#10b981"})
            fig_nshap.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(apply_light_theme(fig_nshap), use_container_width=True)

    # =========================================================================
    # SECTION 12: MODEL CARD & GOVERNANCE
    # =========================================================================
    elif "12. Model Card" in section:
        st.subheader("Model Card & System Lineage")
        card_path = PROJECT_ROOT / "docs" / "MODEL_CARD.md"
        if card_path.exists():
            with open(card_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Model Card document is located at docs/MODEL_CARD.md.")


if __name__ == "__main__":
    main()
