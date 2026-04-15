from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"
COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
    color: white;
    padding: 1.4rem 1.5rem;
    border-radius: 22px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(15,23,42,0.18);
}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: 0.35rem 0 0 0; opacity: 0.95;}
.card {
    background: white;
    color: #0f172a;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    border: 1px solid #e5e7eb;
}
.card h3, .card p, .card ol, .card li, .card b { color: #0f172a; }
.small-label {font-size: 0.9rem; color: #64748b; margin-bottom: 0.2rem;}
.result-box {
    border-radius: 18px;
    padding: 1rem;
    color: white;
    font-weight: 700;
}
.result-yes {background: linear-gradient(135deg, #16a34a, #22c55e);}
.result-no {background: linear-gradient(135deg, #991b1b, #ef4444);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model artifacts, training from scratch if they don't exist yet."""
    if not MODEL_PATH.exists():
        import train as _train
        with st.spinner(
            "🚀 No trained model found — training now (this takes ~2–3 minutes on first run)…"
        ):
            _train.main()

    if not MODEL_PATH.exists():
        return None, None, None, None

    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else None
    comparison = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else None
    return model, metrics, summary, comparison


st.markdown(
    """
    <div class="hero">
        <h1>Bank Term Deposit Subscription Predictor</h1>
        <p>Predict whether a customer is likely to subscribe to a term deposit using a trained machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, metrics, summary, comparison = load_artifacts()

if model is None:
    st.error("❌ Training failed. Check that all dependencies in `requirements.txt` are installed.")
    st.stop()

# Summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"<div class='card'><div class='small-label'>Rows</div><h3>{summary['rows']:,}</h3></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='card'><div class='small-label'>Best model</div><h3>{summary['best_model']}</h3></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='card'><div class='small-label'>Positive rate</div><h3>{summary['positive_rate']:.1%}</h3></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='card'><div class='small-label'>Feature used</div><h3>{len(summary['features_used'])}</h3></div>",
        unsafe_allow_html=True,
    )

# Sidebar inputs
st.sidebar.title("Customer Inputs")
st.sidebar.caption("Enter the customer details and choose a decision threshold.")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

with st.sidebar.form("prediction_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown",
            ],
        )
        marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox(
            "Education",
            [
                "primary", "secondary", "tertiary", "basic.4y", "basic.6y",
                "basic.9y", "high.school", "professional.course", "university.degree",
                "illiterate", "unknown",
            ],
        )
        default = st.selectbox("Credit in default?", ["no", "yes"])
        balance = st.number_input(
            "Average yearly balance (€)",
            value=0,
            min_value=-50000,
            max_value=1000000,
            step=100,
        )
        housing = st.selectbox("Housing loan?", ["no", "yes"])
        loan = st.selectbox("Personal loan?", ["no", "yes"])
        contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"])

    with col_b:
        day = st.number_input("Day of month", min_value=1, max_value=31, value=15)
        month = st.selectbox(
            "Month",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            index=4,
        )
        campaign = st.number_input("Campaign contacts", min_value=1, max_value=63, value=1)
        pdays = st.number_input(
            "Days since previous contact (-1 = never contacted)",
            min_value=-1,
            max_value=999,
            value=-1,
        )
        previous = st.number_input("Previous contacts", min_value=0, max_value=55, value=0)
        poutcome = st.selectbox("Previous campaign outcome", ["unknown", "other", "failure", "success"])

    submitted = st.form_submit_button("Predict Subscription")

input_df = pd.DataFrame(
    [
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day_of_week": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }
    ]
)

# Main tabs
about_tab, predict_tab, eval_tab, insight_tab = st.tabs(
    [
        "Overview",
        "Prediction",
        "Model performance",
        "Dataset insights",
    ]
)

with about_tab:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown(
            """
            <div class='card'>
                <h3>What this system does</h3>
                <p>This app predicts whether a bank customer is likely to subscribe to a term deposit. It is designed as a practical marketing support tool.</p>
                <p><b>Important:</b> the feature <code>duration</code> was intentionally excluded because it is only known after the call ends, which would make the system unrealistic for early decision-making.</p>
                <p>The model uses raw customer and campaign details, then converts them through preprocessing pipelines before prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class='card'>
                <h3>How to use</h3>
                <ol>
                    <li>Fill in the customer details in the sidebar.</li>
                    <li>Choose a threshold for what counts as a positive prediction.</li>
                    <li>Click <b>Predict Subscription</b>.</li>
                    <li>Read the prediction, probability, and supporting charts.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

with predict_tab:
    if submitted:
        proba = float(model.predict_proba(input_df)[0, 1])
        predicted_class = int(proba >= threshold)
        label = "Likely to Subscribe" if predicted_class == 1 else "Unlikely to Subscribe"

        result_class = "result-yes" if predicted_class == 1 else "result-no"
        st.markdown(f"<div class='result-box {result_class}'>{label}</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Subscription probability", f"{proba:.1%}")
        c2.metric("Decision threshold", f"{threshold:.0%}")
        c3.metric("Model decision", "Yes" if predicted_class == 1 else "No")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                title={"text": "Subscription probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 40], "color": "#fecaca"},
                        {"range": [40, 70], "color": "#fde68a"},
                        {"range": [70, 100], "color": "#bbf7d0"},
                    ],
                },
            )
        )
        gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        st.dataframe(input_df, use_container_width=True)
    else:
        st.info("Fill in the customer details in the sidebar and press **Predict Subscription**.")

with eval_tab:
    if comparison is not None:
        display_cols = ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        display_df = comparison[display_cols].copy()
        display_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(
            display_df.style.format(
                {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}
            ),
            use_container_width=True,
        )

        fig = px.bar(
            display_df,
            x="Model",
            y=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            barmode="group",
            title="Model comparison on the test set",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        if (ARTIFACT_DIR / "confusion_matrix.png").exists():
            st.image(
                str(ARTIFACT_DIR / "confusion_matrix.png"),
                caption="Confusion Matrix",
                use_container_width=True,
            )
    with col_right:
        if (ARTIFACT_DIR / "roc_curve.png").exists():
            st.image(
                str(ARTIFACT_DIR / "roc_curve.png"),
                caption="ROC Curve",
                use_container_width=True,
            )

with insight_tab:
    if (ARTIFACT_DIR / "feature_importance.png").exists():
        st.image(
            str(ARTIFACT_DIR / "feature_importance.png"),
            caption="Top feature importance",
            use_container_width=True,
        )

    if summary is not None:
        st.markdown(
            f"""
            <div class='card'>
                <h3>Dataset notes</h3>
                <p><b>Rows:</b> {summary['rows']:,}</p>
                <p><b>Positive cases:</b> {summary['positive_cases']:,}</p>
                <p><b>Negative cases:</b> {summary['negative_cases']:,}</p>
                <p><b>Dropped feature:</b> {summary['dropped_feature']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Why the threshold matters"):
        st.write(
            "A lower threshold predicts more 'Yes' cases, which increases recall. A higher threshold is stricter and usually reduces false positives."
        )
