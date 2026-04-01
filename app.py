
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Wine Quality Assessment",
    page_icon="🍷",
    layout="wide"
)

@st.cache_resource
def load_models():
    rf = joblib.load(MODELS_DIR / "random_forest_wine_model.pkl")
    ada = joblib.load(MODELS_DIR / "adaboost_wine_model.pkl")
    with open(ASSETS_DIR / "model_metrics.json", "r") as f:
        metrics = json.load(f)
    return rf, ada, metrics

rf_model, ada_model, metrics = load_models()
FEATURE_COLUMNS = metrics["feature_columns"]

def validate_dataframe(df_input: pd.DataFrame):
    incoming = list(df_input.columns)
    missing = [c for c in FEATURE_COLUMNS if c not in incoming]
    extra = [c for c in incoming if c not in FEATURE_COLUMNS]
    if missing or extra:
        return False, missing, extra, None
    ordered = df_input[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
    bad_rows = ordered[ordered.isna().any(axis=1)]
    if not bad_rows.empty:
        return False, [], [], bad_rows.index.tolist()
    return True, [], [], ordered

def predict_one(input_df: pd.DataFrame):
    rf_pred = int(rf_model.predict(input_df)[0])
    rf_prob = float(rf_model.predict_proba(input_df)[0][1])
    ada_pred = int(ada_model.predict(input_df)[0])
    ada_prob = float(ada_model.predict_proba(input_df)[0][1])
    return rf_pred, rf_prob, ada_pred, ada_prob

def label_from_pred(pred: int):
    return "Good" if pred == 1 else "Poor"

def probability_delta(rf_prob, ada_prob):
    return abs(rf_prob - ada_prob)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #371b58 0%, #6a2c70 50%, #b83b5e 100%);
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    }
    .hero h1 {margin: 0 0 0.35rem 0; font-size: 2.2rem;}
    .hero p {margin: 0.15rem 0; opacity: 0.95;}
    .metric-card {
        background: #111827;
        padding: 1rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
        text-align: center;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="hero">
        <h1>Wine Quality Assesment</h1>
        <p><strong>Created by Powell Ndlovu</strong></p>
        <p>Portfolio-grade Streamlit app for binary wine quality prediction, model comparison, explainability, and batch scoring.</p>
        <p>Run time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Model Performance Snapshot")
m1, m2, m3, m4, m5 = st.columns(5)
rf_metrics = metrics["random_forest"]
ada_metrics = metrics["adaboost"]

def better(metric):
    if rf_metrics[metric] > ada_metrics[metric]:
        return "RF"
    elif rf_metrics[metric] < ada_metrics[metric]:
        return "Ada"
    return "Tie"

m1.metric("RF ROC-AUC", f'{rf_metrics["roc_auc"]:.3f}', delta=f'Best: {better("roc_auc")}')
m2.metric("Ada ROC-AUC", f'{ada_metrics["roc_auc"]:.3f}')
m3.metric("RF F1", f'{rf_metrics["f1"]:.3f}', delta=f'Best: {better("f1")}')
m4.metric("Ada F1", f'{ada_metrics["f1"]:.3f}')
m5.metric("Accuracy Gap", f'{abs(rf_metrics["accuracy"] - ada_metrics["accuracy"]):.3f}')

tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch CSV", "Model Insights", "About This App"])

with tab1:
    st.markdown("### Enter Wine Chemistry Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        fixed_acidity = st.number_input("fixed_acidity", value=7.0)
        volatile_acidity = st.number_input("volatile_acidity", value=0.39)
        citric_acidity = st.number_input("citric_acidity", value=0.26)
        residual_sugar = st.number_input("residual_sugar", value=1.7)
    with c2:
        chlorides = st.number_input("chlorides", value=0.040)
        free_sulfur_dioxide = st.number_input("free_sulfur_dioxide", value=39.0)
        total_sulfur_dioxide = st.number_input("total_sulfur_dioxide", value=187.0)
        density = st.number_input("density", value=0.990)
    with c3:
        pH = st.number_input("pH", value=3.41)
        sulphates = st.number_input("sulphates", value=0.50)
        alcohol = st.number_input("alcohol", value=10.8)

    input_df = pd.DataFrame([{
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acidity": citric_acidity,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    st.dataframe(input_df, use_container_width=True)

    if st.button("Compare Random Forest vs AdaBoost", type="primary"):
        rf_pred, rf_prob, ada_pred, ada_prob = predict_one(input_df)

        result_df = pd.DataFrame({
            "Model": ["Random Forest", "AdaBoost"],
            "Prediction": [label_from_pred(rf_pred), label_from_pred(ada_pred)],
            "Probability_Good": [round(rf_prob, 4), round(ada_prob, 4)],
            "Probability_Poor": [round(1-rf_prob, 4), round(1-ada_prob, 4)]
        })
        st.markdown("### Side-by-Side Model Output")
        st.dataframe(result_df, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("RF Verdict", label_from_pred(rf_pred), delta=f"Good prob {rf_prob:.3f}")
        r2.metric("Ada Verdict", label_from_pred(ada_pred), delta=f"Good prob {ada_prob:.3f}")
        r3.metric("Model Disagreement", f"{probability_delta(rf_prob, ada_prob):.3f}")

        st.markdown("### SHAP Explanation Panel (Random Forest)")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_df)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        if isinstance(shap_values, list):
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], input_df.iloc[0], show=False)
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], arr[0, :, 1], input_df.iloc[0], show=False)
            else:
                shap.plots._waterfall.waterfall_legacy(explainer.expected_value, arr[0], input_df.iloc[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

with tab2:
    st.markdown("### Batch Prediction with Validation")
    st.caption("Upload a CSV that contains exactly the required feature columns. Wrong columns, missing fields, extra columns, or non-numeric values are rejected.")
    template_path = DATA_DIR / "wine_input_template.csv"
    st.download_button(
        "Download CSV Template",
        data=template_path.read_bytes(),
        file_name="wine_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload input CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            valid, missing, extra, payload = validate_dataframe(raw_df)
            if not valid:
                if missing or extra:
                    st.error("CSV schema validation failed.")
                    if missing:
                        st.write("Missing columns:", missing)
                    if extra:
                        st.write("Unexpected columns:", extra)
                else:
                    st.error(f"CSV contains non-numeric or blank values in rows: {payload[:20]}")
            else:
                batch_df = payload.copy()
                batch_df["RF_Prediction"] = [label_from_pred(x) for x in rf_model.predict(batch_df)]
                batch_df["RF_Probability_Good"] = rf_model.predict_proba(batch_df)[:, 1]
                batch_df["Ada_Prediction"] = [label_from_pred(x) for x in ada_model.predict(batch_df)]
                batch_df["Ada_Probability_Good"] = ada_model.predict_proba(batch_df)[:, 1]
                batch_df["Agreement"] = np.where(
                    batch_df["RF_Prediction"] == batch_df["Ada_Prediction"], "Agree", "Disagree"
                )
                st.success("CSV passed validation and predictions were generated.")
                st.dataframe(batch_df.head(20), use_container_width=True)
                st.download_button(
                    "Download Scored CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="wine_batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")

with tab3:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("### Random Forest Feature Importance")
        st.image(str(ASSETS_DIR / "rf_feature_importance.png"), use_container_width=True)
    with right:
        st.markdown("### Metrics Table")
        metrics_df = pd.DataFrame([
            {"Model": "Random Forest", **rf_metrics},
            {"Model": "AdaBoost", **ada_metrics}
        ])
        st.dataframe(metrics_df, use_container_width=True)
        st.markdown("### Required Input Columns")
        st.code(", ".join(FEATURE_COLUMNS))
        st.markdown('<p class="small-note">SHAP is shown for the Random Forest model because tree-based local explanations are more stable and readable here than for AdaBoost.</p>', unsafe_allow_html=True)

with tab4:
    st.markdown(
        """
        ### What this portfolio app demonstrates
        - Binary wine quality classification where quality >= 6 is labelled as Good.
        - Side-by-side comparison of Random Forest and AdaBoost.
        - Metric cards for fast model screening.
        - Global feature importance for Random Forest.
        - Local SHAP explanation for a single prediction.
        - CSV validation checks to prevent broken uploads.

        ### Why this matters
        This app is designed as a portfolio piece to show model building, evaluation, validation, explainability, and product thinking in one deployable artifact.
        """
    )
