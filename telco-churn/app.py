import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ART_DIR = Path("artifacts")
preprocessor = joblib.load(ART_DIR / "preprocessor.pkl")
model = joblib.load(ART_DIR / "model.pkl")
with open(ART_DIR / "feature_names.json") as f:
    FEATURE_ORDER = json.load(f)

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Telco Customer Churn Predictor")
st.write("Estimate churn risk for a single customer or upload a CSV for batch scoring.")

# Helper: build a single-row dataframe from sidebar inputs
def single_input_form():
    df_schema = {f: None for f in FEATURE_ORDER}
    # Minimal ergonomic UI: infer numeric vs categorical by preprocessor input features
    # We’ll use heuristics based on typical Telco columns:
    numeric_like = {"tenure", "MonthlyCharges", "TotalCharges"}
    form = st.sidebar.form("customer_form")
    values = {}
    for col in FEATURE_ORDER:
        if col in numeric_like:
            default = 1 if col == "tenure" else 70.0 if col == "MonthlyCharges" else 2000.0
            values[col] = form.number_input(col, value=float(default))
        else:
            # common categories; users can type new values too
            choices = {
                "gender": ["Male", "Female"],
                "SeniorCitizen": ["0", "1"],
                "Partner": ["Yes", "No"],
                "Dependents": ["Yes", "No"],
                "PhoneService": ["Yes", "No"],
                "MultipleLines": ["No", "Yes", "No phone service"],
                "InternetService": ["DSL", "Fiber optic", "No"],
                "OnlineSecurity": ["No", "Yes", "No internet service"],
                "OnlineBackup": ["No", "Yes", "No internet service"],
                "DeviceProtection": ["No", "Yes", "No internet service"],
                "TechSupport": ["No", "Yes", "No internet service"],
                "StreamingTV": ["No", "Yes", "No internet service"],
                "StreamingMovies": ["No", "Yes", "No internet service"],
                "Contract": ["Month-to-month", "One year", "Two year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ],
            }
            if col in choices:
                values[col] = form.selectbox(col, choices[col])
            else:
                # fallback text input
                values[col] = form.text_input(col, value="No")
    submit = form.form_submit_button("Predict churn for this customer")
    if submit:
        return pd.DataFrame([values])
    return None

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab_single:
    row = single_input_form()
    if row is not None:
        # Preprocess & predict
        proba = model.predict_proba(preprocessor.transform(row))[:, 1][0]
        churn = proba >= 0.5
        st.subheader("Result")
        st.metric("Churn Probability", f"{proba:.2%}", help="Probability the customer will churn.")
        st.write("**Prediction:**", "⚠️ Likely to churn" if churn else "✅ Likely to stay")

with tab_batch:
    st.write("Upload a CSV matching the **Telco schema** (same columns as training).")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        # Ensure required columns present; extra columns will be ignored by preprocessor if unseen (handled by OHE)
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            Xp = preprocessor.transform(df[FEATURE_ORDER])
            probs = model.predict_proba(Xp)[:, 1]
            out = df.copy()
            out["churn_probability"] = probs
            out["prediction"] = (out["churn_probability"] >= 0.5).astype(int)
            st.success("Predictions generated.")
            st.dataframe(out.head(20))
            st.download_button(
                "Download predictions as CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

st.caption("Model: RandomForest (class_weight=balanced). Threshold = 0.5 (tune for recall if needed).")
