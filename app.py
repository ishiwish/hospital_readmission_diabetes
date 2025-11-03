import json
from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(page_title="Hospital Readmission (Diabetes)", layout="centered")
st.title("🏥 Diabetes Readmission Predictor")

repo_dir = Path(__file__).parent
MODEL_PATH = repo_dir / "readmission_rf_pipeline.joblib"
THRESHOLD_PATH = repo_dir / "threshold.json"

st.caption(f"Working dir: `{repo_dir}`")
st.caption("Looking for model files next to app.py")

model = None
threshold = 0.5
try:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
    else:
        model = joblib.load(MODEL_PATH)
        st.success(f"Loaded model: {MODEL_PATH.name}")

    if THRESHOLD_PATH.exists():
        threshold = float(json.load(open(THRESHOLD_PATH))["threshold"])
        st.success(f"Loaded threshold: {threshold:.3f}")
    else:
        st.warning("No threshold.json found — using 0.50")
except Exception as e:
    st.error("Error while loading model/threshold:")
    st.exception(e)

st.divider()
st.subheader("Enter patient features")

col1, col2 = st.columns(2)
time_in_hospital   = col1.number_input("time_in_hospital", 1, 30, 4)
num_lab_procedures = col1.number_input("num_lab_procedures", 0, 200, 40)
num_procedures     = col1.number_input("num_procedures", 0, 20, 1)
num_medications    = col1.number_input("num_medications", 0, 100, 12)
number_outpatient  = col2.number_input("number_outpatient", 0, 50, 0)
number_emergency   = col2.number_input("number_emergency", 0, 50, 0)
number_inpatient   = col2.number_input("number_inpatient", 0, 50, 0)
number_diagnoses   = col2.number_input("number_diagnoses", 1, 16, 9)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded — see messages above.")
    else:
        import pandas as pd
        X = pd.DataFrame([{
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
        }])
        try:
            proba = float(model.predict_proba(X)[0, 1])
            pred = int(proba >= threshold)
            st.metric("Readmission probability", f"{proba:.3f}")
            st.metric("Prediction (0=no, 1=yes)", pred)
            st.caption(f"Decision threshold = {threshold:.3f}")
        except Exception as e:
            st.error("Prediction failed (likely feature mismatch).")
            st.exception(e)
