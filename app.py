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
# --- add this helper once (near the top, after imports) ---
def expected_columns_from_model(m):
    try:
        # scikit-learn 1.2+: ColumnTransformer stores the original feature names
        return list(getattr(m, "feature_names_in_", [])) or \
               list(getattr(getattr(m, "named_steps", {}).get("preprocessor", None),
                            "feature_names_in_", []))
    except Exception:
        return []

# Show what the model expects (debug)
exp_cols = expected_columns_from_model(model) if model is not None else []
if exp_cols:
    with st.expander("Expected raw feature columns (from pipeline)"):
        st.code(exp_cols, language="python")

st.divider()
st.subheader("Enter patient features")

# Numeric features you already had
col1, col2 = st.columns(2)
time_in_hospital   = col1.number_input("time_in_hospital", 1, 30, 4)
num_lab_procedures = col1.number_input("num_lab_procedures", 0, 200, 40)
num_procedures     = col1.number_input("num_procedures", 0, 20, 1)
num_medications    = col1.number_input("num_medications", 0, 100, 12)
number_outpatient  = col2.number_input("number_outpatient", 0, 50, 0)
number_emergency   = col2.number_input("number_emergency", 0, 50, 0)
number_inpatient   = col2.number_input("number_inpatient", 0, 50, 0)
number_diagnoses   = col2.number_input("number_diagnoses", 1, 16, 9)

st.markdown("### Categorical Features")

# These options match the common UCI Diabetes dataset encodings
age_opt = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
           "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
race_opt = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "?"]
gender_opt = ["Male", "Female", "Unknown/Invalid"]
diabetesMed_opt = ["Yes", "No"]
change_opt = ["Ch", "No"]

# admission_type_id / source / discharge are integer-coded categories in UCI data
# Use integers so they pass exactly as expected by the pipeline
admission_type_id = st.number_input("admission_type_id (int)", 1, 9, 1)
admission_source_id = st.number_input("admission_source_id (int)", 1, 25, 7)
discharge_disposition_id = st.number_input("discharge_disposition_id (int)", 1, 30, 1)

col3, col4, col5, col6 = st.columns(4)
age = col3.selectbox("age", age_opt, index=5)               # default [50-60)
race = col4.selectbox("race", race_opt, index=0)
gender = col5.selectbox("gender", gender_opt, index=0)
diabetesMed = col6.selectbox("diabetesMed", diabetesMed_opt, index=0)

change = st.selectbox("change (med change during stay)", change_opt, index=0)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded — see messages above.")
    else:
        import pandas as pd
        # Build the single-row DataFrame with ALL raw columns expected by the pipeline
        row = {
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "admission_type_id": int(admission_type_id),
            "discharge_disposition_id": int(discharge_disposition_id),
            "admission_source_id": int(admission_source_id),
            "diabetesMed": diabetesMed,   # "Yes"/"No"
            "change": change,             # "Ch"/"No"
            "race": race,                 # e.g., "Caucasian"
            "age": age,                   # e.g., "[50-60)"
            "gender": gender,             # "Male"/"Female"/"Unknown/Invalid"
        }
        X = pd.DataFrame([row])

        try:
            proba = float(model.predict_proba(X)[0, 1])
            pred = int(proba >= threshold)
            st.metric("Readmission probability", f"{proba:.3f}")
            st.metric("Prediction (0=no, 1=yes)", pred)
            st.caption(f"Decision threshold = {threshold:.3f}")
        except Exception as e:
            st.error("Prediction failed. See details below:")
            st.exception(e)



