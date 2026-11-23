from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered",
)

MODEL_PATH = Path("models/best_model.joblib")
REFERENCE_PATHS = [Path("data/processed/train.csv"), Path("data/tourism.csv")]
TARGET_COL = "ProdTaken"


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Ensure Criteria 3 committed models/best_model.joblib."
        )
    return joblib.load(model_path)


def load_feature_schema() -> pd.DataFrame | None:
    for path in REFERENCE_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            if TARGET_COL in df.columns:
                df = df.drop(columns=[TARGET_COL])
            return df
    return None


def collect_inputs_dataframe(ref_df: pd.DataFrame | None) -> pd.DataFrame:
    inputs: dict[str, object] = {}

    if ref_df is not None and not ref_df.empty:
        for col in ref_df.columns:
            series = ref_df[col].dropna()
            if pd.api.types.is_integer_dtype(ref_df[col]):
                default_val = int(series.median()) if not series.empty else 0
                min_val = int(series.min()) if not series.empty else 0
                max_val = int(series.max()) if not series.empty else default_val + 100
                if min_val > default_val:
                    min_val = default_val
                if max_val < default_val:
                    max_val = default_val
                inputs[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=1,
                )
            elif pd.api.types.is_float_dtype(ref_df[col]):
                default_val = float(series.median()) if not series.empty else 0.0
                min_val = float(series.min()) if not series.empty else 0.0
                max_val = float(series.max()) if not series.empty else default_val + 100.0
                if min_val > default_val:
                    min_val = default_val
                if max_val < default_val:
                    max_val = default_val
                inputs[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                )
            else:
                options = sorted(series.astype(str).unique().tolist())
                if options:
                    inputs[col] = st.selectbox(label=col, options=options, index=0)
                else:
                    inputs[col] = st.text_input(label=col, value="")
    else:
        col1, col2 = st.columns(2)
        with col1:
            inputs["Age"] = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
            inputs["MonthlyIncome"] = st.number_input("MonthlyIncome", min_value=0.0, value=25000.0, step=500.0)
            inputs["NumberOfTrips"] = st.number_input("NumberOfTrips", min_value=0, max_value=50, value=2, step=1)
        with col2:
            inputs["Gender"] = st.selectbox("Gender", ["Male", "Female"])
            inputs["MaritalStatus"] = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"])
            inputs["Occupation"] = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])

    return pd.DataFrame([inputs])


st.title("Wellness Tourism Package Uptake Predictor")
st.write("Enter customer details below to save them into a DataFrame and predict whether the wellness package (`ProdTaken`) will be taken.")

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

ref_df = load_feature_schema()
input_df = collect_inputs_dataframe(ref_df)

st.subheader("Input DataFrame")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Package Uptake", type="primary"):
    try:
        prediction = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("Likely to take the Wellness Tourism Package (`ProdTaken = 1`).")
        else:
            st.info("Unlikely to take the Wellness Tourism Package (`ProdTaken = 0`).")

        if proba is not None:
            st.metric("Predicted Uptake Probability", f"{proba:.2%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
