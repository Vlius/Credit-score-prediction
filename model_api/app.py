import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# Get absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing objects using absolute paths
model_path = os.path.join(script_dir, "xgboost_model.json")
imputer_path = os.path.join(script_dir, "imputer.joblib")
scaler_path = os.path.join(script_dir, "scaler.joblib")

# Check if files exist before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

if not os.path.exists(imputer_path):
    raise FileNotFoundError(f"Imputer file not found at: {imputer_path}")

if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

# Load model and transformers
model = xgb.Booster()
model.load_model(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

# Define features
feats = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CREDIT_TO_INCOME_RATIO",
    "CREDIT_TERM",
    "EXT_SOURCE_MEAN",
    "PREV_AMT_CREDIT_mean",
    "PREV_AMT_CREDIT_sum",
    "PREV_SK_ID_PREV_count",
    "PREV_NAME_CONTRACT_STATUS_ApprovalRate",
]


# Prediction function
def predict_credit_risk(input_dict):
    input_dict["CREDIT_TO_INCOME_RATIO"] = (
        input_dict["AMT_CREDIT"] / input_dict["AMT_INCOME_TOTAL"]
        if input_dict["AMT_INCOME_TOTAL"] != 0
        else np.nan
    )
    input_dict["CREDIT_TERM"] = (
        input_dict["AMT_CREDIT"] / input_dict["AMT_ANNUITY"]
        if input_dict["AMT_ANNUITY"] != 0
        else np.nan
    )
    input_dict["EXT_SOURCE_MEAN"] = np.mean(
        [input_dict["EXT_SOURCE_2"], input_dict["EXT_SOURCE_3"]]
    )
    input_data = pd.DataFrame([input_dict])[feats]
    input_data = pd.DataFrame(imputer.transform(input_data), columns=feats)
    input_data = pd.DataFrame(scaler.transform(input_data), columns=feats)
    dmatrix = xgb.DMatrix(input_data)
    return model.predict(dmatrix)[0] * 100  # Return as percentage


# Streamlit interface
st.title("Credit Risk Prediction")

st.sidebar.header("Input Features")
input_dict = {
    "AMT_CREDIT": st.sidebar.number_input("Loan Amount", min_value=0.0, value=100000.0),
    "AMT_INCOME_TOTAL": st.sidebar.number_input(
        "Annual Income", min_value=0.0, value=50000.0
    ),
    "AMT_ANNUITY": st.sidebar.number_input(
        "Monthly Payment", min_value=0.0, value=5000.0
    ),
    "EXT_SOURCE_2": st.sidebar.slider("Credit Score 1", 0.0, 1.0, 0.5),
    "EXT_SOURCE_3": st.sidebar.slider("Credit Score 2", 0.0, 1.0, 0.5),
    "PREV_AMT_CREDIT_mean": st.sidebar.number_input(
        "Avg Previous Loan Amount", min_value=0.0, value=50000.0
    ),
    "PREV_AMT_CREDIT_sum": st.sidebar.number_input(
        "Total Previous Loans", min_value=0.0, value=100000.0
    ),
    "PREV_SK_ID_PREV_count": st.sidebar.number_input(
        "Number of Previous Loans", min_value=0, value=5
    ),
    "PREV_NAME_CONTRACT_STATUS_ApprovalRate": st.sidebar.slider(
        "Previous Approval Rate", 0.0, 1.0, 0.8
    ),
}

if st.sidebar.button("Predict"):
    risk_score = predict_credit_risk(input_dict)
    st.write(f"**Credit Risk Score:** {risk_score:.2f}%")
    if risk_score < 30:
        st.success("Low Risk")
    elif risk_score < 60:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
