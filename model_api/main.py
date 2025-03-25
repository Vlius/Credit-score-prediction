import functions_framework
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer


model = xgb.Booster()
model.load_model("xgboost_model.json")
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")


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


def add_engineered_features(df):
    df["CREDIT_TO_INCOME_RATIO"] = np.where(
        df["AMT_INCOME_TOTAL"] > 0, df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"], np.nan
    )
    df["CREDIT_TERM"] = np.where(
        df["AMT_ANNUITY"] > 0, df["AMT_CREDIT"] / df["AMT_ANNUITY"], np.nan
    )
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    return df


@functions_framework.http
def predict(request):
    """HTTP Cloud Function to predict using an improved XGBoost model."""
    request_json = request.get_json(silent=True)
    if not request_json or "data" not in request_json:
        return json.dumps({"error": "Invalid input: 'data' field required"}), 400

    input_data = pd.DataFrame([request_json["data"]])

    for feat in feats:
        if feat not in input_data.columns:
            input_data[feat] = np.nan

    input_data = add_engineered_features(input_data)
    input_data = input_data[feats]

    input_data = pd.DataFrame(imputer.transform(input_data), columns=feats)
    input_data = pd.DataFrame(scaler.transform(input_data), columns=feats)

    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)[0]

    return (
        json.dumps({"prediction": float(prediction), "probability": float(prediction)}),
        200,
        {"Content-Type": "application/json"},
    )


def train_improved_model(X_train, y_train):
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed), columns=X_train.columns
    )

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "eval_metric": "auc",
    }

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    model.save_model("xgboost_model.json")
    joblib.dump(imputer, "imputer.joblib")
    joblib.dump(scaler, "scaler.joblib")

    return model, imputer, scaler
