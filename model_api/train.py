import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


train_df = pd.read_csv(
    "C:/Users/Vilius/Desktop/Capstone_3/notebook/application_train.csv"
)
test_df = pd.read_csv(
    "C:/Users/Vilius/Desktop/Capstone_3/notebook/application_test.csv"
)
previous_application = pd.read_csv(
    "C:/Users/Vilius/Desktop/Capstone_3/notebook/previous_application.csv"
)


def aggregate_previous_applications(df):
    agg_funcs = {
        "AMT_CREDIT": ["mean", "sum"],
        "SK_ID_PREV": ["count"],
        "NAME_CONTRACT_STATUS": [lambda x: (x == "Approved").mean()],
    }
    agg_df = df.groupby("SK_ID_CURR").agg(agg_funcs)
    agg_df.columns = [
        "PREV_AMT_CREDIT_mean"
        if "AMT_CREDIT" in col and "mean" in col
        else "PREV_AMT_CREDIT_sum"
        if "AMT_CREDIT" in col and "sum" in col
        else "PREV_SK_ID_PREV_count"
        if "SK_ID_PREV" in col
        else "PREV_NAME_CONTRACT_STATUS_ApprovalRate"
        for col in agg_df.columns
    ]
    return agg_df


prev_app_agg = aggregate_previous_applications(previous_application)
train_df = train_df.merge(prev_app_agg, on="SK_ID_CURR", how="left")
test_df = test_df.merge(prev_app_agg, on="SK_ID_CURR", how="left")


def create_features(df):
    df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(
        0, np.nan
    )
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
        axis=1
    )
    return df


train_df = create_features(train_df)
test_df = create_features(test_df)


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


X_train = train_df[feats]
y_train = train_df["TARGET"]
X_test = test_df[feats]


imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feats)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feats)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feats)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feats)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
model = xgb.train(params, dtrain, num_boost_round=1000)


model.save_model("C:/Users/Vilius/Desktop/Capstone_3/model_api/xgboost_model.json")
joblib.dump(imputer, "C:/Users/Vilius/Desktop/Capstone_3/model_api/imputer.joblib")
joblib.dump(scaler, "C:/Users/Vilius/Desktop/Capstone_3/model_api/scaler.joblib")
prev_app_agg.to_csv(
    "C:/Users/Vilius/Desktop/Capstone_3/model_api/prev_app_agg.csv", index=True
)


y_pred_proba_test = model.predict(dtest)
submission = pd.DataFrame(
    {"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_pred_proba_test}
)
submission.to_csv(
    "C:/Users/Vilius/Desktop/Capstone_3/model_api/submission.csv", index=False
)

print("Model training and file generation completed successfully!")
