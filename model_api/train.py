import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save model and preprocessing objects
model.save_model(os.path.join(script_dir, "xgboost_model.json"))
joblib.dump(imputer, os.path.join(script_dir, "imputer.joblib"))
joblib.dump(scaler, os.path.join(script_dir, "scaler.joblib"))
prev_app_agg.to_csv(os.path.join(script_dir, "prev_app_agg.csv"), index=True)

# Make predictions
y_pred_proba_test = model.predict(dtest)
submission = pd.DataFrame(
    {"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_pred_proba_test}
)
submission.to_csv(os.path.join(script_dir, "submission.csv"), index=False)

print("Model training and file generation completed successfully!")
