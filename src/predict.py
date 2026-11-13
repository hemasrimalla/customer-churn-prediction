"""
Production Inference Script: Customer Churn Prediction
Author: B.Tech ECE Student ML Project
Description:
    Takes raw customer dictionary attributes, applies the exact preprocessing
    transformations (one-hot encoding alignment and standard scaling), and outputs:
    1. Churn probability (%)
    2. Churn classification (Will Churn / Will Stay)
    3. Actionable business recommendation
"""

import os
import joblib
import pandas as pd
import numpy as np

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 1. Load artifacts: trained model, fitted scaler, and exact training feature names
MODEL_PATH = os.path.join(MODELS_DIR, "final_churn_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")

# Ensure final model is saved from best random forest if not already copied
if not os.path.exists(MODEL_PATH):
    best_rf_path = os.path.join(MODELS_DIR, "best_random_forest.joblib")
    rf = joblib.load(best_rf_path)
    joblib.dump(rf, MODEL_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_features = joblib.load(FEATURES_PATH)

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn probability for a single customer.
    
    Args:
        customer_data (dict): Dictionary with raw customer attributes.
        
    Returns:
        dict: Churn probability, risk level, binary prediction, and recommendation.
    """
    # Step A: Convert input dictionary to DataFrame
    df_raw = pd.DataFrame([customer_data])

    # Ensure numeric columns are floats
    for col in NUMERIC_COLS:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Step B: One-Hot Encode categorical columns using drop_first=True
    df_encoded = pd.get_dummies(df_raw, drop_first=True, dtype=int)

    # Step C: Align columns with expected training features
    # If a category was not present in this single row, set its dummy column to 0
    df_aligned = pd.DataFrame(0, index=[0], columns=expected_features)
    for col in df_encoded.columns:
        if col in df_aligned.columns:
            df_aligned[col] = df_encoded[col].values

    # Step D: Apply the exact same StandardScaler fitted on the training set
    df_aligned[NUMERIC_COLS] = scaler.transform(df_aligned[NUMERIC_COLS])

    # Step E: Generate churn probability from the model
    # predict_proba returns [prob_class_0, prob_class_1]
    churn_proba = model.predict_proba(df_aligned)[0][1]
    churn_percent = churn_proba * 100

    # Step F: Determine risk tier and business recommendation
    if churn_percent >= 65.0:
        risk_level = "HIGH RISK"
        prediction = "Will Churn (1)"
        action = "Urgent: Customer likely to leave. Dispatch targeted 20% discount offer on a 1-year contract and assign a priority support representative."
    elif churn_percent >= 35.0:
        risk_level = "MEDIUM RISK"
        prediction = "Moderate Risk (Watchlist)"
        action = "Proactive: Customer showing signs of disengagement. Send an automated satisfaction survey and recommend value-add services like TechSupport."
    else:
        risk_level = "LOW RISK"
        prediction = "Will Stay (0)"
        action = "Healthy: Customer is well-retained. Eligible for standard loyalty appreciation perks."

    return {
        "churn_probability": round(churn_percent, 2),
        "risk_level": risk_level,
        "prediction": prediction,
        "recommended_action": action
    }

if __name__ == "__main__":
    print("=" * 65)
    print("STAGE 9: LIVE CUSTOMER CHURN PREDICTION TEST")
    print("=" * 65)

    # Test Customer 1: High Risk Profile
    # Brand new customer (2 months tenure), month-to-month contract,
    # expensive fiber optic, manual electronic check, no tech support.
    high_risk_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 75.50,
        'TotalCharges': 151.00
    }

    # Test Customer 2: Low Risk Profile
    # Long-term customer (65 months tenure), two-year contract,
    # online security, tech support, automatic bank transfer.
    low_risk_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 65,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 55.20,
        'TotalCharges': 3588.00
    }

    print("\n--- INFERENCE ON CUSTOMER 1 (High Risk Profile) ---")
    res1 = predict_churn(high_risk_customer)
    print(f"Churn Probability: {res1['churn_probability']}%")
    print(f"Risk Tier:         {res1['risk_level']}")
    print(f"Model Prediction:  {res1['prediction']}")
    print(f"Business Action:   {res1['recommended_action']}")

    print("\n--- INFERENCE ON CUSTOMER 2 (Low Risk Profile) ---")
    res2 = predict_churn(low_risk_customer)
    print(f"Churn Probability: {res2['churn_probability']}%")
    print(f"Risk Tier:         {res2['risk_level']}")
    print(f"Model Prediction:  {res2['prediction']}")
    print(f"Business Action:   {res2['recommended_action']}")
    print("=" * 65)
