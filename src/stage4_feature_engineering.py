"""
Stage 4: Feature Engineering & Preprocessing Pipeline
Author: B.Tech ECE Student ML Project
Description: 
    1. Separate features (X) and target (y).
    2. One-Hot Encode categorical variables with drop_first=True to avoid dummy variable trap.
    3. Stratified Train/Test split (80/20) with random_state=42.
    4. Feature scaling using StandardScaler (fit on train, transform on test to prevent data leakage).
    5. Save processed train/test datasets and the fitted scaler for inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data"
MODELS_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\models"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_telco_churn.csv")

def feature_engineering():
    print("=" * 65)
    print("STAGE 4: FEATURE ENGINEERING & PREPROCESSING PIPELINE")
    print("=" * 65)

    # 1. Load cleaned data
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"\n[Step 1] Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate feature matrix (X) and target vector (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    print(f"  -> Features count: {X.shape[1]}, Target: 'Churn'")

    # 2. One-Hot Encoding for categorical features
    # Why drop_first=True?
    # Prevents multicollinearity (the "Dummy Variable Trap").
    # If a feature has k categories, k-1 binary indicators fully capture the information.
    # The k-th category is implicitly represented when all k-1 dummy variables are 0.
    print("\n[Step 2] One-Hot Encoding categorical features (drop_first=True)...")
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
    print(f"  -> Feature count expanded from {X.shape[1]} to {X_encoded.shape[1]} columns")

    # 3. Stratified Train / Test Split (80% Train, 20% Test)
    # Why stratify=y?
    # Target class is imbalanced (~73.4% No, ~26.6% Yes).
    # Stratification forces both train and test splits to have the exact same class ratio.
    # random_state=42 ensures exact reproducibility.
    print("\n[Step 3] Splitting dataset (80% Train, 20% Test, Stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, 
        y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )

    print(f"  -> Training Set:   {X_train.shape[0]} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  -> Testing Set:    {X_test.shape[0]} samples ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  -> Train Churn %:  {y_train.mean()*100:.2f}% (Churn=1: {y_train.sum()}, Churn=0: {(y_train==0).sum()})")
    print(f"  -> Test Churn %:   {y_test.mean()*100:.2f}% (Churn=1: {y_test.sum()}, Churn=0: {(y_test==0).sum()})")

    # 4. Feature Scaling (StandardScaler)
    # Continuous numeric columns have wildly different scales:
    # tenure: 1 to 72 months
    # MonthlyCharges: $18.25 to $118.75
    # TotalCharges: $18.80 to $8684.80
    #
    # CRITICAL INTERVIEW PRINCIPLE: PREVENTING DATA LEAKAGE!
    # Fit the scaler ONLY on training data (scaler.fit_transform(X_train)),
    # and use those learned parameters (mean, std) to transform test data (scaler.transform(X_test)).
    # Never fit on the whole dataset or on the test set!
    print("\n[Step 4] Scaling continuous numerical columns with StandardScaler...")
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    print(f"  -> Continuous columns to scale: {numeric_cols}")

    scaler = StandardScaler()
    
    # Create copies to preserve original column types cleanly
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("  -> Sample training stats after scaling (mean approx 0, std approx 1):")
    for col in numeric_cols:
        print(f"     * {col:<15}: mean={X_train_scaled[col].mean():.4f}, std={X_train_scaled[col].std():.4f}")

    # 5. Save artifacts to disk
    print("\n[Step 5] Saving processed train/test datasets and fitted scaler...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  -> Saved StandardScaler to: {scaler_path}")

    # Save feature names list to ensure consistent ordering during inference
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    joblib.dump(list(X_train_scaled.columns), feature_names_path)
    print(f"  -> Saved feature column names ({len(X_train_scaled.columns)}) to: {feature_names_path}")

    # Save train/test datasets
    X_train_scaled.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)
    print("  -> Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv in data/")

    print("=" * 65)
    print("STAGE 4 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    feature_engineering()
