"""
Stage 3: Data Cleaning Pipeline
Author: B.Tech ECE Student ML Project
Description: Clean the Telco Customer Churn dataset:
             1. Convert 'TotalCharges' from object/string to numeric float.
             2. Handle missing values created from whitespace entries.
             3. Drop non-predictive identifier column 'customerID'.
             4. Map binary target 'Churn' from Yes/No to 1/0.
             5. Save the cleaned dataset for downstream feature engineering.
"""

import os
import pandas as pd
import numpy as np

# Define file paths
RAW_DATA_PATH = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data\Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data\cleaned_telco_churn.csv"

def clean_data():
    print("=" * 65)
    print("STAGE 3: DATA CLEANING PIPELINE")
    print("=" * 65)

    # 1. Load the raw dataset
    df = pd.read_csv(RAW_DATA_PATH)
    initial_shape = df.shape
    print(f"\n[Step 1] Loaded raw data: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # 2. Fix TotalCharges column type
    # Why: 11 rows contained ' ' (space), causing Pandas to treat the column as object/string.
    # 'errors=coerce' replaces invalid string values with NaN (Not a Number) floating-point values.
    print("\n[Step 2] Converting 'TotalCharges' to float64 numeric type...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    nan_count = df['TotalCharges'].isnull().sum()
    print(f"  -> Converted dtype: {df['TotalCharges'].dtype}")
    print(f"  -> Successfully converted valid values. Missing values (NaN) generated: {nan_count}")

    # Inspect the rows that generated NaN values
    null_customers = df[df['TotalCharges'].isnull()]
    print(f"  -> Tenure for all missing TotalCharges rows: {null_customers['tenure'].unique().tolist()}")
    print(f"  -> Churn value for all missing TotalCharges rows: {null_customers['Churn'].unique().tolist()}")

    # 3. Handle missing values
    # Decision: Drop the 11 rows.
    # Justification:
    # 1. 11 rows represent only (11 / 7043) = 0.156% of total data (negligible information loss).
    # 2. All 11 have tenure = 0 months (customers who just joined and have no billing history).
    # 3. None of them churned ('No' for all 11), so dropping them does not alter churn patterns.
    print("\n[Step 3] Handling missing values...")
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    print(f"  -> Rows remaining after dropping 11 rows: {len(df)}")

    # 4. Drop non-predictive identifier column 'customerID'
    # Justification:
    # customerID is an arbitrary database primary key (e.g., '7590-VHVEG').
    # Every customer has a unique ID (100% unique cardinality).
    # It contains zero generalizable signal about whether a customer will churn.
    # Leaving it in would either explode memory (one-hot encoding) or cause severe overfitting.
    print("\n[Step 4] Dropping 'customerID' column...")
    df = df.drop(columns=['customerID'])
    print(f"  -> 'customerID' dropped. Remaining features: {df.shape[1] - 1} + 1 target.")

    # 5. Map target variable 'Churn' to binary integers (0 and 1)
    # Standard ML convention: 0 = Negative class (No churn), 1 = Positive class (Churn)
    print("\n[Step 5] Encoding target variable 'Churn' to binary (0 and 1)...")
    print(f"  -> Before encoding: {df['Churn'].value_counts().to_dict()}")
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    print(f"  -> After encoding:  {df['Churn'].value_counts().to_dict()}")
    print(f"  -> Target dtype:    {df['Churn'].dtype}")

    # 6. Verify and save the cleaned dataset
    print("\n[Step 6] Final verification and saving dataset...")
    print(f"  -> Final cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  -> Total missing values remaining: {df.isnull().sum().sum()}")

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"  -> Saved cleaned file to: {CLEANED_DATA_PATH}")
    print("=" * 65)
    print("DATA CLEANING COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return df

if __name__ == "__main__":
    clean_data()
