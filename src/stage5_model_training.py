"""
Stage 5: Model Training Pipeline
Author: B.Tech ECE Student ML Project
Description:
    Train two foundational classification models for customer churn prediction:
    1. Logistic Regression (Linear parametric baseline)
    2. Random Forest Classifier (Non-linear non-parametric ensemble)
    Both models use random_state=42 for exact reproducibility.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Paths
DATA_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data"
MODELS_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\models"

def train_models():
    print("=" * 65)
    print("STAGE 5: MODEL TRAINING (LOGISTIC REGRESSION & RANDOM FOREST)")
    print("=" * 65)

    # 1. Load preprocessed train and test data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    print(f"\n[Step 1] Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"         Loaded testing data:  {X_test.shape[0]} samples")

    # -------------------------------------------------------------
    # Model 1: Logistic Regression
    # -------------------------------------------------------------
    print("\n[Step 2] Training Baseline Model 1: Logistic Regression...")
    # max_iter=1000 ensures solver converges smoothly without iteration warnings
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    print("  -> Logistic Regression trained successfully!")

    # Check quick train/test accuracy to verify learning
    train_acc_lr = lr_model.score(X_train, y_train)
    test_acc_lr = lr_model.score(X_test, y_test)
    print(f"  -> Train Accuracy: {train_acc_lr * 100:.2f}% | Test Accuracy: {test_acc_lr * 100:.2f}%")

    # -------------------------------------------------------------
    # Model 2: Random Forest Classifier
    # -------------------------------------------------------------
    print("\n[Step 3] Training Baseline Model 2: Random Forest Classifier...")
    # n_estimators=100 builds an ensemble of 100 independent decision trees
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("  -> Random Forest trained successfully!")

    train_acc_rf = rf_model.score(X_train, y_train)
    test_acc_rf = rf_model.score(X_test, y_test)
    print(f"  -> Train Accuracy: {train_acc_rf * 100:.2f}% | Test Accuracy: {test_acc_rf * 100:.2f}%")
    print(f"  -> Notice: RF train accuracy is {train_acc_rf*100:.2f}%, showing unconstrained trees tend to overfit the training set.")

    # -------------------------------------------------------------
    # Save Trained Baseline Models
    # -------------------------------------------------------------
    print("\n[Step 4] Saving trained models to models/ directory...")
    lr_save_path = os.path.join(MODELS_DIR, "baseline_logistic_regression.joblib")
    rf_save_path = os.path.join(MODELS_DIR, "baseline_random_forest.joblib")

    joblib.dump(lr_model, lr_save_path)
    joblib.dump(rf_model, rf_save_path)

    print(f"  -> Saved Logistic Regression model: {lr_save_path}")
    print(f"  -> Saved Random Forest model:       {rf_save_path}")

    # Generate and save predictions on the test set for Stage 6 evaluation
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'lr_pred': lr_model.predict(X_test),
        'lr_proba': lr_model.predict_proba(X_test)[:, 1],
        'rf_pred': rf_model.predict(X_test),
        'rf_proba': rf_model.predict_proba(X_test)[:, 1]
    })
    preds_path = os.path.join(DATA_DIR, "baseline_predictions.csv")
    predictions_df.to_csv(preds_path, index=False)
    print(f"  -> Saved baseline predictions to: {preds_path}")

    print("=" * 65)
    print("STAGE 5 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return lr_model, rf_model

if __name__ == "__main__":
    train_models()
