"""
Stage 7: Model Improvement Pipeline
Author: B.Tech ECE Student ML Project
Description:
    Improve churn prediction models by:
    1. Handling class imbalance using 'class_weight="balanced"'.
    2. Systematic hyperparameter tuning using 5-fold Stratified GridSearchCV.
    3. Optimizing for ROC-AUC / Recall to catch more at-risk churners.
    4. Comparing Before vs. After results across Accuracy, Precision, Recall, F1, and ROC-AUC.
"""

import os
import joblib
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Paths
DATA_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data"
MODELS_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\models"
FIGURES_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\figures"

def improve_models():
    print("=" * 65)
    print("STAGE 7: MODEL IMPROVEMENT (CLASS WEIGHTING & GRIDSEARCHCV)")
    print("=" * 65)

    # 1. Load preprocessed datasets
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # Load baseline predictions to compute exact Before metrics
    df_base = pd.read_csv(os.path.join(DATA_DIR, "baseline_predictions.csv"))

    # 5-fold Stratified K-Fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------------------
    # 1. TUNING LOGISTIC REGRESSION
    # -------------------------------------------------------------
    print("\n[Step 1] Tuning Logistic Regression with GridSearchCV...")
    lr_param_grid = {
        'C': [0.01, 0.05, 0.1, 1.0, 5.0, 10.0],
        'class_weight': ['balanced', None],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }

    lr_grid = GridSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_grid=lr_param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)

    best_lr = lr_grid.best_estimator_
    print(f"  -> Best LR Parameters: {lr_grid.best_params_}")
    print(f"  -> Best LR CV ROC-AUC: {lr_grid.best_score_:.4f}")

    # -------------------------------------------------------------
    # 2. TUNING RANDOM FOREST CLASSIFIER
    # -------------------------------------------------------------
    print("\n[Step 2] Tuning Random Forest with GridSearchCV...")
    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [5, 8, 12],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced']
    }

    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_
    print(f"  -> Best RF Parameters: {rf_grid.best_params_}")
    print(f"  -> Best RF CV ROC-AUC: {rf_grid.best_score_:.4f}")

    # -------------------------------------------------------------
    # 3. EVALUATE TUNED MODELS ON TEST SET
    # -------------------------------------------------------------
    print("\n[Step 3] Evaluating Tuned Models on Unseen Test Set...")
    
    # Save Tuned Models
    joblib.dump(best_lr, os.path.join(MODELS_DIR, "best_logistic_regression.joblib"))
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "best_random_forest.joblib"))

    # Predictions
    lr_tuned_pred = best_lr.predict(X_test)
    lr_tuned_proba = best_lr.predict_proba(X_test)[:, 1]

    rf_tuned_pred = best_rf.predict(X_test)
    rf_tuned_proba = best_rf.predict_proba(X_test)[:, 1]

    # Save tuned predictions
    df_tuned = pd.DataFrame({
        'y_true': y_test,
        'lr_tuned_pred': lr_tuned_pred,
        'lr_tuned_proba': lr_tuned_proba,
        'rf_tuned_pred': rf_tuned_pred,
        'rf_tuned_proba': rf_tuned_proba
    })
    df_tuned.to_csv(os.path.join(DATA_DIR, "tuned_predictions.csv"), index=False)

    # -------------------------------------------------------------
    # 4. BEFORE VS. AFTER COMPARISON TABLE
    # -------------------------------------------------------------
    def get_metrics(y_true, y_pred, y_proba):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "Accuracy": accuracy_score(y_true, y_pred) * 100,
            "Precision": precision_score(y_true, y_pred) * 100,
            "Recall": recall_score(y_true, y_pred) * 100,
            "F1-Score": f1_score(y_true, y_pred) * 100,
            "ROC-AUC": roc_auc_score(y_true, y_proba),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }

    lr_before = get_metrics(y_test, df_base['lr_pred'], df_base['lr_proba'])
    lr_after  = get_metrics(y_test, lr_tuned_pred, lr_tuned_proba)

    rf_before = get_metrics(y_test, df_base['rf_pred'], df_base['rf_proba'])
    rf_after  = get_metrics(y_test, rf_tuned_pred, rf_tuned_proba)

    comparison_data = [
        {"Model Stage": "Logistic Regression (Before)", **lr_before},
        {"Model Stage": "Logistic Regression (After)",  **lr_after},
        {"Model Stage": "Random Forest (Before)",        **rf_before},
        {"Model Stage": "Random Forest (After)",         **rf_after},
    ]

    comp_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 80)
    print("BEFORE vs. AFTER IMPROVEMENT METRICS TABLE:")
    print(comp_df[["Model Stage", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "FN"]].to_string(index=False))
    print("=" * 80)

    # -------------------------------------------------------------
    # 5. BEFORE VS. AFTER VISUALIZATION
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Plot for Logistic Regression
    lr_plot_data = pd.DataFrame({
        "Metric": metrics_to_plot,
        "Before": [lr_before[m] for m in metrics_to_plot],
        "After (Tuned + Balanced)": [lr_after[m] for m in metrics_to_plot]
    }).melt(id_vars="Metric", var_name="Version", value_name="Percentage (%)")

    sns.barplot(data=lr_plot_data, x="Metric", y="Percentage (%)", hue="Version", 
                palette=["#7f8c8d", "#2980b9"], ax=axes[0])
    axes[0].set_title("Logistic Regression: Before vs After", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylim(0, 100)
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1.5),
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot for Random Forest
    rf_plot_data = pd.DataFrame({
        "Metric": metrics_to_plot,
        "Before": [rf_before[m] for m in metrics_to_plot],
        "After (Tuned + Balanced)": [rf_after[m] for m in metrics_to_plot]
    }).melt(id_vars="Metric", var_name="Version", value_name="Percentage (%)")

    sns.barplot(data=rf_plot_data, x="Metric", y="Percentage (%)", hue="Version", 
                palette=["#7f8c8d", "#27ae60"], ax=axes[1])
    axes[1].set_title("Random Forest: Before vs After", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_ylim(0, 100)
    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1.5),
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    comp_fig_path = os.path.join(FIGURES_DIR, "07_before_after_improvement.png")
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"\n[Generated Figure]: {comp_fig_path}")

    # Plot Confusion Matrices of Tuned Models
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tuned_models_dict = {
        "Tuned Logistic Regression": lr_tuned_pred,
        "Tuned Random Forest": rf_tuned_pred
    }
    for ax, (name, pred) in zip(axes, tuned_models_dict.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax,
                    xticklabels=['Stayed (0)', 'Churned (1)'],
                    yticklabels=['Stayed (0)', 'Churned (1)'],
                    annot_kws={"fontsize": 13, "fontweight": "bold"})
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Actual Label", fontsize=11)

    plt.tight_layout()
    tuned_cm_path = os.path.join(FIGURES_DIR, "08_confusion_matrices_tuned.png")
    plt.savefig(tuned_cm_path, dpi=300)
    plt.close()
    print(f"[Generated Figure]: {tuned_cm_path}")

    print("\nStage 7 improvement completed successfully.")
    return comp_df

if __name__ == "__main__":
    improve_models()
