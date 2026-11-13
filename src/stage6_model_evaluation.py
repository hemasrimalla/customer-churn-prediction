"""
Stage 6: Model Evaluation Pipeline
Author: B.Tech ECE Student ML Project
Description:
    Evaluate baseline Logistic Regression and Random Forest models across:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - Confusion Matrix (TN, FP, FN, TP)
    - ROC Curve plots saved to figures/
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Paths
DATA_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data"
FIGURES_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\figures"
PREDS_PATH = os.path.join(DATA_DIR, "baseline_predictions.csv")

def evaluate_models():
    print("=" * 65)
    print("STAGE 6: MODEL EVALUATION & PERFORMANCE COMPARISON")
    print("=" * 65)

    df_preds = pd.read_csv(PREDS_PATH)
    y_true = df_preds['y_true'].values

    models = {
        "Logistic Regression": {
            "pred": df_preds['lr_pred'].values,
            "proba": df_preds['lr_proba'].values,
            "color": "#2b5c8f"
        },
        "Random Forest": {
            "pred": df_preds['rf_pred'].values,
            "proba": df_preds['rf_proba'].values,
            "color": "#e74c3c"
        }
    }

    results = []

    for name, data in models.items():
        y_pred = data["pred"]
        y_proba = data["proba"]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "ROC-AUC": f"{auc:.4f}",
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

        print(f"\n--- {name.upper()} EVALUATION ---")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        print(f"ROC-AUC:   {auc:.4f}")
        print("\nConfusion Matrix Breakdown:")
        print(f"  True Negatives  (TN - Correctly identified stayed):   {tn}")
        print(f"  False Positives (FP - False Alarm, stayed flagged):   {fp}")
        print(f"  False Negatives (FN - Missed Churners, left unseen):  {fn}")
        print(f"  True Positives  (TP - Successfully caught churners):  {tp}")
        print("\nFull Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Stayed (0)", "Churned (1)"]))

    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("SUMMARY COMPARISON TABLE:")
    print(summary_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))
    print("=" * 65)

    # -------------------------------------------------------------
    # PLOT 1: Confusion Matrices Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, data) in zip(axes, models.items()):
        cm = confusion_matrix(y_true, data["pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Stayed (0)', 'Churned (1)'],
                    yticklabels=['Stayed (0)', 'Churned (1)'],
                    annot_kws={"fontsize": 13, "fontweight": "bold"})
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Actual Label", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "05_confusion_matrices_baseline.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[Generated Figure]: {cm_path}")

    # -------------------------------------------------------------
    # PLOT 2: Combined ROC Curve
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    for name, data in models.items():
        fpr, tpr, _ = roc_curve(y_true, data["proba"])
        auc = roc_auc_score(y_true, data["proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=data["color"], linewidth=2.2)

    # Plot diagonal 50% random chance line
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.500)", alpha=0.7)
    plt.title("ROC Curves Comparison: Logistic Regression vs Random Forest", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "06_roc_curves_baseline.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Generated Figure]: {roc_path}")

    print("\nStage 6 evaluation finished successfully.")
    return summary_df

if __name__ == "__main__":
    evaluate_models()
