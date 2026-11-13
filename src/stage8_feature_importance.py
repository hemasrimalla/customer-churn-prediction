"""
Stage 8: Feature Importance Pipeline
Author: B.Tech ECE Student ML Project
Description:
    Extract and rank feature importances from the tuned Random Forest model
    and coefficients from Logistic Regression to identify the top 10 churn drivers.
    Save publication-ready visualization to figures/09_top10_feature_importance.png.
"""

import os
import joblib
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODELS_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\models"
FIGURES_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\figures"

def analyze_feature_importance():
    print("=" * 65)
    print("STAGE 8: FEATURE IMPORTANCE ANALYSIS (TOP 10 CHURN DRIVERS)")
    print("=" * 65)

    # 1. Load best model and feature column names
    rf_model = joblib.load(os.path.join(MODELS_DIR, "best_random_forest.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "best_logistic_regression.joblib"))

    # 2. Extract Random Forest Gini Feature Importances (MDI)
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Add cumulative importance and percentage
    fi_df['Importance_Pct'] = fi_df['Importance'] * 100
    fi_df['Cumulative_Pct'] = fi_df['Importance_Pct'].cumsum()

    print("\n[Step 1] Top 10 Features Driving Churn (Random Forest):")
    top10_rf = fi_df.head(10)
    for idx, row in top10_rf.iterrows():
        print(f"  {idx+1:>2}. {row['Feature']:<35} : {row['Importance_Pct']:>6.2f}% (Cumulative: {row['Cumulative_Pct']:>5.1f}%)")

    # 3. Extract Logistic Regression Coefficients for directionality (+ increases churn, - reduces churn)
    lr_coefs = lr_model.coef_[0]
    lr_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_coefs,
        'Odds_Ratio': np.exp(lr_coefs)
    }).sort_values(by='Coefficient', key=abs, ascending=False).reset_index(drop=True)

    print("\n[Step 2] Directionality Insights from Logistic Regression (Top 10 by magnitude):")
    for idx, row in lr_df.head(10).iterrows():
        direction = "INCREASES CHURN (+)" if row['Coefficient'] > 0 else "DECREASES CHURN (-)"
        print(f"  {idx+1:>2}. {row['Feature']:<35} : Coef = {row['Coefficient']:>6.3f} | {direction}")

    # -------------------------------------------------------------
    # 4. PLOT: Top 10 Feature Importance Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis_r", 10)
    
    # Plot horizontal bar chart
    bars = plt.barh(range(10), top10_rf['Importance_Pct'][::-1], color=colors, edgecolor='black', linewidth=0.8)
    plt.yticks(range(10), top10_rf['Feature'][::-1], fontsize=10, fontweight='bold')
    plt.xlabel("Relative Importance (%)", fontsize=11, fontweight='bold')
    plt.title("Top 10 Features Driving Customer Churn (Random Forest)", fontsize=13, fontweight='bold', pad=14)
    plt.xlim(0, max(top10_rf['Importance_Pct']) * 1.18)

    # Add percentage text labels on each bar
    for idx, bar in enumerate(bars):
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", 
                 va='center', ha='left', fontsize=9.5, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    fi_plot_path = os.path.join(FIGURES_DIR, "09_top10_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"\n[Generated Figure]: {fi_plot_path}")

    # Save top features table to CSV
    fi_df.to_csv(os.path.join(MODELS_DIR, "feature_importance_ranking.csv"), index=False)
    print(f"  -> Saved full feature ranking to: {os.path.join(MODELS_DIR, 'feature_importance_ranking.csv')}")

    print("=" * 65)
    print("STAGE 8 FEATURE IMPORTANCE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return top10_rf, lr_df

if __name__ == "__main__":
    analyze_feature_importance()
