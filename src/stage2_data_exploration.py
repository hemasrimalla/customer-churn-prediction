"""
Stage 2: Exploratory Data Analysis (EDA)
Author: B.Tech ECE Student ML Project
Description: Inspect dataset dimensions, column data types, missing values,
             target class distribution, and generate 4 key explanatory charts.
"""

import os
import tempfile

# Configure temporary writable config directory for matplotlib in sandbox
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for publication-ready plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

# Define paths
DATA_PATH = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\data\Telco-Customer-Churn.csv"
FIGURES_DIR = r"C:\Users\HEMA\.gemini\antigravity\scratch\customer-churn-prediction\figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("1. DATASET SHAPE")
print(f"Total Rows (Customers):   {df.shape[0]}")
print(f"Total Columns (Features): {df.shape[1]}")
print("=" * 65)

# 2. Inspect Column Names and Data Types
print("\n2. DATA TYPES SUMMARY:")
dtype_summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Sample Value': df.iloc[0].values
})
print(dtype_summary)

# 3. Check for standard missing values (NaN / Null)
print("\n" + "=" * 65)
print("3. MISSING VALUES CHECK (Standard nulls):")
null_counts = df.isnull().sum()
print(f"Standard null count across all columns: {null_counts.sum()}")

# Check for hidden missing values (spaces/blanks in string columns like TotalCharges)
space_counts = {col: (df[col].astype(str).str.strip() == "").sum() for col in df.columns}
hidden_found = {k: v for k, v in space_counts.items() if v > 0}
print(f"Hidden missing values (whitespace ' '): {hidden_found}")
if 'TotalCharges' in hidden_found:
    print(f"-> Crucial Finding: {hidden_found['TotalCharges']} rows in 'TotalCharges' contain blank whitespace ' '.")
    print(f"-> All of them have tenure = {df[df['TotalCharges'].astype(str).str.strip() == '']['tenure'].unique()} (brand new customers).")

# 4. Class Balance (Target Variable: 'Churn')
print("\n" + "=" * 65)
print("4. CLASS BALANCE FOR TARGET VARIABLE ('Churn'):")
churn_counts = df['Churn'].value_counts()
churn_percent = df['Churn'].value_counts(normalize=True) * 100

for label in churn_counts.index:
    print(f"  {label:<5}: {churn_counts[label]:>5} customers ({churn_percent[label]:.2f}%)")
print("=" * 65)

# -------------------------------------------------------------
# 5. GENERATE 4 CHARTS WITH ONE-LINE INSIGHTS
# -------------------------------------------------------------

# Chart 1: Target Class Distribution (Churn Balance)
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ['#2b5c8f', '#d95f02']
bars = ax.bar(churn_counts.index, churn_counts.values, color=colors, width=0.45, edgecolor='black', linewidth=1)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 60, f"{yval}\n({pct:.1f}%)", 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title("Target Class Distribution: Churn vs No Churn", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Churn Status", fontsize=11)
ax.set_ylabel("Number of Customers", fontsize=11)
ax.set_ylim(0, 6000)
plt.tight_layout()
chart1_path = os.path.join(FIGURES_DIR, "01_churn_distribution.png")
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"\n[Generated Chart 1]: {chart1_path}")

# Chart 2: Churn by Contract Type
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn[['No', 'Yes']].plot(
    kind='bar', 
    stacked=True, 
    color=['#2b5c8f', '#e74c3c'], 
    ax=ax, 
    edgecolor='black', 
    linewidth=0.8
)
ax.set_title("Customer Churn Rate by Contract Type", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Contract Type", fontsize=11)
ax.set_ylabel("Percentage of Customers (%)", fontsize=11)
ax.set_xticklabels(contract_churn.index, rotation=0)
ax.legend(["Stayed (No)", "Churned (Yes)"], loc='upper right')
for idx, row in contract_churn.iterrows():
    ax.text(list(contract_churn.index).index(idx), row['No'] + row['Yes'] / 2, 
            f"{row['Yes']:.1f}% Churn", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
chart2_path = os.path.join(FIGURES_DIR, "02_churn_by_contract.png")
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"[Generated Chart 2]: {chart2_path}")

# Chart 3: Churn vs Tenure (Kernel Density Estimation / Histogram)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=30, 
             palette={'No': '#2b5c8f', 'Yes': '#e74c3c'}, ax=ax, alpha=0.6)
ax.set_title("Customer Tenure (Months) vs Churn", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Tenure (Number of Months with Company)", fontsize=11)
ax.set_ylabel("Customer Count", fontsize=11)
plt.tight_layout()
chart3_path = os.path.join(FIGURES_DIR, "03_churn_by_tenure.png")
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"[Generated Chart 3]: {chart3_path}")

# Chart 4: Churn vs Monthly Charges
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', hue='Churn', data=df, 
            palette={'No': '#2b5c8f', 'Yes': '#e74c3c'}, ax=ax, width=0.4, legend=False)
ax.set_title("Monthly Charges ($) Distribution by Churn Status", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Churn Status", fontsize=11)
ax.set_ylabel("Monthly Charges ($)", fontsize=11)
medians = df.groupby('Churn')['MonthlyCharges'].median()
for idx, val in enumerate(medians):
    ax.text(idx, val + 2.5, f"Median: ${val:.2f}", horizontalalignment='center', 
            fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.tight_layout()
chart4_path = os.path.join(FIGURES_DIR, "04_churn_by_monthly_charges.png")
plt.savefig(chart4_path, dpi=300)
plt.close()
print(f"[Generated Chart 4]: {chart4_path}")

print("\nAll Stage 2 exploration tasks and charts completed successfully.")
