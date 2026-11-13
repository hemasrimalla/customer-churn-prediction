# Telco Customer Churn Prediction (End-to-End ML Pipeline)

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-3.1.0-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning classification system designed to predict customer churn in the telecommunications industry, prioritize high-risk accounts, and maximize customer retention revenue.

---

## 📌 Executive Summary & Problem Statement

In the subscription-based telecom sector, **Customer Acquisition Cost (CAC) is 5x to 7x higher than retention cost**. Losing a customer permanently drains monthly recurring revenue and forces companies to spend heavily on aggressive marketing to replace them.

* **Objective:** Build a robust, interpretable binary classification model to detect customers likely to churn before they leave.
* **Target Metric:** Maximize **Recall** (Sensitivity) and **ROC-AUC** while maintaining balanced precision, minimizing catastrophic **False Negatives** (missed churners).
* **Key Achievement:** Tuned Random Forest with cost-sensitive class balancing boosted Churn Recall from **51.34% to 80.21%**, catching **300 out of 374 at-risk churners** (slashing missed churners from 182 to 74) on unseen test data.

---

## 📊 Dataset Overview

* **Source:** Public [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Dimensions:** 7,043 rows (customers) and 21 attributes.
* **Target Variable:** `Churn` (Binary: `0 = Retained`, `1 = Churned`).
* **Class Balance:** Moderately imbalanced (73.46% Retained vs. 26.54% Churned; ~2.77:1 ratio).
* **Feature Categories:**
  * **Demographics:** Gender, Senior Citizen, Partner, Dependents.
  * **Subscribed Services:** Phone, Multiple Lines, Internet (DSL/Fiber optic), Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
  * **Account Details:** Tenure (months), Contract type, Paperless Billing, Payment Method, Monthly Charges, Total Charges.

---

## 🛠️ Project Architecture & Pipeline

```text
customer-churn-prediction/
│
├── data/                                 # Datasets
│   ├── Telco-Customer-Churn.csv          # Raw data (7,043 rows)
│   ├── cleaned_telco_churn.csv           # Cleaned data (7,032 rows)
│   ├── X_train.csv, X_test.csv           # Processed feature splits (80/20)
│   └── y_train.csv, y_test.csv           # Target splits
│
├── notebooks/                            # Interactive Jupyter Notebooks
│   ├── 01_data_exploration.ipynb         # EDA and distribution charts
│   ├── 02_data_cleaning.ipynb            # Missing values and type conversions
│   ├── 03_feature_engineering.ipynb      # One-hot encoding & standard scaling
│   ├── 04_model_training.ipynb           # Baseline model fits
│   ├── 05_model_evaluation.ipynb         # Confusion matrices & ROC curves
│   ├── 06_model_improvement.ipynb        # Class weighting & GridSearchCV
│   ├── 07_feature_importance.ipynb       # MDI feature importance
│   └── 08_model_prediction.ipynb         # Inference workflow
│
├── src/                                  # Modular Production Code
│   ├── stage2_data_exploration.py        # EDA execution script
│   ├── stage3_data_cleaning.py           # Cleaning pipeline
│   ├── stage4_feature_engineering.py     # Encoding & scaling pipeline
│   ├── stage5_model_training.py          # Model fitting
│   ├── stage6_model_evaluation.py        # Metric evaluation
│   ├── stage7_model_improvement.py       # Hyperparameter tuning
│   ├── stage8_feature_importance.py      # Feature ranking
│   └── predict.py                        # Live inference wrapper
│
├── models/                               # Serialized Artifacts (.joblib)
│   ├── final_churn_model.joblib          # Tuned champion model
│   ├── scaler.joblib                     # Fitted StandardScaler
│   ├── feature_names.joblib              # 30-feature alignment schema
│   └── feature_importance_ranking.csv   # Feature ranking metrics
│
├── figures/                              # Publication-Grade Visualizations
│   ├── 01_churn_distribution.png         # Target imbalance
│   ├── 02_churn_by_contract.png          # Churn rate across contract types
│   ├── 03_churn_by_tenure.png            # Churn vs tenure KDE distribution
│   ├── 04_churn_by_monthly_charges.png   # Monthly charges boxplot
│   ├── 05_confusion_matrices_baseline.png# Baseline CMs
│   ├── 06_roc_curves_baseline.png        # Baseline ROC curves
│   ├── 07_before_after_improvement.png   # Improvement comparison bar chart
│   ├── 08_confusion_matrices_tuned.png   # Tuned model CMs
│   └── 09_top10_feature_importance.png   # Top 10 feature drivers
│
├── requirements.txt                      # Locked Python dependencies
├── .gitignore                            # Git exclusion rules
└── README.md                             # Project documentation
```

---

## 🔬 Methodology & Key Engineering Decisions

### 1. Data Cleaning
* **Whitespace Detection in `TotalCharges`:** Identified 11 rows where `TotalCharges` contained blank strings (`" "`), forcing an `object` dtype. All 11 records had `tenure = 0` (brand-new accounts without a completed billing cycle) and zero churn history (`Churn = 'No'`). Converted using `pd.to_numeric(..., errors='coerce')` and dropped the 11 records (representing a negligible 0.15% data loss).
* **ID Removal:** Dropped `customerID` (100% unique primary key) to prevent the curse of dimensionality and memorization overfitting.

### 2. Feature Engineering & Leakage Prevention
* **One-Hot Encoding with `drop_first=True`:** Encoded categorical features while dropping the first level ($k-1$ dummies for $k$ levels) to avoid the **Dummy Variable Trap** (multicollinearity). Expanded features from 19 to 30.
* **Stratified 80/20 Train/Test Split:** Enforced identical class proportions (26.58% Churn) across training (5,625 rows) and testing (1,407 rows) sets with `random_state=42`.
* **Zero Data Leakage Scaling:** Fitted `StandardScaler` strictly on `X_train` (`scaler.fit_transform`) and applied it to `X_test` (`scaler.transform`).

### 3. Handling Class Imbalance & Hyperparameter Tuning
* **Cost-Sensitive Learning:** Configured `class_weight='balanced'`, assigning a 2.77x higher loss penalty to churn errors ($w_1 \approx 1.88$ vs. $w_0 \approx 0.68$).
* **5-Fold Stratified `GridSearchCV`:** Tuned `RandomForestClassifier` across depth, leaves, splits, and estimators to control tree complexity and prevent the baseline's 99.88% train overfitting.

---

## 📈 Real Performance Results

Evaluated on the held-out test partition of **1,407 customers** (374 Churners, 1,033 Retained):

| Model Stage | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Caught Churn (TP) | Missed Churn (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 80.45% | 65.05% | 57.22% | 60.88% | 0.8361 | 214 | 160 |
| **Random Forest (Baseline)** | 78.82% | 62.34% | 51.34% | 56.30% | 0.8156 | 192 | 182 |
| **Logistic Regression (Tuned)** | 80.03% | 63.88% | 57.22% | 60.37% | 0.8354 | 214 | 160 |
| **Random Forest (Tuned + Balanced)** 🏆 | **73.77%** | **50.42%** | **80.21%** | **61.92%** | **0.8384** | **300** | **74** |

> **Business Impact:** The Tuned Random Forest captures **80.21% of all churners** (catching 108 more customers than the baseline). In a commercial setting, preserving 108 accounts at an average \$80/month bill represents **\$103,680 in annualized recurring revenue saved**.

---

## 💡 Top 10 Churn Drivers (Feature Importance)

Analysis of Gini feature importances (MDI) indicates that 10 features drive **79.6%** of churn predictions:

| Rank | Feature | Importance (%) | Business Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `tenure` | 17.38% | Customers with 0–12 months tenure are at peak flight risk; risk drops sharply after 24 months. |
| 2 | `TotalCharges` | 13.10% | High lifetime spend creates natural ecosystem inertia and customer loyalty. |
| 3 | `Contract_Two year` | 13.05% | Contract commitment acts as a strong protective shield against churn (2.8% churn rate). |
| 4 | `MonthlyCharges` | 8.28% | Higher recurring monthly fees drive customers to compare competitor pricing. |
| 5 | `InternetService_Fiber optic` | 8.04% | Premium Fiber optic users show elevated churn due to pricing pressure or technical outages. |
| 6 | `PaymentMethod_Electronic check`| 5.93% | Manual payment methods introduce monthly friction; automatic bank/card debit reduces churn. |
| 7 | `Contract_One year` | 5.07% | Mid-term contract commitment dampens turnover compared to month-to-month plans. |
| 8 | `OnlineSecurity_Yes` | 3.99% | Cybersecurity add-ons make the account "sticky" and harder to abandon. |
| 9 | `TechSupport_Yes` | 2.52% | Responsive customer support quickly resolves issues before they lead to cancellations. |
| 10| `StreamingMovies_No internet` | 2.25% | Low-cost basic landline accounts rarely churn. |

---

## 🚀 How to Run This Project

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
python -m pip install -r requirements.txt
```

### 2. Execute the Pipeline End-to-End
```bash
# Stage 2: Exploratory Data Analysis
python src/stage2_data_exploration.py

# Stage 3: Data Cleaning
python src/stage3_data_cleaning.py

# Stage 4: Feature Engineering & Preprocessing
python src/stage4_feature_engineering.py

# Stage 5 & 6: Model Training & Evaluation
python src/stage5_model_training.py
python src/stage6_model_evaluation.py

# Stage 7 & 8: Improvement & Feature Importance
python src/stage7_model_improvement.py
python src/stage8_feature_importance.py
```

### 3. Run Real-Time Customer Inference
```bash
python src/predict.py
```

**Example Programmatic Usage:**
```python
from src.predict import predict_churn

sample_customer = {
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

result = predict_churn(sample_customer)
print(result)
# Output: {'churn_probability': 69.82, 'risk_level': 'HIGH RISK', 'prediction': 'Will Churn (1)', ...}
```

---

## 🔮 Future Improvements

1. **Gradient Boosting Architectures:** Benchmark XGBoost, LightGBM, and CatBoost for higher non-linear interaction modeling.
2. **Cost-Optimal Threshold Tuning:** Rather than standard 0.5 thresholding, compute expected financial cost curves ($\text{Cost} = C_{\text{FP}} \cdot \text{FP} + C_{\text{FN}} \cdot \text{FN}$) to locate the exact mathematically optimal decision threshold.
3. **SHAP (SHapley Additive exPlanations):** Implement local sample-level SHAP explainability to provide customer-specific churn explanations to frontline support agents.
4. **API Containerization:** Wrap `predict.py` in a FastAPI microservice and package it into a Docker container for cloud deployment (AWS ECS / GCP Cloud Run).
