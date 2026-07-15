# HR Employee Attrition Prediction — XGBoost

An end-to-end machine learning pipeline that predicts whether an employee is likely to leave the company using **XGBoost**.

## 📌 Problem Statement
- **Objective:** Predict employee attrition (whether an employee will leave or stay) based on satisfaction, workload, and tenure-related features
- **Dataset:** HR Analytics dataset (`HR_comma_sep.csv`) — 14,999 records, 10 columns
- **Problem Type:** Classification
- **Target Variable:** `left` (1 = Employee left, 0 = Employee stayed)
- **Business Relevance:** Helps HR teams proactively identify at-risk employees and intervene early, reducing costly turnover and preserving institutional knowledge
- **Success Metric:** F1-score > 0.80 (due to class imbalance) → **Achieved: F1 = 0.9811**

## 📊 Dataset
- **Original rows:** 14,999 | **After removing duplicates:** 11,991 (3,008 duplicate rows removed)
- **Class distribution:** Stayed 83.4% | Left 16.6% — **imbalanced**, so the pipeline prioritized F1/Recall over raw accuracy
- **Numeric features:** `satisfaction_level`, `last_evaluation`, `number_project`, `average_montly_hours`, `time_spend_company`, `work_accident`, `promotion_last_5years`
- **Categorical features:** `sales` (department), `salary`
- No missing values, no constant columns, no highly correlated features detected

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## 🔍 Pipeline / Methodology
1. **Data Loading & Validation** — shape, info, summary statistics
2. **Data Cleaning** — removed 3,008 duplicate rows, standardized column names
3. **Exploratory Data Analysis** — dashboard overview, class imbalance check, correlation analysis, skewness check
4. **Outlier Treatment** — boxplot-based capping applied to numeric features
5. **Feature Engineering** — checked for highly correlated features (>0.95); none found
6. **Preprocessing Pipeline** — imputation, scaling for numeric features + one-hot encoding for categorical features (via `ColumnTransformer`)
7. **Train-Test Split** — 80/20 split (9,592 train / 2,399 test)
8. **Model Comparison** — benchmarked 10 classification algorithms using 5-fold cross-validation
9. **Model Training** — final pipeline trained using **XGBoost**
10. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
11. **Sanity Checks** — train vs test score comparison, feature importance dominance check
12. **Cross Validation** — 5-fold CV for robustness check
13. **Hyperparameter Tuning** — GridSearchCV & RandomizedSearchCV to tune `learning_rate`, `max_depth`, `n_estimators`

## 📈 Results

| Metric | Value |
|--------|-------|
| **Accuracy (Test)** | 0.9812 |
| **Precision** | 0.9811 |
| **Recall** | 0.9812 |
| **F1-Score** | **0.9811** |
| Cross-Validation Mean Accuracy | 0.9816 (± 0.0029) |
| Best Hyperparameters (GridSearchCV) | `learning_rate = 0.1`, `max_depth = 5`, `n_estimators = 100` |
| Final Tuned Model Score | 0.9825 |
| Train Score vs Test Score | 0.9957 vs 0.9812 |

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 (Stayed) | 0.99 | 0.99 | 0.99 |
| 1 (Left) | 0.96 | 0.93 | 0.94 |

**Model Comparison (Top Results — CV Accuracy):**

| Model | CV Accuracy |
|-------|-------------|
| Random Forest | 0.9829 |
| **XGBoost (selected)** | 0.9816 |
| Gradient Boosting | 0.9797 |
| Extra Trees | 0.9769 |
| Decision Tree | 0.9667 |

## ⚠️ Note on Model Reliability
The pipeline's automated sanity check flagged the train/test scores (0.9957 vs 0.9812) as **"suspiciously high — possible data leakage."** In this case it's more likely a sign of a **genuinely strong, highly predictive feature** (`satisfaction_level`) rather than actual leakage, since:
- The dataset retained a healthy 11,991 rows after deduplication (not reduced to a handful of samples)
- Cross-validation scores were consistently high and stable (± 0.0029), not just a single lucky split

That said, this is worth validating on a fresh, unseen dataset before deploying in a real HR system.

## 🔑 Key Insights
- **Best Model overall:** Random Forest scored marginally higher (98.29%) than XGBoost (98.16%) in comparison, but XGBoost was selected and performed nearly identically after tuning
- The dataset is **imbalanced** (83.4% stayed vs. 16.6% left), so F1-score and recall were prioritized over raw accuracy per the success metric
- Recall on employees who left is 93% — meaning ~7% of employees who actually left were misclassified as likely to stay, which is the more costly error type for HR intervention purposes
- `satisfaction_level` is the most dominant feature (importance: 0.265), consistent with common HR attrition research

## 📂 Folder Structure
```
HR-Attrition-Prediction/
│
├── data/
│   └── HR_comma_sep.csv
├── xgboost-hr-attrition-prediction-classification.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**
