# 🔁 Customer Churn Prediction System

This project implements a machine learning system for predicting customer churn. It includes preprocessing pipelines, classification models, evaluation metrics, and model explanation (including counterfactual analysis).

---

## 📁 Project Structure
```
churn_system/
├── data/ # Raw and processed data
├── models/ # Trained model files (e.g. .pkl, .pt)
├── src/ # Source code: preprocessing, modeling, explainability
│ ├── main.py
│ ├── train.py
│ ├── shap_value.py
│ └── dice.py
├── README.md
├── requirements.txt
└── EDA.pbix
```

---

## 🚀 Key Features

- ✅ Customer churn prediction using various ML models (GradientBoosting, LogisticRegression, RandomForest, etc.)
- ✅ Preprocessing and feature engineering pipeline
- ✅ Model evaluation using confusion matrix, recall, F1-score, and more
- ✅ Explainability via DiCE for counterfactual analysis
- ✅ Power BI dashboard for business reporting
- ✅ Training pipeline with minimal configuration
  
---

## 🎯 Objectives

- Analyze customer churn trends via Power BI.
- Predict customer churn using machine learning models (code not included here).
- Generate churn-based recommendations for retention strategies.
- Integrate model results with reporting (Power BI).
  
---

## 🎯 Effectiveness

The system achieves approximately **95% accuracy** on the test dataset, successfully identifying customers at high risk of churning.
Furthermore, it can generate **counterfactual recommendations** to suggest minimal changes that would have prevented churn in **most cases** — excluding immutable characteristics (e.g., gender, age, state).
This allows businesses to take **targeted, actionable steps** to reduce churn and improve retention.

---

## 📊 Exploratory Data Analysis

The file `EDA.pbix` contains rich visualizations and insights, including:
- Churn rate for plan
- Top N Churn Rate for State
- Churn Rate for Customer Service Calls
➡️ Open `EDA.pbix` with [Power BI Desktop](https://powerbi.microsoft.com/) to explore the dashboard.

---

## 🔍 Recommendations Output

The file `data/Report.csv` includes:
- Index
- Churn risk probability
- Suggested action
This output is intended to be used by marketing or customer success teams.

---

## 💻 Requirements

To install dependencies (for the churn prediction module, if added later):

```bash
pip install -r requirements.txt
```

---

## 📎 Dependencies

The requirements.txt suggests use of:
- pandas, scikit-learn — for data processing and modeling
- dice-ml — for explainability (counterfactual explanations)
- matplotlib, seaborn — for visualizations
- joblib — for saving models

---

## 📄 License
- This project is for educational and demo purposes. Contact the author for production use.
