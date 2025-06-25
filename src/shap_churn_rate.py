import pandas as pd
import shap
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train-80.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test-20.csv'))
    test_raw = pd.read_csv(os.path.join(DATA_DIR, 'test-20.csv'))
    test = test_raw.copy()
    target_column = 'Churn'
    columns = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
    train = train[columns]
    test = test[columns]
    binary_columns = ['International plan', 'Voice mail plan', 'Churn']
    for col in binary_columns:
        train[col] = train[col].map({'Yes': 1, 'No': 0})
        test[col] = test[col].map({'Yes': 1, 'No': 0})
    
    le = LabelEncoder()
    train['State'] = le.fit_transform(train['State'])
    test['State'] = le.fit_transform(test['State'])
    model_name = {
        'XGBoost': 'xgb_model.pkl',
        'logisticRegression': 'logistic_model.pkl',
        'gradientBoosting': 'gb_model.pkl',
        'randomForest': 'rf_model.pkl'
    }
    model = joblib.load(os.path.join(MODEL_DIR, model_name['XGBoost']))
    x_test = test.drop(columns=[target_column])
    x_train = train.drop(columns=[target_column])
    probs = model.predict_proba(x_test)[:, 1]
    top_index = probs.argmin()
    explainer = shap.Explainer(model, x_train)
    joblib.dump(explainer, os.path.join(MODEL_DIR, 'explainer.pkl'))
    shap_values = explainer(x_test)
    shap_df = pd.DataFrame(shap_values.values, columns=x_test.columns)
    shap_df.columns = [f"{col}_shap" for col in shap_df.columns]
    shap_df['base_value'] = shap_values.base_values
    shap_df['shap_sum'] = shap_df[[col for col in shap_df.columns if col.endswith('_shap')]].sum(axis=1)
    shap_df['prediction_logit'] = shap_df['base_value'] + shap_df['shap_sum']
    shap_df['prediction_proba'] = 1 / (1 + np.exp(-shap_df['prediction_logit']))
    x_test_reset = x_test.reset_index(drop=True)
    shap_df = shap_df.reset_index(drop=True) 
    merged_df = pd.concat([x_test_reset, shap_df], axis=1)
    test_reset = test_raw.reset_index(drop=True)
    merged_df['Churn'] = test_reset[target_column]
    final_cols = []
    for col in x_test.columns:
        final_cols.extend([col, f"{col}_shap"])
    final_cols.extend(['base_value', 'shap_sum', 'prediction_logit', 'prediction_proba', 'Churn'])
    merged_df = merged_df[final_cols]
    merged_df.to_csv(os.path.join(DATA_DIR, 'shap_values.csv'), index=False)

if __name__ == "__main__":
    main()
