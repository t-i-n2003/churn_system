import pandas as pd
import dice_ml
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ======= Setup path =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

def generate_churn_report(df_input, model, feature_columns, metadata=None):
    df = df_input.copy()
    
    # Xử lý thiếu cột
    missing_cols = set(feature_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[feature_columns]

    # Dự đoán xác suất churn
    churn_probs = model.predict_proba(df)[:, 1]
    results = []

    # Thiết lập DiCE
    df_dice = df.copy()
    df_dice['Churn'] = 0  # Giả định ban đầu không churn

    if metadata:
        cont_features = metadata.get("continuous_features", [])
        cat_features = metadata.get("categorical_features", [])
    else:
        cont_features, cat_features = [], []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                cont_features.append(col)
            else:
                cat_features.append(col)

    data_dice = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=cont_features,
        categorical_features=cat_features,
        outcome_name="Churn"
    )
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    dice = dice_ml.Dice(data_dice, model_dice)

    for idx, row in df.iterrows():
        prob = churn_probs[idx]
        recommendation = "Không cần hành động"
        
        if prob > 0.6:  # Ngưỡng churn cao
            try:
                cf = dice.generate_counterfactuals(
                    df_dice.iloc[[idx]][feature_columns],
                    total_CFs=3,
                    desired_class=0,
                    verbose=False
                )
                
                if cf.cf_examples_list and not cf.cf_examples_list[0].final_cfs_df.empty:
                    cf_row = cf.cf_examples_list[0].final_cfs_df.iloc[0]
                    changes = []
                    
                    for col in feature_columns:
                        original = row[col]
                        new = cf_row[col]
                        
                        # Kiểm tra thay đổi
                        if (col in cont_features and not np.isclose(original, new, atol=1e-4)) or \
                           (col in cat_features and original != new):
                                changes.append(f"{col}: {original} → {new}")
                    
                    if changes:
                        recommendation = " | ".join(changes)
            except Exception:
                recommendation = "Lỗi khi sinh gợi ý"

        results.append({
            "Index": idx,
            "Churn_Prob": round(prob, 4),
            "Recommendation": recommendation
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load model và features
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    
    # Load metadata nếu có
    metadata_path = os.path.join(MODEL_DIR, "feature_metadata.pkl")
    metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None

    # Load dữ liệu
    df = pd.read_csv(os.path.join(DATA_DIR, "new_customers_minit.csv"))
    
    # Tiền xử lý cơ bản
    if 'State' in df.columns:
        df['State'] = LabelEncoder().fit_transform(df['State'])
    if 'International plan' in df.columns:
        df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
    if 'Voice mail plan' in df.columns:
        df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})

    # Tạo báo cáo
    report = generate_churn_report(
        df_input=df,
        model=model,
        feature_columns=[col for col in feature_columns if col != 'Churn'],
        metadata=metadata
    )
    report.to_csv(os.path.join(DATA_DIR, "churn_report.csv"), index=False)
    print(report)
