import pandas as pd
import dice_ml
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ======= Setup paths =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

def churn_dss_report(df_input, model, feature_columns, metadata=None, output_path=None):
    df = df_input.copy()

    missing_cols = set(feature_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[feature_columns]

    churn_probs = model.predict_proba(df)[:, 1]
    df["Churn_Prob"] = churn_probs

    churn_df = df.copy().reset_index(drop=False)
    df_dice = df[feature_columns].copy()
    df_dice["Churn"] = 0

    # Load metadata
    if metadata is None:
        meta_path = os.path.join(MODEL_DIR, 'feature_metadata.pkl')
        if os.path.exists(meta_path):
            metadata = joblib.load(meta_path)

    if metadata:
        continuous_features = metadata.get("continuous_features", [])
        categorical_features = metadata.get("categorical_features", [])
    else:
        continuous_features, categorical_features = [], []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df_dice[col]):
                if df_dice[col].nunique() <= 10:
                    categorical_features.append(col)
                else:
                    continuous_features.append(col)
            else:
                categorical_features.append(col)

    # Initialize DiCE
    data_dice = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name="Churn"
    )
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    dice = dice_ml.Dice(data_dice, model_dice)

    # Generate counterfactuals for all customers
    results = []
    cf_all = []  # Store counterfactual features for re-evaluation

    for _, row in churn_df.iterrows():
        idx = row['index']
        prob_before = row["Churn_Prob"]
        try:
            x_query = df_dice.loc[[idx], feature_columns]

            # Only generate counterfactuals for high-risk customers
            if prob_before >= 0.5:
                cf = dice.generate_counterfactuals(
                    x_query,
                    total_CFs=1,
                    desired_class=0,  # Always target reduced churn (class 0)
                    verbose=False
                )

                if cf.cf_examples_list and not cf.cf_examples_list[0].final_cfs_df.empty:
                    cf_df = cf.cf_examples_list[0].final_cfs_df[feature_columns]
                    prob_after = model.predict_proba(cf_df)[0][1]
                    delta = prob_before - prob_after

                    # Suggested changes
                    changes = []
                    for col in feature_columns:
                        val_old = x_query[col].values[0]
                        val_new = cf_df[col].values[0]
                        if col in continuous_features:
                            if not np.isclose(val_old, val_new, atol=1e-4):
                                changes.append(f"{col}: {val_old:.2f} → {val_new:.2f}")
                        else:
                            if str(val_old) != str(val_new):
                                changes.append(f"{col}: {val_old} → {val_new}")
                    recommendation = ", ".join(changes) if changes else "No change required"
                    cf_all.append(cf_df.iloc[0].to_dict())
                else:
                    prob_after = prob_before
                    delta = 0
                    recommendation = "No counterfactuals generated"
                    cf_all.append(x_query.iloc[0].to_dict())
            else:
                prob_after = prob_before
                delta = 0
                recommendation = "No change required"
                cf_all.append(x_query.iloc[0].to_dict())

        except Exception as e:
            prob_after = prob_before
            delta = 0
            recommendation = f"DiCE error: {str(e)}"
            cf_all.append(x_query.iloc[0].to_dict())

        results.append({
            "Index": idx,
            "Churn_Prob_Before": round(prob_before, 4),
            "Churn_Prob_After": round(prob_after, 4),
            "Delta": round(delta, 4),
            "Improvement Suggestion": recommendation
        })

    result_df = pd.DataFrame(results)
    cf_df_all = pd.DataFrame(cf_all)

    if output_path:
        result_df.to_csv(output_path, index=False)

    return result_df, cf_df_all


if __name__ == "__main__":
    # Load model and metadata
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    metadata_path = os.path.join(MODEL_DIR, "feature_metadata.pkl")
    metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None

    # Load data
    df_new = pd.read_csv(os.path.join(DATA_DIR, "test-20.csv"))

    # Save actual churn labels if available
    if 'Churn' in df_new.columns:
        churn_actual = df_new['Churn'].astype(int)
        df_new = df_new.drop(columns=["Churn"])
    else:
        churn_actual = None

    # Preprocessing
    le = LabelEncoder()
    df_new['State'] = le.fit_transform(df_new['State'])
    binary_columns = ['International plan', 'Voice mail plan']
    for col in binary_columns:
        df_new[col] = df_new[col].map({'Yes': 1, 'No': 0})

    # Generate churn report
    report, cf_features = churn_dss_report(
        df_input=df_new,
        model=model,
        feature_columns=[col for col in feature_columns if col != 'Churn'],
        metadata=metadata,
        output_path=os.path.join(DATA_DIR, "report.csv")
    )

    print("\n===== Top 5 customers with improvement suggestions =====")
    print(report.head())

    # Summary statistics
    print("\n===== CLASSIFICATION MODEL SUMMARY =====")
    X_before = df_new[[col for col in feature_columns if col != 'Churn']]
    X_after = cf_features[[col for col in feature_columns if col != 'Churn']]

    pred_before = model.predict(X_before)
    pred_after = model.predict(X_after)

    churn_pred_rate = np.mean(pred_before)
    churn_adjusted_rate = np.mean(pred_after)

    print(f"Total customers                  : {len(report)}")
    print(f"Initial predicted churn rate     : {churn_pred_rate:.4f}")
    if churn_actual is not None:
        print(f"Actual churn rate                : {churn_actual.mean():.4f}")
    else:
        print("Actual churn rate                : Ground truth not available")
    print(f"Adjusted churn rate after CF     : {churn_adjusted_rate:.4f}")