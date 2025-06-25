import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, 'train-80.csv')
test_path = os.path.join(DATA_DIR, 'test-20.csv')

# ---------- Các hàm mô hình ----------
def xgboost_model(trainX, trainy, testX, testy):
    model = XGBClassifier(scale_pos_weight=6, use_label_encoder=False, eval_metric='logloss')
    model.fit(trainX, trainy)
    predictions = model.predict(testX)
    print("XGBoost Report:\n", classification_report(testy, predictions))
    return model

def logistic_regression_model(trainX, trainy, testX, testy):
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(trainX, trainy)
    predictions = model.predict(testX)
    print("Logistic Regression Report:\n", classification_report(testy, predictions))
    return model

def gradient_boosting_model(trainX, trainy, testX, testy):
    model = GradientBoostingClassifier()
    model.fit(trainX, trainy)
    predictions = model.predict(testX)
    print("Gradient Boosting Report:\n", classification_report(testy, predictions))
    return model

def random_forest_model(trainX, trainy, testX, testy):
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(trainX, trainy)
    predictions = model.predict(testX)
    print("Random Forest Report:\n", classification_report(testy, predictions))
    print("Confusion Matrix:\n", confusion_matrix(testy, predictions))
    print(f'Accuracy: {accuracy_score(testy, predictions):.2%}')
    return model

def basic_statistics(df):
    print("Basic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

# ---------- Main ----------
def main():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Encode dữ liệu
    le = LabelEncoder()
    train['State'] = le.fit_transform(train['State'])
    test['State'] = le.transform(test['State'])  # dùng transform để giữ mapping cũ

    binary_columns = ['International plan', 'Voice mail plan']
    for col in binary_columns:
        train[col] = train[col].map({'Yes': 1, 'No': 0})
        test[col] = test[col].map({'Yes': 1, 'No': 0})

    train['Churn'] = train['Churn'].astype(int)
    test['Churn'] = test['Churn'].astype(int)

    # Xử lý cột
    target_column = 'Churn'
    drop_cols = ['Account length', 'Area code']
    feature_columns = [col for col in train.columns if col not in drop_cols + [target_column]]

    trainX = train[feature_columns]
    trainy = train[target_column]
    testX = test[feature_columns]
    testy = test[target_column]

    # Thống kê
    basic_statistics(train)

    # Lưu feature columns
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, 'features.pkl'))

    # Lưu metadata DiCE
    continuous_features = [col for col in feature_columns if pd.api.types.is_numeric_dtype(trainX[col])]
    categorical_features = [col for col in feature_columns if col not in continuous_features]
    metadata = {
        "continuous_features": continuous_features,
        "categorical_features": categorical_features
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, 'feature_metadata.pkl'))

    # Huấn luyện & lưu model
    model_funcs = {
        'xgb_model.pkl': xgboost_model,
        'logistic_model.pkl': logistic_regression_model,
        'gb_model.pkl': gradient_boosting_model,
        'rf_model.pkl': random_forest_model
    }

    for filename, func in model_funcs.items():
        print(f"\nTraining {filename}...")
        model = func(trainX, trainy, testX, testy)
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
    print("\n✅ Train script hoàn tất.")
