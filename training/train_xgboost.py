import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score
import xgboost as xgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')

def train_xgboost():
    print("\n--- Training XGBoost Model ---")
    features_path = os.path.join(PROCESSED_DIR, 'features.csv')
    df = pd.read_csv(features_path)
    
    # Exclude non-features if present
    exclude = ['timestamp', 'period', 'result', 'size', 'color']
    feature_cols = [c for c in df.columns if c not in exclude and c != 'size_binary']
    
    X = df[feature_cols].values
    y = df['size_binary'].values
    
    # Split first 70% train, last 30% test (NO SHUFFLE)
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"XGBoost Train Accuracy: {train_acc:.4f}")
    print(f"XGBoost Test Accuracy:  {test_acc:.4f}")
    
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(SAVED_DIR, 'model_xgb.pkl'))
    
    train_probas = model.predict_proba(X_train)
    test_probas = model.predict_proba(X_test)
    
    np.save(os.path.join(SAVED_DIR, 'xgb_train_proba.npy'), train_probas)
    np.save(os.path.join(SAVED_DIR, 'xgb_test_proba.npy'), test_probas)
    
    # Top 10 feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Top 10 Feature Importances:")
    for i in range(min(10, len(indices))):
        print(f"  {i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
        
    return model, test_acc

if __name__ == "__main__":
    pass
