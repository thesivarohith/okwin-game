"""
Step 4: Train XGBoost binary classifier for Big/Small prediction.
Input:  data/features.csv
Output: model/model_xgb.pkl
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# --- Paths ---
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_xgb.pkl')
META_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_columns.json')

# Feature columns (exclude metadata and target)
EXCLUDE_COLS = ['timestamp', 'period', 'result', 'size', 'color', 'color_encoded', 'target']


def main():
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: {FEATURES_PATH} not found. Run feature_engineering.py first.")
        sys.exit(1)

    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].values
    y = df['target'].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")

    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False  # time-series: no shuffle
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Hyperparameter search
    print("\nTraining XGBoost with hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    grid = GridSearchCV(
        base_model, param_grid,
        cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Small', 'Big']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance (top 15)
    importances = best_model.feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]
    print("\nTop 15 Features:")
    for i in top_idx:
        print(f"  {feature_cols[i]:30s} {importances[i]:.4f}")

    # Save model and feature columns
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(META_PATH, 'w') as f:
        json.dump(feature_cols, f)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved feature columns: {META_PATH}")

    return acc


if __name__ == '__main__':
    main()
