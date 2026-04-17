"""
Step 7 & 8: Ensemble — weighted soft voting + evaluation.
Combines XGBoost, Bi-LSTM, and Markov Chain predictions.
Input:  model/model_xgb.pkl, model/model_lstm.keras, model/markov_model.pkl
Output: model/eval_report.txt
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add training dir to path so MarkovChain class can be unpickled
sys.path.insert(0, os.path.dirname(__file__))
from markov_model import MarkovChain  # noqa: E402

# --- Paths ---
BASE = os.path.join(os.path.dirname(__file__), '..')
FEATURES_PATH = os.path.join(BASE, 'data', 'features.csv')
CLEAN_PATH = os.path.join(BASE, 'data', 'clean_data.csv')
XGB_PATH = os.path.join(BASE, 'model', 'model_xgb.pkl')
LSTM_PATH = os.path.join(BASE, 'model', 'model_lstm.keras')
MARKOV_PATH = os.path.join(BASE, 'model', 'markov_model.pkl')
FEATURE_COLS_PATH = os.path.join(BASE, 'model', 'feature_columns.json')
REPORT_PATH = os.path.join(BASE, 'model', 'eval_report.txt')

# Ensemble weights
W_XGB = 0.50
W_LSTM = 0.30
W_MARKOV = 0.20

EXCLUDE_COLS = ['timestamp', 'period', 'result', 'size', 'color', 'color_encoded', 'target']
WINDOW_SIZE = 20


def main():
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Load models
    print("Loading models...")
    xgb_model = joblib.load(XGB_PATH)
    markov_model = joblib.load(MARKOV_PATH)

    from tensorflow.keras.models import load_model
    lstm_model = load_model(LSTM_PATH)

    # Load data
    df_feat = pd.read_csv(FEATURES_PATH)
    df_clean = pd.read_csv(CLEAN_PATH)

    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = json.load(f)

    # Prepare XGBoost test data
    X_all = df_feat[feature_cols].values
    y_all = df_feat['target'].values
    split_xgb = int(len(X_all) * 0.7)
    X_test_xgb = X_all[split_xgb:]
    y_test_xgb = y_all[split_xgb:]

    # Prepare LSTM test data
    results = df_clean['result'].values.astype(float)
    targets = df_clean['target'].values.astype(float)
    colors = df_clean['color_encoded'].values.astype(float)
    results_norm = results / 9.0
    colors_norm = colors / 4.0

    X_lstm, y_lstm = [], []
    for i in range(WINDOW_SIZE, len(df_clean)):
        seq = np.column_stack([
            results_norm[i - WINDOW_SIZE:i],
            targets[i - WINDOW_SIZE:i],
            colors_norm[i - WINDOW_SIZE:i],
        ])
        X_lstm.append(seq)
        y_lstm.append(targets[i])
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    split_lstm = int(len(X_lstm) * 0.7)
    X_test_lstm = X_lstm[split_lstm:]
    y_test_lstm = y_lstm[split_lstm:]

    # Prepare Markov test data
    full_seq = df_clean['target'].values.tolist()
    split_markov = int(len(full_seq) * 0.7)

    # Align test sizes — use the minimum test size
    n_test = min(len(y_test_xgb), len(y_test_lstm))
    print(f"Test samples (aligned): {n_test}")

    # Get predictions from all 3 models
    print("\nRunning XGBoost predictions...")
    xgb_proba = xgb_model.predict_proba(X_test_xgb[-n_test:])  # [P(Small), P(Big)]

    print("Running LSTM predictions...")
    lstm_proba_big = lstm_model.predict(X_test_lstm[-n_test:]).flatten()
    lstm_proba = np.column_stack([1 - lstm_proba_big, lstm_proba_big])  # [P(Small), P(Big)]

    print("Running Markov predictions...")
    markov_proba = []
    for i in range(n_test):
        idx = len(full_seq) - n_test + i
        history = full_seq[:idx]
        p = markov_model.predict_proba(history)  # [P(Small), P(Big)]
        markov_proba.append(p)
    markov_proba = np.array(markov_proba)

    # Ensemble: weighted soft voting
    print("\nComputing ensemble predictions...")
    ensemble_proba = W_XGB * xgb_proba + W_LSTM * lstm_proba + W_MARKOV * markov_proba
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    y_true = y_test_xgb[-n_test:]

    # Individual model predictions
    xgb_pred = np.argmax(xgb_proba, axis=1)
    lstm_pred = np.argmax(lstm_proba, axis=1)
    markov_pred = np.argmax(markov_proba, axis=1)

    # Accuracies
    acc_xgb = accuracy_score(y_true, xgb_pred)
    acc_lstm = accuracy_score(y_true, lstm_pred)
    acc_markov = accuracy_score(y_true, markov_pred)
    acc_ensemble = accuracy_score(y_true, ensemble_pred)

    # Build report
    report_lines = [
        "=" * 60,
        "  OkWin Big/Small Predictor — Evaluation Report",
        "=" * 60,
        "",
        f"Test samples: {n_test}",
        f"Ensemble weights: XGBoost={W_XGB}, LSTM={W_LSTM}, Markov={W_MARKOV}",
        "",
        "--- Individual Model Accuracy ---",
        f"  XGBoost:      {acc_xgb:.4f}",
        f"  Bi-LSTM:      {acc_lstm:.4f}",
        f"  Markov Chain: {acc_markov:.4f}",
        "",
        f"--- Ensemble Accuracy: {acc_ensemble:.4f} ---",
        "",
        "--- Ensemble Classification Report ---",
        classification_report(y_true, ensemble_pred, target_names=['Small', 'Big']),
        "--- Confusion Matrix (Ensemble) ---",
        str(confusion_matrix(y_true, ensemble_pred)),
        "",
        "--- Average Prediction Confidence ---",
        f"  Mean max probability: {np.max(ensemble_proba, axis=1).mean():.4f}",
        f"  Predictions above 60% confidence: {(np.max(ensemble_proba, axis=1) > 0.6).sum()} / {n_test}",
        "",
        "=" * 60,
    ]

    report = "\n".join(report_lines)
    print(report)

    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == '__main__':
    main()
