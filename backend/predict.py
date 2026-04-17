"""
Prediction logic — loads models, computes features from input history, runs ensemble.
"""
import numpy as np
import os
import sys
import json
import joblib

# Add training dir to path for MarkovChain class
TRAINING_DIR = os.path.join(os.path.dirname(__file__), '..', 'training')
sys.path.insert(0, TRAINING_DIR)
from markov_model import MarkovChain  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

# Ensemble weights (Calibrated for current model performance)
W_XGB = 0.60
W_LSTM = 0.20
W_MARKOV = 0.20

WINDOW_SIZE = 20


class PredictionEngine:
    """Loads all 3 models and runs ensemble predictions."""

    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.markov_model = None
        self.feature_cols = None
        self.loaded = False

    def load_models(self):
        """Load all models at startup."""
        print("Loading models...")

        # XGBoost
        xgb_path = os.path.join(MODEL_DIR, 'model_xgb.pkl')
        self.xgb_model = joblib.load(xgb_path)
        print("  XGBoost loaded")

        # Feature columns
        meta_path = os.path.join(MODEL_DIR, 'feature_columns.json')
        with open(meta_path, 'r') as f:
            self.feature_cols = json.load(f)
        print(f"  Feature columns: {len(self.feature_cols)}")

        # Markov Chain
        from train_markov import MarkovChain  # noqa
        markov_path = os.path.join(MODEL_DIR, 'markov_model.pkl')
        self.markov_model = joblib.load(markov_path)
        print("  Markov Chain loaded")

        # LSTM
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import load_model
        lstm_path = os.path.join(MODEL_DIR, 'model_lstm.keras')
        self.lstm_model = load_model(lstm_path)
        print("  Bi-LSTM loaded")

        self.loaded = True
        print("All models ready.")

    def _compute_xgb_features(self, history_results, history_sizes, history_colors, period):
        """Compute XGBoost features from input history."""
        features = {}

        # Recent history (last 10 results and sizes)
        for i in range(1, 11):
            idx = -(i)
            features[f'last_{i}_result'] = history_results[idx] if len(history_results) >= i else 0
            features[f'last_{i}_size'] = history_sizes[idx] if len(history_sizes) >= i else 0

        # Recent colors (last 5)
        for i in range(1, 6):
            features[f'last_{i}_color'] = history_colors[-i] if len(history_colors) >= i else 0

        # Streak
        streak_len = 1
        streak_type = history_sizes[-1] if history_sizes else 0
        for j in range(2, min(len(history_sizes) + 1, 20)):
            if history_sizes[-j] == streak_type:
                streak_len += 1
            else:
                break
        features['streak_length'] = streak_len
        features['streak_type'] = streak_type

        # Frequency counts
        for window, label in [(10, 'short'), (20, 'mid'), (50, 'long')]:
            recent = history_sizes[-window:] if len(history_sizes) >= window else history_sizes
            big_count = sum(recent)
            features[f'big_count_{label}'] = big_count
            features[f'small_count_{label}'] = len(recent) - big_count
            features[f'big_ratio_{label}'] = big_count / max(len(recent), 1)

        # Transitions
        if len(history_sizes) >= 2:
            features['last_2_combo'] = history_sizes[-2] * 2 + history_sizes[-1]
        else:
            features['last_2_combo'] = -1
        if len(history_sizes) >= 3:
            features['last_3_combo'] = history_sizes[-3] * 4 + history_sizes[-2] * 2 + history_sizes[-1]
        else:
            features['last_3_combo'] = -1

        # Gap features
        gap_big, gap_small = len(history_sizes), len(history_sizes)
        for j in range(1, len(history_sizes) + 1):
            if history_sizes[-j] == 1 and gap_big == len(history_sizes):
                gap_big = j
            if history_sizes[-j] == 0 and gap_small == len(history_sizes):
                gap_small = j
        features['gap_since_big'] = gap_big
        features['gap_since_small'] = gap_small

        # Cyclical
        for mod in [3, 5, 10, 20]:
            features[f'period_mod_{mod}'] = period % mod

        # Pattern flags
        if len(history_sizes) >= 4:
            last4 = history_sizes[-4:]
            features['is_alternating'] = int(all(last4[i] != last4[i+1] for i in range(3)))
        else:
            features['is_alternating'] = 0

        if len(history_sizes) >= 3:
            features['is_repeating'] = int(history_sizes[-1] == history_sizes[-2] == history_sizes[-3])
        else:
            features['is_repeating'] = 0

        recent10 = history_sizes[-10:] if len(history_sizes) >= 10 else history_sizes
        features['dominant_last10'] = int(sum(recent10) >= 6)

        # Result stats
        for w in [5, 10]:
            recent_r = history_results[-w:] if len(history_results) >= w else history_results
            features[f'result_mean_{w}'] = np.mean(recent_r)
            features[f'result_std_{w}'] = np.std(recent_r)
            features[f'result_median_{w}'] = np.median(recent_r)

        # Build feature vector in correct column order
        return np.array([[features.get(col, 0) for col in self.feature_cols]])

    def _compute_lstm_input(self, history_results, history_sizes, history_colors):
        """Compute LSTM sliding window input."""
        results_norm = np.array(history_results[-WINDOW_SIZE:]) / 9.0
        sizes = np.array(history_sizes[-WINDOW_SIZE:], dtype=float)
        colors_norm = np.array(history_colors[-WINDOW_SIZE:]) / 4.0

        seq = np.column_stack([results_norm, sizes, colors_norm])
        return seq.reshape(1, WINDOW_SIZE, 3)

    def predict(self, history_results: list, period: int):
        """
        Run ensemble prediction.
        
        Args:
            history_results: list of last 20+ result values (0-9)
            period: current period number
        
        Returns:
            dict with prediction, confidence, probabilities
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Derive sizes and colors from results
        history_sizes = [1 if r >= 5 else 0 for r in history_results]

        color_map = {
            0: 4, 1: 1, 2: 0, 3: 1, 4: 0,
            5: 3, 6: 0, 7: 1, 8: 0, 9: 1
        }
        history_colors = [color_map.get(r, 0) for r in history_results]

        # --- XGBoost ---
        X_xgb = self._compute_xgb_features(history_results, history_sizes, history_colors, period)
        xgb_proba = self.xgb_model.predict_proba(X_xgb)[0]  # [P(Small), P(Big)]

        # --- LSTM ---
        if len(history_results) >= WINDOW_SIZE:
            X_lstm = self._compute_lstm_input(history_results, history_sizes, history_colors)
            lstm_big = float(self.lstm_model.predict(X_lstm, verbose=0)[0][0])
        else:
            lstm_big = 0.5
        lstm_proba = np.array([1 - lstm_big, lstm_big])

        # --- Markov ---
        markov_proba = self.markov_model.predict_proba(history_sizes)

        # --- Ensemble ---
        ensemble_proba = W_XGB * xgb_proba + W_LSTM * lstm_proba + W_MARKOV * markov_proba
        ensemble_proba = ensemble_proba / ensemble_proba.sum()  # normalize

        predicted_class = int(np.argmax(ensemble_proba))
        confidence = float(np.max(ensemble_proba))

        return {
            'prediction': 'Big' if predicted_class == 1 else 'Small',
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'Small': round(float(ensemble_proba[0]) * 100, 2),
                'Big': round(float(ensemble_proba[1]) * 100, 2),
            },
            'model_contributions': {
                'xgboost': {
                    'Small': round(float(xgb_proba[0]) * 100, 2),
                    'Big': round(float(xgb_proba[1]) * 100, 2),
                },
                'lstm': {
                    'Small': round(float(lstm_proba[0]) * 100, 2),
                    'Big': round(float(lstm_proba[1]) * 100, 2),
                },
                'markov': {
                    'Small': round(float(markov_proba[0]) * 100, 2),
                    'Big': round(float(markov_proba[1]) * 100, 2),
                },
            }
        }
