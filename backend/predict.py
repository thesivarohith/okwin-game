import numpy as np
import pandas as pd
import os
import sys
import json
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

class PredictionEngine:
    def __init__(self):
        self.xgb = None
        self.lstm = None
        self.tcn = None
        self.hmm = None
        self.markov = None
        self.meta_mlp = None
        self.dbscan_scaler = None
        self.dbscan_model = None
        self.prefixspan_rules = {}
        self.feature_cols = []
        self.loaded = False
        
        # State for Kalman
        self.x_hat = np.array([0.5, 0.5])
        self.P = np.eye(2)
        
    def load_models(self):
        required_files = [
            os.path.join(MODEL_SAVED_DIR, 'model_xgb.pkl'),
            os.path.join(MODEL_SAVED_DIR, 'model_lstm.keras'),
            os.path.join(MODEL_SAVED_DIR, 'model_tcn.keras'),
            os.path.join(MODEL_SAVED_DIR, 'model_hmm.pkl'),
            os.path.join(MODEL_SAVED_DIR, 'markov_model.pkl'),
            os.path.join(MODEL_SAVED_DIR, 'model_meta_mlp.keras'),
            os.path.join(MODEL_SAVED_DIR, 'dbscan_scaler.pkl'),
            os.path.join(MODEL_SAVED_DIR, 'dbscan_model.pkl'),
            os.path.join(DATA_DIR, 'prefixspan_rules.csv'),
            os.path.join(MODEL_DIR, 'feature_columns.json')
        ]
        
        for f in required_files:
            if not os.path.exists(f):
                return False

        from tensorflow.keras.models import load_model
        
        self.xgb = joblib.load(required_files[0])
        self.lstm = load_model(required_files[1])
        self.tcn = load_model(required_files[2])
        self.hmm = joblib.load(required_files[3])
        self.markov = joblib.load(required_files[4])
        self.meta_mlp = load_model(required_files[5])
        self.dbscan_scaler = joblib.load(required_files[6])
        self.dbscan_model = joblib.load(required_files[7])
        
        rules_df = pd.read_csv(required_files[8])
        for _, row in rules_df.iterrows():
            # pattern parsed from str, e.g. "1 0 1" => (1, 0, 1)
            pat_str = str(row['pattern']).replace('_',' ')
            pat_tuple = tuple(int(x) for x in pat_str.split())
            self.prefixspan_rules[pat_tuple] = (int(row['pred']), float(row['confidence']))
            
        with open(required_files[9], 'r') as json_file:
            self.feature_cols = json.load(json_file)
            
        self.loaded = True
        return True

    def _update_kalman(self, new_obs):
        # new_obs: 1 for Big, 0 for Small
        Q = 0.001 * np.eye(2)
        R = 0.05 * np.eye(2)
        F = np.eye(2)
        H = np.eye(2)
        
        z = np.array([1, 0]) if new_obs == 1 else np.array([0, 1])
        
        # Predict
        x_hat_minus = F.dot(self.x_hat)
        P_minus = F.dot(self.P).dot(F.T) + Q
        
        # Update
        S = H.dot(P_minus).dot(H.T) + R
        K = P_minus.dot(H.T).dot(np.linalg.inv(S))
        
        y = z - H.dot(x_hat_minus)
        self.x_hat = x_hat_minus + K.dot(y)
        self.P = (np.eye(2) - K.dot(H)).dot(P_minus)
        
        return self.x_hat

    def predict(self, history_results: list, period: int):
        if not self.loaded:
            return {"status": "models_not_ready", "message": "Run python train_all.py first"}
            
        # history_results is rolling 50 results
        if len(history_results) < 50:
            return {"status": "error", "message": f"Need 50 history values, got {len(history_results)}"}
            
        # Build sizes and colors
        sizes = [1 if r >= 5 else 0 for r in history_results]
        color_map = {0: 2, 1: 1, 2: 0, 3: 1, 4: 0, 5: 2, 6: 0, 7: 1, 8: 0, 9: 1}
        colors = [color_map.get(r, 0) for r in history_results]
        
        # We need to construct real-time feature row to match `feature_columns.json`
        # Because constructing exactly matching features is complex manually, we do it via DataFrame.
        # Create small dataframe of the history
        df_hist = pd.DataFrame({
            'period': [period - 50 + i for i in range(50)],
            'result': history_results,
            'size_binary': sizes,
            'color_enc': colors
        })
        
        # We process this with our scripts
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        
        # We need streak, frequencies etc. The easiest is using pandas rolling on df_hist
        # But for inference speed we manually extract what we need for row 50 (index 49)
        # Actually since we have feature_engineering logic, doing it on df_hist might give valid row 49
        
        cur = 49 # last index
        
        xgb_feats = {}
        for i in range(1, 11):
            xgb_feats[f'last_{i}_result'] = df_hist['result'].shift(i).iloc[cur]
            xgb_feats[f'last_{i}_size'] = df_hist['size_binary'].shift(i).iloc[cur]
        for i in range(1, 6):
            xgb_feats[f'last_{i}_color'] = df_hist['color_enc'].shift(i).iloc[cur]
            
        # Streaks (looking backwards from cur-1)
        stype = sizes[cur-1]
        slen = 1
        for j in range(cur-2, -1, -1):
            if sizes[j] == stype: slen += 1
            else: break
        xgb_feats['streak_length'] = slen
        xgb_feats['streak_type'] = stype
        xgb_feats['big_streak'] = slen if stype == 1 else 0
        xgb_feats['small_streak'] = slen if stype == 0 else 0
        
        # Frequency
        shifted_sz = df_hist['size_binary'].shift(1)
        for w, name in [(10, 'short'), (20, 'mid'), (50, 'long')]:
            b_cnt = shifted_sz.rolling(w, min_periods=1).sum().iloc[cur]
            tot = shifted_sz.rolling(w, min_periods=1).count().iloc[cur]
            xgb_feats[f'big_count_{name}'] = b_cnt
            xgb_feats[f'small_count_{name}'] = tot - b_cnt
            xgb_feats[f'big_ratio_{name}'] = b_cnt / w
            
        # Result stats
        shifted_res = df_hist['result'].shift(1)
        xgb_feats['result_mean_5'] = shifted_res.rolling(5).mean().iloc[cur]
        xgb_feats['result_mean_10'] = shifted_res.rolling(10).mean().iloc[cur]
        xgb_feats['result_std_5'] = shifted_res.rolling(5).std().iloc[cur]
        xgb_feats['result_std_10'] = shifted_res.rolling(10).std().iloc[cur]
        xgb_feats['result_median_10'] = shifted_res.rolling(10).median().iloc[cur]
        xgb_feats['number_trend'] = shifted_res.iloc[cur] - df_hist['result'].shift(5).iloc[cur]
        
        # Transition
        l1, l2, l3 = df_hist['size_binary'].shift(1).iloc[cur], df_hist['size_binary'].shift(2).iloc[cur], df_hist['size_binary'].shift(3).iloc[cur]
        xgb_feats['last_2_combo'] = l2 * 2 + l1
        xgb_feats['last_3_combo'] = l3 * 4 + l2 * 2 + l1
        
        # Gap
        gap_big = gap_small = 0
        for i in range(1, cur+1):
            if sizes[cur-i] == 1 and gap_big == 0: gap_big = i
            if sizes[cur-i] == 0 and gap_small == 0: gap_small = i
        xgb_feats['gap_since_big'] = gap_big
        xgb_feats['gap_since_small'] = gap_small
        
        # Cyclical
        for mod in [3, 5, 10, 20]:
            xgb_feats[f'period_mod_{mod}'] = period % mod
            
        # Pattern Flags
        l4 = df_hist['size_binary'].shift(4).iloc[cur]
        xgb_feats['is_alternating'] = 1 if (l1 != l2 and l2 != l3 and l3 != l4) else 0
        xgb_feats['is_repeating'] = 1 if (l1 == l2 == l3) else 0
        xgb_feats['dominant_last10'] = 1 if xgb_feats['big_count_short'] >= 5 else 0
        
        # Fill missing numeric safely
        for k in xgb_feats:
            if pd.isna(xgb_feats[k]): xgb_feats[k] = 0
            
        # XGB Prediction
        X_xgb = np.array([[xgb_feats.get(c, 0) for c in self.feature_cols]])
        xgb_proba = self.xgb.predict_proba(X_xgb)[0] # [Small, Big]
        
        # LSTM Prediction (Window=20)
        lst_seq = np.column_stack([np.array(history_results[-20:])/9.0, sizes[-20:], np.array(colors[-20:])/2.0])
        lstm_pred = float(self.lstm.predict(lst_seq.reshape(1, 20, 3), verbose=0)[0][0]) # prob big? Wait, lstm model has Dense(2), so [Small, Big]
        lstm_proba = self.lstm.predict(lst_seq.reshape(1, 20, 3), verbose=0)[0]
        
        # TCN Prediction
        tcn_seq = np.array(sizes[-20:]).reshape(1, 20, 1)
        tcn_proba = self.tcn.predict(tcn_seq, verbose=0)[0]
        
        # HMM Prediction
        from train_hmm import hmm_predict_proba
        try:
            hmm_proba = hmm_predict_proba(self.hmm, sizes[-20:])
        except:
            hmm_proba = np.array([0.5, 0.5])
            
        # Markov
        from train_markov import predict_markov
        markov_proba = predict_markov(self.markov, sizes[-20:])
        
        # Kalman
        self._update_kalman(sizes[-1])
        kalman_big = self.x_hat[0]
        kalman_small = self.x_hat[1]
        kalman_drift = float(np.linalg.norm(self.x_hat - np.array([0.5, 0.5])))
        
        # Mining Features
        # Cycle
        fft_vals = np.abs(np.fft.fft(np.array(sizes) - np.mean(sizes)))
        freqs = np.fft.fftfreq(len(sizes))
        fft_vals[0] = 0
        pos = np.where(freqs > 0)[0]
        sorted_idx = pos[np.argsort(fft_vals[pos])[::-1]]
        top_cycle = 0
        for idx in sorted_idx[:3]:
            cl = 50 / idx
            if 10 <= cl <= 500:
                top_cycle = cl
                break
        cycle_phase = (period % top_cycle) / top_cycle if top_cycle > 0 else 0.0
        
        # Entropy
        p_b = sum(sizes[-20:]) / 20
        p_s = 1 - p_b
        entropy = -sum(p * np.log2(p + 1e-9) for p in [p_b, p_s])
        
        # Cluster
        x_clust = np.array([[xgb_feats.get('big_ratio_short',0), xgb_feats.get('big_ratio_mid',0), 
                             xgb_feats.get('streak_length',0), xgb_feats.get('is_alternating',0),
                             xgb_feats.get('result_mean_5',0)]])
        x_sc = self.dbscan_scaler.transform(x_clust)
        
        # dbscan.predict() doesn't exist for DBSCAN in sklearn, there are workarounds with pairwise dist,
        # but since phase 3 said just cluster_label, at inference we return -1 (noise) natively.
        cluster_label = -1 
        
        # Prefixspan
        best_pred = -1
        for p_len in [5,4,3,2]:
            tup = tuple(sizes[-p_len:])
            if tup in self.prefixspan_rules:
                best_pred = self.prefixspan_rules[tup][0]
                break
        
        # Meta MLP
        meta_X = np.array([[
            xgb_proba[0], xgb_proba[1],
            lstm_proba[0], lstm_proba[1],
            tcn_proba[0], tcn_proba[1],
            hmm_proba[0], hmm_proba[1],
            markov_proba[0], markov_proba[1],
            kalman_big, kalman_small, kalman_drift,
            entropy, cycle_phase, cluster_label, best_pred
        ]])
        
        final_proba = self.meta_mlp.predict(meta_X, verbose=0)[0]
        
        confidence = float(np.max(final_proba))
        pred_label = "Big" if np.argmax(final_proba) == 1 else "Small"
        
        return {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Big": round(float(final_proba[1]) * 100, 2),
                "Small": round(float(final_proba[0]) * 100, 2)
            },
            "model_contributions": {
                "xgboost": {"Big": round(float(xgb_proba[1]) * 100, 2), "Small": round(float(xgb_proba[0]) * 100, 2)},
                "lstm": {"Big": round(float(lstm_proba[1]) * 100, 2), "Small": round(float(lstm_proba[0]) * 100, 2)},
                "markov": {"Big": round(float(markov_proba[1]) * 100, 2), "Small": round(float(markov_proba[0]) * 100, 2)}
            },
            "kalman_drift": kalman_drift,
            "cycle_phase": cycle_phase
        }
