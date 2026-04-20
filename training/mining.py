import numpy as np
import pandas as pd
import os
import joblib
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')

def compute_entropy_features(df):
    """FUNCTION 1: Rolling Shannon entropy on size_binary column."""
    def calc_entropy(window_series):
        if len(window_series) == 0:
            return 0.0
        p_big = window_series.sum() / len(window_series)
        p_small = 1.0 - p_big
        return -sum(p * np.log2(p + 1e-9) for p in [p_big, p_small])
        
    df['entropy_last_20'] = df['size_binary'].rolling(20, min_periods=1).apply(calc_entropy).fillna(0)
    df['entropy_last_50'] = df['size_binary'].rolling(50, min_periods=1).apply(calc_entropy).fillna(0)
    
    # Fill first N rows with 0 is handled implicitly by filling early NaNs or we can explicitly zero them out.
    # Instruction says: Fill first N rows with 0. Let's explicitly set first 19/49 rows if needed, 
    # but the above rolling(min_periods=1) handles prefix nicely. To strictly follow:
    df.loc[:19, 'entropy_last_20'] = 0.0
    df.loc[:49, 'entropy_last_50'] = 0.0
    
    global_entropy = calc_entropy(df['size_binary'])
    # Randomness % is (entropy / max_entropy) * 100 where max for binary is 1.0
    print(f"Global Entropy score: {global_entropy:.4f} ({global_entropy * 100:.2f}% randomness)")
    return df

def compute_fft_features(df):
    """FUNCTION 2: FFT on size_binary sequence to find dominant frequency peaks."""
    N = len(df)
    size_vals = df['size_binary'].values
    cycle_phase = np.zeros(N)
    dominant_cycle_arr = np.zeros(N, dtype=int)
    
    if N > 50:
        # We do a global FFT just to find the general cycle, applying it backward or to the whole series.
        # "If strongest cycle is between 10 and 500 rounds... cycle_phase = (row_idx % cycle_len) / cycle_len"
        fft_vals = np.abs(np.fft.fft(size_vals - np.mean(size_vals)))
        freqs = np.fft.fftfreq(N)
        
        # skip index 0
        fft_vals[0] = 0
        # only look at positive frequencies
        pos_indices = np.where(freqs > 0)[0]
        
        # Sort indices by fft magnitude descending
        sorted_pos_idx = pos_indices[np.argsort(fft_vals[pos_indices])[::-1]]
        
        top_cycle_length = 0
        for idx in sorted_pos_idx[:3]:
            cycle_length = N / idx
            if 10 <= cycle_length <= 500:
                top_cycle_length = cycle_length
                break
                
        if top_cycle_length > 0:
            dom_int = int(top_cycle_length)
            for i in range(N):
                cycle_phase[i] = (i % top_cycle_length) / top_cycle_length
                dominant_cycle_arr[i] = dom_int
            print(f"Dominant cycle: every {dom_int} rounds")
        else:
            print("No cycle found")
    else:
        print("No cycle found (insufficient data)")
        
    df['cycle_phase'] = cycle_phase
    df['dominant_cycle'] = dominant_cycle_arr
    return df

def compute_prefixspan_features(df):
    """FUNCTION 3: Build pattern lookup from size_binary sequence."""
    size_seq = df['size_binary'].values
    N = len(size_seq)
    
    pattern_counts = defaultdict(lambda: {0: 0, 1: 0})
    
    for pat_len in [2, 3, 4, 5]:
        for i in range(N - pat_len):
            window = tuple(size_seq[i : i + pat_len])
            next_val = size_seq[i + pat_len]
            pattern_counts[window][next_val] += 1
            
    # Filter rules
    rules = {}
    for pat, counts in pattern_counts.items():
        total = counts[0] + counts[1]
        if total >= 15:
            conf_0 = counts[0] / total
            conf_1 = counts[1] / total
            if conf_0 >= 0.55:
                rules[pat] = (0, conf_0)
            elif conf_1 >= 0.55:
                rules[pat] = (1, conf_1)
                
    # Save rules
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, 'prefixspan_rules.csv'), 'w') as f:
        f.write("pattern,pred,confidence\n")
        # Sort rules by confidence string representation
        for pat, (pred, conf) in rules.items():
            f.write(f"{str(pat).replace(', ','_').strip('()')},{pred},{conf:.4f}\n")
            
    print(f"Total rules found: {len(rules)}")
    sorted_rules = sorted(rules.items(), key=lambda x: x[1][1], reverse=True)
    for pat, (pred, conf) in sorted_rules[:5]:
        print(f"  Pattern {pat} -> {pred} (Conf: {conf:.2f})")
        
    preds = np.full(N, -1)
    # Apply rules
    for i in range(N):
        best_pred = -1
        # Look backwards for the longest matching pattern among lengths 5,4,3,2
        for pat_len in [5, 4, 3, 2]:
            if i >= pat_len:
                hist_window = tuple(size_seq[i - pat_len : i])
                if hist_window in rules:
                    best_pred = rules[hist_window][0]
                    break
        preds[i] = best_pred
        
    df['prefixspan_pred'] = preds
    return df

def compute_dbscan_features(df):
    """FUNCTION 4: Apply DBSCAN."""
    # Ensure required features exist (if called sequentially properly, they might not exist yet, 
    # since instructed feature_engineering runs AFTER mining. Let's compute inline if missing.)
    
    if 'big_ratio_short' not in df.columns:
        df['big_ratio_short'] = df['size_binary'].rolling(10, min_periods=1).mean().fillna(0)
    
    if 'big_ratio_mid' not in df.columns:
        df['big_ratio_mid'] = df['size_binary'].rolling(20, min_periods=1).mean().fillna(0)
        
    if 'streak_length' not in df.columns:
        # Simple inline streak calc
        streak = np.zeros(len(df))
        s_vals = df['size_binary'].values
        cur_len = 1
        for i in range(1, len(df)):
            if s_vals[i] == s_vals[i-1]:
                cur_len += 1
            else:
                cur_len = 1
            streak[i] = cur_len
        df['streak_length'] = streak
        
    if 'is_alternating' not in df.columns:
        alt = np.zeros(len(df))
        s_vals = df['size_binary'].values
        for i in range(3, len(df)):
            if s_vals[i] != s_vals[i-1] and s_vals[i-1] != s_vals[i-2] and s_vals[i-2] != s_vals[i-3]:
                alt[i] = 1
        df['is_alternating'] = alt
        
    if 'result_mean_5' not in df.columns:
        if 'result' in df.columns:
            df['result_mean_5'] = df['result'].rolling(5, min_periods=1).mean().fillna(0)
        else:
            df['result_mean_5'] = 0

    features_cols = ['big_ratio_short', 'big_ratio_mid', 'streak_length', 'is_alternating', 'result_mean_5']
    X = df[features_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(SAVED_DIR, 'dbscan_scaler.pkl'))
    
    db = DBSCAN(eps=0.4, min_samples=10)
    cluster_labels = db.fit_predict(X_scaled)
    
    joblib.dump(db, os.path.join(SAVED_DIR, 'dbscan_model.pkl'))
    
    df['cluster_label'] = cluster_labels
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"DBSCAN clusters: {n_clusters}, noise points: {n_noise}")
    
    return df

def run_all_mining(df):
    df = compute_entropy_features(df)
    df = compute_fft_features(df)
    df = compute_prefixspan_features(df)
    df = compute_dbscan_features(df)
    return df
