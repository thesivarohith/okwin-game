import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from preprocess import load_clean_data
from mining import run_all_mining

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')
FEATURES_CSV_PATH = os.path.join(PROCESSED_DIR, 'features.csv')
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_columns.json')

def build_features():
    df = load_clean_data()
    
    # Run Layer 0 Data Mining (returns df with new cols)
    df = run_all_mining(df)
    
    # Core feature engineering
    # === LAG FEATURES ===
    for i in range(1, 11):
        df[f'last_{i}_result'] = df['result'].shift(i)
        df[f'last_{i}_size'] = df['size_binary'].shift(i)
        
    for i in range(1, 6):
        df[f'last_{i}_color'] = df['color_enc'].shift(i)
        
    # === STREAK FEATURES ===
    streak_length = np.zeros(len(df))
    streak_type = np.zeros(len(df))
    big_streak = np.zeros(len(df))
    small_streak = np.zeros(len(df))
    
    s_vals = df['size_binary'].values
    # Note: we need streak ending at i-1 to avoid data leakage for prediction target i
    for i in range(1, len(df)):
        # Check consecutive identical values backwards starting from i-1
        count = 1
        stype = s_vals[i-1]
        for j in range(i-2, -1, -1):
            if s_vals[j] == stype:
                count += 1
            else:
                break
        streak_length[i] = count
        streak_type[i] = stype
        if stype == 1:
            big_streak[i] = count
            small_streak[i] = 0
        else:
            big_streak[i] = 0
            small_streak[i] = count
            
    # If no history yet
    streak_length[0] = 0
    streak_type[0] = -1
    
    df['streak_length'] = streak_length
    df['streak_type'] = streak_type
    df['big_streak'] = big_streak
    df['small_streak'] = small_streak
    
    # === FREQUENCY FEATURES ===
    for w, name in [(10, 'short'), (20, 'mid'), (50, 'long')]:
        # Using shift(1) to avoid data leakage
        shifted_size = df['size_binary'].shift(1)
        df[f'big_count_{name}'] = shifted_size.rolling(w, min_periods=1).sum()
        df[f'small_count_{name}'] = shifted_size.rolling(w, min_periods=1).count() - df[f'big_count_{name}']
        df[f'big_ratio_{name}'] = df[f'big_count_{name}'] / w
        
    # === RESULT NUMBER FEATURES ===
    shifted_result = df['result'].shift(1)
    df['result_mean_5'] = shifted_result.rolling(5, min_periods=1).mean()
    df['result_mean_10'] = shifted_result.rolling(10, min_periods=1).mean()
    df['result_std_5'] = shifted_result.rolling(5, min_periods=1).std()
    df['result_std_10'] = shifted_result.rolling(10, min_periods=1).std()
    df['result_median_10'] = shifted_result.rolling(10, min_periods=1).median()
    df['number_trend'] = shifted_result - df['result'].shift(5)
    
    # === TRANSITION FEATURES ===
    df['last_2_combo'] = df['last_2_size'] * 2 + df['last_1_size']
    df['last_3_combo'] = df['last_3_size'] * 4 + df['last_2_size'] * 2 + df['last_1_size']
    
    # === GAP FEATURES ===
    gap_big = np.zeros(len(df))
    gap_small = np.zeros(len(df))
    for i in range(1, len(df)):
        if s_vals[i-1] == 1:
            gap_big[i] = 0
            gap_small[i] = gap_small[i-1] + 1
        else:
            gap_small[i] = 0
            gap_big[i] = gap_big[i-1] + 1
    df['gap_since_big'] = gap_big
    df['gap_since_small'] = gap_small
    
    # === CYCLICAL FEATURES ===
    for mod in [3, 5, 10, 20]:
        df[f'period_mod_{mod}'] = df['period'] % mod
        
    # === PATTERN FLAGS ===
    df['is_alternating'] = 0
    df['is_repeating'] = 0
    
    for i in range(4, len(df)):
        l1, l2, l3, l4 = s_vals[i-1], s_vals[i-2], s_vals[i-3], s_vals[i-4]
        if l1 != l2 and l2 != l3 and l3 != l4:
            df.loc[i, 'is_alternating'] = 1
            
    for i in range(3, len(df)):
        l1, l2, l3 = s_vals[i-1], s_vals[i-2], s_vals[i-3]
        if l1 == l2 == l3:
            df.loc[i, 'is_repeating'] = 1
            
    df['dominant_last10'] = (df['big_count_short'] >= 5).astype(int)
    
    # TARGET
    # The target is df['size_binary'] at index i (automatically set by definition).
    # All features are computed looking at i-k, so we predict y[i] from row i's features.
    
    # Drop first 50 rows
    df = df.iloc[50:]
    df = df.fillna(0)
    
    # Feature columns exclusion (don't train on timestamp, period, result, size, color, size_binary which is target)
    exclude_cols = ['timestamp', 'period', 'result', 'size', 'color', 'size_binary']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['size_binary']
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(FEATURES_CSV_PATH, index=False)
    
    os.makedirs(os.path.dirname(FEATURE_COLS_PATH), exist_ok=True)
    with open(FEATURE_COLS_PATH, 'w') as f:
        json.dump(feature_cols, f)
        
    print(f"Total features: {len(feature_cols)}, Total rows: {len(df)}")
    print(f"NaN count: {df.isna().sum().sum()}")
    
    return X, y

if __name__ == "__main__":
    build_features()
