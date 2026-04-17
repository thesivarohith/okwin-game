"""
Step 2: Feature engineering for Big/Small prediction.
Input:  data/clean_data.csv
Output: data/features.csv
"""
import pandas as pd
import numpy as np
import os
import sys

# --- Configuration ---
CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')

# Lookback windows
HISTORY_DEPTH = 10       # last N results/sizes/colors
SHORT_WINDOW = 10
MID_WINDOW = 20
LONG_WINDOW = 50
MIN_ROWS_NEEDED = LONG_WINDOW  # skip first N rows (not enough history)


def add_recent_history(df):
    """Last N result values, size values, and color values as features."""
    for i in range(1, HISTORY_DEPTH + 1):
        df[f'last_{i}_result'] = df['result'].shift(i)
        df[f'last_{i}_size'] = df['target'].shift(i)
    for i in range(1, 6):  # only 5 for color
        df[f'last_{i}_color'] = df['color_encoded'].shift(i)
    return df


def add_streak_features(df):
    """Current streak length and type (Big=1, Small=0)."""
    targets = df['target'].values
    streak_lengths = np.zeros(len(targets), dtype=int)
    streak_types = np.zeros(len(targets), dtype=int)

    for i in range(1, len(targets)):
        if targets[i - 1] == targets[i - 2] if i >= 2 else True:
            # Count backwards from i-1
            count = 1
            val = targets[i - 1]
            j = i - 2
            while j >= 0 and targets[j] == val:
                count += 1
                j -= 1
            streak_lengths[i] = count
            streak_types[i] = val
        else:
            streak_lengths[i] = 1
            streak_types[i] = targets[i - 1]

    df['streak_length'] = streak_lengths
    df['streak_type'] = streak_types
    return df


def add_frequency_features(df):
    """Big/Small frequency counts over short/mid/long windows."""
    for window, label in [(SHORT_WINDOW, 'short'), (MID_WINDOW, 'mid'), (LONG_WINDOW, 'long')]:
        df[f'big_count_{label}'] = df['target'].rolling(window=window, min_periods=1).sum().shift(1)
        df[f'small_count_{label}'] = window - df[f'big_count_{label}']
        df[f'big_ratio_{label}'] = df[f'big_count_{label}'] / window
    return df


def add_transition_features(df):
    """Last 2 and 3 size combo encodings."""
    # Last 2 combo: 4 possibilities (BB=0, BS=1, SB=2, SS=3)
    t = df['target'].values
    last2 = np.full(len(t), -1, dtype=int)
    last3 = np.full(len(t), -1, dtype=int)

    for i in range(2, len(t)):
        last2[i] = t[i - 2] * 2 + t[i - 1]
    for i in range(3, len(t)):
        last3[i] = t[i - 3] * 4 + t[i - 2] * 2 + t[i - 1]

    df['last_2_combo'] = last2
    df['last_3_combo'] = last3
    return df


def add_gap_features(df):
    """Rounds since last Big and last Small."""
    targets = df['target'].values
    gap_big = np.zeros(len(targets), dtype=int)
    gap_small = np.zeros(len(targets), dtype=int)

    last_big = -1
    last_small = -1

    for i in range(len(targets)):
        if i > 0:
            gap_big[i] = (i - last_big) if last_big >= 0 else i
            gap_small[i] = (i - last_small) if last_small >= 0 else i
        if targets[i] == 1:
            last_big = i
        else:
            last_small = i

    df['gap_since_big'] = gap_big
    df['gap_since_small'] = gap_small
    return df


def add_cyclical_features(df):
    """Period-based modulo features to detect algorithmic cycles."""
    # Convert period to numeric, coercion will turn 'SYNTH_0' to NaN
    # We'll use the index + some offset as a fallback for synthetic periods
    p_numeric = pd.to_numeric(df['period'], errors='coerce')
    is_synth = p_numeric.isna()
    
    # For real periods, use the last few digits to avoid overflow issues
    # For synthetic, use the index
    period_val = p_numeric.fillna(0).astype(np.int64) % 1000000
    synth_val = df.index.to_series()
    
    for mod in [3, 5, 10, 20]:
        df[f'period_mod_{mod}'] = np.where(is_synth, synth_val % mod, period_val % mod)
    return df


def add_pattern_flags(df):
    """Meta-pattern detection flags."""
    t = df['target'].values

    # Is alternating (last 4: BSBS or SBSB)
    is_alt = np.zeros(len(t), dtype=int)
    for i in range(4, len(t)):
        if (t[i-1] != t[i-2]) and (t[i-2] != t[i-3]) and (t[i-3] != t[i-4]):
            is_alt[i] = 1
    df['is_alternating'] = is_alt

    # Is repeating (last 3 same)
    is_rep = np.zeros(len(t), dtype=int)
    for i in range(3, len(t)):
        if t[i-1] == t[i-2] == t[i-3]:
            is_rep[i] = 1
    df['is_repeating'] = is_rep

    # Dominant in last 10
    df['dominant_last10'] = (df['target'].rolling(10, min_periods=1).sum().shift(1) >= 6).astype(int)

    return df


def add_result_stats(df):
    """Statistical features from the raw result values."""
    for w in [5, 10]:
        df[f'result_mean_{w}'] = df['result'].rolling(window=w, min_periods=1).mean().shift(1)
        df[f'result_std_{w}'] = df['result'].rolling(window=w, min_periods=1).std().shift(1).fillna(0)
        df[f'result_median_{w}'] = df['result'].rolling(window=w, min_periods=1).median().shift(1)
    return df


def main():
    if not os.path.exists(CLEAN_PATH):
        print(f"ERROR: {CLEAN_PATH} not found. Run preprocess.py first.")
        sys.exit(1)

    print(f"Loading: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    print(f"  Rows: {len(df)}")

    # Add all feature groups
    print("Engineering features...")
    df = add_recent_history(df)
    df = add_streak_features(df)
    df = add_frequency_features(df)
    df = add_transition_features(df)
    df = add_gap_features(df)
    df = add_cyclical_features(df)
    df = add_pattern_flags(df)
    df = add_result_stats(df)

    # Drop rows without enough history
    before = len(df)
    df = df.iloc[MIN_ROWS_NEEDED:].reset_index(drop=True)
    print(f"  Dropped first {MIN_ROWS_NEEDED} rows (insufficient history)")
    print(f"  Remaining rows: {len(df)}")

    # Fill any remaining NaN with 0
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} remaining NaN values with 0")
        df = df.fillna(0)

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Report
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'period', 'result', 'size', 'color', 'color_encoded', 'target']]
    print(f"\nSaved: {OUT_PATH}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
