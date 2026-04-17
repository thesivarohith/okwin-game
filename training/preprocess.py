"""
Step 1: Load raw CSV, clean data, encode target.
Input:  data/raw_data.csv (or any CSV with: timestamp, period, result, size, color)
Output: data/clean_data.csv
"""
import pandas as pd
import numpy as np
import os
import sys

# --- Configuration ---
RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented_15k.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')


def load_and_clean(path):
    """Load CSV and perform basic cleaning."""
    print(f"Loading: {path}")
    df = pd.read_csv(path)

    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    print(f"  Raw rows: {len(df)}")

    # Drop rows with missing critical values
    critical_cols = ['period', 'result', 'size', 'color']
    before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing values")

    # Ensure result is integer
    df['result'] = df['result'].astype(int)

    # Sort by period (chronological)
    df.sort_values('period', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def encode(df):
    """Encode target and categorical columns."""
    # Target: size → binary (Big=1, Small=0)
    df['target'] = (df['size'] == 'Big').astype(int)

    # Color encoding
    color_map = {
        'Red': 0, 'Green': 1, 'Violet': 2,
        'Green+Violet': 3, 'Red+Violet': 4
    }
    df['color_encoded'] = df['color'].map(color_map)

    # Handle unmapped colors
    unmapped = df['color_encoded'].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} unmapped color values, filling with -1")
        df['color_encoded'] = df['color_encoded'].fillna(-1).astype(int)
    else:
        df['color_encoded'] = df['color_encoded'].astype(int)

    return df


def main():
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found.")
        print("Place your CSV in data/raw_data.csv or run generate_synthetic.py first.")
        sys.exit(1)

    df = load_and_clean(RAW_PATH)
    df = encode(df)

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(f"  Total rows: {len(df)}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()} (1=Big, 0=Small)")
    print(f"  Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
