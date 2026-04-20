import pandas as pd
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'okwin_30s_dataset.csv')
CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clean_data.csv')

def load_clean_data() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    
    # Cleaning
    # 1. Sort ascending by period
    df = df.sort_values(by='period', ascending=True)
    
    # 2. Remove duplicate periods
    df = df.drop_duplicates(subset=['period'], keep='first')
    
    # 3. Normalize size column & drop invalid
    df = df[df['size'].isin(['Big', 'Small'])]
    
    # 4. Normalize color
    df['color'] = df['color'].replace({'Red+Violet': 'Violet', 'Green+Violet': 'Violet'})
    
    # 5. Normalize result (0-9)
    # Ensure it's integer and valid
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df = df.dropna(subset=['result'])
    df['result'] = df['result'].astype(int)
    df = df[df['result'].between(0, 9)]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Encoding
    df['size_binary'] = df['size'].map({'Big': 1, 'Small': 0})
    df['color_enc'] = df['color'].map({'Red': 0, 'Green': 1, 'Violet': 2})
    
    # Print metrics
    total_rows = len(df)
    big_count = df['size_binary'].sum()
    small_count = total_rows - big_count
    
    big_pct = (big_count / total_rows * 100) if total_rows > 0 else 0
    small_pct = (small_count / total_rows * 100) if total_rows > 0 else 0
    
    first_ts = df['timestamp'].iloc[0] if total_rows > 0 else "N/A"
    last_ts = df['timestamp'].iloc[-1] if total_rows > 0 else "N/A"
    
    print(f"Total rows after cleaning: {total_rows}")
    print(f"Big count: {big_count} ({big_pct:.2f}%) / Small count: {small_count} ({small_pct:.2f}%)")
    print(f"Date range: {first_ts} to {last_ts}")
    
    # Save
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    
    return df

if __name__ == "__main__":
    load_clean_data()
