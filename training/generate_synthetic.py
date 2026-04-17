"""
Generate synthetic OkWin data for pipeline development.
Replace with real CSV data when available.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
N = 1000

# Generate timestamps (every ~30s going backward)
base_time = datetime(2026, 4, 14, 10, 33, 49)
timestamps = [base_time - timedelta(seconds=30 * i) for i in range(N)]
timestamps.reverse()

# Generate period IDs (incrementing)
base_period = 20260414100050108
periods = [base_period + i for i in range(N)]

# Generate results 0-9 with slight bias patterns
results = np.random.choice(range(10), size=N, p=[
    0.08, 0.09, 0.10, 0.10, 0.11, 0.11, 0.10, 0.10, 0.09, 0.12
])

# Derive size from result
sizes = ['Big' if r >= 5 else 'Small' for r in results]

# Generate colors with compound types
color_map = {
    0: 'Red+Violet', 1: 'Green', 2: 'Red', 3: 'Green',
    4: 'Red', 5: 'Green+Violet', 6: 'Red', 7: 'Green',
    8: 'Red', 9: 'Green'
}
colors = [color_map[r] for r in results]

df = pd.DataFrame({
    'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
    'period': periods,
    'result': results,
    'size': sizes,
    'color': colors
})

df.to_csv('data/raw_data.csv', index=False)
print(f"Generated {len(df)} rows → data/raw_data.csv")
print(f"Size distribution: {df['size'].value_counts().to_dict()}")
print(f"Color distribution: {df['color'].value_counts().to_dict()}")
print(df.head(10))
