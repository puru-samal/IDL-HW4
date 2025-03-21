import json
import pandas as pd
import numpy as np

# Read JSON data
with open('hw4p2_sol.json', 'r') as f:
    data = json.load(f)

# Create base DataFrame
df = pd.DataFrame({
    'id': range(len(data)),
    'transcription': data
})

# Generate Usage column with 50-50 split between Public and Private
np.random.seed(11785)  # for reproducibility
n_samples = len(df)
usage_values = np.random.choice(
    ['Public', 'Private'],
    size=n_samples,
    p=[0.4, 0.6]  # 40% Public, 60% Private
)
df['Usage'] = usage_values

# Reorder columns
df = df[['id', 'Usage', 'transcription']]
    
# Save to CSV
df.to_csv('hw4p2_sol.csv', index=False)


