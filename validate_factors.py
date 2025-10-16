import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

conn = sqlite3.connect(DB_PATH)

# Load raw monthly data
df = pd.read_sql("SELECT * FROM factors_monthly", conn)
df_wide = df.pivot(index='date', columns='proxy', values='value')

# Load factor mapping
mapping_df = pd.read_sql("SELECT proxy, category FROM instruments", conn)
conn.close()

# Calculate correlation within each category
category_coherence = {}
for category in mapping_df['category'].unique():
    proxies = mapping_df[mapping_df['category'] == category]['proxy'].tolist()
    available = [p for p in proxies if p in df_wide.columns]
    
    if len(available) > 1:
        corr_matrix = df_wide[available].corr()
        # Average absolute correlation (exclude diagonal)
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        avg_corr = np.abs(corr_matrix.values[mask]).mean()
        category_coherence[category] = avg_corr

# Sort by coherence
coherence_df = pd.DataFrame.from_dict(category_coherence, orient='index', columns=['avg_abs_correlation'])
print(coherence_df.sort_values('avg_abs_correlation'))
